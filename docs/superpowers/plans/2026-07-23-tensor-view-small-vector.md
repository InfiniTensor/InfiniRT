# TensorView Inline Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `TensorView`'s two heap-backed metadata vectors with a narrow in-tree inline container, then choose inline capacity 4 or 8 from allocation, latency, and object-size evidence.

**Architecture:** Add a header-only `infini::rt::detail::SmallVector<T, N>` restricted to trivial element types. Keep `TensorView`'s owned metadata and existing constructors, but construct generic ranges through iterators. Benchmark the post-#33 vector implementation, capacity 4, and capacity 8 from independent source trees before retaining exactly one source constant.

**Tech Stack:** C++17, CMake/CTest, the existing InfiniRT performance runner, clang-format 21, Linux allocation instrumentation, Docker, and the `accelerator-dev/nvidia:latest` image.

---

The approved design at `docs/superpowers/specs/2026-07-23-tensor-view-small-vector-design.md` is the source of truth. Do not add a version or `SOVERSION` change, a public capacity option, a third-party container, borrowed metadata, or unrelated `TensorView` behavior changes.

Because CMake writes generated public headers into the source tree, the vector baseline, capacity-4 candidate, capacity-8 candidate, CPU validation, and NVIDIA validation must use independent source copies. Reusing one source tree with multiple build directories is invalid for this work.

## Linux TDD Execution Protocol

`test_tensor_view_allocations` exists only on Linux. Run every build/test command in Tasks 1 through 6 inside `accelerator-dev/nvidia:latest` on `nvidia`, never directly in the Windows worktree.

Before each red or green cycle, publish the current working tree, including uncommitted tests, to a new remote source directory. Set `$SnapshotName` to exactly one of `task1-baseline`, `task2-red`, `task3-green`, `task4-red`, `task5-cap4`, `task6-red`, or `task6-cap8`:

```powershell
$SnapshotName = 'task2-red'
$AllowedSnapshots = @(
  'task1-baseline',
  'task2-red',
  'task3-green',
  'task4-red',
  'task5-cap4',
  'task6-red',
  'task6-cap8'
)
if ($SnapshotName -notin $AllowedSnapshots) {
  throw "Unexpected TensorView snapshot name: $SnapshotName"
}
$ArchivePath = Join-Path $env:TEMP "infinirt-tv-$SnapshotName.tar.gz"
tar -czf $ArchivePath --exclude=.git --exclude=generated --exclude='build-*' .
ssh nvidia "test ! -e /tmp/infinirt-tv-$SnapshotName && mkdir /tmp/infinirt-tv-$SnapshotName"
scp $ArchivePath "nvidia:/tmp/infinirt-tv-$SnapshotName.tar.gz"
ssh nvidia "tar -xzf /tmp/infinirt-tv-$SnapshotName.tar.gz -C /tmp/infinirt-tv-$SnapshotName"
```

Run that cycle's commands from a container shell mounted on the matching directory:

```powershell
ssh -t nvidia "docker run --rm -it -v /tmp/infinirt-tv-$SnapshotName:/workspace/InfiniRT -w /workspace/InfiniRT accelerator-dev/nvidia:latest bash"
```

Each snapshot is created once and never overwritten. The task steps below name the required snapshot before each command block.

## Task 1: Freeze the Expanded Vector Baseline

**Files:**

- Modify: `tests/performance/perf_tensor_view.cc`

- [ ] **Step 1: Replace the narrow benchmark set with the common 58-result matrix**

Register these nine benchmark names at ranks 1, 2, 4, 5, 8, and 9:

```text
perf_tensor_view.construct_lvalue_explicit
perf_tensor_view.construct_rvalue_explicit
perf_tensor_view.construct_default_strides
perf_tensor_view.construct_initializer_list
perf_tensor_view.construct_tensor_like
perf_tensor_view.copy
perf_tensor_view.operator_index
perf_tensor_view.pass_by_value
perf_tensor_view.numel
```

Keep four rank-2 controls:

```text
perf_tensor_view.transpose
perf_tensor_view.is_contiguous_true
perf_tensor_view.is_contiguous_false
perf_tensor_view.hash
```

Use a vector-backed fixture so the generic path remains independent of the candidate metadata type:

```cpp
struct VectorTensorLike {
  void* data_value;

  std::vector<std::size_t> shape_value;

  DataType dtype_value;

  Device device_value;

  std::vector<std::ptrdiff_t> strides_value;

  void* data() const { return data_value; }

  const std::vector<std::size_t>& shape() const { return shape_value; }

  DataType dtype() const { return dtype_value; }

  Device device() const { return device_value; }

  const std::vector<std::ptrdiff_t>& strides() const {
    return strides_value;
  }
};
```

Prevent the by-value call from being optimized into the caller:

```cpp
#if defined(_MSC_VER)
#define INFINI_RT_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define INFINI_RT_NOINLINE __attribute__((noinline))
#else
#define INFINI_RT_NOINLINE
#endif

INFINI_RT_NOINLINE std::size_t ConsumeTensorView(TensorView tensor) {
  perf::DoNotOptimize(tensor.data());
  return tensor.ndim() + tensor.size(0) +
         static_cast<std::size_t>(tensor.stride(0));
}
```

Emit layout information to `stderr` so it never becomes a duplicate JSON key:

```cpp
std::cerr << "sizeof(TensorView)=" << sizeof(TensorView)
          << " sizeof(Shape)=" << sizeof(TensorView::Shape)
          << " sizeof(Strides)=" << sizeof(TensorView::Strides) << '\n';
```

Use `__has_include(<infini/rt/detail/common/small_vector.h>)` only for candidate-only `SmallVector<Size, 4>` and `SmallVector<Size, 8>` size lines. The vector baseline must compile when that generated detail header does not exist.

All benchmark input lifetimes are fixed. Preconstruct lvalue metadata, TensorLike metadata, and the source view outside the measured closure. Reconstruct exact rvalue metadata inside every iteration; never repeatedly move one preconstructed object:

```cpp
template <std::size_t Rank>
void RunRankBenchmarks(float* data, const Device& device) {
  const auto shape_values = MakeShape<Rank>();
  const auto stride_values = MakeStrides(shape_values);
  const TensorView::Shape shape{shape_values.begin(), shape_values.end()};
  const TensorView::Strides strides{stride_values.begin(),
                                    stride_values.end()};
  const VectorTensorLike tensor_like{
      data,
      {shape_values.begin(), shape_values.end()},
      DataType::kFloat32,
      device,
      {stride_values.begin(), stride_values.end()}};
  const TensorView source{data, shape, DataType::kFloat32, device, strides};
  const auto params =
      std::vector<perf::Param>{perf::NumberParam("ndim", Rank)};

  perf::RunBenchmark(
      "perf_tensor_view.construct_lvalue_explicit", params, kIterations, "ns",
      [&] {
        TensorView tensor{data, shape, DataType::kFloat32, device, strides};
        perf::DoNotOptimize(tensor);
      });

  perf::RunBenchmark(
      "perf_tensor_view.construct_rvalue_explicit", params, kIterations, "ns",
      [&] {
        TensorView tensor{
            data,
            TensorView::Shape{shape_values.begin(), shape_values.end()},
            DataType::kFloat32,
            device,
            TensorView::Strides{stride_values.begin(), stride_values.end()}};
        perf::DoNotOptimize(tensor);
      });

  perf::RunBenchmark(
      "perf_tensor_view.construct_default_strides", params, kIterations, "ns",
      [&] {
        TensorView tensor{
            data,
            TensorView::Shape{shape_values.begin(), shape_values.end()},
            DataType::kFloat32, device};
        perf::DoNotOptimize(tensor);
      });

  perf::RunBenchmark(
      "perf_tensor_view.construct_tensor_like", params, kIterations, "ns",
      [&] {
        TensorView tensor{tensor_like};
        perf::DoNotOptimize(tensor);
      });

  perf::RunBenchmark("perf_tensor_view.copy", params, kIterations, "ns", [&] {
    TensorView tensor{source};
    perf::DoNotOptimize(tensor);
  });

  perf::RunBenchmark(
      "perf_tensor_view.operator_index", params, kIterations, "ns", [&] {
        const auto tensor = source[0];
        perf::DoNotOptimize(tensor);
      });

  perf::RunBenchmark(
      "perf_tensor_view.pass_by_value", params, kIterations, "ns", [&] {
        const auto value = ConsumeTensorView(source);
        perf::DoNotOptimize(value);
      });

  perf::RunBenchmark("perf_tensor_view.numel", params, kIterations, "ns", [&] {
    const auto value = source.numel();
    perf::DoNotOptimize(value);
  });
}
```

`MakeShape<Rank>` returns an all-2 shape and `MakeStrides` returns its contiguous strides. Register initializer-list construction separately inside the measured closure with these exact pairs:

| Rank | Shape | Strides |
| ---: | --- | --- |
| 1 | `{2}` | `{1}` |
| 2 | `{2, 2}` | `{2, 1}` |
| 4 | `{2, 2, 2, 2}` | `{8, 4, 2, 1}` |
| 5 | `{2, 2, 2, 2, 2}` | `{16, 8, 4, 2, 1}` |
| 8 | `{2, 2, 2, 2, 2, 2, 2, 2}` | `{128, 64, 32, 16, 8, 4, 2, 1}` |
| 9 | `{2, 2, 2, 2, 2, 2, 2, 2, 2}` | `{256, 128, 64, 32, 16, 8, 4, 2, 1}` |

The initializer-list benchmark calls a rank-specific helper that returns `TensorView{data, shape_list, DataType::kFloat32, device, stride_list}`. Rank-2 transpose and contiguity/hash controls reuse the preconstructed rank-2 source.

- [ ] **Step 2: Build and run the unchanged vector implementation**

Publish and enter snapshot `task1-baseline`, then run:

```bash
cmake -S . -B build-perf-baseline \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=OFF \
  -DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON
cmake --build build-perf-baseline --target perf_tensor_view -j2
python3 scripts/run_performance_tests.py \
  --build-dir build-perf-baseline \
  --backend cpu \
  --test perf_tensor_view \
  --output /tmp/tensor-view-baseline-smoke.json
```

Expected output:

```text
wrote 58 performance results to /tmp/tensor-view-baseline-smoke.json
```

Check that the array contains 58 unique backend, benchmark, params, and unit keys.

- [ ] **Step 3: Commit the common harness and record the baseline ref**

```bash
git add tests/performance/perf_tensor_view.cc
git commit -m "perf: expand TensorView metadata benchmarks"
git update-ref refs/benchmarks/tensor-view/baseline HEAD
```

The production `src/` tree at this ref must still match commit `656b941938a1f2b23604a6165b0faa12554db6dc`.

## Task 2: Specify SmallVector Before Implementing It

**Files:**

- Create: `tests/test_small_vector.cc`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Register the focused test**

Add immediately after `test_core`:

```cmake
add_infini_rt_test(test_small_vector test_small_vector.cc)
```

- [ ] **Step 2: Add compile-time and constructor coverage**

Use `infini::rt::detail::SmallVector` from `common/small_vector.h`. Cover all required constructors, accessors, iterators, and equality in both directions with `std::vector`:

```cpp
using Inline4 = infini::rt::detail::SmallVector<std::size_t, 4>;
using Inline8 = infini::rt::detail::SmallVector<std::size_t, 8>;

static_assert(std::is_copy_constructible_v<Inline4>);
static_assert(std::is_move_constructible_v<Inline4>);
static_assert(std::is_copy_assignable_v<Inline4>);
static_assert(std::is_move_assignable_v<Inline4>);

Inline4 empty;
context->Expect(empty.empty(), "A default SmallVector should be empty.");
context->ExpectEqual(empty.capacity(), std::size_t{4},
                     "A default SmallVector should expose inline capacity.");

Inline4 inline_values{1, 2, 3, 4};
Inline4 overflow_values{1, 2, 3, 4, 5};
context->ExpectEqual(inline_values.capacity(), std::size_t{4},
                     "Inline values should keep inline storage.");
context->Expect(overflow_values.capacity() >= 5,
                "Overflow values should use sufficient heap storage.");
```

- [ ] **Step 3: Add mutation and Rule-of-Five coverage**

Exercise `clear`, geometric `reserve`, grow/shrink `resize`, repeated `push_back`, and `assign`. Explicitly test:

```cpp
Inline4 self_assigned{1, 2, 3};
self_assigned = self_assigned;
context->Expect(self_assigned == std::vector<std::size_t>({1, 2, 3}),
                "Self-assignment should preserve values.");

Inline4 heap_to_inline{1, 2, 3, 4, 5};
heap_to_inline.assign({7, 8});
context->ExpectEqual(heap_to_inline.capacity(), std::size_t{4},
                     "Assigning a small range should restore inline storage.");

Inline4 inline_to_heap{1, 2};
inline_to_heap.assign({1, 2, 3, 4, 5});
context->Expect(inline_to_heap.capacity() >= 5,
                "Assigning an overflow range should use heap storage.");
```

For copies, prove that changing the source does not change the destination. For moves, cover inline and overflow storage, then assign a valid value to each moved-from object without assuming it became empty.

- [ ] **Step 4: Run the RED build**

Publish and enter snapshot `task2-red`, then run:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=ON \
  -DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON
cmake --build build-cpu --target test_small_vector -j2
```

Expected result: compilation fails because `common/small_vector.h` or `SmallVector` does not exist.

## Task 3: Implement the Narrow Header-Only Container

**Files:**

- Create: `src/common/small_vector.h`

- [ ] **Step 1: Add the constrained class and representation**

```cpp
namespace infini::rt::detail {

template <typename T, std::size_t InlineCapacity>
class SmallVector {
  static_assert(InlineCapacity > 0,
                "SmallVector requires a positive inline capacity.");
  static_assert(std::is_trivially_copyable_v<T>,
                "SmallVector requires trivially copyable elements.");
  static_assert(std::is_trivially_destructible_v<T>,
                "SmallVector requires trivially destructible elements.");

 private:
  union Storage {
    T inline_data[InlineCapacity];

    T* heap_data;

    constexpr Storage() : inline_data{} {}
  };

  Storage storage_;

  std::size_t size_{0};

  std::size_t capacity_{InlineCapacity};
};

}  // namespace infini::rt::detail
```

Heap mode must always have `capacity_ > InlineCapacity`, so capacity identifies the active union member without a boolean. Use `std::allocator<T>`. Keep initializer order identical to declaration order.

- [ ] **Step 2: Implement construction, destruction, and assignment**

Implement this surface:

```cpp
SmallVector();

explicit SmallVector(std::size_t count);

SmallVector(std::initializer_list<T> values);

template <typename Iterator,
          std::enable_if_t<!std::is_integral_v<Iterator>, int> = 0>
SmallVector(Iterator first, Iterator last);

template <typename Container,
          std::enable_if_t<!std::is_same_v<std::decay_t<Container>,
                                           SmallVector>,
                           int> = 0>
explicit SmallVector(const Container& values);

SmallVector(const SmallVector& other);

SmallVector(SmallVector&& other) noexcept;

SmallVector& operator=(const SmallVector& other);

SmallVector& operator=(SmallVector&& other) noexcept;

~SmallVector();
```

Forward/random-access ranges must allocate once. Input iterators must append without first consuming the range. Heap copies allocate independent storage; heap moves transfer the pointer; inline moves copy at most `InlineCapacity` elements. Save the old heap pointer before activating inline storage during heap-to-inline assignment. Self-copy and self-move assignment return immediately.

- [ ] **Step 3: Implement the vector-like surface**

Implement:

```cpp
std::size_t size() const noexcept;

std::size_t capacity() const noexcept;

bool empty() const noexcept;

T* data() noexcept;

const T* data() const noexcept;

T& front() noexcept;

const T& front() const noexcept;

T& back() noexcept;

const T& back() const noexcept;

T& operator[](std::size_t index) noexcept;

const T& operator[](std::size_t index) const noexcept;

T* begin() noexcept;

const T* begin() const noexcept;

const T* cbegin() const noexcept;

T* end() noexcept;

const T* end() const noexcept;

const T* cend() const noexcept;

void clear() noexcept;

void reserve(std::size_t requested_capacity);

void resize(std::size_t requested_size);

void push_back(const T& value);

template <typename Iterator>
void assign(Iterator first, Iterator last);

void assign(std::initializer_list<T> values);
```

Repeated growth is geometric. `reserve` never shrinks. `assign` of at most `InlineCapacity` values restores inline mode. Do not add allocator APIs, arbitrary insertion, `shrink_to_fit`, or explicit exception handling.

Match `std::vector` value semantics for this trivial subset: the count constructor and newly grown `resize` elements are value-initialized to `T{}`.

- [ ] **Step 4: Implement constrained equality**

```cpp
template <typename T, std::size_t LeftCapacity, std::size_t RightCapacity>
bool operator==(const SmallVector<T, LeftCapacity>& left,
                const SmallVector<T, RightCapacity>& right);

template <typename T, std::size_t LeftCapacity, std::size_t RightCapacity>
bool operator!=(const SmallVector<T, LeftCapacity>& left,
                const SmallVector<T, RightCapacity>& right);
```

Add constrained overloads for `SmallVector == compatible range` and `compatible range == SmallVector`. Compare size before elements and reject scalar or unrelated types during substitution.

- [ ] **Step 5: Run the focused test**

Publish and enter snapshot `task3-green`, then run:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=ON \
  -DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON
cmake --build build-cpu --target test_small_vector -j2
ctest --test-dir build-cpu -R '^test_small_vector$' --output-on-failure
```

Expected result:

```text
100% tests passed, 0 tests failed out of 1
```

- [ ] **Step 6: Commit the standalone container**

```bash
git add src/common/small_vector.h tests/test_small_vector.cc tests/CMakeLists.txt
git commit -m "perf: add inline metadata container"
```

## Task 4: Add Failing TensorView Integration Tests

**Files:**

- Modify: `tests/test_tensor_view_allocations.cc`
- Modify: `tests/test_core.cc`
- Modify: `tests/install_consumer_smoke.cc`

- [ ] **Step 1: Expand Linux allocation tests while TensorView still uses vector**

Prepare inputs outside `CountAllocations` except exact temporaries and initializer lists. Encode:

| Path | rank <= 4 | rank 5 |
| --- | ---: | ---: |
| lvalue explicit metadata | 0 | 2 |
| exact-type temporaries created inside scope | 0 | 2 |
| initializer-list metadata | 0 | 2 |
| vector-backed generic TensorLike | 0 | 2 |
| ordinary default strides | 0 | 2 |
| preconstructed metadata moved into explicit constructor | 0 | 0 |
| preconstructed shape moved while generating strides | 0 | 1 |

Also require zero allocations for inline copy, inline/overflow move, rank-4 indexing, rank-5 indexing to rank 4, and rank-2 transpose. Require two allocations for overflow copy.

```cpp
ExpectAllocationCount(
    context,
    CountAllocations([&] {
      TensorView tensor{data, shape4, DataType::kFloat32, cpu, strides4};
      (void)tensor;
    }),
    0, "Rank-4 lvalue metadata should stay inline.");
```

- [ ] **Step 2: Expand portable semantics tests**

In `tests/test_core.cc`, cover ranks 0, 1, 2, 3, 4, 5, 8, and 9 for `ndim`, shape, strides, `numel`, and contiguity. Add a vector-backed TensorLike fixture and preserve indexing, transpose, hashing, and equality.

```cpp
static_assert(std::is_copy_constructible_v<TensorView>);
static_assert(std::is_move_constructible_v<TensorView>);
static_assert(!std::is_copy_assignable_v<TensorView>);
static_assert(!std::is_move_assignable_v<TensorView>);
```

In `tests/install_consumer_smoke.cc`, construct default-stride and explicit-stride views from `std::vector` and verify both.

- [ ] **Step 3: Run the RED allocation test**

Publish and enter snapshot `task4-red`, then run:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=ON \
  -DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON
cmake --build build-cpu --target test_core test_tensor_view_allocations -j2
ctest --test-dir build-cpu \
  -R '^(test_core|test_tensor_view_allocations)$' \
  --output-on-failure
```

Expected result: `test_core` remains green and `test_tensor_view_allocations` fails on rank-4 zero-allocation assertions because `TensorView` still uses `std::vector`.

## Task 5: Integrate Capacity 4 Into TensorView

**Files:**

- Modify: `src/tensor_view.h`
- Modify only if compilation requires it: `src/tensor_view.cc`

- [ ] **Step 1: Change the aliases behind one private source constant**

Add `common/small_vector.h` and `<iterator>`, remove `<vector>`, and define:

```cpp
namespace tensor_view_detail {

inline constexpr std::size_t kInlineMetadataCapacity = 4;

template <typename Metadata, typename Range>
Metadata CopyMetadata(const Range& range) {
  return Metadata(std::begin(range), std::end(range));
}

}  // namespace tensor_view_detail
```

Change the aliases:

```cpp
using Shape =
    detail::SmallVector<Size, tensor_view_detail::kInlineMetadataCapacity>;

using Strides =
    detail::SmallVector<Stride, tensor_view_detail::kInlineMetadataCapacity>;
```

- [ ] **Step 2: Make generic constructors range-based**

Always generate default strides from `shape_`:

```cpp
template <typename ShapeLike>
TensorView(void* data, const ShapeLike& shape)
    : data_{data},
      shape_{std::begin(shape), std::end(shape)},
      dtype_{DefaultDataType()},
      device_{DefaultDevice()},
      strides_{DefaultStrides(shape_)} {}

template <typename ShapeLike, typename StridesLike>
TensorView(void* data, const ShapeLike& shape, const DataType& dtype,
           const Device& device, const StridesLike& strides)
    : data_{data},
      shape_{std::begin(shape), std::end(shape)},
      dtype_{dtype},
      device_{device},
      strides_{std::begin(strides), std::end(strides)} {}
```

For TensorLike construction, bind each returned range once through `CopyMetadata` so accessors returning by value remain valid. Keep exact by-value and initializer-list overloads.

- [ ] **Step 3: Build and run focused tests**

Publish and enter snapshot `task5-cap4`, then run:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=ON \
  -DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON
cmake --build build-cpu \
  --target test_small_vector test_core test_tensor_view_allocations \
           perf_tensor_view -j2
ctest --test-dir build-cpu \
  -R '^(test_small_vector|test_core|test_tensor_view_allocations)$' \
  --output-on-failure
```

Expected result: all 3 tests pass.

- [ ] **Step 4: Verify installed public headers and vector consumers**

Re-enter snapshot `task5-cap4` so this step uses the build from Step 3:

```bash
ctest --test-dir build-cpu \
  -R '^test_install(_consumer)?$' \
  --output-on-failure
test -f generated/include/infini/rt/detail/common/small_vector.h
test -f build-cpu/tests/install_consumer_prefix/include/infini/rt/detail/common/small_vector.h
```

Expected result: both CTest cases and both file checks pass.

- [ ] **Step 5: Commit and record capacity 4**

```bash
git add src/tensor_view.h tests/test_core.cc \
  tests/test_tensor_view_allocations.cc tests/install_consumer_smoke.cc
git commit -m "perf: inline TensorView metadata"
git update-ref refs/benchmarks/tensor-view/cap4 HEAD
```

Stage `src/tensor_view.cc` only if it has a real diff.

## Task 6: Drive Capacity 8 With a Second RED Cycle

**Files:**

- Modify: `tests/test_tensor_view_allocations.cc`
- Modify: `tests/test_core.cc`
- Modify: `src/tensor_view.h`

- [ ] **Step 1: Add rank-8/rank-9 thresholds before changing capacity**

Require ranks 0 through 8 to be allocation-free for all inline construction paths. For rank 9 encode:

| Path | Expected allocations |
| --- | ---: |
| lvalue explicit metadata | 2 |
| exact-type temporaries created inside scope | 2 |
| initializer-list metadata | 2 |
| vector-backed generic TensorLike | 2 |
| ordinary default strides | 2 |
| preconstructed metadata moved into explicit constructor | 0 |
| preconstructed shape moved while generating strides | 1 |

Publish and enter snapshot `task6-red`, then run:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=ON \
  -DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON
cmake --build build-cpu --target test_core test_tensor_view_allocations -j2
ctest --test-dir build-cpu \
  -R '^test_tensor_view_allocations$' \
  --output-on-failure
```

Expected result: RED because capacity 4 allocates at rank 8.

- [ ] **Step 2: Change only the source constant**

```cpp
inline constexpr std::size_t kInlineMetadataCapacity = 8;
```

- [ ] **Step 3: Rebuild and rerun the same tests**

Publish and enter snapshot `task6-cap8`, then run:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=ON \
  -DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON
cmake --build build-cpu \
  --target test_small_vector test_core test_tensor_view_allocations \
           perf_tensor_view -j2
ctest --test-dir build-cpu \
  -R '^(test_small_vector|test_core|test_tensor_view_allocations)$' \
  --output-on-failure
```

Expected result: all 3 tests pass.

- [ ] **Step 4: Commit and record capacity 8**

```bash
git add src/tensor_view.h tests/test_core.cc \
  tests/test_tensor_view_allocations.cc
git commit -m "perf: evaluate eight inline dimensions"
git update-ref refs/benchmarks/tensor-view/cap8 HEAD
```

Verify performance-relevant source differs only by capacity:

```bash
git diff --name-only \
  refs/benchmarks/tensor-view/cap4 \
  refs/benchmarks/tensor-view/cap8 \
  -- src tests/performance
```

Expected output:

```text
src/tensor_view.h
```

## Task 7: Run the Three-Way Capacity Experiment

**Files:**

- No repository files are modified.
- Store raw results under `/tmp/tensor-view-small-vector/results`.

- [ ] **Step 1: Bundle exact refs and create independent remote sources**

From PowerShell:

```powershell
$BundlePath = Join-Path $env:TEMP 'tensor-view-small-vector.bundle'
git bundle create $BundlePath refs/benchmarks/tensor-view/baseline refs/benchmarks/tensor-view/cap4 refs/benchmarks/tensor-view/cap8
ssh nvidia "test ! -e /tmp/tensor-view-small-vector && mkdir /tmp/tensor-view-small-vector"
scp $BundlePath nvidia:/tmp/tensor-view-small-vector/source.bundle
```

On `nvidia`:

```bash
for variant in baseline cap4 cap8; do
  git init "/tmp/tensor-view-small-vector/$variant"
  git -C "/tmp/tensor-view-small-vector/$variant" fetch \
    /tmp/tensor-view-small-vector/source.bundle \
    "refs/benchmarks/tensor-view/$variant"
  git -C "/tmp/tensor-view-small-vector/$variant" \
    checkout --detach FETCH_HEAD
done
mkdir /tmp/tensor-view-small-vector/results
```

- [ ] **Step 2: Build all variants with identical image and flags**

Run for `baseline`:

```bash
docker run --rm \
  -v /tmp/tensor-view-small-vector:/workspace \
  -w /workspace/baseline \
  accelerator-dev/nvidia:latest \
  cmake -S . -B build-perf \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_CPU=ON \
    -DINFINI_RT_BUILD_TESTING=OFF \
    -DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON
docker run --rm \
  -v /tmp/tensor-view-small-vector:/workspace \
  -w /workspace/baseline \
  accelerator-dev/nvidia:latest \
  cmake --build build-perf --target perf_tensor_view -j2
```

Repeat exactly with `-w /workspace/cap4` and `-w /workspace/cap8`. Record the image ID, host model, compiler path/version, and all three Git SHAs.

- [ ] **Step 3: Execute five round-robin process groups on CPU 0**

```text
run 1: baseline, cap4, cap8
run 2: cap4, cap8, baseline
run 3: cap8, baseline, cap4
run 4: baseline, cap8, cap4
run 5: cap4, baseline, cap8
```

For the first process:

```bash
docker run --rm --cpuset-cpus 0 \
  -v /tmp/tensor-view-small-vector:/workspace \
  -w /workspace/baseline \
  accelerator-dev/nvidia:latest \
  python3 scripts/run_performance_tests.py \
    --build-dir build-perf \
    --backend cpu \
    --test perf_tensor_view \
    --output /workspace/results/baseline-1.json
```

Change only the working directory, output prefix, and run number according to the fixed order. Every invocation must report 58 results. Never concatenate files.

- [ ] **Step 4: Apply the existing comparison script to every matched pair**

For each run number 1 through 5:

```bash
python3 scripts/compare_performance_results.py \
  --baseline /tmp/tensor-view-small-vector/results/baseline-1.json \
  --candidate /tmp/tensor-view-small-vector/results/cap4-1.json
python3 scripts/compare_performance_results.py \
  --baseline /tmp/tensor-view-small-vector/results/cap4-1.json \
  --candidate /tmp/tensor-view-small-vector/results/cap8-1.json
python3 scripts/compare_performance_results.py \
  --baseline /tmp/tensor-view-small-vector/results/baseline-1.json \
  --candidate /tmp/tensor-view-small-vector/results/cap8-1.json
```

Repeat with run numbers 2 through 5. Any missing or new key invalidates the experiment.

- [ ] **Step 5: Aggregate the five paired median changes**

```bash
python3 - /tmp/tensor-view-small-vector/results <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])


def key(item):
    params = json.dumps(item.get("params") or {}, sort_keys=True, separators=(",", ":"))
    return item.get("backend", ""), item["benchmark"], params, item["unit"]


def load(name, run):
    path = root / f"{name}-{run}.json"
    raw = json.loads(path.read_text())
    if len(raw) != 58:
        raise SystemExit(f"expected 58 results in {path}, found {len(raw)}")
    indexed = {}
    for item in raw:
        result_key = key(item)
        if result_key in indexed:
            raise SystemExit(f"duplicate benchmark key in {path}: {result_key}")
        indexed[result_key] = item
    if len(indexed) != len(raw):
        raise SystemExit(f"result indexing lost entries in {path}")
    return indexed


for baseline_name, candidate_name in (
    ("baseline", "cap4"),
    ("cap4", "cap8"),
    ("baseline", "cap8"),
):
    changes = {}
    for run in range(1, 6):
        baseline = load(baseline_name, run)
        candidate = load(candidate_name, run)
        if baseline.keys() != candidate.keys():
            raise SystemExit(f"key mismatch: {baseline_name} {candidate_name} run {run}")
        for result_key in baseline:
            old = baseline[result_key]["median"]
            new = candidate[result_key]["median"]
            changes.setdefault(result_key, []).append((new - old) / old * 100.0)
    for result_key in sorted(changes):
        values = changes[result_key]
        backend, benchmark, params, unit = result_key
        print(
            baseline_name,
            candidate_name,
            benchmark,
            params,
            unit,
            f"median={statistics.median(values):+.2f}%",
            f"range=[{min(values):+.2f}%,{max(values):+.2f}%]",
            sep="\t",
        )
PY
```

- [ ] **Step 6: Apply all approved decision gates**

The experiment passes only if:

- capacity 4 versus vector baseline is at most +5% for every applicable rank-1/2/4 explicit/default construction, copy, derived-view, and by-value result;
- capacity-4 construction and copy at ranks 1/2/4 are below 0%;
- capacity 8 versus capacity 4 is at most +5% for the same rank-1/2/4 paths;
- each candidate versus vector baseline is at most +5% for rank-9 explicit/default construction, copy, and by-value results;
- capacity 8 is allocation-free at ranks 5 and 8, and its construction, copy, and by-value results there are below 0% versus capacity 4;
- every `numel` control has absolute change at most 5%.

Report all `stderr` layout lines. Report that InfiniOps `Add` has six `Tensor::Shape/Strides` members and `FlashAttnVarlenFunc` has twelve, then multiply those counts by the measured `sizeof(SmallVector<Size, 8>) - sizeof(SmallVector<Size, 4>)`. This footprint is disclosed beside latency; it does not silently override the approved preference for capacity 8 when every numerical gate passes.

- [ ] **Step 7: Retain exactly one capacity**

If every gate passes:

```bash
git update-ref refs/benchmarks/tensor-view/selected \
  refs/benchmarks/tensor-view/cap8
```

If capacity 8 fails but capacity 4 passes, change the constant and rank-dependent expectations back to 4, rerun the focused suite, commit that measured choice, and point `selected` at the new commit.

If capacity 4 fails a baseline gate, stop the integration and return to the combined-metadata fallback. Do not publish an allocation-only regression.

## Task 8: Document the Compatibility Boundary

**Files:**

- Modify: `docs/api/core-types.md`
- Modify: `docs/compatibility.md`

- [ ] **Step 1: Keep public examples source-compatible**

Retain the `std::vector` example in `docs/api/core-types.md`. State that shape and strides are owned, use inline storage through the selected low-rank capacity, and fall back to heap storage above it.

- [ ] **Step 2: State the rebuilding requirement**

Add this substance under `ABI Notes`:

```text
TensorView::Shape and TensorView::Strides are concrete C++ aliases whose
representation can affect TensorView layout. Consumers must rebuild after an
alias or layout change and must not mix headers and libraries from different
builds.
```

Do not add release-version or `SOVERSION` policy.

- [ ] **Step 3: Commit documentation**

```bash
git add docs/api/core-types.md docs/compatibility.md
git commit -m "docs: document TensorView metadata compatibility"
```

## Task 9: Consolidate and Run Final InfiniRT Validation

**Files:**

- Modify only for demonstrated defects in files already listed.

- [ ] **Step 1: Consolidate checkpoints before final validation**

The local benchmark refs preserve every measured tree. Convert the feature branch to one `CONTRIBUTING.md`-compliant commit before collecting final validation evidence:

```bash
git fetch origin
TV_BASE_SHA="$(git merge-base HEAD origin/master)"
if ! git diff --quiet "$TV_BASE_SHA"..origin/master -- \
  src \
  tests/performance \
  tests/CMakeLists.txt \
  scripts/run_performance_tests.py \
  scripts/compare_performance_results.py \
  CMakeLists.txt; then
  echo "Performance-relevant upstream files changed; replay Tasks 1-7." >&2
  exit 1
fi
git reset --soft "$TV_BASE_SHA"
git commit \
  -m "perf!: inline TensorView metadata" \
  -m "BREAKING CHANGE: TensorView::Shape and TensorView::Strides now use an inline metadata container. Rebuild consumers against matching InfiniRT headers and libraries."
git rebase origin/master
git update-ref refs/benchmarks/tensor-view/selected HEAD
```

Verify `git diff refs/benchmarks/tensor-view/selected^..refs/benchmarks/tensor-view/selected` contains the complete intended tree and no benchmark result artifacts.

Bundle that exact ref and create three independent sources from it. From PowerShell:

```powershell
$SelectedBundle = Join-Path $env:TEMP 'infinirt-tv-selected.bundle'
git bundle create $SelectedBundle refs/benchmarks/tensor-view/selected
scp $SelectedBundle nvidia:/tmp/tensor-view-small-vector/selected.bundle
ssh nvidia mkdir /tmp/tensor-view-small-vector/selected
ssh nvidia mkdir /tmp/infinirt-small-vector-cpu
ssh nvidia mkdir /tmp/infinirt-small-vector-nvidia
ssh nvidia git -C /tmp/tensor-view-small-vector/selected init
ssh nvidia git -C /tmp/tensor-view-small-vector/selected fetch /tmp/tensor-view-small-vector/selected.bundle refs/benchmarks/tensor-view/selected
ssh nvidia git -C /tmp/tensor-view-small-vector/selected checkout --detach FETCH_HEAD
ssh nvidia git -C /tmp/infinirt-small-vector-cpu init
ssh nvidia git -C /tmp/infinirt-small-vector-cpu fetch /tmp/tensor-view-small-vector/selected.bundle refs/benchmarks/tensor-view/selected
ssh nvidia git -C /tmp/infinirt-small-vector-cpu checkout --detach FETCH_HEAD
ssh nvidia git -C /tmp/infinirt-small-vector-nvidia init
ssh nvidia git -C /tmp/infinirt-small-vector-nvidia fetch /tmp/tensor-view-small-vector/selected.bundle refs/benchmarks/tensor-view/selected
ssh nvidia git -C /tmp/infinirt-small-vector-nvidia checkout --detach FETCH_HEAD
```

All three remote `git rev-parse HEAD` results must equal the local final SHA.

- [ ] **Step 2: Run the full CPU Release suite in a clean selected source**

```bash
ssh nvidia docker run --rm \
  -v /tmp/infinirt-small-vector-cpu:/workspace/InfiniRT \
  -w /workspace/InfiniRT accelerator-dev/nvidia:latest \
  cmake -S . -B build-cpu \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_CPU=ON \
    -DINFINI_RT_BUILD_TESTING=ON \
    -DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON
ssh nvidia docker run --rm \
  -v /tmp/infinirt-small-vector-cpu:/workspace/InfiniRT \
  -w /workspace/InfiniRT accelerator-dev/nvidia:latest \
  cmake --build build-cpu -j2
ssh nvidia docker run --rm --entrypoint ctest \
  -v /tmp/infinirt-small-vector-cpu:/workspace/InfiniRT \
  -w /workspace/InfiniRT accelerator-dev/nvidia:latest \
  --test-dir build-cpu --output-on-failure
```

Expected result after adding `test_small_vector`:

```text
100% tests passed, 0 tests failed out of 11
```

- [ ] **Step 3: Run NVIDIA Release build and non-performance tests separately**

Use an independent checkout of `refs/benchmarks/tensor-view/selected` at `/tmp/infinirt-small-vector-nvidia`:

```bash
ssh nvidia docker run --rm --gpus all \
  -v /tmp/infinirt-small-vector-nvidia:/workspace/InfiniRT \
  -w /workspace/InfiniRT accelerator-dev/nvidia:latest \
  cmake -S . -B build-nvidia \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_NVIDIA=ON \
    -DINFINI_RT_BUILD_TESTING=ON \
    -DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON
ssh nvidia docker run --rm --gpus all \
  -v /tmp/infinirt-small-vector-nvidia:/workspace/InfiniRT \
  -w /workspace/InfiniRT accelerator-dev/nvidia:latest \
  cmake --build build-nvidia -j2
ssh nvidia docker run --rm --gpus all --entrypoint ctest \
  -v /tmp/infinirt-small-vector-nvidia:/workspace/InfiniRT \
  -w /workspace/InfiniRT accelerator-dev/nvidia:latest \
  --test-dir build-nvidia -E '^perf_' --output-on-failure
```

Expected result:

```text
100% tests passed, 0 tests failed out of 9
```

- [ ] **Step 4: Run formatting and whitespace checks**

```bash
ssh nvidia docker run --rm --entrypoint clang-format \
  -v /tmp/tensor-view-small-vector/selected:/workspace/InfiniRT \
  -w /workspace/InfiniRT \
  ghcr.io/jidicula/clang-format:21 \
  --dry-run --Werror \
  src/common/small_vector.h \
  src/tensor_view.h \
  tests/test_small_vector.cc \
  tests/test_core.cc \
  tests/test_tensor_view_allocations.cc \
  tests/performance/perf_tensor_view.cc \
  tests/install_consumer_smoke.cc
git diff --check
```

Add `src/tensor_view.cc` to the formatter command only if it changed.

- [ ] **Step 5: Execute downstream validation**

Complete `docs/superpowers/plans/2026-07-23-tensor-view-small-vector-downstream.md` against the installed selected candidate. InfiniOps CPU/pybind and torch-infini adapter evidence are required before proposing the InfiniRT PR.

## Task 10: Prepare and Publish the Pull Request

**Files:**

- Verify: `CONTRIBUTING.md`
- Verify: `.github/PULL_REQUEST_TEMPLATE.md`

- [ ] **Step 1: Audit scope and checkpoints**

```bash
git status --short
git diff --stat origin/master...HEAD
git diff --check origin/master...HEAD
git log --oneline origin/master..HEAD
```

The final diff contains only the design, container, TensorView integration, tests, benchmarks, and compatibility docs. It contains no JSON results, build trees, capacity toggles, version changes, or unrelated refactors.

- [ ] **Step 2: Fill every PR template section with observed evidence**

Create `docs/superpowers/pr-body.md` as a temporary untracked file by copying the repository template and replacing every prompt with observed evidence:

- `Summary`: container, TensorView integration, tests, and selected capacity.
- `Motivation`: shape/stride allocations remaining after #33; state that this is a follow-up and that no issue is closed.
- `Type of Change`: check `perf` and breaking change.
- `Platforms Affected`: check every backend, generated headers, and public headers.
- `Smoke Build and Test Result`: paste exact CPU and NVIDIA commands with trimmed output.
- `Test Results on Supported Platforms`: mark CPU full passed and NVIDIA non-performance passed; identify each unavailable accelerator and request maintainer validation.
- `Benchmark / Performance Impact`: include host, image ID, compiler, ranks, five-run order, all SHAs, paired median/range, allocation counts, and object sizes.
- `Notes for Reviewers`: call out the API/ABI break, matching-header requirement, selected-capacity tradeoff, and separate downstream compatibility work.

Never claim a platform or downstream test passed unless its exact command completed at the final commit.

- [ ] **Step 3: Push and verify the published pull request**

The branch `perf/inline-tensor-view-metadata` already matches `CONTRIBUTING.md`. Push the final commit and create a ready pull request titled `perf!: inline TensorView metadata` using the fully populated repository template. Then verify the published title and body:

```bash
git push --set-upstream origin perf/inline-tensor-view-metadata
test -s docs/superpowers/pr-body.md
gh pr create \
  --title "perf!: inline TensorView metadata" \
  --body-file docs/superpowers/pr-body.md
gh pr view --json title,body,url,isDraft,headRefOid
```

The returned body must contain every template heading and no template placeholder text, `isDraft` must be `false`, and `headRefOid` must equal `git rev-parse HEAD`.

- [ ] **Step 4: Remove experiment refs after evidence is captured**

```bash
git update-ref -d refs/benchmarks/tensor-view/baseline
git update-ref -d refs/benchmarks/tensor-view/cap4
git update-ref -d refs/benchmarks/tensor-view/cap8
git update-ref -d refs/benchmarks/tensor-view/selected
```

Run `git status --short --branch`. Expected result: a clean feature branch with one commit relative to `origin/master`.

Delete the temporary `docs/superpowers/pr-body.md` with `apply_patch` before that final status check.
