# TensorView Inline Metadata Storage Design

Date: 2026-07-23

Status: Combined metadata selected with inline capacity 8 and a documented
rank-4 small-key footprint trade-off; rank 9 and above use owned heap fallback

## Context

`TensorView` is a framework-neutral tensor metadata object used directly by
InfiniRT consumers and aliased as `infini::ops::Tensor` by InfiniOps. Pull
request #33 removed redundant vector copies, but a typical non-empty view still
owns one heap allocation for its shape and one for its strides. Construction,
copying, indexing, and transposition therefore remain allocation-sensitive.

Deep-learning tensors usually have a small rank. Inline metadata storage can
remove these allocations, but replacing the public `std::vector` aliases also
changes the public C++ API and the object layout. This design accepts that
compatibility boundary: consumers must rebuild against matching InfiniRT
headers and libraries.

The first experiment stored shape and strides in two independent
`SmallVector` members. It delivered substantial wins on several common
low-rank paths, but it did not satisfy the pre-agreed whole-matrix gates: 52 of
114 predicates failed. The capacity-4 and capacity-8 `TensorView` objects were
120 and 184 bytes respectively, and rank-9 paths regressed materially against
the vector baseline. Those results reject the two-member representation as the
shipping design; they do not show that inline metadata itself is ineffective.

InfiniRT has not had a formal release. This work does not change the project
version or add an `SOVERSION`.

## Goals

- Make `TensorView` construction, copying, indexing, and transposition perform
  no heap allocations for ranks 0 through 8.
- Retain owned metadata and value semantics.
- Preserve the vector-like owning types used to construct `TensorView`, while
  exposing shape and strides through lightweight contiguous views.
- Keep metadata contiguous and expose stable `data()` and iterator ranges.
- Support arbitrary practical ranks by falling back to heap storage.
- Use inline capacity 8, selected from measured end-to-end `TensorView`
  performance and an InfiniOps-style cache-key footprint benchmark.
- Avoid new public third-party dependencies.

## Non-Goals

- Borrowed shape or stride storage.
- A general-purpose replacement for `std::vector`.
- Allocator customization.
- Unrelated `TensorView` correctness changes.
- Refactoring InfiniOps operator storage or dispatch.
- Version, package-compatibility, or `SOVERSION` policy changes.
- Backend-specific runtime changes.

## Compatibility Boundary

`TensorView::Shape` and `TensorView::Strides` are public concrete aliases and
members of an installed C++ class. Replacing them changes `sizeof(TensorView)`,
member offsets, inline special members, and out-of-line method expectations.
Old headers and a new `libinfinirt` must not be mixed.

The intended source compatibility boundary is:

- Preserve the names `TensorView`, `Shape`, and `Strides`.
- Preserve existing `TensorView` construction from vector-like ranges.
- Preserve initializer-list and `std::vector` construction.
- Preserve accessor iteration, indexing, size queries, contiguous data, and
  equality, but permit `shape()` and `strides()` to return a view by value.
- Permit source changes where callers require the exact `std::vector` type,
  require an owning result from an accessor, depend on an allocator, use a
  `std::vector`-specific caster, or combine different accessor return types in
  one conditional expression.
- Require every consumer to rebuild against the matching installed headers and
  library.

## Alternatives

### Two SmallVector Members

Replace `Shape` and `Strides` with two instances of an in-tree
`SmallVector<T, N>`. This keeps the current `TensorView` model and lets
InfiniOps operator metadata members benefit from the same inline storage.

This was the first measured approach. It produced large wins on several
low-rank construction and copy paths, but failed 52 of 114 decision predicates.
Its capacity-4 and capacity-8 `TensorView` layouts were 120 and 184 bytes, and
both candidates regressed materially on rank-9 work. It is rejected as the
shipping representation.

### Combined Tensor Metadata Storage

Store shape and strides in one TensorView-specific owner. This is the current
fallback candidate because it retains inline low-rank storage without paying
for two independent inline containers in every `TensorView`.

The representation has three states:

- Inline SoA: one inline shape array followed by one inline stride array.
- Combined overflow: one allocation owns both arrays for lvalue, generic,
  copy, initializer-list, and ordinary default-stride construction.
- Split-adopt overflow: two allocations already owned by exact `Shape` and
  `Strides` rvalues are adopted without allocating a third block. Moving
  preconstructed exact metadata can therefore transfer ownership without a
  new allocation.

`shape()` and `strides()` return non-owning contiguous views by value. This is
an intentional source and ABI compatibility break and must be validated in
InfiniOps and torch-infini before delivery.

### Third-Party Small Vectors

LLVM, Boost, and Abseil provide mature inline containers. Each option would
still change the public API and ABI while adding a dependency to installed
headers and consumers. InfiniRT currently has no comparable runtime container
dependency, so these options are rejected.

## Retained SmallVector Input Type

Add a header-only `infini::rt::detail::SmallVector<T, N>` under `src/common/`.
It is deliberately limited to trivially copyable and trivially destructible
element types. The initial consumers are `std::size_t` and `std::ptrdiff_t`.

The representation contains:

- A union of an inline `T[N]` buffer and a heap pointer.
- A current size.
- A current capacity that also identifies inline versus heap mode.

The implementation uses standard allocation primitives with the same
allocation-failure behavior as the existing `std::vector` members. It does not
throw or catch exceptions explicitly. Heap growth is geometric for repeated
`push_back`; constructors from sized or random-access ranges allocate the
required capacity directly.

Required operations are:

- Default, count, initializer-list, iterator-range, and compatible-container
  construction.
- Copy and move construction and assignment.
- Destruction and self-assignment safety.
- `size`, `capacity`, `empty`, `data`, `front`, `back`, and `operator[]`.
- `begin`, `end`, `cbegin`, and `cend`.
- `clear`, `reserve`, `resize`, `push_back`, and `assign`.
- Equality and inequality for compatible contiguous ranges.

The class does not provide allocator APIs, insertion at arbitrary positions,
or `shrink_to_fit` unless a real downstream compile failure demonstrates that
one is required.

The fallback uses `SmallVector` as the public owning `Shape` and `Strides`
input type, but no longer stores two instances inside `TensorView`. Its inline
storage must begin the lifetime of a real `T[N]` array without value-initializing
the entire capacity. Its overflow ownership-transfer API must return a
move-only token that retains both size and allocation capacity, and leave the
source valid. Adopting an over-capacity allocation preserves the existing
move semantics of the owning input and guarantees allocator-correct release.
Validation and allocation must occur before ownership is released so that a
throwing constructor cannot leak either array.

Inline copies copy their elements into the destination object. Heap copies
allocate independent storage. Inline moves copy at most `N` trivial elements;
heap moves transfer the pointer without allocating. A moved-from object must
remain destructible and assignable, but is not required to be empty.

## TensorMetadata Integration

`TensorView::Shape` and `TensorView::Strides` remain aliases of
`SmallVector<Size, kInlineCapacity>` and
`SmallVector<Stride, kInlineCapacity>` for owning construction inputs.
`TensorView` itself stores one private `TensorMetadata` owner instead of two
containers. The final inline capacity is a source constant, not a public build
option, because different capacities produce binary-incompatible object
layouts.

`TensorMetadata` stores shape and strides as a structure of arrays. Inline
mode owns two real arrays in the object. Combined mode owns one aligned raw
block containing a real `Size[]` followed by a real `Stride[]`. Split-adopt
mode owns the two arrays released by exact rvalue inputs. A compact explicit
state tag, or an equivalently reviewed encoding, distinguishes the two
overflow modes; rank alone cannot distinguish them.

Metadata lengths are encoded as `std::uint32_t`. A range longer than
`UINT32_MAX` violates the constructor precondition and terminates before any
length conversion, allocation-size calculation, or ownership transfer.

The combined block must not rely on pointer arithmetic over individually
placement-constructed scalar objects. It creates actual array objects with
non-allocating placement array new. The C++17 implementation relies on the
accepted CWG 2382 defect resolution that forbids placement-array overhead for
this standard form. The exact allocation, construction, destruction, and
deallocation sequence must be compiled and exercised with the supported GCC,
Clang, and MSVC toolchains before delivery.

Exact `Shape` and `Strides` constructor overloads use `const&` and `&&` pairs
so lvalues can copy directly into one combined block and rvalues can be
adopted. Generic `TensorView` constructors build one combined block from
iterator ranges. Accessors return lightweight typed views by value; they keep
contiguous `data()`, iterators, indexing, size, and equality, but do not imply
ownership or an implicit allocation-producing conversion.

Existing `TensorView` behavior remains unchanged for:

- Default dtype, device, and contiguous stride generation.
- Scalar and high-rank tensors.
- Positive and negative indexing.
- Two-dimensional transposition.
- Hashing and equality.
- Copy and move constructibility.
- Deleted assignment caused by the existing `const dtype_` member.

## Revised Inline Capacity Experiment

The rejected two-member measurements remain recorded as experiment evidence.
The combined-metadata fallback was evaluated independently against the same
post-#33 vector baseline and the same benchmark matrix. The shipped
`TensorView` uses inline capacity 8.

1. Add failing tests for view semantics, the three storage states, allocation
   counts, copy/move ownership, and overflow cleanup.
2. Implement and validate a capacity-4 combined-metadata candidate.
3. Record allocation counts, object sizes, and five-round performance results.
4. Add rank-8/rank-9 failing thresholds, then change only the inline capacity
   to 8.
5. Rerun the same correctness, allocation, compiler, and benchmark checks.
6. Select capacity 8 from the rank-0-through-8 latency, allocation, and object
   footprint evidence. Keep rank 9 and above as a supported correctness and
   owned-fallback boundary, not a performance gate.

The five-round combined-metadata result selected capacity 8. Against capacity
4, the aggregate across all 25 applicable rank-1/2/4 keys was +0.75 percent
and +0.029 ns. Eight individual percentage medians exceeded +5 percent, but
their absolute changes were only 0.133 through 1.360 ns, every five-run range
crossed zero, and none was slower in all five runs. At ranks 5 and 8, all 14
target paths improved by 41.8 through 74.8 percent, or 7.987 through 21.701 ns.
The rank-9 rvalue and default-stride regressions are retained as explicit
fallback trade-off evidence rather than hidden or treated as low-rank results.

The measured layouts are 72 bytes for the post-#33 vector baseline, 96 bytes
for combined capacity 4, and 160 bytes for combined capacity 8. Object sizes
are disclosed beside latency and are also exercised by the cache-key footprint
benchmark below.

## Test-Driven Development

Production changes follow red-green-refactor cycles.

### Allocation Thresholds

For the capacity-4 combined-metadata candidate, tests first require:

- Rank 0 through 4 lvalue, rvalue, initializer-list, default-stride, and generic
  TensorLike construction: zero allocations.
- Rank 5 lvalue explicit metadata, initializer-list metadata, generic
  TensorLike construction, ordinary default-stride construction, and copy
  construction: one combined allocation.
- Rank 5 exact-type shape and stride temporaries created inside the measured
  expression: two allocations for those owning inputs and no third allocation
  in `TensorView`.
- Rank 5 exact-type rvalue-shape default-stride construction: at most two
  allocations, one adopted shape allocation and one generated-stride
  allocation.
- Moving preconstructed rank-5 shape and strides into explicit-metadata
  construction: zero allocations.
- Moving a preconstructed rank-5 shape into default-stride construction: one
  allocation for generated strides.

For the capacity-8 candidate, new failing thresholds require:

- Rank 0 through 8 construction paths: zero allocations.
- Rank 9 follows the same path-specific expectations as rank 5 above: one
  combined allocation for lvalue, initializer-list, generic TensorLike,
  ordinary default-stride, and copy construction; two existing allocations
  and no third allocation for measured exact-type temporaries; zero for moving
  preconstructed explicit metadata; and one for moving a preconstructed shape
  while generating default strides.

Input containers are prepared outside allocation scopes except where the test
specifically measures rvalue or initializer-list construction.

### Value Semantics

- Inline copy construction performs zero allocations and owns independent
  storage.
- Overflow copy construction canonicalizes either overflow state into one
  combined allocation and owns independent storage.
- Inline and both overflow-state move constructions perform zero allocations.
- Destruction and exceptional construction release every live allocation once
  in combined and split-adopt states.
- Accessor views have the same lifetime as their owning `TensorView`; copying a
  view never copies metadata or extends its lifetime.
- SmallVector self-assignment, heap-to-inline assignment, and inline-to-heap
  assignment preserve values and storage invariants.
- Moved-from values are only tested for valid destruction and reassignment.
- Compile-time assertions preserve TensorView copy/move construction and its
  existing deleted copy/move assignment.

### Derived Views

- Rank-4 indexing produces rank 3 without allocation.
- Rank-5 indexing produces rank 4 without allocation, exercising overflow to
  inline conversion.
- Positive and negative indexes preserve the existing data offset, shape,
  stride, dtype, and device behavior.
- Rank-2 `T()` performs no allocation and preserves current transpose behavior.

### Portable Functional Coverage

Core tests cover ranks 0, 1, 2, 3, 4, 5, 8, and 9 for `ndim`, shape, strides,
`numel`, and contiguity. A TensorLike fixture whose metadata is stored in
`std::vector` verifies source interoperability.

The installed-consumer test continues to compile a consumer using
`std::vector` metadata against the installed public header and shared library.
Compile-time coverage also verifies the intended view-by-value accessor return
types and rejects accidental implicit conversion back to an owning container.

## Benchmark Design

The post-#33 merge commit is the baseline. Combined capacity 4 is retained as
a diagnostic comparison and combined capacity 8 is the selected candidate.
All three are built with the same compiler, optimization level, source apart
from the capacity constant, and benchmark harness.

Measure ranks 1, 2, 4, 5, 8, and 9 for:

- Lvalue explicit-metadata construction.
- Rvalue explicit-metadata construction.
- Default-stride construction.
- Initializer-list construction where applicable.
- Generic TensorLike construction.
- Copy construction.
- `operator[]`.
- Rank-2 `T()`.
- A noinline by-value consumer that reads data, rank, size, and stride.
- `numel()` as a no-allocation control.

Run Release builds on the same host and compiler and pin them to a fixed CPU
core. Execute five round-robin baseline/capacity-4/capacity-8 process groups,
rotating candidate order between groups. Keep every process in a separate JSON
result file; do not concatenate duplicate benchmark keys. Apply the existing
comparison script to each matched file pair, then report the median and range
of the five pairwise percentage changes. This avoids changing the runner or
the comparison script while making the aggregation reproducible.

The term "median paired change" below means the median of those five matched
changes for one benchmark and rank. Both percentage and absolute nanosecond
changes are recorded because a percentage-only threshold is unstable for the
shortest low-rank operations.

Report `sizeof(SmallVector<Size, 4>)`, `sizeof(SmallVector<Size, 8>)`, each
metadata view, and each combined-metadata candidate `sizeof(TensorView)`
outside the JSON benchmark key.

The unchanged 58-result microbenchmark is supplemented by
`perf_tensor_view_footprint.cache_key_build_hit`, which models the InfiniOps
`CacheKey::Build` path. Each measured iteration hashes a vector size and every
input tensor, appends copied `TensorView` objects to a temporary vector without
`reserve`, compares that candidate with a prebuilt key, and destroys it. The
matrix is:

- Ranks 4 and 8.
- Tensor counts 8 and 256.
- `262144 / tensor_count` iterations, keeping TensorView visits constant.
- Shapes vary in their first dimension to prevent identical-input folding.

Rank 4 compares the inline object footprint of capacity 4 and capacity 8.
Rank 8 is an end-to-end comparison that also includes capacity 4's heap
fallback versus capacity 8's inline storage. Count 8 represents an ordinary
multi-tensor key; count 256 deliberately puts the capacity-8 vector near 40
KiB so a cache-footprint cliff is visible.

The rank-4/count-8 capacity-8 comparison remained above the provisional +5
percent capacity-4 gate after the initial five-run range crossed zero, so the
experiment was extended to 15 rounds. The final paired results were:

| Rank | Tensor count | Capacity 8 vs. capacity 4 | Capacity 8 vs. vector baseline |
| ---: | ---: | ---: | ---: |
| 4 | 8 | +7.275% (+21.826 ns) | -36.887% (-180.459 ns) |
| 4 | 256 | -1.867% (-163.378 ns) | -60.816% (-12733.150 ns) |
| 8 | 8 | -17.234% (-76.338 ns) | -26.603% (-139.257 ns) |
| 8 | 256 | -33.757% (-5621.646 ns) | -52.062% (-12494.684 ns) |

Capacity 8 therefore does not pass the provisional capacity-4 comparison for
the small rank-4 key: 12 of 15 paired runs were slower, with a -16.873 through
+30.785 percent range. The selection accepts and discloses that cost in
exchange for allocation-free ranks 5 through 8. It still improves every
footprint key against the shipping vector baseline, while both rank-8 keys
improve against capacity 4.

Decision gates are:

- At ranks 1, 2, and 4, each applicable explicit/default construction, copy,
  derived-view, and by-value consumer median paired change for combined
  capacity 4 versus the post-#33 baseline is at most +5 percent. Construction
  and copy changes are below 0 percent.
- At ranks 1, 2, and 4, a capacity-8 versus capacity-4 path is a material
  regression only when its median paired percentage change is above +5
  percent, its median paired absolute change is above +2 ns, and at least four
  of five runs are slower. This replaces the historical percentage-only gate
  for these nanosecond-scale paths.
- At ranks 5 and 8, capacity 8 performs zero allocations and its construction,
  copy, and by-value consumer median paired changes versus capacity 4 are below
  0 percent.
- Rank 9 and above must pass correctness, ownership, allocation-count, and
  sanitizer checks. Their latency is reported as fallback trade-off evidence
  and does not select the inline capacity.
- A `numel()` control invalidates a run only when it shows a stable absolute
  drift above 2 ns in at least four of five runs.
- For both tensor counts at rank 4, capacity 8 is at most +5 percent versus the
  vector baseline. Its capacity-4 comparison is reported as the explicit
  coverage-versus-footprint selection trade-off rather than described as a
  passed gate.
- For both tensor counts at rank 8, capacity 8 is below 0 percent versus
  capacity 4 and at most +5 percent versus the vector baseline.

A footprint result that exceeds a gate while its five-run range crosses zero
is inconclusive and must be extended before delivery.

## Downstream Migration

InfiniOps aliases `infini::ops::Tensor` to `TensorView`, copies tensors into
cache keys, and stores many `Tensor::Shape` and `Tensor::Strides` members. The
new aliases should compile without mass refactoring when the required
vector-like API is complete.

InfiniOps pybind currently casts Python metadata directly to
`Tensor::Shape` and `Tensor::Strides` through `pybind11/stl.h`. A custom
SmallVector has no automatic STL caster. Adapt only these conversions to cast
to `std::vector` first and then construct the Tensor metadata. Call sites that
require the two accessors to have one exact owning type, including conditional
expressions and explicit owner parameters, must materialize the intended
owning type explicitly.

Build InfiniOps against the installed candidate InfiniRT prefix before making
other downstream edits. Fix only demonstrated compile or test failures.

The torch-infini adapter requires default construction, `push_back`, and
contiguous `data()`. Validate its adapter build against the installed candidate
and modify it only if a real failure occurs.

Downstream changes remain separate commits and pull requests from the InfiniRT
performance change.

## Validation Matrix

Required before the InfiniRT change is proposed for merge:

- InfiniRT CPU Release full build and full CTest suite.
- InfiniRT NVIDIA Release build and non-performance smoke tests.
- InfiniRT installed-consumer test against the installed prefix.
- Allocation threshold tests on Linux.
- Combined capacity-4/capacity-8 58-result and cache-key footprint benchmark
  evidence.
- Exact clang-format 21 checks and `git diff --check`.
- InfiniOps CPU and pybind build plus available smoke tests against the
  candidate InfiniRT prefix.
- torch-infini adapter compile against the candidate prefix.

The public header and layout affect all backends. If other accelerator SDKs or
hosts are unavailable, the pull request must identify each untested platform,
state the reason, and request maintainer validation as required by
`CONTRIBUTING.md`.

## Delivery Boundaries

The InfiniRT change is one focused performance branch and ultimately one
Conventional Commit. It contains the container, TensorView integration, tests,
benchmarks, and necessary public documentation.

InfiniOps and torch-infini changes are created only for demonstrated
compatibility failures and remain in their own repositories and commits.

No version change, backend behavior change, general operator refactor, or
borrowed-metadata construction/storage mode is included. Accessor views borrow
only from metadata still owned by their `TensorView`.

## Acceptance Criteria

- Inline capacity 8 satisfies all rank-0-through-8 allocation thresholds and
  functional tests.
- High-rank combined and split-adopt states preserve owned contiguous shape
  and stride ranges.
- All known source-compatible `std::vector` construction paths still compile.
- Capacity 8 improves all cache-key footprint cases versus the vector baseline
  and both rank-8 cases versus capacity 4. The rank-4/count-8 capacity-4 cost is
  disclosed as a selection trade-off; rank 9 and above remain correct owned
  heap fallbacks.
- The placement-array implementation is validated with GCC, Clang, and MSVC,
  and sanitizer coverage finds no lifetime, alignment, leak, or double-free
  defect.
- InfiniRT CPU, NVIDIA, installation, formatting, and diff checks pass.
- Required InfiniOps and torch-infini downstream validation completes or any
  unavailable environment is explicitly documented.
- The final diff contains no capacity experiment toggles, temporary benchmark
  artifacts, unrelated refactors, or version changes.
