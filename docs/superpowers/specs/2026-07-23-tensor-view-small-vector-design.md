# TensorView Inline Metadata Storage Design

Date: 2026-07-23

Status: Approved

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

InfiniRT has not had a formal release. This work does not change the project
version or add an `SOVERSION`.

## Goals

- Make `TensorView` construction, copying, indexing, and transposition perform
  no heap allocations while rank fits the selected inline capacity.
- Retain owned metadata and value semantics.
- Preserve the vector-like operations used by InfiniRT, InfiniOps, and the
  known framework adapters.
- Keep metadata contiguous and expose stable `data()` and iterator ranges.
- Support arbitrary practical ranks by falling back to heap storage.
- Select inline capacity 4 or 8 using measured end-to-end `TensorView`
  performance rather than rank frequency alone.
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
- Preserve existing `TensorView` constructors and accessors at the source
  level where their arguments use vector-like ranges.
- Preserve initializer-list and `std::vector` construction.
- Preserve iteration, indexing, size queries, contiguous data, and equality.
- Permit source changes where callers require the exact `std::vector` type,
  depend on its allocator, or use a `std::vector`-specific caster.
- Require every consumer to rebuild against the matching installed headers and
  library.

## Alternatives

### Two SmallVector Members

Replace `Shape` and `Strides` with two instances of an in-tree
`SmallVector<T, N>`. This keeps the current `TensorView` model and lets
InfiniOps operator metadata members benefit from the same inline storage.

This is the selected approach. Its main cost is a larger `TensorView` object,
especially at inline capacity 8. That cost is part of the benchmark decision.

### Combined Tensor Metadata Storage

Store shape and strides in one TensorView-specific inline or heap block. This
could reduce object size and use one overflow allocation, but would change the
accessor model more deeply and would not improve InfiniOps members typed as
`Tensor::Shape` or `Tensor::Strides`.

This remains a fallback only if both SmallVector capacities fail the measured
performance gates.

### Third-Party Small Vectors

LLVM, Boost, and Abseil provide mature inline containers. Each option would
still change the public API and ABI while adding a dependency to installed
headers and consumers. InfiniRT currently has no comparable runtime container
dependency, so these options are rejected.

## SmallVector Design

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

Inline copies copy their elements into the destination object. Heap copies
allocate independent storage. Inline moves copy at most `N` trivial elements;
heap moves transfer the pointer without allocating. A moved-from object must
remain destructible and assignable, but is not required to be empty.

## TensorView Integration

`TensorView::Shape` and `TensorView::Strides` become aliases of
`SmallVector<Size, kInlineCapacity>` and
`SmallVector<Stride, kInlineCapacity>`. The final inline capacity is a source
constant, not a public build option, because different capacities produce
binary-incompatible object layouts.

Generic `TensorView` constructors build metadata from iterator ranges instead
of relying on exact-type conversion. This preserves construction from
`std::vector`, framework shape objects, and the new SmallVector type.

Existing `TensorView` behavior remains unchanged for:

- Default dtype, device, and contiguous stride generation.
- Scalar and high-rank tensors.
- Positive and negative indexing.
- Two-dimensional transposition.
- Hashing and equality.
- Copy and move constructibility.
- Deleted assignment caused by the existing `const dtype_` member.

## Inline Capacity Experiment

The generic container supports both capacities, but the shipped `TensorView`
uses exactly one.

1. Implement and validate a capacity-4 TensorView candidate.
2. Record allocation counts, object sizes, and performance results.
3. Change only the TensorView capacity constant to 8.
4. Extend the threshold tests and rerun the same commands and benchmarks.
5. Keep capacity 8 only when it satisfies every benchmark decision gate below,
   including the 5 percent low-rank regression limit.

Capacity 8 must also preserve correctness through rank 9 and satisfy the
rank-5 and rank-8 benchmark gates below. Object sizes are reported separately;
they are not hidden in benchmark parameters or allocation counts.

If capacity 4 regresses any listed low-rank benchmark median paired change by
more than 5 percent relative to the post-#33 baseline, stop the SmallVector
integration and revisit combined metadata storage rather than merging an
allocation-only win.

## Test-Driven Development

Production changes follow red-green-refactor cycles.

### Allocation Thresholds

For the capacity-4 candidate, tests first require:

- Rank 0 through 4 lvalue, rvalue, initializer-list, default-stride, and generic
  TensorLike construction: zero allocations.
- Rank 5 lvalue explicit metadata, exact-type rvalue temporaries created inside
  the measured expression, initializer-list metadata, and generic TensorLike
  construction: two allocations, one for each overflow container.
- Rank 5 lvalue and exact-type rvalue-temporary default-stride construction:
  two allocations, one for shape and one for generated strides.
- Moving preconstructed rank-5 shape and strides into explicit-metadata
  construction: zero allocations.
- Moving a preconstructed rank-5 shape into default-stride construction: one
  allocation for generated strides.

For the capacity-8 candidate, new failing thresholds require:

- Rank 0 through 8 construction paths: zero allocations.
- Rank 9 follows the same path-specific expectations as rank 5 above: two
  allocations for lvalue, measured exact-type temporaries, initializer-list,
  generic TensorLike, and ordinary default-stride construction; zero for
  moving preconstructed explicit metadata; and one for moving a preconstructed
  shape while generating default strides.

Input containers are prepared outside allocation scopes except where the test
specifically measures rvalue or initializer-list construction.

### Value Semantics

- Inline copy construction performs zero allocations and owns independent
  storage.
- Overflow copy construction performs two allocations and owns independent
  storage.
- Inline and overflow move construction perform zero allocations.
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

## Benchmark Design

The post-#33 merge commit is the baseline. Capacity 4 and capacity 8 are built
with the same compiler, optimization level, source apart from the capacity
constant, and benchmark harness.

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
percentage changes for one benchmark and rank.

Report `sizeof(SmallVector<Size, 4>)`, `sizeof(SmallVector<Size, 8>)`, and each
candidate `sizeof(TensorView)` outside the JSON benchmark key.

Decision gates are:

- At ranks 1, 2, and 4, each applicable explicit/default construction, copy,
  derived-view, and by-value consumer median paired change for capacity 4
  versus the post-#33 baseline is at most +5 percent. Construction and copy
  changes are below 0 percent.
- At ranks 1, 2, and 4, the same capacity-8 versus capacity-4 median paired
  changes are at most +5 percent.
- At rank 9, each candidate's explicit/default construction, copy, and by-value
  consumer median paired changes versus the post-#33 vector baseline are at
  most +5 percent.
- At ranks 5 and 8, capacity 8 performs zero allocations and its construction,
  copy, and by-value consumer median paired changes versus capacity 4 are below
  0 percent.
- Every `numel()` control median paired change has an absolute value of at most
  5 percent.

## Downstream Migration

InfiniOps aliases `infini::ops::Tensor` to `TensorView`, copies tensors into
cache keys, and stores many `Tensor::Shape` and `Tensor::Strides` members. The
new aliases should compile without mass refactoring when the required
vector-like API is complete.

InfiniOps pybind currently casts Python metadata directly to
`Tensor::Shape` and `Tensor::Strides` through `pybind11/stl.h`. A custom
SmallVector has no automatic STL caster. Adapt only these conversions to cast
to `std::vector` first and then construct the Tensor metadata.

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
- Capacity-4 and capacity-8 benchmark evidence.
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
borrowed-metadata API is included.

## Acceptance Criteria

- The selected capacity satisfies all allocation thresholds and functional
  tests.
- High-rank fallback preserves owned contiguous metadata.
- All known source-compatible `std::vector` construction paths still compile.
- The selected capacity satisfies the benchmark decision gates.
- InfiniRT CPU, NVIDIA, installation, formatting, and diff checks pass.
- Required InfiniOps and torch-infini downstream validation completes or any
  unavailable environment is explicitly documented.
- The final diff contains no capacity experiment toggles, temporary benchmark
  artifacts, unrelated refactors, or version changes.
