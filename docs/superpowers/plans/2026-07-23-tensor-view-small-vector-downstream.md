# TensorView SmallVector Downstream Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that the selected InfiniRT metadata layout can be consumed by current InfiniOps and torch-infini, making only compile-failure-driven compatibility edits in their own repositories.

**Architecture:** Install one selected InfiniRT candidate into an isolated prefix, build clean pinned downstream snapshots against that prefix, and preserve repository ownership boundaries. InfiniOps converts Python metadata through `std::vector` before constructing `TensorView` metadata; torch-infini should compile unchanged against the required vector-like API.

**Tech Stack:** C++17, CMake/Ninja, pybind11, Python, pytest, PyTorch CPU wheels, pip wheel, `readelf`, and isolated Linux source/build/install directories.

---

This plan follows `docs/superpowers/specs/2026-07-23-tensor-view-small-vector-design.md` and runs only after capacity selection by `docs/superpowers/plans/2026-07-23-tensor-view-small-vector.md`. Downstream edits are separate commits and pull requests. Do not copy `SmallVector` into another repository, add a generic pybind caster, pin a project version, or refactor operator metadata.

Use these audited snapshots:

- InfiniOps: `fd15321d7849a4bde7595414afdbe46c95b62241`
- torch-infini: `8e96889b1d8329afa49ec3d43428dacd14e3ced9`

The existing local torch-infini branch is not a validation base because it has unmerged commits and a deleted upstream. Use a clean detached `origin/master` snapshot.

Run Tasks 1 through 3 in one disposable shell so Python installation and loader changes do not affect the host:

```powershell
ssh -t nvidia docker run --rm -it `
  -v /tmp/tensor-view-small-vector:/tmp/tensor-view-small-vector `
  -v /tmp/infinirt-small-vector-cpu:/tmp/infinirt-small-vector-cpu:ro `
  -v /tmp/infinirt-small-vector-nvidia:/tmp/infinirt-small-vector-nvidia:ro `
  accelerator-dev/nvidia:latest bash
```

Before starting, run this preflight in that container and use the same `python3` executable throughout:

```bash
export TV_PYTHON="$(command -v python3)"
test -n "$TV_PYTHON"
cmake --version
ninja --version
c++ --version
"$TV_PYTHON" -c 'import clang, pybind11, pytest, torch, wheel, yaml'
readelf --version
```

Treat a missing tool or module as environment setup failure, not a project failure.

## Task 1: Install the Selected InfiniRT Candidate

**Files:**

- No source edits.
- Install under `/tmp/tensor-view-small-vector/infinirt`.

- [ ] **Step 1: Verify selected source identity**

Run on the `nvidia` Linux host after the main plan creates `/tmp/tensor-view-small-vector/selected` from the final selected bundle:

```bash
export TV_RT_SRC=/tmp/tensor-view-small-vector/selected
export TV_RT_BUILD=/tmp/tensor-view-small-vector/build-infinirt-downstream
export TV_RT_PREFIX=/tmp/tensor-view-small-vector/infinirt
TV_RT_SHA="$(git -C "$TV_RT_SRC" rev-parse HEAD)"
test "$TV_RT_SHA" = \
  "$(git -C /tmp/infinirt-small-vector-cpu rev-parse HEAD)"
test "$TV_RT_SHA" = \
  "$(git -C /tmp/infinirt-small-vector-nvidia rev-parse HEAD)"
git -C "$TV_RT_SRC" status --short
```

The three SHAs must equal the final one-commit InfiniRT branch SHA recorded after consolidation, and the selected worktree must be clean.

- [ ] **Step 2: Configure, build, test installation, and install**

```bash
export TV_RT_SRC=/tmp/tensor-view-small-vector/selected
export TV_RT_BUILD=/tmp/tensor-view-small-vector/build-infinirt-downstream
export TV_RT_PREFIX=/tmp/tensor-view-small-vector/infinirt

cmake -S "$TV_RT_SRC" -B "$TV_RT_BUILD" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$TV_RT_PREFIX" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DAUTO_DETECT_DEVICES=OFF \
  -DINFINI_RT_BUILD_TESTING=ON \
  -DINFINI_RT_BUILD_PERFORMANCE_TESTING=OFF \
  -DWITH_CPU=ON
cmake --build "$TV_RT_BUILD" --parallel 2
ctest --test-dir "$TV_RT_BUILD" \
  -R '^test_install(_consumer)?$' \
  --output-on-failure
cmake --install "$TV_RT_BUILD"
```

Expected result: both install tests pass.

- [ ] **Step 3: Verify installed artifacts**

```bash
test -f /tmp/tensor-view-small-vector/infinirt/include/infini/rt.h
test -f /tmp/tensor-view-small-vector/infinirt/include/infini/rt/detail/common/small_vector.h
test -f /tmp/tensor-view-small-vector/infinirt/lib/libinfinirt.so
```

`CMAKE_INSTALL_LIBDIR=lib` fixes the library directory used by every later command.

## Task 2: Reproduce and Fix the InfiniOps Pybind Boundary

**Files:**

- Modify only after RED: `src/pybind11_utils.h`

- [ ] **Step 1: Create a clean pinned source**

```bash
git init /tmp/tensor-view-small-vector/InfiniOps
git -C /tmp/tensor-view-small-vector/InfiniOps remote add origin \
  https://github.com/InfiniTensor/InfiniOps.git
git -C /tmp/tensor-view-small-vector/InfiniOps fetch --depth=1 origin \
  fd15321d7849a4bde7595414afdbe46c95b62241
git -C /tmp/tensor-view-small-vector/InfiniOps checkout --detach FETCH_HEAD
test "$(git -C /tmp/tensor-view-small-vector/InfiniOps rev-parse HEAD)" = \
  fd15321d7849a4bde7595414afdbe46c95b62241
```

Read that checkout's `CONTRIBUTING.md` before editing. Do not use `scripts/dev/build.sh` because it forces `WITH_TORCH=ON`.

- [ ] **Step 2: Run the unmodified CPU and pybind build**

```bash
export TV_RT_PREFIX=/tmp/tensor-view-small-vector/infinirt
export TV_OPS_SRC=/tmp/tensor-view-small-vector/InfiniOps
export TV_OPS_BUILD=/tmp/tensor-view-small-vector/build-infiniops
export TV_OPS_PYROOT=/tmp/tensor-view-small-vector/infiniops-python
export TV_OPS_PREFIX="$TV_OPS_PYROOT/infini"
export TV_PYTHON="$(command -v python3)"

cmake -S "$TV_OPS_SRC" -B "$TV_OPS_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$TV_OPS_PREFIX" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DPython_EXECUTABLE="$TV_PYTHON" \
  -DINFINI_RT_ROOT="$TV_RT_PREFIX" \
  -DAUTO_DETECT_DEVICES=OFF \
  -DAUTO_DETECT_BACKENDS=OFF \
  -DWITH_CPU=ON \
  -DWITH_TORCH=OFF \
  -DGENERATE_OPERATOR_CALL_INSTANTIATIONS=ON \
  -DGENERATE_PYTHON_BINDINGS=ON \
  -DINFINI_OPS_SMOKE_BUILD=ON
cmake --build "$TV_OPS_BUILD" --target ops --parallel 2
```

Expected RED: compilation reaches `TensorFromPybind11Handle` and rejects direct `pybind11/stl.h` conversion to `Tensor::Shape` or `Tensor::Strides`. Save the first diagnostic. If the build succeeds, do not edit; continue to smoke tests and record that no patch is needed.

- [ ] **Step 3: Replace only the two exact-type casts**

Create the repository-compliant branch after observing RED:

```bash
git -C /tmp/tensor-view-small-vector/InfiniOps \
  switch -c fix/tensor-view-metadata-pybind
```

Change:

```cpp
auto shape{obj.attr("shape").cast<typename Tensor::Shape>()};
auto strides{obj.attr("stride")().cast<typename Tensor::Strides>()};
```

to:

```cpp
auto shape_values{obj.attr("shape").cast<std::vector<Tensor::Size>>()};
Tensor::Shape shape{shape_values.begin(), shape_values.end()};

auto strides_values{
    obj.attr("stride")().cast<std::vector<Tensor::Stride>>()};
Tensor::Strides strides{strides_values.begin(), strides_values.end()};
```

Keep:

```cpp
return Tensor{data, std::move(shape), dtype, device, std::move(strides)};
```

Do not register a general `type_caster<SmallVector>`.

- [ ] **Step 4: Rebuild, install, and run CPU smoke tests**

```bash
export TV_RT_PREFIX=/tmp/tensor-view-small-vector/infinirt
export TV_OPS_SRC=/tmp/tensor-view-small-vector/InfiniOps
export TV_OPS_BUILD=/tmp/tensor-view-small-vector/build-infiniops
export TV_OPS_PYROOT=/tmp/tensor-view-small-vector/infiniops-python
export TV_OPS_PREFIX="$TV_OPS_PYROOT/infini"
export TV_PYTHON="$(command -v python3)"

cmake --build "$TV_OPS_BUILD" --target ops --parallel 2
cmake --install "$TV_OPS_BUILD"

cd "$TV_OPS_SRC"
PYTHONPATH="$TV_OPS_PYROOT${PYTHONPATH:+:$PYTHONPATH}" \
CPLUS_INCLUDE_PATH="$TV_RT_PREFIX/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}" \
LIBRARY_PATH="$TV_RT_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
LD_LIBRARY_PATH="$TV_OPS_PREFIX:$TV_OPS_PREFIX/lib:$TV_RT_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
INFINI_OPS_INSTALL_PREFIX="$TV_OPS_PREFIX" \
"$TV_PYTHON" -m pytest tests -m smoke -q --devices cpu
```

The smoke set must include `tests/test_add.py`, which executes `TensorFromPybind11Handle`, and `tests/test_cpp_api.py`, which compiles a public-header consumer.

- [ ] **Step 5: Record downstream footprint context**

```bash
rg -o 'Tensor::(Shape|Strides)' \
  /tmp/tensor-view-small-vector/InfiniOps/src/base/add.h
rg -o 'Tensor::(Shape|Strides)' \
  /tmp/tensor-view-small-vector/InfiniOps/src/base/flash_attn_varlen_func.h
```

Expected counts are 6 and 12. Include them and the capacity-4 versus capacity-8 per-container size delta in the InfiniRT report. Do not refactor these members.

- [ ] **Step 6: Commit only after the reproduced failure**

Create `fix/tensor-view-metadata-pybind` and commit:

```bash
git -C /tmp/tensor-view-small-vector/InfiniOps \
  add src/pybind11_utils.h
git -C /tmp/tensor-view-small-vector/InfiniOps \
  commit -m "fix: adapt Tensor metadata pybind conversion"
```

If the unmodified build passed, leave InfiniOps detached and clean and record `no source change required`.

## Task 3: Validate the torch-infini Adapter From an Installed Wheel

**Files:**

- Expected source changes: none.
- Diagnose: `csrc/infini_ops.cpp`
- Diagnose: `csrc/infini_ops.h`

- [ ] **Step 1: Create a clean pinned source**

```bash
git init /tmp/tensor-view-small-vector/torch-infini
git -C /tmp/tensor-view-small-vector/torch-infini remote add origin \
  https://github.com/InfiniTensor/torch-infini.git
git -C /tmp/tensor-view-small-vector/torch-infini fetch --depth=1 origin \
  8e96889b1d8329afa49ec3d43428dacd14e3ced9
git -C /tmp/tensor-view-small-vector/torch-infini checkout --detach FETCH_HEAD
test "$(git -C /tmp/tensor-view-small-vector/torch-infini rev-parse HEAD)" = \
  8e96889b1d8329afa49ec3d43428dacd14e3ced9
```

Use `README.md` and `.github/workflows/cpu.yml` as repository-native guidance; this repository has no `CONTRIBUTING.md` or `DEV.md`. Do not reuse the stale InfiniRT/InfiniOps SHAs pinned in that workflow.

- [ ] **Step 2: Build the wheel without source edits**

```bash
export TV_RT_PREFIX=/tmp/tensor-view-small-vector/infinirt
export TV_OPS_PYROOT=/tmp/tensor-view-small-vector/infiniops-python
export TV_OPS_PREFIX="$TV_OPS_PYROOT/infini"
export TV_TORCH_SRC=/tmp/tensor-view-small-vector/torch-infini
export TV_WHEELHOUSE=/tmp/tensor-view-small-vector/wheelhouse
export TV_TORCH_RUN=/tmp/tensor-view-small-vector/installed-test
export TV_PYTHON="$(command -v python3)"
mkdir "$TV_WHEELHOUSE"
mkdir "$TV_TORCH_RUN"

INFINI_RT_PREFIX="$TV_RT_PREFIX" \
INFINI_OPS_PREFIX="$TV_OPS_PREFIX" \
LD_LIBRARY_PATH="$TV_OPS_PREFIX/lib:$TV_RT_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$TV_PYTHON" -m pip wheel "$TV_TORCH_SRC" \
  --wheel-dir "$TV_WHEELHOUSE" \
  --no-build-isolation \
  --no-deps
```

The adapter's `to_shape` and `to_strides` paths require default construction, `reserve`, `push_back`, copying, and contiguous iteration. A successful wheel build proves those C++ uses compile.

- [ ] **Step 3: Install and test outside the source checkout**

```bash
export TV_RT_PREFIX=/tmp/tensor-view-small-vector/infinirt
export TV_OPS_PYROOT=/tmp/tensor-view-small-vector/infiniops-python
export TV_OPS_PREFIX="$TV_OPS_PYROOT/infini"
export TV_TORCH_SRC=/tmp/tensor-view-small-vector/torch-infini
export TV_WHEELHOUSE=/tmp/tensor-view-small-vector/wheelhouse
export TV_TORCH_RUN=/tmp/tensor-view-small-vector/installed-test
export TV_PYTHON="$(command -v python3)"

"$TV_PYTHON" -m pip install --force-reinstall --no-deps \
  "$TV_WHEELHOUSE"/torch_infini-*.whl

cd "$TV_TORCH_RUN"
LD_LIBRARY_PATH="$TV_OPS_PREFIX/lib:$TV_RT_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
"$TV_PYTHON" -c 'import pathlib, torch_infini; source = pathlib.Path("/tmp/tensor-view-small-vector/torch-infini").resolve(); loaded = pathlib.Path(torch_infini.__file__).resolve(); assert source not in loaded.parents; print(loaded)'

INFINI_RT_PREFIX="$TV_RT_PREFIX" \
INFINI_OPS_PREFIX="$TV_OPS_PREFIX" \
LD_LIBRARY_PATH="$TV_OPS_PREFIX/lib:$TV_RT_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
TORCH_INFINI_TEST_EXPECTED_BACKEND=cpu \
"$TV_PYTHON" -m pytest -q \
  "$TV_TORCH_SRC/tests/test_infini_ops.py" \
  "$TV_TORCH_SRC/tests/test_add.py"
```

Expected result: the wheel imports from site-packages and both selected test files pass.

- [ ] **Step 4: Inspect native linkage**

```bash
export TV_RT_PREFIX=/tmp/tensor-view-small-vector/infinirt
export TV_OPS_PYROOT=/tmp/tensor-view-small-vector/infiniops-python
export TV_OPS_PREFIX="$TV_OPS_PYROOT/infini"
export LD_LIBRARY_PATH="$TV_OPS_PREFIX/lib:$TV_RT_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TV_PYTHON="$(command -v python3)"
TV_EXTENSION="$("$TV_PYTHON" -c 'import torch_infini._C; print(torch_infini._C.__file__)')"
readelf -d "$TV_EXTENSION"
```

Record `DT_NEEDED` entries for `libinfiniops.so` and `libinfinirt.so`. Record `RPATH` or `RUNPATH`; a source/build-tree path is a failure.

- [ ] **Step 5: Respond only to demonstrated failures**

If compilation fails because an approved vector-like operation is missing, fix `src/common/small_vector.h` in InfiniRT, rerun InfiniRT CPU/NVIDIA/install validation, reinstall both prefixes, and restart this downstream plan.

If torch-infini depends on a concrete `std::vector` behavior outside the approved surface, create `fix/tensor-view-metadata-compat` and make the smallest adapter-local conversion:

```bash
git -C /tmp/tensor-view-small-vector/torch-infini \
  switch -c fix/tensor-view-metadata-compat
git -C /tmp/tensor-view-small-vector/torch-infini \
  add csrc/infini_ops.cpp csrc/infini_ops.h
git -C /tmp/tensor-view-small-vector/torch-infini commit \
  -m "fix: adapt TensorView metadata construction"
```

Do not stage a header that did not change. If the wheel and tests pass unchanged, create no torch-infini branch, commit, or pull request.

## Task 4: Return Verified Evidence to the InfiniRT PR Task

**Files:**

- No repository source edits.
- Return a complete evidence block before the main plan creates the InfiniRT pull request.

- [ ] **Step 1: Capture reproducibility data**

Record:

- selected InfiniRT SHA, installed prefix, compiler, and `sizeof` lines;
- InfiniOps SHA, exact configure/build/test commands, test count, and compatibility commit if needed;
- torch-infini SHA, wheel filename, installed module path, pytest result, and `readelf -d` evidence;
- first failure diagnostic for every source edit;
- `no source change required` for a repository that compiled unchanged.

- [ ] **Step 2: Prepare separate downstream pull requests only where needed**

For InfiniOps, use its `CONTRIBUTING.md` and PR template. For torch-infini, use its available template and CI conventions. State that consumers must build against the selected InfiniRT commit.

- [ ] **Step 3: Hand evidence to the main plan**

Provide the InfiniOps metadata-member footprint counts for `Benchmark / Performance Impact`. Provide exact downstream results and links to required compatibility PRs for `Smoke Build and Test Result` and `Notes for Reviewers`. The main plan writes this evidence into `docs/superpowers/pr-body.md` before creating the InfiniRT PR. Do not advance to PR creation while a required downstream build is failing.
