"""Compare every allocation strategy across the CPU and device backends.

The question this answers is three questions, and they need different builds to
answer honestly:

  direct vs arena      Is the arena worth having at all on this backend?
  pool vs arena        Which pool design wins, and on which shapes?
  arena vs cuda_async  Does a hand-written arena beat the vendor's own pool?

The third only exists on a device, and the first has a completely different
answer on each backend -- a host `malloc` costs tens of nanoseconds, so nothing
the arena saves can pay for its bookkeeping, while a `cudaMalloc` costs hundreds
of microseconds and the same arena wins by orders of magnitude. Reporting one
number for "the allocator" would average those into something true of neither.
So this configures and runs both builds and prints each comparison per backend.

Usage:

    python scripts/compare_allocators.py                 # build, run, compare
    python scripts/compare_allocators.py --quick         # shorter arms
    python scripts/compare_allocators.py --no-build      # reuse what is built
    python scripts/compare_allocators.py --backend cpu   # one backend only

A note on why the builds are serialized rather than parallel: `generated/` is
written into the *source* tree at configure time and its contents depend on which
backends are enabled, so two configured build trees cannot be compiled
concurrently -- the second would compile against the first's headers. Each
backend is therefore configured, built, and only then is anything run.
"""

import argparse
import json
import pathlib
import platform
import re
import subprocess
import sys

# The executables that emit per-allocator rows. Only `perf_allocator_matrix`
# does: it is the one binary that runs every arm over one set of workloads, which
# is what makes its rows pivotable into the tables below. `perf_memory_pool`
# covers the general shapes, but only over `direct`/`arena`, and it is driven by
# `scripts/run_performance_tests.py` instead.
_MATRIX_TESTS = ("perf_allocator_matrix",)

# Each backend's build tree, the CMake options that select it, and whether a
# device is required. Ordered CPU-first so a machine without a GPU still gets a
# useful run before anything fails.
_BACKENDS = (
    {
        "name": "cpu",
        "build_dir": "build-perf-cpu",
        "options": ["-DWITH_CPU=ON", "-DWITH_NVIDIA=OFF"],
    },
    {
        "name": "nvidia",
        "build_dir": "build-perf-cuda",
        "options": ["-DWITH_CPU=OFF", "-DWITH_NVIDIA=ON"],
    },
)

# The two comparisons, as (baseline, candidate) arm pairs. `arena` is the
# candidate in every one -- including against `cuda_async`, where the natural
# phrasing would put the vendor's pool second. Keeping the arena in the candidate
# column means the win column always answers the same question, "should we adopt
# this thing", instead of flipping direction in the middle of the report.
_COMPARISONS = (
    ("direct", "arena", "the backend allocator vs the arena"),
    ("cuda_async", "arena", "the vendor's stream-ordered pool vs the arena"),
)

# Units where a larger number is the better outcome. Everything else here is a
# cost -- a duration, a byte count, a call count -- and lower wins.
_HIGHER_IS_BETTER = frozenset({"GiB/s", "seq_per_s"})

# Ratio rows are already a ratio; comparing two of them is meaningless.
#
# `x` covers both the retention-amplification rows and the fragmentation probe
# success rate. The latter is the one row here where a *ratio* is the primary
# result rather than a derived one, and it is deliberately not turned into a win
# column: 8/8 vs 8/8 is the expected outcome for both designs, and a 1.00x win
# column would read as "no difference measured" rather than "both succeeded".
_RATIO_UNITS = frozenset({"x"})

# Diagnostics rather than costs. `LedgerAccuracy` asks whether one allocator's
# self-reported retention matches what the driver says it took -- a question about
# that allocator's honesty, answered by the binary's own table. Racing two arms'
# answers would print a win column for a row where neither arm is competing.
_DIAGNOSTIC_BENCHMARKS = frozenset({"allocator_matrix.LedgerAccuracy"})


def _repo_root():
    return pathlib.Path(__file__).resolve().parents[1]


def _run(command, **kwargs):
    printable = " ".join(str(part) for part in command)
    print(f"$ {printable}", flush=True)
    return subprocess.run(command, check=True, **kwargs)


def _configure_and_build(backend, jobs):
    """Configure and compile one backend's tree.

    Configuring rewrites `generated/` in the source tree, so this must complete
    before another backend is configured.
    """
    build_dir = _repo_root() / backend["build_dir"]
    _run(
        [
            "cmake",
            "-S",
            str(_repo_root()),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DINFINI_RT_BUILD_PERFORMANCE_TESTING=ON",
            *backend["options"],
        ],
        stdout=subprocess.DEVNULL,
    )
    for test in _MATRIX_TESTS:
        _run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--target",
                test,
                "-j",
                str(jobs),
            ],
            stdout=subprocess.DEVNULL,
        )
    return build_dir


def _find_executable(build_dir, name):
    for candidate in (
        build_dir / "tests" / "performance" / name,
        build_dir / "tests" / "performance" / "Release" / name,
        build_dir / name,
    ):
        if candidate.exists():
            return candidate
    return None


def _run_matrix(build_dir, quick):
    """Run one backend's benchmarks and return the parsed JSON result lines.

    stderr is forwarded rather than captured: it carries the binary's own
    per-backend table and its skip messages, and a run that skipped an arm is
    something the reader needs to see next to the numbers.
    """
    results = []
    for test in _MATRIX_TESTS:
        executable = _find_executable(build_dir, test)
        if executable is None:
            print(f"  {test}: not built, skipping", file=sys.stderr)
            continue

        command = [str(executable)]
        if quick:
            command.append("--quick")
        completed = subprocess.run(
            command, cwd=executable.parent, text=True, capture_output=True,
            check=False,
        )
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{executable} exited with status {completed.returncode}"
            )
        for line in completed.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("{"):
                results.append(json.loads(stripped))
    return results


def _git(args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_repo_root(), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _row_key(result):
    """Identity of a measurement with the allocator dimension removed.

    Two rows collapse to the same key exactly when they are the same workload at
    the same parameters measured on different arms, which is what makes them
    comparable. `arena_config` is deliberately part of the key: the reduced host
    config and the production device config are not the same measurement, and
    merging them would silently compare a 8 MB ramp against a 64 MB one.
    """
    params = {
        name: value
        for name, value in result.get("params", {}).items()
        if name != "allocator"
    }
    # `arena_config` stays in the key but out of the rendered label: it is
    # constant within a backend, so printing it on every row is noise, while
    # keeping it in the key is what stops a reduced-config host row from being
    # merged with a production-config device row.
    config = params.pop("arena_config", "")
    rendered = ", ".join(f"{name}={value}" for name, value in sorted(params.items()))
    return (result["benchmark"], rendered, config)


def _pivot(results):
    """Group results into {(benchmark, params, config): {arm: result}}."""
    table = {}
    for result in results:
        arm = result.get("params", {}).get("allocator")
        if arm is None:
            continue
        table.setdefault(_row_key(result), {})[arm] = result
    return table


def _shorten(benchmark):
    return re.sub(r"^allocator_matrix\.", "", benchmark)


def _format_value(value, unit):
    if unit == "count":
        return f"{value:,.0f}"
    if unit == "bytes":
        return f"{value / (1024 * 1024):,.1f} MiB"
    # Sub-microsecond latencies and single-digit-percent probe rates both lose
    # their meaning at two decimals, so small magnitudes get more.
    if unit in ("us", "ms", "x") and 0.0 < abs(value) < 1.0:
        return f"{value:,.4f}"
    return f"{value:,.2f}"


def _speedup(baseline, candidate, unit):
    """How many times better the candidate is. None where that has no meaning.

    A zero on either side is not reported as a ratio: for a call count zero is a
    real and important outcome (the arena served a whole workload without going
    upstream once), and dividing by it would turn the best result in the table
    into a blank.
    """
    if unit in _RATIO_UNITS:
        return None
    good, bad = (candidate, baseline) if unit in _HIGHER_IS_BETTER else (
        baseline,
        candidate,
    )
    if good <= 0.0 or bad <= 0.0:
        return None
    return good / bad


def _describe_zero(baseline, candidate, baseline_arm, candidate_arm):
    if baseline == 0.0 and candidate == 0.0:
        return "both zero"
    if candidate == 0.0:
        return f"{candidate_arm} zero"
    if baseline == 0.0:
        return f"{baseline_arm} zero"
    return "-"


def _print_comparison(backend, table, baseline_arm, candidate_arm, caption):
    rows = []
    configs = set()
    for (benchmark, params, config), arms in table.items():
        if benchmark in _DIAGNOSTIC_BENCHMARKS:
            continue
        if baseline_arm not in arms or candidate_arm not in arms:
            continue
        configs.add(config)
        rows.append(
            (_shorten(benchmark), params, arms[baseline_arm], arms[candidate_arm])
        )

    if not rows:
        print(
            f"\n[{backend}] {baseline_arm} vs {candidate_arm}: "
            "no comparable rows (an arm did not run on this backend)."
        )
        return

    rows.sort(key=lambda row: (row[0], row[1]))

    # Sized to the content rather than to a guess, so a long parameter list
    # cannot push the numeric columns out of alignment.
    name_width = max(len("workload"), *(len(row[0]) for row in rows)) + 2
    params_width = max(len("params"), *(len(row[1]) for row in rows)) + 2

    config_note = f" (arena config: {', '.join(sorted(configs))})" if configs else ""
    print(f"\n=== [{backend}] {caption}{config_note} ===")
    win_label = f"{candidate_arm} win"
    win_width = max(len(win_label), 12) + 2
    header = (
        f"{'workload':<{name_width}}{'params':<{params_width}}"
        f"{baseline_arm:>16}{candidate_arm:>16}{'unit':>9}"
        f"{win_label:>{win_width}}"
    )
    print(header)
    print("-" * len(header))

    for workload, params, baseline, candidate in rows:
        unit = baseline.get("unit", "")
        base_value = baseline["median"]
        cand_value = candidate["median"]
        ratio = _speedup(base_value, cand_value, unit)
        if ratio is None:
            verdict = _describe_zero(
                base_value, cand_value, baseline_arm, candidate_arm
            )
        else:
            verdict = f"{ratio:,.2f}x"
        print(
            f"{workload:<{name_width}}{params:<{params_width}}"
            f"{_format_value(base_value, unit):>16}"
            f"{_format_value(cand_value, unit):>16}"
            f"{unit:>9}{verdict:>{win_width}}"
        )

    print(
        f"\n`{candidate_arm} win` > 1 means {candidate_arm} is better on that row. "
        "Rows in\nthe `x` unit are already ratios, so no win column is computed "
        "for them."
    )
    if candidate_arm == "cuda_async" or baseline_arm == "cuda_async":
        print(
            "cuda_async is stream-ordered: its release does not wait for pending\n"
            "device work, so it offers a weaker guarantee than the other arms and\n"
            "these ratios are not a drop-in speedup."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare allocation strategies across backends."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer iterations and thread counts. Shapes hold, noise is higher.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Run whatever is already built instead of configuring first.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        default=None,
        choices=[backend["name"] for backend in _BACKENDS],
        help="Limit to one backend. Can be passed more than once.",
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Write the merged raw results here as JSON.",
    )
    args = parser.parse_args()

    selected = [
        backend
        for backend in _BACKENDS
        if args.backends is None or backend["name"] in args.backends
    ]

    metadata = {
        "commit": _git(["rev-parse", "HEAD"]),
        "system": platform.platform(),
        "quick": args.quick,
    }

    # Build every selected backend before running any of them. Not an
    # optimization -- configuring rewrites the shared `generated/` tree, so a
    # build interleaved with another backend's configure would compile against
    # the wrong headers.
    build_dirs = {}
    for backend in selected:
        if args.no_build:
            build_dirs[backend["name"]] = _repo_root() / backend["build_dir"]
            continue
        try:
            build_dirs[backend["name"]] = _configure_and_build(backend, args.jobs)
        except subprocess.CalledProcessError:
            print(
                f"{backend['name']}: configure or build failed, skipping it.",
                file=sys.stderr,
            )

    merged = []
    for backend in selected:
        build_dir = build_dirs.get(backend["name"])
        if build_dir is None or not build_dir.exists():
            continue

        print(f"\n### running {backend['name']} ###", flush=True)
        try:
            results = _run_matrix(build_dir, args.quick)
        except (RuntimeError, OSError) as exc:
            print(f"{backend['name']}: {exc}", file=sys.stderr)
            continue

        for result in results:
            result.update(metadata)
        merged.extend(results)

        table = _pivot(results)
        for baseline_arm, candidate_arm, caption in _COMPARISONS:
            _print_comparison(
                backend["name"], table, baseline_arm, candidate_arm, caption
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(merged, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {len(merged)} raw results to {args.output}")

    if not merged:
        print("\nno results: nothing ran successfully.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
