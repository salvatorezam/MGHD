#!/usr/bin/env python
"""MGHD preflight and validator.

Runs a compact readiness suite:
- Dependency checks (versions of PyMatching/Stim and CUDA‑Q availability)
- Optional pytest run
- Stim + DEM correlated matching A/B with LER thresholds
- Optional CUDA‑Q smoke (teacher_eval with cudaq sampler)

Artifacts are written under ``artifacts/preflight`` and a final summary is
printed and saved as JSON. Non‑blocking failures are reported but do not abort
the entire run; blocking failures raise PreflightError early.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
import os
import shutil
from pathlib import Path


@dataclass
class StepResult:
    """One row in the preflight summary table."""

    name: str
    status: str
    details: dict[str, object]

    def to_summary(self) -> dict[str, object]:
        data = {"status": self.status}
        data.update(self.details)
        return data


def section(title: str) -> None:
    """Print a section header to the console."""
    print(f"\n=== {title} ===")


class PreflightError(RuntimeError):
    """Raised when a blocking preflight error occurs."""


def check_deps(log_path: Path, *, expect_conda_env: str | None = None) -> dict[str, object]:
    """Verify core dependencies and emit a log; return version info dict.

    Ensures PyMatching≥2.3.0 and Stim are importable; probes CUDA‑Q presence
    without failing the entire run when unavailable.
    """
    import importlib.metadata as im

    from packaging.version import Version  # type: ignore

    log_path.parent.mkdir(parents=True, exist_ok=True)
    versions: dict[str, object] = {"cudaq_available": False}

    # Record active conda env (if any)
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    versions["conda_env"] = conda_env

    def emit(line: str) -> None:
        print(line)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")

    log_path.write_text("", encoding="utf-8")

    pm_version = None
    for dist in ("pymatching", "PyMatching"):
        try:
            pm_version = im.version(dist)
        except im.PackageNotFoundError:
            continue
        else:
            versions["pymatching"] = pm_version
            break
    if pm_version is None:
        emit("ERROR: PyMatching is not installed; install PyMatching>=2.3.0.")
        raise PreflightError("pymatching missing")

    if Version(pm_version) < Version("2.3.0"):
        emit(f"ERROR: PyMatching version {pm_version} < 2.3.0; upgrade to run correlated matching.")
        raise PreflightError("pymatching too old")

    try:
        stim_version = im.version("stim")
    except im.PackageNotFoundError as exc:
        emit(f"ERROR: stim import failed ({exc}); install stim>=1.13.")
        raise PreflightError("stim missing") from exc
    else:
        versions["stim"] = stim_version

    emit(f"PyMatching version: {pm_version}")
    emit(f"Stim version:       {stim_version}")

    try:
        __import__("cudaq")
    except Exception:
        emit("CUDA-Q:            not installed (smoke test will be skipped)")
        versions["cudaq"] = None
    else:
        versions["cudaq_available"] = True
        try:
            cudaq_version = im.version("cudaq")
        except im.PackageNotFoundError:
            cudaq_version = "installed"
        versions["cudaq"] = cudaq_version
        emit(f"CUDA-Q version:   {cudaq_version}")

    # Optionally probe a specific conda env for CUDA-Q if not available here
    if not versions.get("cudaq_available") and expect_conda_env:
        conda_bin = shutil.which("conda")
        if conda_bin:
            try:
                cp = subprocess.run(
                    [
                        conda_bin,
                        "run",
                        "-n",
                        expect_conda_env,
                        "python",
                        "-c",
                        "import importlib.metadata as im; import cudaq; print(im.version('cudaq'))",
                    ],
                    capture_output=True,
                    text=True,
                )
                if cp.returncode == 0:
                    v = cp.stdout.strip().splitlines()[-1] if cp.stdout else "installed"
                    versions["cudaq_in_env"] = v
                    emit(f"CUDA-Q in env '{expect_conda_env}': {v}")
                else:
                    versions["cudaq_in_env"] = None
                    emit(f"CUDA-Q in env '{expect_conda_env}': not installed or import failed")
            except Exception:
                versions["cudaq_in_env"] = None
        else:
            versions["cudaq_in_env"] = None

    return versions


def run(cmd: Iterable[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, tee output to a log, and return the CompletedProcess."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_list = list(cmd)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd_list) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output_chunks: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            output_chunks.append(line)
        retcode = process.wait()
    return subprocess.CompletedProcess(cmd_list, retcode, "".join(output_chunks), "")


LER_PATTERN = re.compile(
    r"LER(?:_(?P<kind>mix))?\s*[:=]\s*(?P<value>[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"
)


def parse_ler(stdout: str) -> dict[str, float | None]:
    """Extract LER_mix (if present) from teacher_eval output."""
    results: dict[str, float | None] = {"mix": None}
    for match in LER_PATTERN.finditer(stdout):
        kind = match.group("kind") or "mix"
        try:
            results[kind] = float(match.group("value"))
        except ValueError:
            continue
    return results


def write_skip_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message + "\n", encoding="utf-8")
    print(message)


def summarize(rows: Iterable[StepResult]) -> None:
    """Print a compact text table for the collected StepResults."""
    section("Summary")
    header = f"{'Stage':<18}{'Status':<8}Details"
    print(header)
    print("-" * len(header))
    for row in rows:
        status = row.status.upper()
        detail = row.details.get("note") or row.details.get("message") or ""
        ler = row.details.get("ler")
        if isinstance(ler, dict):
            ler_parts = [
                f"LER_mix={ler['mix']:.4f}" if ler.get("mix") is not None else None,
            ]
            ler_text = ", ".join(filter(None, ler_parts))
            detail = f"{detail} {ler_text}".strip()
        print(f"{row.name:<18}{status:<8}{detail}")


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entrypoint for MGHD preflight.

    Returns process‑style exit code (0 on pass, 1 on fail)."""
    parser = argparse.ArgumentParser("MGHD preflight and validator")
    parser.add_argument("--families", default="surface")
    parser.add_argument("--distances", default="5")
    parser.add_argument("--shots-per-batch", type=int, default=64)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--qpu-profile", default="mghd/qpu/profiles/iqm_garnet_example.json")
    parser.add_argument("--context-source", default="qiskit")
    parser.add_argument("--max-ler-mix", type=float, default=0.10)
    parser.add_argument(
        "--pytest",
        dest="run_pytest",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--skip-cudaq", action="store_true")
    parser.add_argument(
        "--expect-conda-env",
        default=None,
        help="If set, also probe CUDA-Q inside this conda env (e.g., 'mlqec-env')",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(__file__).resolve().parents[1]
    artifact_root = repo_root / "artifacts" / "preflight"
    artifact_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[StepResult] = []
    summary_json: dict[str, object] = {"arguments": vars(args)}

    # 1) Dependencies
    section("Dependency Check")
    deps_log = artifact_root / "deps.txt"
    try:
        versions = check_deps(deps_log, expect_conda_env=args.expect_conda_env)
    except PreflightError as exc:
        summary_json["versions"] = {}
        summary_rows.append(StepResult("dependencies", "fail", {"message": str(exc)}))
        summary_json["steps"] = {row.name: row.to_summary() for row in summary_rows}
        summary_path = artifact_root / "summary.json"
        summary_path.write_text(json.dumps(summary_json, indent=2) + "\n", encoding="utf-8")
        print(f"Summary written to {summary_path}")
        summarize(summary_rows)
        return 1

    summary_json["versions"] = {k: v for k, v in versions.items() if k != "cudaq_available"}
    summary_rows.append(StepResult("dependencies", "pass", {"message": "versions ok"}))

    # 2) PyTest
    pytest_result: subprocess.CompletedProcess[str] | None = None
    if args.run_pytest:
        section("PyTest")
        pytest_result = run([sys.executable, "-m", "pytest", "-q"], artifact_root / "pytest.txt")
        pytest_status = "pass" if pytest_result.returncode == 0 else "fail"
        pytest_details: dict[str, object] = {"returncode": pytest_result.returncode}
        if pytest_status == "fail":
            pytest_details["note"] = "pytest failed"
        summary_rows.append(StepResult("pytest", pytest_status, pytest_details))
    else:
        section("PyTest")
        write_skip_log(artifact_root / "pytest.txt", "pytest skipped by --no-pytest flag")
        summary_rows.append(StepResult("pytest", "skip", {"note": "skipped"}))

    # 3) Stim teacher A/B
    section("Stim Teacher A/B")
    stim_cmd = [
        sys.executable,
        "-m",
        "mghd.tools.teacher_eval",
        "--families",
        args.families,
        "--distances",
        args.distances,
        "--sampler",
        "stim",
        "--rounds",
        str(args.rounds),
        "--shots-per-batch",
        str(args.shots_per_batch),
        "--batches",
        str(args.batches),
    ]
    stim_result = run(stim_cmd, artifact_root / "stim.txt")
    stim_ler = parse_ler(stim_result.stdout)
    stim_ok = stim_result.returncode == 0
    notes: list[str] = []
    if stim_ler["mix"] is not None and stim_ler["mix"] > args.max_ler_mix:
        stim_ok = False
        notes.append(f"LER_mix {stim_ler['mix']:.4f} > {args.max_ler_mix:.4f}")
    if stim_ler["mix"] is None:
        notes.append("LER metrics not found")
    stim_status = "pass" if stim_ok else "fail"
    stim_detail = {"returncode": stim_result.returncode, "ler": stim_ler}
    if notes:
        stim_detail["note"] = "; ".join(notes)
    summary_rows.append(StepResult("stim", stim_status, stim_detail))

    # 4) CUDA-Q smoke
    cudaq_log = artifact_root / "cudaq.txt"
    cudaq_available = bool(versions.get("cudaq_available")) and not args.skip_cudaq
    if cudaq_available:
        section("CUDA-Q Smoke")
        cudaq_cmd = [
            sys.executable,
            "-m",
            "mghd.tools.teacher_eval",
            "--families",
            args.families,
            "--distances",
            args.distances,
            "--sampler",
            "cudaq",
            "--qpu-profile",
            args.qpu_profile,
            "--context-source",
            args.context_source,
            "--shots-per-batch",
            str(args.shots_per_batch),
            "--batches",
            str(args.batches),
        ]
        cudaq_result = run(cudaq_cmd, cudaq_log)
        cudaq_ler = parse_ler(cudaq_result.stdout)
        cudaq_status = "pass" if cudaq_result.returncode == 0 else "fail"
        cudaq_detail = {"returncode": cudaq_result.returncode, "ler": cudaq_ler}
        if cudaq_status == "fail":
            cudaq_detail["note"] = "non-zero exit (ignored for overall pass)"
        summary_rows.append(StepResult("cudaq", cudaq_status, cudaq_detail))
    else:
        section("CUDA-Q Smoke")
        reason = (
            "CUDA-Q not available"
            if not versions.get("cudaq_available")
            else "skipped via --skip-cudaq"
        )
        write_skip_log(cudaq_log, reason)
        summary_rows.append(StepResult("cudaq", "skip", {"note": reason}))

    summary_json["steps"] = {row.name: row.to_summary() for row in summary_rows}
    summary_json.setdefault("versions", {})
    summary_json["cudaq_available"] = bool(versions.get("cudaq_available"))
    if "conda_env" in versions:
        summary_json["conda_env"] = versions.get("conda_env")
    if "cudaq_in_env" in versions:
        summary_json["cudaq_in_env"] = versions.get("cudaq_in_env")

    summary_path = artifact_root / "summary.json"
    summary_path.write_text(json.dumps(summary_json, indent=2) + "\n", encoding="utf-8")
    print(f"Summary written to {summary_path}")

    summarize(summary_rows)

    overall_ok = True
    for row in summary_rows:
        if row.status == "skip":
            continue
        if row.status != "pass":
            overall_ok = False
            break
    if not any(row.name == "stim" and row.status == "pass" for row in summary_rows):
        overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
