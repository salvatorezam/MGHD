#!/usr/bin/env python
"""MGHD preflight and validator script."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_deps() -> Dict[str, Optional[str]]:
    import importlib.metadata as im

    versions: Dict[str, Optional[str]] = {}

    try:
        pm_version = im.version("PyMatching")
        versions["pymatching"] = pm_version
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ERROR: Unable to determine PyMatching version ({exc})", file=sys.stderr)
        raise SystemExit(1)

    from packaging.version import Version

    if Version(pm_version) < Version("2.3.0"):
        print("ERROR: PyMatching >= 2.3.0 is required for correlated matching.", file=sys.stderr)
        raise SystemExit(1)

    try:
        stim_version = im.version("stim")
        versions["stim"] = stim_version
    except Exception as exc:
        print(f"ERROR: stim is required ({exc})", file=sys.stderr)
        raise SystemExit(1)

    try:
        __import__("cudaq")
    except Exception:
        versions["cudaq"] = None
    else:
        try:
            versions["cudaq"] = im.version("cudaq")
        except Exception:
            versions["cudaq"] = "available"

    print(f"PyMatching: {pm_version}")
    print(f"Stim:       {stim_version}")
    if versions["cudaq"] is None:
        print("CUDA-Q:     not installed (smoke will be skipped)")
    else:
        print(f"CUDA-Q:     {versions['cudaq']}")

    return versions


def run(cmd: Iterable[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_list = list(cmd)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd_list) + "\n")
        log.flush()
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
            log.write(line)
            output_chunks.append(line)
        retcode = process.wait()
    return subprocess.CompletedProcess(cmd_list, retcode, "".join(output_chunks), "")


LER_PATTERN = re.compile(r"LER_(dem|mix)\s*=\s*([0-9]+\.?[0-9eE+-]*)")


def parse_ler(stdout: str) -> Dict[str, Optional[float]]:
    results: Dict[str, Optional[float]] = {"dem": None, "mix": None}
    for match in LER_PATTERN.finditer(stdout):
        key, value = match.groups()
        try:
            results[key] = float(value)
        except ValueError:
            continue
    if results["mix"] is None:
        plain_match = re.search(r"LER\s*=\s*([0-9]+\.?[0-9eE+-]*)", stdout)
        if plain_match:
            try:
                results["mix"] = float(plain_match.group(1))
            except ValueError:
                pass
    return results


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MGHD preflight and validator")
    parser.add_argument("--families", default="surface")
    parser.add_argument("--distances", default="5")
    parser.add_argument("--shots-per-batch", type=int, default=64)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--dem-rounds", type=int, default=5)
    parser.add_argument("--qpu-profile", default="qpu_profiles/iqm_garnet_example.json")
    parser.add_argument("--context-source", default="qiskit")
    parser.add_argument("--max-ler-dem", type=float, default=0.10)
    parser.add_argument("--max-ler-mix", type=float, default=0.10)
    parser.add_argument("--pytest", dest="run_pytest", action="store_true", default=True)
    parser.add_argument("--no-pytest", dest="run_pytest", action="store_false")
    parser.add_argument("--skip-cudaq", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)

    artifact_root = Path("artifacts/preflight")
    artifact_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "versions": {},
        "pytest": None,
        "stim_dem": None,
        "cudaq": None,
    }

    section("Dependency check")
    versions = check_deps()
    summary["versions"] = versions

    pytest_pass = True
    if args.run_pytest:
        section("PyTest suite")
        result = run([sys.executable, "-m", "pytest", "-q"], artifact_root / "pytest.txt")
        pytest_pass = result.returncode == 0
        summary["pytest"] = {"returncode": result.returncode}
        if not pytest_pass:
            print("PyTest failed.")
    else:
        summary["pytest"] = {"skipped": True}

    section("Stim + DEM validator")
    stim_cmd = [
        sys.executable,
        "-m",
        "tools.train_core",
        "--families",
        args.families,
        "--distances",
        args.distances,
        "--sampler",
        "stim",
        "--dem-enable",
        "--dem-correlated",
        "--dem-rounds",
        str(args.dem_rounds),
        "--shots-per-batch",
        str(args.shots_per_batch),
        "--batches",
        str(args.batches),
    ]
    stim_result = run(stim_cmd, artifact_root / "stim_dem.txt")
    stim_lers = parse_ler(stim_result.stdout)
    stim_ok = stim_result.returncode == 0
    if stim_lers["dem"] is not None:
        stim_ok &= stim_lers["dem"] <= args.max_ler_dem
    if stim_lers["mix"] is not None:
        stim_ok &= stim_lers["mix"] <= args.max_ler_mix
    summary["stim_dem"] = {
        "returncode": stim_result.returncode,
        "ler": stim_lers,
        "pass": stim_ok,
    }

    cudaq_info = {
        "skipped": False,
        "returncode": None,
        "ler": None,
    }
    cudaq_available = versions.get("cudaq") is not None and not args.skip_cudaq
    if cudaq_available:
        section("CUDA-Q smoke")
        cudaq_cmd = [
            sys.executable,
            "-m",
            "tools.train_core",
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
        cudaq_result = run(cudaq_cmd, artifact_root / "cudaq.txt")
        cudaq_lers = parse_ler(cudaq_result.stdout)
        cudaq_info.update(
            {
                "returncode": cudaq_result.returncode,
                "ler": cudaq_lers,
            }
        )
        if cudaq_result.returncode != 0:
            print("CUDA-Q smoke returned a non-zero exit code (ignored for overall pass).", file=sys.stderr)
    else:
        section("CUDA-Q smoke")
        print("CUDA-Q not installed or explicitly skipped; marking as skipped.")
        cudaq_info["skipped"] = True
    summary["cudaq"] = cudaq_info

    summary_path = artifact_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary written to {summary_path}")

    ok = pytest_pass and summary["stim_dem"]["pass"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
'''
path.write_text(new_text)
