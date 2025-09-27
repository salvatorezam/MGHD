#!/usr/bin/env python3
import os, json, subprocess, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

def run(cmd):
    return subprocess.check_output(cmd, text=True).strip()

def tracked(pattern):
    try:
        out = run(["git", "ls-files", pattern])
        return [p for p in out.splitlines() if p]
    except Exception:
        # fallback: walk the tree
        out = []
        for d, _, fs in os.walk(ROOT):
            for f in fs:
                p = os.path.relpath(os.path.join(d, f), ROOT)
                if p.endswith(".py"):
                    out.append(p)
        return out

EXCLUDES = ("archive/", "results/", "Plots and Data/", "experiments/", "fastpath/", ".pytest_cache/", "__pycache__/", ".git/")

def is_active(path):
    return not any(seg in path for seg in EXCLUDES)

py = tracked("*.py")
active = sorted(p for p in py if is_active(p))
archived = sorted(set(py) - set(active))

summary = {
    "root": ROOT,
    "counts": {"python_all": len(py), "python_active": len(active), "python_archived": len(archived)},
    "active_top": [p for p in active if p.split(os.sep)[0] in {"mghd_public", "mghd_clustered", "teachers", "cudaq_backend", "training", "tools", "tests"}],
    "by_dir": {}
}

for p in active:
    head = p.split(os.sep)[0]
    summary["by_dir"].setdefault(head, 0)
    summary["by_dir"][head] += 1

print(json.dumps(summary, indent=2))
