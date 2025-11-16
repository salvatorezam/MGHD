#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _parse_teacher_eval(txt_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not txt_path.exists():
        return out
    try:
        txt = txt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out
    # Extract LER mean if present like: "LER=1.23e-03"
    m = re.search(r"LER=([0-9eE+\-\.]+)", txt)
    if m:
        try:
            out["ler_mean"] = float(m.group(1))
        except Exception:
            pass
    # Extract teacher-usage dict if present
    m2 = re.search(r"teacher-usage=\{([^}]*)\}", txt)
    if m2:
        try:
            entries = [p.strip() for p in m2.group(1).split(",") if p.strip()]
            usage: Dict[str, int] = {}
            for e in entries:
                if ":" in e:
                    k, v = e.split(":", 1)
                    usage[k.strip().strip("'\"")] = int(v.strip())
            out["teacher_usage"] = usage
        except Exception:
            pass
    return out


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def plot_run(run_dir: Path, out_dir: Path | None = None, title: str | None = None) -> List[Path]:
    out_dir = out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_log = _load_json(run_dir / "train_log.json") or []
    meta = _load_json(run_dir / "run_meta.json") or {}
    teacher = _parse_teacher_eval(run_dir / "teacher_eval.txt")

    plt = _safe_import_matplotlib()
    outputs: List[Path] = []
    if plt is None:
        # Fallback: write a compact JSON summary if matplotlib is unavailable
        summary = {
            "n_epochs": len(train_log),
            "loss_first": (train_log[0]["loss"] if train_log else None),
            "loss_last": (train_log[-1]["loss"] if train_log else None),
            "meta": meta,
            "teacher": teacher,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return outputs

    # Plot: loss vs epoch; optional p schedule on twin axis
    epochs = [e.get("epoch", i + 1) for i, e in enumerate(train_log)]
    losses = [float(e.get("loss", 0.0)) for e in train_log]
    ps = [e.get("p") for e in train_log]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, losses, marker="o", label="train loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    if any(p is not None for p in ps):
        ax2 = ax.twinx()
        ax2.plot(
            epochs,
            [p if p is not None else float("nan") for p in ps],
            color="tab:orange",
            label="p (online)",
        )
        ax2.set_ylabel("p")
        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax.legend(loc="best")

    ttl = title or f"MGHD run — {meta.get('family', '?')} d={meta.get('distance', '?')}"
    ax.set_title(ttl)
    fig.tight_layout()
    out1 = out_dir / "loss.png"
    fig.savefig(out1)
    plt.close(fig)
    outputs.append(out1)

    # Optional: teacher usage bar if present
    usage = teacher.get("teacher_usage")
    if usage:
        fig2, axb = plt.subplots(figsize=(5, 3))
        keys = list(usage)
        vals = [usage[k] for k in keys]
        axb.bar(
            keys, vals, color=["#6699cc" if k.strip("'\"") == "mwpf" else "#99cc66" for k in keys]
        )
        axb.set_title("Teacher usage (post-eval)")
        for i, v in enumerate(vals):
            axb.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        fig2.tight_layout()
        out2 = out_dir / "teacher_usage.png"
        fig2.savefig(out2)
        plt.close(fig2)
        outputs.append(out2)

    # Write a small manifest of generated assets
    manifest = {"outputs": [str(p) for p in outputs], "meta": meta, "teacher": teacher}
    (out_dir / "plots_manifest.json").write_text(json.dumps(manifest, indent=2))
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot MGHD training run artifacts.")
    ap.add_argument(
        "run_dirs", nargs="+", help="One or more run directories containing train_log.json"
    )
    ap.add_argument(
        "--out-dir", type=str, default=None, help="Output directory (defaults to each run dir)"
    )
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    out: List[str] = []
    for rd in args.run_dirs:
        rd_path = Path(rd)
        od = Path(args.out_dir) if args.out_dir else rd_path
        outs = plot_run(rd_path, out_dir=od, title=args.title)
        out.extend([str(p) for p in outs])
    print(json.dumps({"generated": out}, separators=(",", ":")))


if __name__ == "__main__":
    main()
