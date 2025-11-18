import json
from pathlib import Path

from mghd.tools import plot_run


def test_plot_run_fallback_summary(tmp_path: Path, monkeypatch):
    # Ensure matplotlib is unavailable to exercise JSON summary fallback
    monkeypatch.setitem(plot_run.__dict__, "_safe_import_matplotlib", lambda: None)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_log.json").write_text(
        json.dumps(
            [
                {"epoch": 1, "loss": 0.3},
                {"epoch": 2, "loss": 0.2},
            ]
        )
    )
    (run_dir / "run_meta.json").write_text(json.dumps({"family": "surface", "distance": 3}))
    # No teacher_eval.txt

    plot_run.plot_run(run_dir)
    # No images in fallback; summary.json is written
    assert (run_dir / "summary.json").exists()
