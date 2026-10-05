"""Train the expanded baseline, then evaluate both checkpoints on common splits."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from xray_workbench.vision.error_analysis import per_class_average_precision

CLASSES = ("gun", "knife", "scissors", "lighter")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "state.json").exists():
        raise SystemExit("Run already has state; use a new run directory")
    state: dict[str, Any] = {
        "phase": "initializing",
        "pid": os.getpid(),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "epochs_completed": 0,
        "planned_epochs": 20,
        "device": "cpu",
        "image_size": 320,
        "batch_size": 16,
        "seed": 42,
        "preprocessing": "raw",
    }

    def save_state(**changes: Any) -> None:
        state.update(changes)
        state["updated_at_utc"] = datetime.now(UTC).isoformat()
        temporary = run_dir / "state.json.tmp"
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(run_dir / "state.json")

    save_state()
    try:
        audit_path = root / "data/processed/pidray-expanded/manifests/audit.json"
        audit = json.loads(audit_path.read_text())
        if any(audit["content_overlap_counts"].values()):
            raise RuntimeError("Dataset audit contains image-content leakage")
        if (audit["splits"]["train"]["images"], audit["splits"]["val"]["images"]) != (2000, 400):
            raise RuntimeError("Unexpected training or validation subset size")
        base = root / "models/base/yolo26n.pt"
        original = root / "training/experiments/pidray-baseline-cpu/weights/best.pt"
        planned_experiment = root / "training/experiments/pidray-expanded-cpu"
        if planned_experiment.exists():
            raise RuntimeError(f"Training directory already exists: {planned_experiment}")
        if not base.is_file() or not original.is_file():
            raise RuntimeError("Both local base weights and original baseline are required")
        save_state(dataset_audit=audit, base_weights_sha256=_hash(base))

        import ultralytics
        from ultralytics import YOLO  # type: ignore[attr-defined]

        started = perf_counter()

        def epoch_finished(trainer: Any) -> None:
            completed = int(trainer.epoch) + 1
            elapsed = perf_counter() - started
            save_state(
                phase="training",
                epochs_completed=completed,
                elapsed_training_seconds=elapsed,
                estimated_remaining_training_seconds=(20 - completed) * elapsed / completed,
                experiment_dir=str(trainer.save_dir),
                last_checkpoint=str(Path(trainer.save_dir) / "weights/last.pt"),
                epoch_metrics={key: float(value) for key, value in trainer.metrics.items()},
            )
            print(f"Epoch {completed}/20 completed; elapsed {elapsed:.1f}s", flush=True)

        model = YOLO(str(base))
        model.add_callback("on_fit_epoch_end", epoch_finished)
        save_state(phase="training")
        model.train(
            data=str(root / "config/datasets/pidray-expanded.yaml"),
            epochs=20,
            imgsz=320,
            device="cpu",
            batch=16,
            workers=0,
            seed=42,
            deterministic=True,
            project=str(root / "training/experiments"),
            name="pidray-expanded-cpu",
        )
        elapsed = perf_counter() - started
        if model.trainer is None:
            raise RuntimeError("Training completed without a trainer")
        best = Path(model.trainer.save_dir) / "weights/best.pt"
        if not best.is_file():
            raise RuntimeError("Training completed without a best checkpoint")
        save_state(phase="evaluating", training_duration_seconds=elapsed, best_checkpoint=str(best))
        evaluations: dict[str, Any] = {}
        error_summaries: dict[str, Any] = {}
        for label, checkpoint in (("original", original), ("expanded", best)):
            detector = YOLO(str(checkpoint))
            if tuple(detector.names[index] for index in range(len(detector.names))) != CLASSES:
                raise RuntimeError(f"Unexpected checkpoint class mapping: {checkpoint}")
            evaluations[label] = {}
            for split, yaml_path in (
                ("validation", "config/datasets/pidray-expanded.yaml"),
                ("hard", "config/datasets/pidray-hard-eval.yaml"),
                ("hidden", "config/datasets/pidray-hidden-eval.yaml"),
            ):
                metrics = detector.val(
                    data=str(root / yaml_path),
                    imgsz=320,
                    device="cpu",
                    workers=0,
                    plots=False,
                    verbose=False,
                    project=str(run_dir / "validation"),
                    name=f"{label}-{split}",
                )
                evaluations[label][split] = {
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                    "map50": float(metrics.box.map50),
                    "map50_95": float(metrics.box.map),
                    **per_class_average_precision(metrics.box, CLASSES),
                }
                save_state(current_evaluation=f"{label}/{split}")
            error_dir = run_dir / f"{label}-errors"
            subprocess.run(
                [
                    sys.executable,
                    str(root / "training/analyze_errors.py"),
                    "--model",
                    str(checkpoint),
                    "--output-dir",
                    str(error_dir),
                    "--device",
                    "cpu",
                ],
                check=True,
                cwd=root,
            )
            error_summaries[label] = json.loads((error_dir / "report.json").read_text())[
                "summaries"
            ]
        report = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset_audit": audit,
            "training": {
                "epochs": 20,
                "seed": 42,
                "image_size": 320,
                "batch_size": 16,
                "device": "cpu",
                "duration_seconds": elapsed,
                "base_weights": str(base),
                "base_weights_sha256": _hash(base),
            },
            "weights": {
                "original": {"path": str(original), "sha256": _hash(original)},
                "expanded": {"path": str(best), "sha256": _hash(best)},
            },
            "ultralytics": ultralytics.__version__,
            "evaluations": evaluations,
            "fixed_threshold_errors": error_summaries,
            "metric_deltas_expanded_minus_original": {
                split: {
                    key: evaluations["expanded"][split][key] - evaluations["original"][split][key]
                    for key in ("precision", "recall", "map50", "map50_95")
                }
                for split in ("validation", "hard", "hidden")
            },
            "limitations": [
                "Validation changed from easy-split images to 400 held-out train.json images.",
                "Both checkpoints were re-evaluated on the same common splits.",
                "Hard and hidden are diagnostic 160-image subsets used for iterative decisions.",
            ],
        }
        report_path = run_dir / "comparison.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        save_state(phase="complete", comparison_report=str(report_path))
        print(f"Experiment complete: {report_path}", flush=True)
    except BaseException:
        save_state(phase="failed", error=traceback.format_exc())
        raise


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    main()
