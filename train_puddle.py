"""
Train baseline and improved YOLOv10 models for rainwater puddle detection.

Examples:
    python train_puddle.py --data D:/datasets/puddle/data.yaml --mode baseline
    python train_puddle.py --data D:/datasets/puddle/data.yaml --mode improved
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLOv10
from ultralytics.utils import LOGGER


ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
BASELINE_MODEL = ROOT / "ultralytics/cfg/models/v10/yolov10n.yaml"
IMPROVED_MODEL = ROOT / "ultralytics/cfg/models/v10/yolov10n_puddle.yaml"
ABLATION_P2 = ROOT / "ultralytics/cfg/models/v10/yolov10n_ablation_p2.yaml"
ABLATION_CBAM = ROOT / "ultralytics/cfg/models/v10/yolov10n_ablation_cbam.yaml"

MODE_MAP = {
    "baseline": (BASELINE_MODEL, "yolov10n_baseline"),
    "ablation_p2": (ABLATION_P2, "yolov10n_ablation_p2"),
    "ablation_cbam": (ABLATION_CBAM, "yolov10n_ablation_cbam"),
    "improved": (IMPROVED_MODEL, "yolov10n_puddle_improved"),
}
YOLOV10_RELEASE_URL = "https://github.com/THU-MIG/yolov10/releases/download/v1.1"
YOLOV10_RELEASE_WEIGHTS = {
    "yolov10n.pt",
    "yolov10s.pt",
    "yolov10m.pt",
    "yolov10b.pt",
    "yolov10l.pt",
    "yolov10x.pt",
}


def resolve_pretrained(pretrained: str) -> str | None:
    """Resolve pretrained argument to a local file or official release URL."""
    raw = (pretrained or "").strip()
    if not raw or raw.lower() in {"none", "scratch"}:
        return None

    path = Path(raw)
    if path.exists():
        return str(path)

    # Try common local locations first to avoid network instability.
    local_candidates = [ROOT / raw, WEIGHTS_DIR / raw]
    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate)

    name = path.name
    if name in YOLOV10_RELEASE_WEIGHTS:
        return f"{YOLOV10_RELEASE_URL}/{name}"

    return raw


def resolve_device(device: str) -> str:
    """Resolve training device with safe fallback when CUDA is unavailable."""
    raw = (device or "").strip()
    lower = raw.lower()

    if lower in {"", "auto"}:
        return "0" if torch.cuda.is_available() else "cpu"

    if lower != "cpu" and not torch.cuda.is_available():
        LOGGER.warning(
            f"Requested device='{raw}', but CUDA is unavailable. "
            "Fallback to CPU. Use '--device cpu' to silence this warning."
        )
        return "cpu"

    return raw


def train(
    data: str,
    mode: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    pretrained: str,
    project: str,
    strict_pretrained: bool,
):
    if mode not in MODE_MAP:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(MODE_MAP.keys())}")
    model_cfg, run_name = MODE_MAP[mode]
    model_cfg = str(model_cfg)

    model = YOLOv10(model_cfg)
    pretrained_source = resolve_pretrained(pretrained)
    if pretrained_source:
        LOGGER.info(f"Loading pretrained weights from: {pretrained_source}")
        try:
            model.load(pretrained_source)
        except Exception as e:  # noqa: BLE001
            if strict_pretrained:
                raise
            LOGGER.warning(f"Failed to load pretrained weights: {e}")
            LOGGER.warning(
                f"Continue with training from scratch. If needed, manually place weights at "
                f"'{WEIGHTS_DIR / 'yolov10n.pt'}' and rerun with --pretrained <local_path>."
            )
    else:
        LOGGER.warning("No pretrained weights provided, training from scratch.")

    resolved_device = resolve_device(device)
    LOGGER.info(f"Training device: {resolved_device}")

    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved_device,
        workers=workers,
        project=project,
        name=run_name,
        cos_lr=True,
        hsv_h=0.01,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=2.0,
        translate=0.08,
        scale=0.5,
        shear=0.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Puddle detection training (baseline/improved).")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--mode", choices=list(MODE_MAP.keys()), default="improved")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto", help="'auto', CUDA device id, or 'cpu'")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--pretrained",
        default="yolov10n.pt",
        help="Local .pt path, yolov10{n/s/m/b/l/x}.pt, URL, or 'none'/'scratch'.",
    )
    parser.add_argument(
        "--strict-pretrained",
        action="store_true",
        help="Raise error if pretrained weights cannot be loaded.",
    )
    parser.add_argument("--project", default="runs/puddle")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data=args.data,
        mode=args.mode,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        pretrained=args.pretrained,
        project=args.project,
        strict_pretrained=args.strict_pretrained,
    )
