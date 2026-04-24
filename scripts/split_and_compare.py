from __future__ import annotations

import csv
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
IMAGES_DIR = DATASET / "images"
LABELS_DIR = DATASET / "labels"
TRAIN_IMAGES = IMAGES_DIR / "train"
VAL_IMAGES = IMAGES_DIR / "val"
TRAIN_LABELS = LABELS_DIR / "train"
VAL_LABELS = LABELS_DIR / "val"
RUNS_DIR = ROOT / "runs"
PLOTS_DIR = ROOT / "plots"
RESULTS_DIR = ROOT / "results"
YOLO_CONFIG_DIR = ROOT / ".ultralytics"
SEED = 42
EPOCHS = 80
MODELS = {
    "yolov8n": "yolov8n.yaml",
    "yolov8s": "yolov8s.yaml",
}


def iter_image_files(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts])


def class_from_name(path: Path) -> str:
    stem = path.stem
    idx = next((i for i, ch in enumerate(stem) if ch.isdigit()), len(stem))
    return stem[:idx]


def clear_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for item in directory.iterdir():
        if item.is_file():
            item.unlink()


def stratified_split() -> dict[str, dict[str, int]]:
    clear_directory(VAL_IMAGES)
    clear_directory(VAL_LABELS)

    pairs_by_class: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for image_path in iter_image_files(TRAIN_IMAGES):
        label_path = TRAIN_LABELS / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path.name}")
        pairs_by_class[class_from_name(image_path)].append((image_path, label_path))

    rng = random.Random(SEED)
    summary: dict[str, dict[str, int]] = {}
    for class_name, pairs in sorted(pairs_by_class.items()):
        rng.shuffle(pairs)
        val_count = max(1, math.floor(len(pairs) * 0.2))
        val_pairs = pairs[:val_count]

        for image_path, label_path in val_pairs:
            shutil.move(str(image_path), str(VAL_IMAGES / image_path.name))
            shutil.move(str(label_path), str(VAL_LABELS / label_path.name))

        summary[class_name] = {
            "total": len(pairs),
            "train": len(pairs) - val_count,
            "val": val_count,
        }

    classes_txt = TRAIN_LABELS / "classes.txt"
    if classes_txt.exists():
        shutil.copy2(classes_txt, VAL_LABELS / "classes.txt")

    return summary


def train_and_evaluate(model_name: str, model_cfg: str) -> dict[str, float | str]:
    model = YOLO(model_cfg)
    train_results = model.train(
        data=str(ROOT / "data.yaml"),
        epochs=EPOCHS,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        cache=False,
        pretrained=False,
        project=str(RUNS_DIR),
        name=f"{model_name}_e{EPOCHS}",
        exist_ok=True,
    )

    best_weight = Path(train_results.save_dir) / "weights" / "best.pt"
    val_model = YOLO(str(best_weight))
    metrics = val_model.val(
        data=str(ROOT / "data.yaml"),
        split="val",
        imgsz=640,
        batch=1,
        device=0,
        workers=0,
        project=str(RUNS_DIR),
        name=f"{model_name}_e{EPOCHS}_val",
        exist_ok=True,
    )

    inference_ms = float(metrics.speed["inference"])
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    return {
        "model": model_name,
        "train_dir": str(train_results.save_dir),
        "best_weight": str(best_weight),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "inference_ms": inference_ms,
        "fps": fps,
    }


def plot_comparison(results: list[dict[str, float | str]]) -> Path:
    PLOTS_DIR.mkdir(exist_ok=True)
    models = [str(item["model"]) for item in results]
    map50 = [float(item["map50"]) for item in results]
    fps = [float(item["fps"]) for item in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(models, map50, color=["#4C78A8", "#F58518"])
    axes[0].set_title("Validation mAP@0.50")
    axes[0].set_ylabel("mAP")
    axes[0].set_ylim(0, max(map50 + [0.1]) * 1.2)
    for i, value in enumerate(map50):
        axes[0].text(i, value, f"{value:.3f}", ha="center", va="bottom")

    axes[1].bar(models, fps, color=["#54A24B", "#E45756"])
    axes[1].set_title("Inference Speed (FPS)")
    axes[1].set_ylabel("FPS")
    axes[1].set_ylim(0, max(fps + [1.0]) * 1.2)
    for i, value in enumerate(fps):
        axes[1].text(i, value, f"{value:.2f}", ha="center", va="bottom")

    fig.tight_layout()
    output_path = PLOTS_DIR / "yolov8n_vs_yolov8s_comparison.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def write_summary(split_summary: dict[str, dict[str, int]], results: list[dict[str, float | str]], plot_path: Path) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    csv_path = RESULTS_DIR / "model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "precision", "recall", "map50", "map50_95", "inference_ms", "fps", "train_dir", "best_weight"],
        )
        writer.writeheader()
        writer.writerows(results)

    payload = {
        "split_summary": split_summary,
        "epochs": EPOCHS,
        "results": results,
        "plot": str(plot_path),
        "notes": {
            "fps_definition": "1000 / inference_ms_per_image from Ultralytics validation output",
            "pretrained": False,
            "device": "cuda:0",
        },
    }
    with (RESULTS_DIR / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    random.seed(SEED)
    YOLO_CONFIG_DIR.mkdir(exist_ok=True)
    split_summary = stratified_split()
    results = [train_and_evaluate(name, cfg) for name, cfg in MODELS.items()]
    plot_path = plot_comparison(results)
    write_summary(split_summary, results, plot_path)
    print(json.dumps({"split_summary": split_summary, "results": results, "plot": str(plot_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
