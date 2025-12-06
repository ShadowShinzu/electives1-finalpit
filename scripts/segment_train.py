#!/usr/bin/env python3
"""
Batch-segment durian images in a folder using the repository's deployment model.

Usage (PowerShell):
    python -m pip install -r requirements.txt
    python scripts/segment_train.py \
            --input "DSP_NewImages/train" \
            --output "DSP_NewImages/segmented" \
            --model "DSP_NewImages/checkpoints/deployment_model/durian_cnn_model.h5" \
            --class-info "DSP_NewImages/checkpoints/deployment_model/class_info.json" \
            --threshold 0.60

The script copies images into subfolders `ripe/`, `unripe/` and `uncertain/` under the output directory
and produces a CSV `segmentation_report.csv` with per-file predictions.
"""
import argparse
import os
import sys
import shutil
import csv
from pathlib import Path
import importlib.util


def load_classifier_from_file(classifier_py_path, model_path, class_info_path):
    """Dynamically load DurianRipenessClassifier from a python source file."""
    spec = importlib.util.spec_from_file_location("durian_classifier", str(classifier_py_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Expect class DurianRipenessClassifier to exist
    cls = getattr(module, "DurianRipenessClassifier")
    return cls(model_path=model_path, class_info_path=class_info_path)


def find_image_files(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in Path(folder).iterdir():
        if p.suffix.lower() in exts and p.is_file():
            yield p


def main():
    parser = argparse.ArgumentParser(description="Segment durian train images into ripe/unripe/uncertain")
    parser.add_argument("--input", default=os.path.join("DSP_NewImages", "train"), required=False, help="Input images folder (default: DSP_NewImages/train)")
    parser.add_argument("--output", default="segmented", required=False, help="Output folder to create segmented subfolders (default: ./segmented)")
    parser.add_argument("--classifier-py", default=os.path.join("DSP_NewImages", "checkpoints", "deployment_model", "durian_classifier.py"), help="Path to durian_classifier.py")
    parser.add_argument("--model", default=os.path.join("DSP_NewImages", "checkpoints", "deployment_model", "durian_cnn_model.h5"), help="Keras model file path")
    parser.add_argument("--class-info", default=os.path.join("DSP_NewImages", "checkpoints", "deployment_model", "class_info.json"), help="JSON file with class info")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold to accept prediction (0-1)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate input directory early and exit with helpful message
    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        print("Tip: run from the repository root or provide an absolute path to --input.")
        sys.exit(1)

    # destinations
    ripe_dir = output_dir / "ripe"
    unripe_dir = output_dir / "unripe"
    uncertain_dir = output_dir / "uncertain"
    for d in (ripe_dir, unripe_dir, uncertain_dir):
        d.mkdir(exist_ok=True)

    # load classifier
    classifier_py = Path(args.classifier_py)
    if not classifier_py.exists():
        print(f"Error: classifier source not found: {classifier_py}")
        sys.exit(1)

    if not Path(args.model).exists() or not Path(args.class_info).exists():
        print("Error: model file or class_info.json not found. Check paths given to --model and --class-info")
        sys.exit(1)

    print("Loading classifier (this may take a few seconds)...")
    classifier = load_classifier_from_file(classifier_py, args.model, args.class_info)
    print("Classifier loaded. Scanning images...")

    report_rows = []
    total = 0
    moved = {"ripe": 0, "unripe": 0, "uncertain": 0}

    for img_path in find_image_files(input_dir):
        total += 1
        try:
            res = classifier.predict(str(img_path))
            cls = res.get("class")
            conf = float(res.get("confidence", 0.0))

            if conf >= args.threshold:
                # map predicted class name to folder name (lowercase)
                target_name = cls.lower().replace(" ", "_")
                if "ripe" in target_name:
                    dst = ripe_dir / img_path.name
                    moved_key = "ripe"
                else:
                    dst = unripe_dir / img_path.name
                    moved_key = "unripe"
            else:
                dst = uncertain_dir / img_path.name
                moved_key = "uncertain"

            shutil.copy2(img_path, dst)
            moved[moved_key] += 1
            report_rows.append((str(img_path), cls, conf, moved_key))
            print(f"{img_path.name} -> {moved_key} ({cls} {conf:.2f})")

        except Exception as e:
            print(f"Skipping {img_path.name}: error during prediction: {e}")
            report_rows.append((str(img_path), "ERROR", 0.0, "error"))

    # write CSV report
    report_csv = output_dir / "segmentation_report.csv"
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "predicted_class", "confidence", "assigned_folder"])
        for r in report_rows:
            w.writerow(r)

    print("\nDone.")
    print(f"Total images scanned: {total}")
    print(f"Moved into: ripe={moved['ripe']}, unripe={moved['unripe']}, uncertain={moved['uncertain']}")
    print(f"Report written to: {report_csv}")


if __name__ == "__main__":
    main()
