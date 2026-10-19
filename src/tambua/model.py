"""A single fastPLS SIMPLS-LDA cell classifier with patient-disjoint evaluation."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

import numpy as np

from .table import CELL_TYPES, MARKERS, rows

FASTPLS_VERSION = "0.4"
FASTPLS_SOURCE = "https://github.com/tkcaccia/fastPLS"
FASTPLS_COMMIT = "265108b6aa250c2534f98ec89037bb5cd001dca0"


def feature_names(header: list[str]) -> list[str]:
    names = [name for part in ("nuclei", "cyto", "cell") for name in
             (f"{part}_{marker}" for marker in MARKERS) if name in header]
    for part in ("nuclei", "cyto", "cell"):
        present = [name for name in names if name.startswith(part + "_")]
        if present and len(present) != len(MARKERS):
            raise ValueError(f"Incomplete {part} marker panel")
    if not names:
        raise ValueError("No complete CellPhenotyper GigaTIME marker panel")
    return names


def matrix(records: list[dict[str, str]], names: list[str]) -> np.ndarray:
    x = np.asarray([[float(row[name]) for name in names] for row in records], dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("Expected finite virtual-marker scores in [0, 1]")
    return np.asfortranarray(x)


def metrics(y: np.ndarray, predicted: np.ndarray, classes: list[str]) -> dict:
    recalls, f1s, matrix_counts = [], [], []
    for actual in classes:
        tp = int(np.sum((y == actual) & (predicted == actual)))
        fn = int(np.sum((y == actual) & (predicted != actual)))
        fp = int(np.sum((y != actual) & (predicted == actual)))
        recalls.append(tp / (tp + fn) if tp + fn else None)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
        matrix_counts.append([int(np.sum((y == actual) & (predicted == name))) for name in classes])
    missing_classes = [name for name, recall in zip(classes, recalls) if recall is None]
    return dict(n=len(y), accuracy=float(np.mean(y == predicted)),
                balanced_accuracy=(None if missing_classes else float(np.mean(recalls))),
                macro_f1=(None if missing_classes else float(np.mean(f1s))),
                missing_reference_classes=missing_classes,
                support={name: int(np.sum(y == name)) for name in classes},
                recall=dict(zip(classes, recalls)), f1=dict(zip(classes, f1s)),
                confusion_matrix=matrix_counts, classes=classes)


def patient_summary(records: list[dict[str, str]], y: np.ndarray,
                    predicted: np.ndarray, classes: list[str]) -> dict:
    patients = sorted({row["patient_id"] for row in records})
    per_patient = {}
    for patient in patients:
        indices = np.asarray([index for index, row in enumerate(records)
                              if row["patient_id"] == patient], dtype=np.int64)
        per_patient[patient] = metrics(y[indices], predicted[indices], classes)
    accuracies = [item["accuracy"] for item in per_patient.values()]
    complete = [item["balanced_accuracy"] for item in per_patient.values()
                if item["balanced_accuracy"] is not None]
    return dict(n_patients=len(patients), equal_weight_accuracy=float(np.mean(accuracies)),
                complete_class_patients=len(complete),
                equal_weight_balanced_accuracy=(float(np.mean(complete)) if complete else None),
                per_patient=per_patient)


def _bridge(mode: str, input_path: Path, model_path: Path, output_path: Path) -> None:
    script = Path(__file__).with_name("fastpls_bridge.R")
    try:
        result = subprocess.run([os.environ.get("TAMBUA_RSCRIPT", "Rscript"), "--vanilla",
                                 str(script), mode, str(input_path), str(model_path),
                                 str(output_path)], capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("Rscript and GitHub fastPLS 0.4 are required") from exc
    if result.returncode:
        raise RuntimeError(f"fastPLS 0.4 {mode} failed: {result.stderr.strip()}")


def _write_features(path: Path, x: np.ndarray, labels: np.ndarray | None = None) -> None:
    names = [f"f_{index:03d}" for index in range(x.shape[1])]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(names + (["label"] if labels is not None else []))
        for index, values in enumerate(x):
            writer.writerow([float(value) for value in values] +
                            ([str(labels[index])] if labels is not None else []))


def _predict_fastpls(x: np.ndarray, model_path: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="tambua-fastpls-") as temporary:
        input_path, output_path = Path(temporary) / "input.csv", Path(temporary) / "predictions.csv"
        _write_features(input_path, x)
        _bridge("predict", input_path, model_path, output_path)
        with output_path.open(newline="", encoding="utf-8") as stream:
            values = [row["predicted"] for row in csv.DictReader(stream)]
    if len(values) != len(x):
        raise RuntimeError("fastPLS prediction row count differs from input")
    return np.asarray(values)


def train(table_path: str | Path, model_path: str | Path, report_path: str | Path) -> dict:
    path = Path(model_path)
    report_file = Path(report_path)
    if path.exists():
        raise FileExistsError(path)
    if report_file.exists():
        raise FileExistsError(report_file)
    if path.suffix != ".rds":
        raise ValueError("The fitted fastPLS 0.4 model must have a .rds extension")
    metadata_path = Path(str(path) + ".json")
    if metadata_path.exists():
        raise FileExistsError(metadata_path)
    records = rows(table_path)
    if not records:
        raise ValueError("Empty cell table")
    names = feature_names(list(records[0]))
    training = [row for row in records if row["split"] == "train"]
    testing = [row for row in records if row["split"] == "test"]
    train_patients = {row["patient_id"] for row in training}
    test_patients = {row["patient_id"] for row in testing}
    if not training or not testing or train_patients & test_patients:
        raise ValueError("Nonempty patient-disjoint train and test cohorts required")
    if "organ" in records[0]:
        train_organs = {row["organ"].strip().lower() for row in training}
        test_organs = {row["organ"].strip().lower() for row in testing}
        if "" in train_organs | test_organs or not test_organs <= train_organs:
            raise ValueError("Every test tissue must be represented in training")
    classes = sorted({row["cell_type"] for row in training})
    if len(classes) < 2 or any(label not in CELL_TYPES for label in classes):
        raise ValueError("At least two supported cell types required")
    if any(row["cell_type"] not in classes for row in testing):
        raise ValueError("Test class missing from training")
    y_train = np.asarray([row["cell_type"] for row in training])
    if any(np.sum(y_train == name) < 2 for name in classes):
        raise ValueError("At least two training cells per class required")
    x_train = matrix(training, names)
    x_test = matrix(testing, names)
    components = min(len(classes) - 1, len(training) - 1, len(names))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tambua-fastpls-", dir=path.parent) as temporary:
        input_path = Path(temporary) / "train.csv"
        temporary_model = Path(temporary) / "model.rds"
        fit_info = Path(temporary) / "fit.csv"
        _write_features(input_path, x_train, y_train)
        start = perf_counter()
        _bridge("train", input_path, temporary_model, fit_info)
        fit_seconds = perf_counter() - start
        with fit_info.open(newline="", encoding="utf-8") as stream:
            fitted_components = int(next(csv.DictReader(stream))["components"])
        if fitted_components != components:
            raise RuntimeError("fastPLS fitted an unexpected component count")
        predicted_names = _predict_fastpls(x_test, temporary_model)
        if any(label not in classes for label in predicted_names):
            raise RuntimeError("fastPLS predicted an unknown cell type")
        os.replace(temporary_model, path)
    actual_names = np.asarray([row["cell_type"] for row in testing])
    majority = max(classes, key=lambda name: int(np.sum(y_train == name)))
    report = dict(method="fastPLS SIMPLS-LDA on CellPhenotyper virtual-marker scores",
                  feature_names=names, fastpls_version=FASTPLS_VERSION,
                  fastpls_source=FASTPLS_SOURCE, fastpls_commit=FASTPLS_COMMIT,
                  components=components,
                  train_patients=sorted(train_patients), test_patients=sorted(test_patients),
                  train_cells=len(training), test_cells=len(testing), fit_seconds=fit_seconds,
                  test=metrics(actual_names, predicted_names, classes),
                  test_by_patient=patient_summary(testing, actual_names, predicted_names, classes),
                  majority_baseline=metrics(actual_names, np.full(len(testing), majority), classes),
                  claim_limit="Virtual-marker scores are not measured protein abundance; this is research cell-type prediction.")
    metadata_path.write_text(json.dumps(dict(format="tambua-fastpls-r-v3-all-23-channels", classes=classes,
                                             feature_names=names, components=components,
                                             fastpls_version=FASTPLS_VERSION,
                                             fastpls_source=FASTPLS_SOURCE,
                                             fastpls_commit=FASTPLS_COMMIT), indent=2) + "\n",
                             encoding="utf-8")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def predict(table_path: str | Path, model_path: str | Path, output_path: str | Path) -> int:
    model_path = Path(model_path)
    metadata = json.loads(Path(str(model_path) + ".json").read_text(encoding="utf-8"))
    if metadata.get("format") != "tambua-fastpls-r-v3-all-23-channels" or metadata.get("fastpls_version") != FASTPLS_VERSION:
        raise ValueError("Incompatible Tambua fastPLS 0.4 classifier; retrain with all 23 GigaTIME channels")
    records = rows(table_path)
    if not records:
        raise ValueError("Empty cell table")
    names = metadata["feature_names"]
    if names != feature_names(list(records[0])):
        raise ValueError("Inference feature panel differs from the full-channel fitted model")
    values = _predict_fastpls(matrix(records, names), model_path)
    classes = metadata["classes"]
    if any(value not in classes for value in values):
        raise RuntimeError("fastPLS predicted an unknown cell type")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "patient_id", "label_id", "predicted_cell_type"])
        writer.writeheader()
        for row, prediction in zip(records, values):
            writer.writerow(dict(sample_id=row["sample_id"], patient_id=row["patient_id"],
                                 label_id=row["label_id"], predicted_cell_type=str(prediction)))
    return len(values)
