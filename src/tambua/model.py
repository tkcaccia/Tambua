"""A single fastPLS SIMPLS-LDA cell classifier with patient-disjoint evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .table import CELL_TYPES, MARKERS, rows


def feature_names(header: list[str]) -> list[str]:
    names = [name for part in ("nuclei", "cyto") for name in
             (f"{part}_{marker}" for marker in MARKERS) if name in header]
    for part in ("nuclei", "cyto"):
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


def _fit(x: np.ndarray, y: np.ndarray, components: int):
    from fastpls import PLS, __version__
    if __version__ != "0.3.0":
        raise RuntimeError(f"Tambua requires audited fastPLS 0.3.0, found {__version__}")
    start = perf_counter()
    model = PLS(n_components=components, method="simpls", classifier="lda",
                scaling="autoscaling", backend="cpu", seed=314159).fit(x, y)
    return model, perf_counter() - start


def train(table_path: str | Path, model_path: str | Path, report_path: str | Path) -> dict:
    path = Path(model_path)
    report_file = Path(report_path)
    if path.exists():
        raise FileExistsError(path)
    if report_file.exists():
        raise FileExistsError(report_file)
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
    classes = sorted({row["cell_type"] for row in training})
    if len(classes) < 2 or any(label not in CELL_TYPES for label in classes):
        raise ValueError("At least two supported cell types required")
    if any(row["cell_type"] not in classes for row in testing):
        raise ValueError("Test class missing from training")
    y_train = np.asarray([classes.index(row["cell_type"]) for row in training], dtype=np.int64)
    if any(np.sum(y_train == index) < 2 for index in range(len(classes))):
        raise ValueError("At least two training cells per class required")
    x_train = matrix(training, names)
    x_test = matrix(testing, names)
    components = min(len(classes) - 1, len(training) - 1, len(names))
    model, fit_seconds = _fit(x_train, y_train, components)
    predicted = np.asarray(model.predict(x_test), dtype=np.int64).reshape(-1)
    if np.any((predicted < 0) | (predicted >= len(classes))):
        raise RuntimeError("fastPLS predicted an unknown cell type")
    actual_names = np.asarray([row["cell_type"] for row in testing])
    predicted_names = np.asarray(classes)[predicted]
    majority = int(np.bincount(y_train).argmax())
    report = dict(method="fastPLS SIMPLS-LDA on CellPhenotyper virtual-marker scores",
                  feature_names=names, fastpls_version="0.3.0", components=components,
                  train_patients=sorted(train_patients), test_patients=sorted(test_patients),
                  train_cells=len(training), test_cells=len(testing), fit_seconds=fit_seconds,
                  test=metrics(actual_names, predicted_names, classes),
                  test_by_patient=patient_summary(testing, actual_names, predicted_names, classes),
                  majority_baseline=metrics(actual_names, np.full(len(testing), classes[majority]), classes),
                  claim_limit="Virtual-marker scores are not measured protein abundance; this is research cell-type prediction.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x_train, y=y_train,
                        metadata=json.dumps(dict(format="tambua-fastpls-v1", classes=classes,
                                                 feature_names=names, components=components,
                                                 fit_refits_on_load=True)))
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def predict(table_path: str | Path, model_path: str | Path, output_path: str | Path) -> int:
    with np.load(model_path, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata"].item()))
        if metadata.get("format") != "tambua-fastpls-v1":
            raise ValueError("Incompatible Tambua classifier")
        model, _ = _fit(bundle["x"], bundle["y"], int(metadata["components"]))
    records = rows(table_path)
    names = metadata["feature_names"]
    values = np.asarray(model.predict(matrix(records, names)), dtype=np.int64).reshape(-1)
    classes = metadata["classes"]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "patient_id", "label_id", "predicted_cell_type"])
        writer.writeheader()
        for row, prediction in zip(records, values):
            writer.writerow(dict(sample_id=row["sample_id"], patient_id=row["patient_id"],
                                 label_id=row["label_id"], predicted_cell_type=classes[int(prediction)]))
    return len(values)
