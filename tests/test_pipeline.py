from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tambua.model import predict, train
from tambua.table import MARKERS, build_table, build_inference_table


def write(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def specimen(tmp_path: Path, sample: str, patient: str, split: str, offset: int) -> dict:
    labels = []
    tables = {}
    for part in ("nuclei", "cyto"):
        records = []
        for index in range(40):
            name = ("epithelial", "lymphocyte", "macrophage", "neutrophil")[index % 4]
            signal = (index % 4 + 1) / 5
            row = dict(label_id=index + 1, mask_name=part,
                       value_semantics="uncalibrated_virtual_marker_score",
                       measured_protein_abundance="False", area_px=30)
            row.update({marker: signal if marker == ("CK", "CD3", "CD68", "CD16")[index % 4]
                        else .05 + .001 * offset for marker in MARKERS})
            records.append(row)
            if part == "nuclei":
                labels.append(dict(label_id=index + 1, cell_type=name))
        path = tmp_path / f"{sample}_{part}.csv"
        write(path, records)
        tables[f"{part}_csv"] = str(path)
    label_path = tmp_path / f"{sample}_labels.csv"
    write(label_path, labels)
    return dict(sample_id=sample, patient_id=patient, split=split,
                labels_csv=str(label_path), **tables)


def test_full_cpu_handoff_and_patient_disjoint_evaluation(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    write(manifest, [specimen(tmp_path, "a", "p1", "train", 0),
                     specimen(tmp_path, "b", "p2", "train", 1),
                     specimen(tmp_path, "c", "p3", "test", 2)])
    table = tmp_path / "cells.csv"
    result = build_table(manifest, table)
    assert result["matched_cells"] == 120
    assert len(result["feature_names"]) == 42
    assert "nuclei_TRITC" not in result["feature_names"]
    model = tmp_path / "classifier.npz"
    report = train(table, model, tmp_path / "report.json")
    assert report["test"]["n"] == 40
    assert report["test"]["balanced_accuracy"] > .9
    assert report["test"]["balanced_accuracy"] > report["majority_baseline"]["balanced_accuracy"]
    assert json.loads((tmp_path / "report.json").read_text())["test_patients"] == ["p3"]
    assert predict(table, model, tmp_path / "predictions.csv") == 120
    future_manifest = tmp_path / "future_manifest.csv"
    new_specimen = specimen(tmp_path, "future", "p4", "test", 3)
    write(future_manifest, [{key: value for key, value in new_specimen.items()
                             if key in ("sample_id", "patient_id", "nuclei_csv", "cyto_csv")}])
    future = tmp_path / "future_cells.csv"
    assert build_inference_table(future_manifest, future)["cells"] == 40
    assert predict(future, model, tmp_path / "future_predictions.csv") == 40


def test_patient_leakage_rejected(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    write(manifest, [specimen(tmp_path, "a", "p1", "train", 0),
                     specimen(tmp_path, "b", "p1", "test", 1)])
    with pytest.raises(ValueError, match="both train and test"):
        build_table(manifest, tmp_path / "cells.csv")


def test_marker_semantics_rejected(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    entry = specimen(tmp_path, "a", "p1", "train", 0)
    records = list(csv.DictReader(Path(entry["nuclei_csv"]).open()))
    records[0]["measured_protein_abundance"] = "True"
    write(Path(entry["nuclei_csv"]), records)
    write(manifest, [entry])
    with pytest.raises(ValueError, match="cannot be declared measured"):
        build_table(manifest, tmp_path / "cells.csv")
