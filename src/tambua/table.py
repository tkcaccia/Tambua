"""Strict, audit-friendly joins of CellPhenotyper outputs and independent labels."""
from __future__ import annotations

import csv
import json
from pathlib import Path

MARKERS = (
    "DAPI", "PD-1", "CD14", "CD4", "T-bet", "CD34", "CD68", "CD16",
    "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1", "CK", "Ki67",
    "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B", "Transgelin",
)
BACKGROUND = ("TRITC", "Cy5")
CELL_TYPES = ("epithelial", "lymphocyte", "macrophage", "neutrophil")
VALUE_SEMANTICS = "uncalibrated_virtual_marker_score"


def rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def unique_index(records: list[dict[str, str]], key: str, source: str) -> dict[int, dict[str, str]]:
    result = {}
    for row in records:
        index = int(row[key])
        if index <= 0 or index in result:
            raise ValueError(f"{source}: cell IDs must be unique positive integers")
        result[index] = row
    return result


def marker_values(row: dict[str, str], compartment: str) -> dict[str, float]:
    import math

    if row.get("mask_name") != compartment or row.get("value_semantics") != VALUE_SEMANTICS:
        raise ValueError(f"Expected CellPhenotyper {compartment} virtual-marker table")
    if row.get("measured_protein_abundance", "").lower() not in ("false", "0"):
        raise ValueError("GigaTIME scores cannot be declared measured protein abundance")
    if any(marker not in row for marker in MARKERS):
        raise ValueError(f"Missing GigaTIME channels for {compartment}")
    values = {f"{compartment}_{marker}": float(row[marker]) for marker in MARKERS}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        raise ValueError("Virtual-marker scores must be finite in [0, 1]")
    return values


def build_table(manifest_path: str | Path, output_path: str | Path,
                *, compartments: tuple[str, ...] = ("nuclei", "cyto")) -> dict:
    """Join by specimen and cell ID; annotations must be independent of GigaTIME."""
    if not compartments or any(c not in ("nuclei", "cyto") for c in compartments):
        raise ValueError("Choose nuclei, cyto or both")
    manifest = rows(manifest_path)
    if not manifest:
        raise ValueError("Empty specimen manifest")
    feature_names = [f"{part}_{marker}" for part in compartments for marker in MARKERS]
    output = []
    seen_specimens = set()
    split_by_patient: dict[str, str] = {}
    for specimen in manifest:
        sample = specimen["sample_id"].strip()
        patient = specimen["patient_id"].strip()
        split = specimen["split"].strip().lower()
        if not sample or not patient or sample in seen_specimens or split not in ("train", "test"):
            raise ValueError("Unique sample_id, patient_id and train/test split required")
        if patient in split_by_patient and split_by_patient[patient] != split:
            raise ValueError("Patient appears in both train and test")
        seen_specimens.add(sample)
        split_by_patient[patient] = split
        annotations = unique_index(rows(specimen["labels_csv"]), "label_id", "expert labels")
        tables = {}
        for part in compartments:
            key = f"{part}_csv"
            if not specimen.get(key):
                raise ValueError(f"Missing {key} for {sample}")
            tables[part] = unique_index(rows(specimen[key]), "label_id", key)
        for cell_id, annotation in annotations.items():
            cell_type = annotation["cell_type"].strip().lower()
            if cell_type not in CELL_TYPES:
                continue  # Ambiguous and out-of-ontology cells are accounted for below.
            if any(cell_id not in table for table in tables.values()):
                continue
            merged = dict(sample_id=sample, patient_id=patient, split=split,
                          label_id=cell_id, cell_type=cell_type)
            for part, table in tables.items():
                merged.update(marker_values(table[cell_id], part))
            output.append(merged)
    if not output:
        raise ValueError("No independently labelled cells matched all requested compartments")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "patient_id", "split", "label_id", "cell_type", *feature_names])
        writer.writeheader()
        writer.writerows(output)
    report = dict(samples=len(seen_specimens), patients=len(split_by_patient),
                  matched_cells=len(output), train_cells=sum(r["split"] == "train" for r in output),
                  test_cells=sum(r["split"] == "test" for r in output),
                  feature_names=feature_names, excluded_background_channels=list(BACKGROUND),
                  value_semantics=VALUE_SEMANTICS,
                  reference_labels="independent expert cell types, supplied by manifest")
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_inference_table(manifest_path: str | Path, output_path: str | Path,
                          *, compartments: tuple[str, ...] = ("nuclei", "cyto")) -> dict:
    """Join unlabelled CellPhenotyper cells for application of a frozen model."""
    if not compartments or any(c not in ("nuclei", "cyto") for c in compartments):
        raise ValueError("Choose nuclei, cyto or both")
    manifest = rows(manifest_path)
    if not manifest:
        raise ValueError("Empty inference manifest")
    features = [f"{part}_{marker}" for part in compartments for marker in MARKERS]
    output = []
    seen = set()
    for specimen in manifest:
        sample = specimen["sample_id"].strip()
        patient = specimen["patient_id"].strip()
        if not sample or not patient or sample in seen:
            raise ValueError("Unique sample_id and patient_id required")
        seen.add(sample)
        tables = {}
        for part in compartments:
            key = f"{part}_csv"
            if not specimen.get(key):
                raise ValueError(f"Missing {key} for {sample}")
            tables[part] = unique_index(rows(specimen[key]), "label_id", key)
        shared = set.intersection(*(set(table) for table in tables.values()))
        for cell_id in sorted(shared):
            item = dict(sample_id=sample, patient_id=patient, label_id=cell_id)
            for part, table in tables.items():
                item.update(marker_values(table[cell_id], part))
            output.append(item)
    if not output:
        raise ValueError("No cells shared by the requested compartments")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "patient_id", "label_id", *features])
        writer.writeheader()
        writer.writerows(output)
    report = dict(samples=len(seen), cells=len(output), feature_names=features,
                  reference_labels_used=False, value_semantics=VALUE_SEMANTICS)
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
