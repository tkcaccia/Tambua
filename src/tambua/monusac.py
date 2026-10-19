"""Extract provenance-preserving MoNuSAC examples from a local Parquet mirror."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .table import CELL_TYPES


def extract(train_parquet: str | Path, test_parquet: str | Path,
            output_dir: str | Path, per_split: int = 4) -> dict:
    import pyarrow.parquet as pq
    from PIL import Image

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    count_by_split = {}
    for split, parquet_path in (("train", Path(train_parquet)), ("test", Path(test_parquet))):
        parquet = pq.ParquetFile(parquet_path)
        candidates = []
        for group in range(parquet.num_row_groups):
            for index, row in enumerate(parquet.read_row_group(group).to_pylist()):
                patient = row["patient"]
                if row["num_nuclei"] < 50:
                    continue
                types = np.asarray(Image.open(io.BytesIO(row["type_map"]["bytes"])))
                instances = np.asarray(Image.open(io.BytesIO(row["inst_map"]["bytes"])))
                if types.shape != instances.shape:
                    raise ValueError("MoNuSAC image and annotation planes disagree")
                counts = {category: 0 for category in range(1, 5)}
                for cell_id in np.unique(instances):
                    if cell_id == 0:
                        continue
                    values, frequencies = np.unique(types[instances == cell_id], return_counts=True)
                    category = int(values[np.argmax(frequencies)])
                    if category in counts:
                        counts[category] += 1
                if sum(counts.values()) < 50 or sum(value > 0 for value in counts.values()) < 2:
                    continue
                candidates.append(dict(group=group, index=index, patient=patient, counts=counts))
        # Select the widest class coverage first, then favor minority myeloid
        # examples while retaining one specimen per patient in each split.
        candidates.sort(key=lambda item: (
            sum(value > 0 for value in item["counts"].values()),
            item["counts"][3] + item["counts"][4],
            sum(item["counts"].values()),
        ), reverse=True)
        selected = []
        seen_patients = set()
        for candidate in candidates:
            if candidate["patient"] in seen_patients:
                continue
            selected.append(candidate)
            seen_patients.add(candidate["patient"])
            if len(selected) == per_split:
                break
        cached_group = None
        cached_rows = None
        for accepted, item in enumerate(selected):
                group, index, patient = item["group"], item["index"], item["patient"]
                if group != cached_group:
                    cached_rows = parquet.read_row_group(group).to_pylist()
                    cached_group = group
                row = cached_rows[index]
                image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                types = np.asarray(Image.open(io.BytesIO(row["type_map"]["bytes"])))
                instances = np.asarray(Image.open(io.BytesIO(row["inst_map"]["bytes"])))
                if image.size != types.shape[::-1]:
                    raise ValueError("MoNuSAC image and annotation planes disagree")
                labels = []
                scored = instances.copy()
                for cell_id in np.unique(instances):
                    if cell_id == 0:
                        continue
                    values, frequencies = np.unique(types[instances == cell_id], return_counts=True)
                    category = int(values[np.argmax(frequencies)])
                    if category not in (1, 2, 3, 4):
                        scored[instances == cell_id] = 0
                        continue
                    labels.append(dict(label_id=int(cell_id), cell_type=CELL_TYPES[category - 1]))
                sample = f"monusac_{split}_{accepted + 1:02d}"
                folder = root / sample
                folder.mkdir(exist_ok=False)
                image.save(folder / "image.png")
                Image.fromarray(scored.astype(np.uint16)).save(folder / "expert_instances.tif")
                with (folder / "expert_cell_types.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["label_id", "cell_type"])
                    writer.writeheader()
                    writer.writerows(labels)
                records.append(dict(sample_id=sample, patient_id=patient, split=split,
                                    source_split=split, image=str(folder / "image.png"),
                                    expert_instances=str(folder / "expert_instances.tif"),
                                    labels_csv=str(folder / "expert_cell_types.csv"),
                                    scored_cells=len(labels), source_row_group=group,
                                    source_row_in_group=index))
        count_by_split[split] = len(selected)
    if min(count_by_split.values()) == 0:
        raise ValueError("No multiclass annotated MoNuSAC examples in a split")
    by_split = {split: set() for split in ("train", "test")}
    for row in records:
        with Path(row["labels_csv"]).open(newline="", encoding="utf-8") as stream:
            by_split[row["split"]].update(item["cell_type"] for item in csv.DictReader(stream))
    if any(classes != set(CELL_TYPES) for classes in by_split.values()):
        raise ValueError("Extracted examples must represent all four cell types in both splits")
    manifest = root / "examples_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = dict(source="MedOtter/MoNuSAC2020 Parquet mirror of official MoNuSAC",
                  canonical_source="https://monusac-2020.grand-challenge.org/Data/",
                  counts=count_by_split, scored_cells=sum(r["scored_cells"] for r in records),
                  patients=len({r["patient_id"] for r in records}),
                  patient_disjoint=len({r["patient_id"] for r in records}) == len(records),
                  train_parquet_sha256=hashlib.sha256(Path(train_parquet).read_bytes()).hexdigest(),
                  test_parquet_sha256=hashlib.sha256(Path(test_parquet).read_bytes()).hexdigest(),
                  resolution="40x nominal; exact micrometres per pixel not verified in this mirror",
                  label_source="expert MoNuSAC nucleus identities; no GigaTIME inference performed")
    (root / "DATASET_PROVENANCE.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
