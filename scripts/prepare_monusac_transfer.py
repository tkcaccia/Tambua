"""Prepare a calibrated, patient-separated MoNuSAC cross-organ pilot.

The original organizer ZIPs supply H&E TIFFs, expert XML nuclear polygons and
pixel spacing. The expanded `cell` mask is an explicit 3 µm perinuclear proxy,
not expert-drawn cell boundaries. It is used only for conditional typing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
import tifffile

CLASSES = {"Epithelial": "epithelial", "Lymphocyte": "lymphocyte",
           "Macrophage": "macrophage", "Neutrophil": "neutrophil"}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def select(train_audit: dict, test_audit: dict,
           train_organ: str = "lung", test_organ: str = "prostate",
           min_side: int = 512) -> list[dict]:
    if train_organ == test_organ:
        raise ValueError("Training and target organs must differ")
    all_test_patients = {item["patient_id"] for item in test_audit["records"]}
    train = [dict(item, split="train") for item in train_audit["records"]
             if item["organ"] == train_organ and
             min(item["height"], item["width"]) >= min_side and
             item["patient_id"] not in all_test_patients]
    test = [dict(item, split="test") for item in test_audit["records"]
            if item["organ"] == test_organ and
            min(item["height"], item["width"]) >= min_side]
    if not train or not test or {r["patient_id"] for r in train} & {r["patient_id"] for r in test}:
        raise ValueError("Expected nonempty patient-disjoint cross-organ cohorts")
    train_types = {name for record in train for name, count in record["expert_types"].items() if count}
    test_types = {name for record in test for name, count in record["expert_types"].items() if count}
    if not test_types <= train_types:
        raise ValueError("Target tissue has a class absent from training")
    return train + test


def _write_ome(path: Path, image: np.ndarray, mpp: float, *, rgb: bool = False) -> None:
    metadata = {"axes": "YXS" if rgb else "YX", "PhysicalSizeX": mpp,
                "PhysicalSizeY": mpp, "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"}
    tifffile.imwrite(path, image, ome=True, metadata=metadata,
                     photometric="rgb" if rgb else "minisblack", compression="zlib")


def prepare_one(archive: ZipFile, item: dict, folder: Path,
                *, halo_um: float = 3.0) -> dict:
    folder.mkdir(parents=True, exist_ok=False)
    tiff_member = item["svs_member"][:-4] + ".tif"
    if tiff_member not in archive.namelist():
        raise ValueError(f"No organizer companion TIFF for {item['sample_id']}")
    image = np.asarray(Image.open(io.BytesIO(archive.read(tiff_member))).convert("RGB"))
    height, width = image.shape[:2]
    if (height, width) != (item["height"], item["width"]):
        raise ValueError(f"Companion TIFF has different geometry: {item['sample_id']}")
    root = ET.fromstring(archive.read(item["xml_member"]))
    if abs(float(root.attrib["MicronsPerPixel"]) - item["mpp"]) > 0.0001:
        raise ValueError(f"Source/annotation pixel spacing changed: {item['sample_id']}")
    canvas = Image.new("I", (width, height), 0)
    painter = ImageDraw.Draw(canvas)
    typed_ids = {}
    for annotation in root.findall("Annotation"):
        attribute = annotation.find("Attributes/Attribute")
        cell_type = CLASSES.get(attribute.attrib.get("Name", "")) if attribute is not None else None
        if cell_type is None:
            continue
        for region in annotation.findall("Regions/Region"):
            vertices = [(round(float(vertex.attrib["X"])), round(float(vertex.attrib["Y"])))
                        for vertex in region.findall("Vertices/Vertex")]
            if len(vertices) < 3:
                continue
            cell_id = len(typed_ids) + 1
            painter.polygon(vertices, fill=cell_id)
            typed_ids[cell_id] = cell_type
    nuclei = np.asarray(canvas, dtype=np.int32)
    if nuclei.max() > np.iinfo(np.uint16).max:
        raise ValueError("Nucleus ID exceeds uint16 capacity")
    areas = np.bincount(nuclei.ravel(), minlength=len(typed_ids) + 1)
    surviving = {cell_id: cell_type for cell_id, cell_type in typed_ids.items() if areas[cell_id] > 0}
    if not surviving:
        raise ValueError(f"No visible expert nuclei: {item['sample_id']}")
    # Closest annotated nucleus claims only a physical 3 µm halo. This is a
    # proxy cell footprint, never an expert cytoplasm mask.
    distance, nearest = distance_transform_edt(nuclei == 0, sampling=(item["mpp"], item["mpp"]),
                                                return_indices=True)
    nearest_id = nuclei[nearest[0], nearest[1]]
    cell_proxy = np.where((nuclei > 0) | (distance <= halo_um), nearest_id, 0).astype(np.uint16)
    nuclei = nuclei.astype(np.uint16)
    _write_ome(folder / "he.ome.tif", image, item["mpp"], rgb=True)
    _write_ome(folder / "expert_nuclei.ome.tif", nuclei, item["mpp"])
    _write_ome(folder / "expert_anchored_cell_proxy.ome.tif", cell_proxy, item["mpp"])
    with (folder / "expert_types.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["label_id", "cell_type"])
        writer.writeheader()
        writer.writerows({"label_id": key, "cell_type": value} for key, value in surviving.items())
    result = {key: item[key] for key in ("sample_id", "patient_id", "organ", "split", "width", "height", "mpp")}
    result.update(expert_nuclei=len(surviving), halo_um=halo_um,
                  quantification_mask="expert-nucleus-anchored 3 µm cell proxy",
                  source_tiff_member=tiff_member,
                  source_tiff_sha256=hashlib.sha256(archive.read(tiff_member)).hexdigest(),
                  expert_xml_sha256=hashlib.sha256(archive.read(item["xml_member"])).hexdigest())
    (folder / "provenance.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--test-zip", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--test-audit", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--train-organ", default="lung")
    parser.add_argument("--test-organ", default="prostate")
    parser.add_argument("--min-side", type=int, default=512)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    train_audit = json.loads(args.train_audit.read_text())
    test_audit = json.loads(args.test_audit.read_text())
    selection = select(train_audit, test_audit, args.train_organ, args.test_organ, args.min_side)
    args.outdir.mkdir(parents=True)
    records = []
    with ZipFile(args.train_zip) as train_zip, ZipFile(args.test_zip) as test_zip:
        for item in selection:
            archive = train_zip if item["split"] == "train" else test_zip
            records.append(prepare_one(archive, item, args.outdir / item["sample_id"]))
    report = {"training_archive_sha256": digest(args.train_zip),
              "testing_archive_sha256": digest(args.test_zip),
              "selection": {"train_organ": args.train_organ, "test_organ": args.test_organ,
                            "min_native_side_px": args.min_side,
                            "excluded_train_patients_if_anywhere_in_test": True},
              "records": records}
    (args.outdir / "COHORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(records), "train": sum(r["split"] == "train" for r in records),
                      "test": sum(r["split"] == "test" for r in records)}))


if __name__ == "__main__":
    main()
