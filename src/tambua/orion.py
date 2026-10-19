"""Extract a physically calibrated, same-section ORION validation window."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


def _spacing(tif) -> float:
    root = ET.fromstring(tif.ome_metadata or "")
    pixels = root.find(".//{*}Pixels")
    if pixels is None:
        raise ValueError("Source lacks OME physical pixel metadata")
    values = []
    for axis in "XY":
        if pixels.get(f"PhysicalSize{axis}Unit", "µm") not in ("µm", "um"):
            raise ValueError("Physical pixel size must be expressed in micrometres")
        value = float(pixels.get(f"PhysicalSize{axis}", "nan"))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid physical pixel size")
        values.append(value)
    if not math.isclose(values[0], values[1], rel_tol=0.001):
        raise ValueError("Anisotropic source pixels need explicit resampling")
    return float(values[0])


def _level_zero(tif):
    import zarr

    result = zarr.open(tif.series[0].aszarr(), mode="r")
    return result["0"] if isinstance(result, zarr.hierarchy.Group) else result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_window(he_path: str | Path, mask_path: str | Path,
                   measured_csv: str | Path, output_dir: str | Path,
                   *, x: int, y: int, size: int = 2048, sample: str) -> dict:
    """Use level-zero coordinates and omit cells cut by the window edge."""
    import tifffile

    he_path, mask_path = Path(he_path), Path(mask_path)
    measured_csv, destination = Path(measured_csv), Path(output_dir)
    if destination.exists() or size < 256 or x < 0 or y < 0 or not sample.strip():
        raise ValueError("New output directory and valid level-zero window required")
    with tifffile.TiffFile(he_path) as he_tif, tifffile.TiffFile(mask_path) as mask_tif:
        he, mask = _level_zero(he_tif), _level_zero(mask_tif)
        if len(he.shape) != 3 or he.shape[2] != 3 or mask.shape != he.shape[:2]:
            raise ValueError("Registered H&E and cell mask must share one RGB grid")
        mpp = _spacing(he_tif)
        if not math.isclose(_spacing(mask_tif), mpp, rel_tol=0.001):
            raise ValueError("H&E and cell mask pixel spacing disagree")
        if x + size > he.shape[1] or y + size > he.shape[0]:
            raise ValueError("Window exceeds source image")
        rgb = np.asarray(he[y:y + size, x:x + size, :])
        cells = np.asarray(mask[y:y + size, x:x + size])
    if rgb.dtype != np.uint8 or cells.dtype.kind not in "iu" or np.any(cells < 0):
        raise ValueError("Expected uint8 RGB and nonnegative integer cell IDs")
    edge_ids = set(int(v) for v in np.unique(np.concatenate(
        (cells[0], cells[-1], cells[:, 0], cells[:, -1])))) - {0}
    if edge_ids:
        cells[np.isin(cells, list(edge_ids))] = 0
    selected = set(int(v) for v in np.unique(cells)) - {0}
    if not selected:
        raise ValueError("Window has no complete annotated cells")
    measured = []
    with measured_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"CellID", "X_centroid", "Y_centroid"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("ORION measured table lacks cell ID or centroid coordinates")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            if int(row["CellID"]) in selected:
                measured.append(row)
    present = {int(row["CellID"]) for row in measured}
    if present != selected or len(measured) != len(selected):
        raise ValueError("Cell mask IDs do not uniquely match measured ORION rows")
    yy, xx = np.nonzero(cells)
    ids, inverse = np.unique(cells[yy, xx], return_inverse=True)
    counts = np.bincount(inverse)
    xsum = np.bincount(inverse, weights=xx)
    ysum = np.bincount(inverse, weights=yy)
    errors = []
    for row in measured:
        index = int(np.searchsorted(ids, int(row["CellID"])))
        dx = float(xsum[index] / counts[index] + x - float(row["X_centroid"]))
        dy = float(ysum[index] / counts[index] + y - float(row["Y_centroid"]))
        error = math.hypot(dx, dy)
        if not math.isfinite(error) or error > 0.5:
            raise ValueError("Measured-cell centroid disagrees with registered mask")
        errors.append(error)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".orion-window-", dir=destination.parent) as temp:
        folder = Path(temp)
        physical = dict(PhysicalSizeX=mpp, PhysicalSizeXUnit="µm",
                        PhysicalSizeY=mpp, PhysicalSizeYUnit="µm")
        tifffile.imwrite(folder / "he.ome.tif", rgb, ome=True, photometric="rgb",
                         metadata=dict(axes="YXS", **physical))
        tifffile.imwrite(folder / "expert_cells.ome.tif", cells, ome=True,
                         metadata=dict(axes="YX", **physical))
        with (folder / "measured_cells.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(measured)
        report = dict(dataset="ORION CRC", sample=sample, source_section="same-section registered H&E and IF",
                      source_mpp=mpp, source_window_xywh=[x, y, size, size],
                      retained_complete_cells=len(selected), excluded_edge_cells=len(edge_ids),
                      max_centroid_error_px=float(max(errors)),
                      measured_protein_values="measured ORION IF, not GigaTIME outputs",
                      input_sha256={p.name: _sha256(p) for p in (he_path, mask_path, measured_csv)})
        (folder / "provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(folder, destination)
    return report
