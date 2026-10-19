from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile

from tambua.orion import extract_window


def fixture(tmp_path: Path, mask_mpp: float = .325) -> tuple[Path, Path, Path]:
    rgb = np.full((512, 512, 3), 180, dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.int32)
    mask[100:120, 100:120] = 7
    mask[0:8, 200:210] = 8
    he, cells, table = (tmp_path / name for name in ("he.ome.tif", "mask.ome.tif", "measured.csv"))
    for path, image, mpp, axes in ((he, rgb, .325, "YXS"), (cells, mask, mask_mpp, "YX")):
        tifffile.imwrite(path, image, ome=True, photometric="rgb" if image.ndim == 3 else None,
                         metadata=dict(axes=axes, PhysicalSizeX=mpp, PhysicalSizeXUnit="µm",
                                       PhysicalSizeY=mpp, PhysicalSizeYUnit="µm"))
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["CellID", "X_centroid", "Y_centroid", "CD68", "Pan-CK"])
        writer.writeheader()
        writer.writerows([{"CellID": 7, "X_centroid": 109.5, "Y_centroid": 109.5,
                           "CD68": .2, "Pan-CK": .7},
                          {"CellID": 8, "X_centroid": 204.5, "Y_centroid": 3.5,
                           "CD68": .8, "Pan-CK": .1}])
    return he, cells, table


def test_orion_window_keeps_only_complete_cells_and_physical_scale(tmp_path: Path):
    source = fixture(tmp_path)
    out = tmp_path / "window"
    report = extract_window(*source, out, x=0, y=0, size=512, sample="synthetic")
    assert report["source_mpp"] == .325
    assert report["retained_complete_cells"] == 1
    assert report["excluded_edge_cells"] == 1
    assert report["max_centroid_error_px"] == 0
    assert len(report["input_sha256"]) == 3
    with tifffile.TiffFile(out / "he.ome.tif") as tf:
        assert 'PhysicalSizeX="0.325"' in tf.ome_metadata
    cells = tifffile.imread(out / "expert_cells.ome.tif")
    assert set(np.unique(cells)) == {0, 7}
    with (out / "measured_cells.csv").open(newline="", encoding="utf-8") as stream:
        assert [row["CellID"] for row in csv.DictReader(stream)] == ["7"]


def test_orion_window_rejects_misaligned_physical_grid(tmp_path: Path):
    source = fixture(tmp_path, mask_mpp=.5)
    with pytest.raises(ValueError, match="spacing disagree"):
        extract_window(*source, tmp_path / "window", x=0, y=0, size=512, sample="synthetic")
    assert not (tmp_path / "window").exists()


def test_orion_window_rejects_wrong_measured_coordinates(tmp_path: Path):
    he, mask, table = fixture(tmp_path)
    content = table.read_text().replace("109.5,109.5", "120.0,109.5")
    table.write_text(content)
    with pytest.raises(ValueError, match="centroid disagrees"):
        extract_window(he, mask, table, tmp_path / "window", x=0, y=0, size=512,
                       sample="synthetic")
