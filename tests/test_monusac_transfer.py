from pathlib import Path
from zipfile import ZipFile
import csv
import io
import sys
import types

import numpy as np
from PIL import Image
import tifffile

from scripts.prepare_monusac_transfer import prepare_one, select, select_matched_tissues


def test_cross_tissue_selection_excludes_every_test_patient():
    item = lambda sample, patient, organ: dict(sample_id=sample, patient_id=patient,
                                                 organ=organ, width=600, height=600,
                                                 expert_types={"epithelial": 2, "lymphocyte": 2})
    train = {"records": [item("safe", "p1", "lung"), item("leak", "p2", "lung")]}
    test = {"records": [item("target", "p3", "prostate"), item("elsewhere", "p2", "breast")]}
    selected = select(train, test)
    assert [(row["sample_id"], row["split"]) for row in selected] == [
        ("safe", "train"), ("target", "test")]


def test_matched_tissue_selection_is_patient_disjoint_and_supports_test_classes():
    def item(sample, patient, organ, types, side=600):
        return dict(sample_id=sample, patient_id=patient, organ=organ,
                    width=side, height=side, expert_types=types)
    epithelial = {"epithelial": 2, "lymphocyte": 0}
    both = {"epithelial": 2, "lymphocyte": 2}
    train = {"records": [item("breast_train", "p1", "breast", both),
                         item("lung_train", "p2", "lung", epithelial),
                         item("leak", "p3", "breast", both)]}
    test = {"records": [item("breast_test", "p4", "breast", epithelial),
                        item("lung_test", "p5", "lung", epithelial),
                        item("small_leak", "p3", "breast", epithelial, side=300)]}
    selected = select_matched_tissues(train, test)
    assert {r["sample_id"] for r in selected} == {
        "breast_train", "lung_train", "breast_test", "lung_test"}
    test["records"][1]["expert_types"] = both
    import pytest
    with pytest.raises(ValueError, match="class absent"):
        select_matched_tissues(train, test)


def test_original_polygons_keep_ids_and_physical_scale(tmp_path: Path):
    array = np.full((30, 30, 3), 180, dtype=np.uint8)
    image = Image.fromarray(array, "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="TIFF")
    base = "source/patient/sample"
    xml = b'''<Annotations MicronsPerPixel="0.25">
      <Annotation><Attributes><Attribute Name="Epithelial"/></Attributes><Regions>
        <Region><Vertices><Vertex X="4" Y="4"/><Vertex X="8" Y="4"/><Vertex X="8" Y="8"/><Vertex X="4" Y="8"/></Vertices></Region>
      </Regions></Annotation>
      <Annotation><Attributes><Attribute Name="Lymphocyte"/></Attributes><Regions>
        <Region><Vertices><Vertex X="20" Y="20"/><Vertex X="23" Y="20"/><Vertex X="23" Y="23"/><Vertex X="20" Y="23"/></Vertices></Region>
      </Regions></Annotation>
    </Annotations>'''
    archive_path = tmp_path / "source.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(base + ".tif", buffer.getvalue())
        archive.writestr(base + ".xml", xml)
    item = dict(sample_id="sample", patient_id="patient", organ="lung", split="train",
                width=30, height=30, mpp=0.25, svs_member=base + ".svs",
                xml_member=base + ".xml")
    with ZipFile(archive_path) as archive:
        report = prepare_one(archive, item, tmp_path / "sample")
    assert report["expert_nuclei"] == 2
    with tifffile.TiffFile(tmp_path / "sample/he.ome.tif") as slide:
        assert 'PhysicalSizeX="0.25"' in slide.ome_metadata
        assert slide.asarray().shape == (30, 30, 3)
    nuclei = tifffile.imread(tmp_path / "sample/expert_nuclei.ome.tif")
    cell = tifffile.imread(tmp_path / "sample/expert_anchored_cell_proxy.ome.tif")
    assert nuclei[5, 5] == 1 and nuclei[21, 21] == 2
    assert np.array_equal(nuclei[nuclei > 0], cell[nuclei > 0])
    with (tmp_path / "sample/expert_types.csv").open(newline="") as source:
        assert [row["cell_type"] for row in csv.DictReader(source)] == ["epithelial", "lymphocyte"]


def test_companion_tiff_bgr_is_corrected_from_scanner_svs(tmp_path: Path, monkeypatch):
    canonical = np.zeros((30, 30, 3), dtype=np.uint8)
    canonical[:] = [190, 85, 35]
    companion = Image.fromarray(canonical[..., ::-1], "RGB")
    tiff = io.BytesIO()
    companion.save(tiff, format="TIFF")

    class FakeSlide:
        dimensions = (30, 30)

        def __init__(self, path):
            assert Path(path).read_bytes() == b"scanner-svs"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read_region(self, *_):
            return Image.fromarray(canonical, "RGB")

    monkeypatch.setitem(sys.modules, "openslide", types.SimpleNamespace(OpenSlide=FakeSlide))
    base = "source/patient/sample"
    xml = b'''<Annotations MicronsPerPixel="0.25"><Annotation><Attributes>
      <Attribute Name="Epithelial"/></Attributes><Regions><Region><Vertices>
      <Vertex X="4" Y="4"/><Vertex X="8" Y="4"/><Vertex X="8" Y="8"/>
      </Vertices></Region></Regions></Annotation></Annotations>'''
    archive_path = tmp_path / "source.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(base + ".svs", b"scanner-svs")
        archive.writestr(base + ".tif", tiff.getvalue())
        archive.writestr(base + ".xml", xml)
    item = dict(sample_id="sample", patient_id="patient", organ="lung", split="train",
                width=30, height=30, mpp=.25, svs_member=base + ".svs",
                xml_member=base + ".xml")
    with ZipFile(archive_path) as archive:
        report = prepare_one(archive, item, tmp_path / "sample")
    assert report["source_image_member"] == base + ".svs"
    assert report["companion_tiff_channel_order"] == "BGR_mistagged_as_RGB"
    assert np.array_equal(tifffile.imread(tmp_path / "sample/he.ome.tif"), canonical)
