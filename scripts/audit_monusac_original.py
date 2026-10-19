"""Audit the original MoNuSAC test ZIP before calibrated cell experiments.

The patient-to-organ mapping follows the published RationAI MoNuSAC conversion
manifest (https://huggingface.co/datasets/RationAI/MoNuSAC/blob/main/gen_script.py).
The H&E pixel spacing and expert type counts come directly from the organizer ZIP.
"""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import tifffile


TEST_ORGAN = {
    **dict.fromkeys("TCGA-49-6743 TCGA-50-6591 TCGA-55-7570 TCGA-55-7573 TCGA-73-4662 TCGA-78-7152 TCGA-MP-A4T7".split(), "lung"),
    **dict.fromkeys("TCGA-2Z-A9JG TCGA-2Z-A9JN TCGA-DW-7838 TCGA-DW-7963 TCGA-F9-A8NY TCGA-IZ-A6M9 TCGA-MH-A55W".split(), "kidney"),
    **dict.fromkeys("TCGA-A2-A04X TCGA-A2-A0ES TCGA-D8-A3Z6 TCGA-E2-A108 TCGA-EW-A6SB".split(), "breast"),
    **dict.fromkeys("TCGA-G9-6356 TCGA-G9-6367 TCGA-VP-A87E TCGA-VP-A87H TCGA-X4-A8KS TCGA-YL-A9WL".split(), "prostate"),
}
CLASSES = {"Epithelial", "Lymphocyte", "Macrophage", "Neutrophil"}


def audit(zip_path: Path, organ_map: dict[str, str] | None = None) -> dict:
    organ_map = organ_map or TEST_ORGAN
    records = []
    with ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for xml_name in sorted(name for name in names if name.endswith(".xml")):
            basename = xml_name[:-4]
            svs_name = basename + ".svs"
            if svs_name not in names:
                raise ValueError(f"Missing source SVS for {xml_name}")
            patient_id = Path(xml_name).parent.name[:12]
            if patient_id not in organ_map:
                raise ValueError(f"Unknown organ mapping for {patient_id}")
            root = ET.fromstring(archive.read(xml_name))
            xml_mpp = float(root.attrib["MicronsPerPixel"])
            counts = Counter()
            for annotation in root.findall("Annotation"):
                attribute = annotation.find("Attributes/Attribute")
                if attribute is None:
                    continue
                cell_type = attribute.attrib.get("Name", "")
                if cell_type not in CLASSES:
                    continue
                counts[cell_type] += len(annotation.findall("Regions/Region"))
            with tifffile.TiffFile(io.BytesIO(archive.read(svs_name))) as slide:
                page = slide.pages[0]
                height, width = page.shape[:2]
                description = page.description or ""
            match = re.search(r"\bMPP\s*=\s*([0-9.]+)", description)
            if match is None:
                raise ValueError(f"Missing Aperio MPP for {svs_name}")
            svs_mpp = float(match.group(1))
            if abs(svs_mpp - xml_mpp) > 0.0001:
                raise ValueError(f"Source/annotation MPP disagree for {svs_name}")
            records.append({
                "sample_id": Path(basename).name,
                "patient_id": patient_id,
                "organ": organ_map[patient_id],
                "width": width,
                "height": height,
                "mpp": xml_mpp,
                "expert_nuclei": sum(counts.values()),
                "expert_types": {name.lower(): counts[name] for name in sorted(CLASSES)},
                "eligible_512": width >= 512 and height >= 512,
                "svs_member": svs_name,
                "xml_member": xml_name,
            })
    return {
        "source": str(zip_path),
        "source_type": "original MoNuSAC organizer ZIP",
        "organ_mapping": "https://huggingface.co/datasets/RationAI/MoNuSAC/blob/main/gen_script.py",
        "images": len(records),
        "patients": len({item["patient_id"] for item in records}),
        "eligible_512_images": sum(item["eligible_512"] for item in records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--organ-map-json", type=Path)
    args = parser.parse_args()
    organ_map = json.loads(args.organ_map_json.read_text()) if args.organ_map_json else None
    report = audit(args.zip_path, organ_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("images", "patients", "eligible_512_images")}))


if __name__ == "__main__":
    main()
