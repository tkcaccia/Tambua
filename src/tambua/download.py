"""Download the pinned public MoNuSAC mirror outside the Git repository."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request


FILES = {
    "train": ("https://huggingface.co/datasets/MedOtter/MoNuSAC2020/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
              "c0d7c0c99600e960a38ed92e67cfa93a80eafb68186d8b958d90a8452dc9cbda"),
    "test": ("https://huggingface.co/datasets/MedOtter/MoNuSAC2020/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
             "196566e3bf4ddc06d880e244864c27f1b60b3db91a421b0daa0465ff81bf382f"),
}

ORION_CRC33_FILES = {
    "he_registered.ome.tif": ("19510_C33_US_SCAN_OR_001__160715-2-registered.ome.tif", 390336926,
                              "e5be3cf68c9adc8bada72a98e681d0ed852abe7b7cbe9b30d4102bea42b9da43"),
    "cell_mask.ome.tif": ("P37_S76_01-CRC33_01-mask.ome.tif", 84491354,
                          "fd1b6876c714d37a4391ffd735e1e8a6bed2b52b4ac03c028315894697563aa2"),
    "measured_cells.csv": ("P37_S76_01-CRC33_01.csv", 222836315,
                           "42683f328cedc134426b45179de5e95c2f14481b9e1f6962c729c2362a37a021"),
    "markers.csv": ("markers.csv", 128,
                    "e7587ca37d48370fcc1d7c0246bef4d8bd9d119df7dede9a6e18a7bf29fcffd3"),
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def download(destination: str | Path) -> dict:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    record = {}
    for split, (url, expected_sha) in FILES.items():
        path = root / f"monusac-{split}-0000.parquet"
        if not path.exists():
            pending = path.with_suffix(".parquet.part")
            with urllib.request.urlopen(url, timeout=60) as source, pending.open("wb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
            if digest(pending) != expected_sha:
                raise ValueError(f"{split} download SHA256 mismatch; partial file retained for review")
            pending.replace(path)
        if digest(path) != expected_sha:
            raise ValueError(f"{split} SHA256 mismatch")
        record[split] = dict(path=str(path.resolve()), sha256=expected_sha, bytes=path.stat().st_size)
    (root / "DOWNLOAD_PROVENANCE.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def download_orion_crc33(destination: str | Path) -> dict:
    """Acquire only a pinned H&E, cell mask and table; never the 44-GiB IF slide."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    record = {}
    base = "https://lin-2023-orion-crc.s3.amazonaws.com/data/CRC33_01/"
    for local_name, (source_name, size, expected_sha) in ORION_CRC33_FILES.items():
        path = root / local_name
        if not path.exists():
            pending = root / f"{local_name}.part"
            count = 0
            with urllib.request.urlopen(base + source_name, timeout=60) as source, pending.open("wb") as output:
                while block := source.read(1024 * 1024):
                    count += len(block)
                    if count > size:
                        raise ValueError(f"{local_name} exceeds pinned source size")
                    output.write(block)
            if count != size or digest(pending) != expected_sha:
                raise ValueError(f"{local_name} size/SHA256 mismatch; partial file retained")
            pending.replace(path)
        if path.stat().st_size != size or digest(path) != expected_sha:
            raise ValueError(f"{local_name} size/SHA256 mismatch")
        record[local_name] = dict(path=str(path.resolve()), source_url=base + source_name,
                                  bytes=size, sha256=expected_sha)
    (root / "DOWNLOAD_PROVENANCE.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
