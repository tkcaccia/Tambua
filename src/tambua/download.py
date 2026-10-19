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
