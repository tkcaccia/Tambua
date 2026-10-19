"""Command line entry point for labelled cell-table preparation and fastPLS."""
from __future__ import annotations

import argparse
import json

from .model import predict, train
from .table import build_table, build_inference_table
from .monusac import extract
from .download import download, download_orion_crc33
from .orion import extract_window


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tambua")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-table", help="Join CellPhenotyper virtual-marker tables with independent cell labels")
    build.add_argument("--manifest", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--compartments", choices=("nuclei", "cyto", "cell", "both"), default="both")
    future = sub.add_parser("build-inference", help="Join new CellPhenotyper cells without reference labels")
    future.add_argument("--manifest", required=True)
    future.add_argument("--output", required=True)
    future.add_argument("--compartments", choices=("nuclei", "cyto", "cell", "both"), default="both")
    fit = sub.add_parser("train", help="Fit fastPLS and evaluate only held-out patients")
    fit.add_argument("--table", required=True)
    fit.add_argument("--model", required=True)
    fit.add_argument("--report", required=True)
    inference = sub.add_parser("predict", help="Apply the fitted fastPLS classifier")
    inference.add_argument("--table", required=True)
    inference.add_argument("--model", required=True)
    inference.add_argument("--output", required=True)
    dataset = sub.add_parser("extract-monusac", help="Extract small annotated examples from the downloaded MoNuSAC Parquet files")
    dataset.add_argument("--train-parquet", required=True)
    dataset.add_argument("--test-parquet", required=True)
    dataset.add_argument("--output-dir", required=True)
    dataset.add_argument("--per-split", type=int, default=4)
    acquire = sub.add_parser("download-monusac", help="Download and verify the pinned MoNuSAC Parquet mirror")
    acquire.add_argument("--output-dir", required=True)
    acquire_orion = sub.add_parser("download-orion-crc33", help="Download pinned calibrated ORION CRC33_01 inputs, excluding IF WSI")
    acquire_orion.add_argument("--output-dir", required=True)
    orion = sub.add_parser("extract-orion", help="Make a calibrated ORION H&E/cell-mask window")
    orion.add_argument("--he", required=True)
    orion.add_argument("--mask", required=True)
    orion.add_argument("--measured-csv", required=True)
    orion.add_argument("--sample", required=True)
    orion.add_argument("--x", type=int, required=True)
    orion.add_argument("--y", type=int, required=True)
    orion.add_argument("--size", type=int, default=2048)
    orion.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "build-table":
        parts = ("nuclei", "cyto") if args.compartments == "both" else (args.compartments,)
        result = build_table(args.manifest, args.output, compartments=parts)
    elif args.command == "build-inference":
        parts = ("nuclei", "cyto") if args.compartments == "both" else (args.compartments,)
        result = build_inference_table(args.manifest, args.output, compartments=parts)
    elif args.command == "train":
        result = train(args.table, args.model, args.report)
    elif args.command == "extract-monusac":
        result = extract(args.train_parquet, args.test_parquet, args.output_dir, args.per_split)
    elif args.command == "download-monusac":
        result = download(args.output_dir)
    elif args.command == "download-orion-crc33":
        result = download_orion_crc33(args.output_dir)
    elif args.command == "extract-orion":
        result = extract_window(args.he, args.mask, args.measured_csv, args.output_dir,
                                x=args.x, y=args.y, size=args.size, sample=args.sample)
    else:
        result = {"predictions": predict(args.table, args.model, args.output)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
