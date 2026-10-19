# Dataset acquisition and evaluation contract

## MoNuSAC 2020

The [organizer release](https://monusac-2020.grand-challenge.org/Data/) provides H&E images and expert nuclei contours for epithelial cells, lymphocytes, macrophages and neutrophils. The [MedOtter Parquet conversion](https://huggingface.co/datasets/MedOtter/MoNuSAC2020) preserves instance and type maps with patient barcodes. Both sources state CC BY-NC-SA 4.0; raw images and annotations are intentionally excluded from this MIT code repository. The two pinned Parquet files total 277,453,464 bytes, representing 209 train and 101 test images from 71 patients. SHA256 is checked by `tambua download-monusac` before use:

- train `c0d7c0c99600e960a38ed92e67cfa93a80eafb68186d8b958d90a8452dc9cbda`;
- test `196566e3bf4ddc06d880e244864c27f1b60b3db91a421b0daa0465ff81bf382f`.

The eight locally extracted examples contain 2,079 scored nuclei across eight distinct patients. Each split includes all four classes, but the small test example has very few neutrophils; it is a data-path demonstration, not a powered benchmark. The mirror gives 40× nominal magnification but no verified pixel size for these files. CellPhenotyper/GigaTIME inference requires a separately verified physical calibration and must not silently assume 0.25 µm/pixel.

MoNuSAC has expert nuclei, not full cell bodies or measured protein. With its expert instance mask as the quantification substrate, the experiment estimates virtual-marker/classifier performance conditional on perfect localization. A second end-to-end endpoint must use CellPhenotyper-detected cells, predeclared instance matching and a count of unmatched detections and references. Never score only easy matched cells without accounting for failures.

## ORION colorectal cancer

The [first-party release](https://github.com/labsyspharm/orion-crc/blob/main/datarelease-README.md) contains registered H&E and 19-channel Orion IF of the same tissue section, labelled cell masks and measured marker tables. It is suitable for biological validation of virtual markers and measured-marker-derived cell classes. Full multiplex WSIs are typically 44–147 GiB each, so select bounded registered windows and document all file ranges, coordinates, registration error and patient IDs. The model must be trained on separate patients and evaluated against labels defined from the **measured** channels without looking at virtual GigaTIME outputs. A serial-section dataset must not be used as one-to-one cell truth.

## NuCLS

The [first-party NuCLS repository](https://github.com/PathologyDataScience/NuCLS) contains pathologist-reviewed breast H&E nuclei annotations under CC0. Its ontology and annotation shapes differ from MoNuSAC. Use only after a documented label mapping and a TCGA patient-barcode overlap check.

## Split and metric plan

Lock patient-disjoint train, validation and test IDs before model selection. Treat MoNuSAC's official test patients as untouched test cases. Evaluate macro F1 and balanced accuracy as primary descriptive scores, per-class recall/F1 and confusion as secondary, with patient-bootstrap intervals after sufficient patients. Report cell detection/matching, GigaTIME generation, marker quantification, fastPLS fit and inference times separately, plus pixel spacing and marker QC. Compare with majority and measured-marker fastPLS references when ORION is included. Cell-level counts are not independent biological replicates.
