# Dataset acquisition and evaluation contract

## MoNuSAC 2020

The [organizer release](https://monusac-2020.grand-challenge.org/Data/) provides H&E images and expert nuclei contours for epithelial cells, lymphocytes, macrophages and neutrophils. The [MedOtter Parquet conversion](https://huggingface.co/datasets/MedOtter/MoNuSAC2020) preserves instance and type maps with patient barcodes. Both sources state CC BY-NC-SA 4.0; raw images and annotations are intentionally excluded from this MIT code repository. The two pinned Parquet files total 277,453,464 bytes, representing 209 train and 101 test images from 71 patients. SHA256 is checked by `tambua download-monusac` before use:

- train `c0d7c0c99600e960a38ed92e67cfa93a80eafb68186d8b958d90a8452dc9cbda`;
- test `196566e3bf4ddc06d880e244864c27f1b60b3db91a421b0daa0465ff81bf382f`.

The eight locally extracted examples contain 2,079 scored nuclei across eight distinct patients. Each split includes all four classes, but the small test example has very few neutrophils; it is a data-path demonstration, not a powered benchmark. The mirror gives 40× nominal magnification but no verified pixel size for these files. CellPhenotyper/GigaTIME inference requires a separately verified physical calibration and must not silently assume 0.25 µm/pixel.

The original organizer test archive was subsequently checked on Chiamaka. It contains 101 `.svs` ROI images with scanner MPP metadata and paired expert XML; **all 101 header MPP values exactly match the corresponding XML `MicronsPerPixel` values** (range 0.2456–0.2527 µm/pixel). There are 25 patient folders; 32 ROIs are at least 512 pixels wide and high. This provides calibrated candidates for the expert test without relying on the mirror's nominal 40× label. Source slide and patient identifiers must still be checked against any future model-training manifest before calling the test independent.

MoNuSAC has expert nuclei, not full cell bodies or measured protein. With its expert instance mask as the quantification substrate, the experiment estimates virtual-marker/classifier performance conditional on perfect localization. A second end-to-end endpoint must use CellPhenotyper-detected cells, predeclared instance matching and a count of unmatched detections and references. Never score only easy matched cells without accounting for failures.

## ORION colorectal cancer

The [first-party release](https://github.com/labsyspharm/orion-crc/blob/main/datarelease-README.md) contains registered H&E and 19-channel Orion IF of the same tissue section, computationally generated labelled cell masks and measured marker tables. It is suitable for biological validation of virtual markers and measured-marker-derived cell classes. Full multiplex WSIs are typically 44–147 GiB each, so select bounded registered windows and document all file ranges, coordinates, registration error and patient IDs. The model must be trained on separate patients and evaluated against labels defined from the **measured** channels without looking at virtual GigaTIME outputs. A serial-section dataset must not be used as one-to-one cell truth.

ORION's measured panel includes Pan-CK, CD3e, CD20, CD68 and CD163, which can support carefully gated epithelial, lymphoid and macrophage references. It does not supply an equally specific neutrophil marker, so ORION alone must not be presented as a four-class reference. Define gates, ambiguous-cell exclusions and quality thresholds before viewing virtual-marker predictions; validate the four-class endpoint separately on MoNuSAC or another independently typed cohort.

A bounded local CRC33_01 case was acquired from that public S3 release without downloading the 43.9 GiB IF WSI. Its registered H&E, cell mask and measured-cell table have SHA256 values `e5be3cf68c9adc8bada72a98e681d0ed852abe7b7cbe9b30d4102bea42b9da43`, `fd1b6876c714d37a4391ffd735e1e8a6bed2b52b4ac03c028315894697563aa2`, and `42683f328cedc134426b45179de5e95c2f14481b9e1f6962c729c2362a37a021`. Both OME image planes are 52,740×36,354 at **0.325 µm/pixel**. The reproducible level-zero window `(x=24576, y=39424, width=2048, height=2048)` has 4,104 complete cell IDs and matching measured rows; 226 edge-truncated cells were excluded. The maximum discrepancy between centroids recomputed from the retained masks and published table coordinates was `1.5×10⁻¹¹` pixels. Local files are under `data/orion_crc33_01/window_x24576_y39424_verified/` and ignored by Git. This is an acquisition/alignment check, not marker inference or biological classification. An independent, predeclared phenotyping gate and multiple patients remain necessary for a validation result. The 19-channel IF WSI is deliberately not downloaded to the local machine.

## NuCLS

The [first-party NuCLS repository](https://github.com/PathologyDataScience/NuCLS) contains pathologist-reviewed breast H&E nuclei annotations under CC0. Its ontology and annotation shapes differ from MoNuSAC. Use only after a documented label mapping and a TCGA patient-barcode overlap check.

## Split and metric plan

Lock patient-disjoint train, validation and test IDs before model selection. Treat MoNuSAC's official test patients as untouched test cases. Evaluate macro F1 and balanced accuracy as primary descriptive scores, per-class recall/F1 and confusion as secondary, with patient-bootstrap intervals after sufficient patients. Report cell detection/matching, GigaTIME generation, marker quantification, fastPLS fit and inference times separately, plus pixel spacing and marker QC. Compare with majority and measured-marker fastPLS references when ORION is included. Cell-level counts are not independent biological replicates.
