# Tambua

Tambua is a research pipeline for identifying individual cell types from **CellPhenotyper** per-cell GigaTIME virtual-marker scores. It uses the corrected, MIT-licensed [fastPLS 0.3.0](https://github.com/tkcaccia/fastPLS-py/tree/b4ea261902244bfb84ad933a236d364d19f905ee) SIMPLS–LDA classifier. Its initial ontology is epithelial cell, lymphocyte, macrophage and neutrophil. These are predictions from H&E-derived **virtual scores**, not measured protein expression, calibrated probabilities or clinical diagnoses.

## Inputs and alignment

CellPhenotyper produces instance masks and GigaTIME `*_nuclei_gigatime_mean_intensity.csv` and `*_cyto_gigatime_mean_intensity.csv` files. Tambua joins these by the **same positive `label_id` within each sample**. It takes expert cell-type annotations from a separate CSV (`label_id,cell_type`) and requires a specimen manifest with `sample_id,patient_id,split,nuclei_csv,cyto_csv,labels_csv`. Splits are patient-disjoint. The 21 virtual biological/nuclear channels from each selected compartment are used; TRITC and Cy5 are omitted because they are background channels. DAPI is retained as a nuclear morphology control. The model file contains training features and labels because the current fastPLS native fitted object is not safely serializable; keep it under the source dataset's access terms and do not commit it.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[datasets]'
tambua build-table --manifest specimens.csv --output results/cells.csv --compartments both
tambua train --table results/cells.csv --model results/cell_classifier.npz --report results/test_report.json
tambua build-inference --manifest new_specimens.csv --output results/new_cells.csv --compartments both
tambua predict --table results/new_cells.csv --model results/cell_classifier.npz --output results/predictions.csv
```

For inference without reference labels, `new_specimens.csv` needs `sample_id,patient_id,nuclei_csv,cyto_csv` and no expert annotation. Use the same compartments as training; the frozen model rejects missing feature columns. The training command reports accuracy, balanced accuracy, macro F1, each class recall/F1, confusion matrix, a training-majority baseline, and equal-weight per-patient accuracy on **test patients only**. A patient or entire test split missing a reference class is listed explicitly; its balanced accuracy and macro F1 are undefined rather than averaged over the remaining classes. It refuses patient leakage or test classes absent from training. Cell-level scores are descriptive; patient-level uncertainty intervals and external validation remain necessary before a performance claim.

## Dataset strategy

1. [MoNuSAC 2020](https://monusac-2020.grand-challenge.org/Data/) is the initial four-type H&E benchmark. Its expert nucleus contours and identities provide independent reference labels. A controlled first study can quantify GigaTIME over the expert nuclei, separating marker/classifier performance from CellPhenotyper's detection error. An end-to-end study must match CellPhenotyper cells to expert instances without using the expert mask at inference. Its CC BY-NC-SA 4.0 data are downloaded separately and not redistributed here.
2. [ORION colorectal cancer](https://github.com/labsyspharm/orion-crc/blob/main/datarelease-README.md) provides same-section registered H&E, multiplex protein images, cell masks and measured single-cell tables. Use it to test whether virtual-marker-derived predictions agree with **measured-marker-defined** phenotypes, with patient-disjoint splits and a locked registration gate. Its full protein WSIs are tens to hundreds of GiB; download bounded, documented cases or image windows only. A measured-marker label must never be derived from Tambua's own virtual scores.
3. [NuCLS](https://github.com/PathologyDataScience/NuCLS) is an additional breast H&E nucleus-label benchmark, but its ontology must be mapped prospectively; it is not automatically interchangeable with MoNuSAC.

The repository contains code and a small controlled demonstration only. No real held-out biological accuracy is claimed until GigaTIME has been run on eligible, independently annotated cases and the complete manifest, model and per-patient results have been audited. Record pixel spacing and physical registration for every case; cell-level matching across serial sections is insufficient.

To acquire the pinned public MoNuSAC mirror and extract a provenance-checked, patient-disjoint example cohort:

```bash
tambua download-monusac --output-dir data
tambua extract-monusac --train-parquet data/monusac-train-0000.parquet \
  --test-parquet data/monusac-test-0000.parquet \
  --output-dir data/monusac_examples --per-split 4
```

The downloaded Parquet contains all 310 released images and annotation maps. The extraction command produces eight example H&E images, scored instance masks and independent cell-type CSVs from eight distinct patients. It does **not** generate GigaTIME scores or a trained biological classifier. Details and checksums are in [dataset notes](docs/DATASETS.md).

For a physically calibrated ORION validation window, download the registered H&E, matching cell mask and measured-cell table from the [first-party public S3 release](https://github.com/labsyspharm/orion-crc/blob/main/datarelease-README.md), then use level-zero coordinates:

```bash
tambua extract-orion --he he_registered.ome.tif --mask cell_mask.ome.tif \
  --measured-csv measured_cells.csv --sample CRC33_01 --x 24576 --y 39424 \
  --size 2048 --output-dir data/orion_crc33_01/window_x24576_y39424_verified
```

This checks the OME physical pixel sizes and native image/mask alignment, removes instances cut by the crop boundary, joins every retained instance to exactly one **measured** protein row, checks that its published centroid matches the mask, and writes calibrated OME-TIFFs with source hashes. Those rows are a separate reference for future evaluation; `extract-orion` does not assign cell types or run GigaTIME. The demonstrated CRC33_01 window is 0.325 µm/pixel and contains 4,104 complete cells. It is one patient and therefore cannot establish held-out classification accuracy.

## Relationship to Bora

CellPhenotyper's **Virchow2** route produces non-overlapping tile features for KODAMA and a full spatial-token field for Bora tissue segmentation. Tambua reads CellPhenotyper's separate cell masks and GigaTIME quantification tables. The tissue and cell classifiers share fastPLS infrastructure, but have different observation units, labels and validation endpoints.

