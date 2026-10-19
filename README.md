# Tambua

Tambua is a research pipeline for identifying individual cell types from **CellPhenotyper** per-cell GigaTIME virtual-marker scores. It uses the MIT-licensed [fastPLS 0.4 R package](https://github.com/tkcaccia/fastPLS/tree/265108b6aa250c2534f98ec89037bb5cd001dca0) for SIMPLS–LDA classification. Its initial ontology is epithelial cell, lymphocyte, macrophage and neutrophil. These are predictions from H&E-derived **virtual scores**, not measured protein expression, calibrated probabilities or clinical diagnoses.

## Inputs and alignment

CellPhenotyper produces instance masks and GigaTIME `*_nuclei_gigatime_mean_intensity.csv`, `*_cyto_gigatime_mean_intensity.csv`, or whole-cell `*_cell_gigatime_mean_intensity.csv` files. Tambua joins these by the **same positive `label_id` within each sample**. It takes expert cell-type annotations from a separate CSV (`label_id,cell_type`) and requires a specimen manifest with `sample_id,patient_id,split` plus the chosen compartment CSV and `labels_csv`. Splits are patient-disjoint. The 21 virtual biological/nuclear channels from each selected compartment are used; TRITC and Cy5 are omitted because they are background channels. DAPI is retained as a nuclear morphology control. The native R fitted model is saved as `.rds` with a `.rds.json` feature/version sidecar. Any distributed trained model must preserve the source dataset's attribution and access terms; the code's MIT license does not relicense it.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[datasets]'
# In R >= 4.6, install float and gpuinfo, then the pinned GitHub fastPLS source:
Rscript -e 'install.packages(c("float", "gpuinfo"), repos="https://cloud.r-project.org")'
fastpls_source_dir=$(mktemp -d)
git init "$fastpls_source_dir"
git -C "$fastpls_source_dir" remote add origin https://github.com/tkcaccia/fastPLS.git
git -C "$fastpls_source_dir" fetch --depth 1 origin 265108b6aa250c2534f98ec89037bb5cd001dca0
git -C "$fastpls_source_dir" checkout --detach FETCH_HEAD
R CMD INSTALL "$fastpls_source_dir"
Rscript -e 'stopifnot(as.character(packageVersion("fastPLS")) == "0.4")'
tambua build-table --manifest specimens.csv --output results/cells.csv --compartments cell
tambua train --table results/cells.csv --model results/cell_classifier.rds --report results/test_report.json
tambua build-inference --manifest new_specimens.csv --output results/new_cells.csv --compartments cell
tambua predict --table results/new_cells.csv --model results/cell_classifier.rds --output results/predictions.csv
```

If fastPLS 0.4 is installed in a project-specific R library, set `TAMBUA_R_LIB=/absolute/path/to/library` for the `tambua train` and `tambua predict` commands. The bridge adds that library explicitly even under `Rscript --vanilla`, avoiding an older system fastPLS package.

For whole-cell inference without reference labels, `new_specimens.csv` needs `sample_id,patient_id,cell_csv` and no expert annotation. Use the same compartments as training; the frozen model rejects missing feature columns. The training command reports accuracy, balanced accuracy, macro F1, each class recall/F1, confusion matrix, a training-majority baseline, and equal-weight per-patient accuracy on **test patients only**. A patient or entire test split missing a reference class is listed explicitly; its balanced accuracy and macro F1 are undefined rather than averaged over the remaining classes. It refuses patient leakage or test classes absent from training. Cell-level scores are descriptive; patient-level uncertainty intervals and external validation remain necessary before a performance claim.

## Dataset strategy

1. [MoNuSAC 2020](https://monusac-2020.grand-challenge.org/Data/) is the initial four-type H&E benchmark. Its expert nucleus contours and identities provide independent reference labels. A controlled first study can quantify GigaTIME over the expert nuclei, separating marker/classifier performance from CellPhenotyper's detection error. An end-to-end study must match CellPhenotyper cells to expert instances without using the expert mask at inference. Its CC BY-NC-SA 4.0 data are downloaded separately and not redistributed here.
2. [ORION colorectal cancer](https://github.com/labsyspharm/orion-crc/blob/main/datarelease-README.md) provides same-section registered H&E, multiplex protein images, cell masks and measured single-cell tables. Use it to test whether virtual-marker-derived predictions agree with **measured-marker-defined** phenotypes, with patient-disjoint splits and a locked registration gate. Its full protein WSIs are tens to hundreds of GiB; download bounded, documented cases or image windows only. A measured-marker label must never be derived from Tambua's own virtual scores.
3. [NuCLS](https://github.com/PathologyDataScience/NuCLS) is an additional breast H&E nucleus-label benchmark, but its ontology must be mapped prospectively; it is not automatically interchangeable with MoNuSAC.

An initial cross-organ study has now completed on 27 MoNuSAC lung training ROIs (9 patients) and 10 prostate test ROIs (6 different patients), with original pixel spacing verified at 0.19448–0.2527 µm/pixel and GigaTIME run at exactly 0.25 µm/pixel. fastPLS 0.4 reached **89.0% accuracy**, **44.5% balanced accuracy** and **44.7% macro F1** on 3,584 expert-typed prostate nuclei; it missed all 20 test macrophages and 2 neutrophils. The result is conditional on expert nuclear localization and a 3 µm cell proxy, and several virtual-DAPI previews are visually diffuse. The [complete gallery and score report](https://tkcaccia.github.io/Tambua-Analysis/#monusac) show every test image. This is not an end-to-end cell-detection result or a head-to-head comparison with other methods. The original MoNuSAC train/test archives share two patient IDs, so this experiment excluded overlap and used patient-disjoint lung/prostate subsets. Record pixel spacing and physical registration for every case; cell-level matching across serial sections is insufficient.

To acquire the pinned public MoNuSAC mirror and extract a provenance-checked, patient-disjoint example cohort:

```bash
tambua download-monusac --output-dir data
tambua extract-monusac --train-parquet data/monusac-train-0000.parquet \
  --test-parquet data/monusac-test-0000.parquet \
  --output-dir data/monusac_examples --per-split 4
```

The downloaded Parquet contains all 310 released images and annotation maps. The extraction command produces eight example H&E images, scored instance masks and independent cell-type CSVs from eight distinct patients. It does **not** generate GigaTIME scores or a trained biological classifier. Details and checksums are in [dataset notes](docs/DATASETS.md).

For a physically calibrated ORION validation window, acquire the pinned registered H&E, matching cell mask and measured-cell table from the [first-party public S3 release](https://github.com/labsyspharm/orion-crc/blob/main/datarelease-README.md), then use level-zero coordinates. The downloader deliberately excludes the 43.9 GiB multiplex IF WSI:

```bash
tambua download-orion-crc33 --output-dir data/orion_crc33_01
tambua extract-orion --he data/orion_crc33_01/he_registered.ome.tif \
  --mask data/orion_crc33_01/cell_mask.ome.tif \
  --measured-csv data/orion_crc33_01/measured_cells.csv \
  --sample CRC33_01 --x 24576 --y 39424 \
  --size 2048 --output-dir data/orion_crc33_01/window_x24576_y39424_verified
```

This checks the OME physical pixel sizes and native image/mask alignment, removes instances cut by the crop boundary, joins every retained instance to exactly one **measured** protein row, checks that its published centroid matches the mask, and writes calibrated OME-TIFFs with source hashes. Those rows are a separate reference for future evaluation; `extract-orion` does not assign cell types or run GigaTIME. The demonstrated CRC33_01 window is 0.325 µm/pixel and contains 4,104 complete cells. It is one patient and therefore cannot establish held-out classification accuracy.

## Relationship to Bora

CellPhenotyper's **Virchow2** route produces non-overlapping tile features for KODAMA and a full spatial-token field for Bora tissue segmentation. Tambua reads CellPhenotyper's separate cell masks and GigaTIME quantification tables. The tissue and cell classifiers share fastPLS infrastructure, but have different observation units, labels and validation endpoints.

