# Solar Filament Segmentation 2026

Instance segmentation of solar filaments in GONG H-alpha full-disk images for the Kaggle competition `filament-segmentation-2026` (MAGFiLO 1.0 data), scored by Panoptic Quality.

## How to Run

1. Install Python 3.12 and the dependencies: `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.
2. Download the competition data (you must accept the competition rules first) and unzip it so that `data/MAGFiLO_1.0_Kaggle_2026/train/` and `data/MAGFiLO_1.0_Kaggle_2026/test/` exist:
   `kaggle competitions download -c filament-segmentation-2026 -p data && unzip data/*.zip -d data`.
3. Tune the threshold and minimum area on a fixed subset of training images: `.venv/bin/python src/baseline.py tune --n 40`.
4. Write the submission: `.venv/bin/python src/baseline.py predict --T 0.94 --A 45` → `submissions/submission_thresh_T0.94_A45.csv`.

## Features

- Training-free baseline: solar-disk mask, limb-darkening flattening (image divided by a masked Gaussian background), dark-region threshold and connected components, at 1/4 resolution.
- Local Panoptic Quality (unique matching at IoU > 0.5) for tuning against the COCO polygon annotations.
- Output in the leaderboard format: one row per filament, COCO compressed RLE at 2048 x 2048.

## Results

| version | method | public leaderboard PQ |
|---|---|---|
| baseline (2026-10-20) | threshold T 0.94, area 45 at 1/4 scale, sigma 20 | 0.12 |

A trained segmentation model will follow; it will be added here with training code, inference code and the hardware used.

## Data

The competition data is not included. Its licence allows non-commercial competition and research use only, so download it from Kaggle yourself.

## Licence

Code: MIT (see `LICENSE`), as the competition rules require an OSI-approved licence that does not limit commercial use.
