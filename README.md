# Solar Filament Segmentation 2026 — U-Net baseline with a label-noise analysis

Instance segmentation of solar filaments in GONG H-alpha full-disk images (MAGFiLO 1.0, Kaggle
`filament-segmentation-2026`). A ResNet U-Net trained at full 2048 x 2048 resolution on 1024 crops,
instances by connected components, scored with Panoptic Quality (IoU > 0.5).

## How to run

1. Create an environment and install: `pip install -r requirements.txt` (PyTorch with CUDA; an RTX 4090 trains one model in ~40-75 min).
2. Download the competition data from Kaggle and unzip it so that `data/MAGFiLO_1.0_Kaggle_2026/train/` and `.../test/` exist
   (or point `FILAMENT_DATA` at the unzipped folder). The data licence is non-commercial; it is not part of this repository.
3. Build the cache and the fixed split (by capture date, seed 20261020; identical to `src/split_seed20261020.json`):
   `FILAMENT_SCALE=1 python src/unet.py cache`
4. Train two seeds (one change per run; see the ablation in the notebook):
   `FILAMENT_SCALE=1 python src/unet.py train --tag v3_full_crop1024 --epochs 100 --bs 4 --crop 1024 --min-area 160`
   `FILAMENT_SCALE=1 python src/unet.py train --tag v3_full_crop1024_s2 --epochs 100 --bs 4 --crop 1024 --min-area 160 --seed 20261021`
5. Cache validation probabilities, tune post-processing on the validation split and evaluate:
   `FILAMENT_SCALE=1 python src/postproc.py probs --tags v3_full_crop1024,v3_full_crop1024_s2`
   `FILAMENT_SCALE=1 python src/postproc.py sweep --tags v3_full_crop1024,v3_full_crop1024_s2 --thr 0.7 --min-area 320`
   `FILAMENT_SCALE=1 python src/evaluate.py --tags v3_full_crop1024,v3_full_crop1024_s2 --thr 0.5 --min-area 240 --merge-r 12`
6. Write the submission (COCO compressed RLE at 2048 x 2048, one row per filament):
   `FILAMENT_SCALE=1 python src/postproc.py predict --tags v3_full_crop1024,v3_full_crop1024_s2 --thr 0.5 --min-area 240 --merge-r 12`

Or open `solar_filament_unet.ipynb`, which runs steps 3-6 in order.

## Features

- Reproducible: fixed by-date split, fixed seeds, every number in the notebook comes from a script in `src/`.
- Honest validation: the public leaderboard tracks the validation PQ computed against the merged annotation sets
  (v1 0.342 val / 0.34 LB; v3 0.359 val / 0.35 LB).
- Label-noise analysis: on validation images annotated twice, one annotator scored against the other reaches PQ 0.33,
  below the model (0.36); `src/evaluate.py` reports PQ = SQ x RQ, IoU and Dice distributions and one-to-many relations
  for the model and for the annotators.
- Ablation (one change per run): epochs, seeds, resolution, encoder, post-processing (limb mask, fragment merging,
  mean-probability filter, hysteresis), with the ones that do not help recorded as well.

## Layout

```
README.md  requirements.txt  solar_filament_unet.ipynb
src/unet.py        cache | train | eval | predict  (ResNet-18/34/50 U-Net, 1-channel input, BCE + Dice, bf16)
src/postproc.py    probs | sweep | predict        (instances, model averaging, submission writer)
src/evaluate.py    judged quantities on the validation split
src/split_seed20261020.json
```

## Training-free baseline

`src/baseline.py` is the first submission (public LB 0.12): limb-flattened dark-region threshold at 1/4 resolution,
`python src/baseline.py tune` then `python src/baseline.py predict --T 0.94 --A 45`.

## Licence

Code: MIT (see `LICENSE`). Encoder weights: torchvision ImageNet (BSD-3). Competition data: not included (non-commercial licence).
