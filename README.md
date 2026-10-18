# SF-GSDA

Implementation of **Anatomy-guided graph modelling with source-free self-supervision for cross-dataset EEG emotion recognition.**

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0, pandas 2.2.3. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Use [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), SEED or DREAMER with harmonized binary valence. Exclude midpoint DEAP/DREAMER ratings before creating labels. Source NPZ arrays are `x: (N,C,B,T)`, `y`, `group`, `split: train/val/test`; bands are delta/theta/alpha/beta/gamma. `regions` assigns each channel to frontal, temporal, central, parietal or occipital nodes. Target adaptation input contains unlabeled `x`, subject `group` IDs and optional availability only. Keep target adaptation and evaluation subjects separate for inductive evaluation. The EEG importer accepts trial CSV columns `file,key,sample_rate,channels,subject,split,label` and supports MAT/NPZ/NPY waveform arrays.

For raw EEG import, install the data extra and export the documented trial manifest:

```bash
python -m pip install -e ".[data]"
python -m agg_sf.prepare_eeg --manifest data/source_trials.csv --output data/source.npz --bands
python -m agg_sf prepare --input data/source.npz --config config.json --output runs/source
```

## Training and inference

```bash
python -m agg_sf train --config config.json --output runs/source
python -m agg_sf.adapt --input data/unlabeled_target.npz --source runs/source/model.pt --config config.json --output runs/target
```

`config.json` records seed 13, channel-to-region order and graph/adapter settings. Adaptation omits labels and uses subjects separate from target evaluation.

## Evaluation

```bash
python -m agg_sf run --input data/target_test.npz --config config.json --output runs/target
python -m agg_sf evaluate --config config.json --output runs/target
```

## Comparisons and ablations

Set `variant` in a separate config per pipeline: `full`, `no-anatomy`, `no-reconstruction`, `no-band`, `no-neighbor`, `entropy-only`, `source-only`. Keep montage, splits and seed fixed.

## Outputs

`runs/source/` contains prepared `data.npz`, normalization and source `model.pt`. Target outputs include adapted `model.pt`, `target_normalization.npz`, `adaptation.csv`, `settings.json`, `predictions.npz`, `metrics.csv` and `risk_coverage.csv`.

```bash
python -m agg_sf plot --config config.json --output runs/target
```
