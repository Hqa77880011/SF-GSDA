# SF-GSDA

Implementation of **Anatomy-guided graph modelling with source-free self-supervision for cross-dataset EEG emotion recognition.**

## Install

Use Python 3.11 in an isolated environment, then run from the repository root:

```bash
python -m pip install -e .
```

## Data

Use [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/), SEED or DREAMER with harmonized binary valence. Exclude midpoint DEAP/DREAMER ratings before creating labels. Source NPZ arrays are `x: (N,C,B,T)`, `y`, `group`, `split: train/val/test`; bands are delta/theta/alpha/beta/gamma. `regions` assigns each channel to frontal, temporal, central, parietal or occipital nodes. Target adaptation input contains unlabeled `x`, subject `group` IDs and optional availability only. Keep target adaptation and evaluation subjects separate for inductive evaluation. The EEG importer accepts trial CSV columns `file,key,sample_rate,channels,subject,split,label` and supports MAT/NPZ/NPY waveform arrays.

## Workflow

```bash
python -m agg_sf prepare --input data/source.npz
python -m agg_sf train
python -m agg_sf run
python -m agg_sf evaluate
python -m agg_sf plot
```

Main commands accept `--config config.json` and `--output runs/default`. Keep each seed/condition in its own output directory. The resolved configuration is written to `settings.json`.

## Configuration

`regions` and `region_adjacency` define montage-to-region pooling and anatomical connections. `anatomy_weight` mixes fixed/functional graphs. `bottleneck` sizes adapters. Reconstruction, band, information and neighbor weights define source/target objectives. `confidence_threshold` and `neighbors` are fixed from source development. `adaptation_epochs` sets a label-free target stopping epoch. Training epochs, batch size, learning rate and seed use the defaults shown in `config.json` where applicable.

## Outputs

`model.pt` stores the source or adapted model. Source outputs use `predictions.npz`, `metrics.csv` and `risk_coverage.csv`. Target adaptation writes `adaptation.csv`, adapted weights and target normalization. Classification metrics include accuracy, Macro-F1, balanced accuracy, calibration and selective risk.

## Implementation and controls

Region pooling masks missing regions. The functional graph is feature-responsive, while adaptation freezes graph/classifier weights and changes only bottleneck adapters and normalization parameters. Target losses combine confident pseudo-labels, entropy/diversity, masked reconstruction, band consistency and minibatch-neighbor agreement. Neither target labels nor source feature banks enter adaptation.

Controls: `full`, `no-anatomy`, `no-reconstruction`, `no-band`, `no-neighbor`, `entropy-only`, `source-only`. Choose `--variant` and use the same preparation and fitting choices in each separate output directory.

```bash
python -m agg_sf.compare --inputs "runs/*/metrics.csv" --reference full --metrics accuracy macro_f1 balanced_accuracy --unit seed
```

The comparison command averages repeated rows within the specified independent unit, bootstraps units and applies paired Wilcoxon/Holm comparisons per metric.

## Source-free adaptation

```bash
python -m pip install -e ".[data]"
python -m agg_sf.prepare_eeg --manifest data/source_trials.csv --output data/source.npz --bands
python -m agg_sf.adapt --input data/unlabeled_target.npz --source runs/default/model.pt --output runs/target
python -m agg_sf run --input data/target_test.npz --output runs/target
python -m agg_sf evaluate --output runs/target
```

The adaptation archive omits labels. Target evaluation archives contain `x,y,group`; their signals are normalized with the saved adaptation statistics. Use a target configuration with the correct channel-to-region order when montages differ. Fixed adapter training epochs avoid target-label checkpoint selection.

