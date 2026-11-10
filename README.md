# MAGNET (book-chapter version): inference demo and reproduction check

This repository shows how the MAGNET numbers of the book chapter (`original_records/xiao.tex`, Tables 1 and 2)
come out of the saved checkpoints, and lets anyone re-run the original inference code on real evaluation samples.
The MAGNET runs of Table 2 were trained between 2025-09-22 and 2025-10-03 (the Transformer-only baselines of
Table 1 in 2026-05). Training is not reproduced here.

It ships the original model and evaluation code, the checkpoints and training records, one complete evaluation batch
(8 samples) from the original validation split of each of the two scenarios whose original data survives
(16 samples in total, 5.6 MB), and reference values computed on the complete validation splits.

## 1. What can be checked here

| Claim | How it is checked | Result |
|---|---|---|
| Every Table 2 value is the per-epoch minimum of the validation ADE / FDE in the run's training record | `check_table2.py` reads the 22 shipped training records | 20 / 20 cells |
| Loading a checkpoint with the original code and running the original `validate()` reproduces the numbers of the training run | the shipped batches are evaluated with each of the 8 checkpoints and compared with the reference values of the same batch | 8 / 8 (within 1e-4) |
| The reference values are those of the training run | the sample-weighted mean of the reference values over all batches of the complete validation split equals the training record at the checkpoint epoch | 8 / 8, to 4 decimals |

The 16 shipped samples cannot by themselves give the means over the 1500 / 3000 validation samples. Those means
were obtained by evaluating the complete validation splits with a larger reproduction package (8.3 GB of
evaluation inputs, not published); their per-batch and per-sample values are shipped in
`reference/full_validation_reference.json`, and the shipped batches show that the same code and checkpoints give
the same values. Results of that package for all of Tables 1 and 2 are in `reference/full_package_results.txt`
(section 4.2).

## 2. Quick start

```bash
git clone <this repository> && cd <repository folder>

# Environment: Python 3.9 (reference versions in requirements.txt)
conda create -n magnet python=3.9 -y && conda activate magnet
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build you need
pip install -r requirements.txt

# Whole check (about 1 minute on CPU; give a GPU id instead of "cpu" to use a GPU)
bash reproduce.sh cpu

# Step-by-step inference of a random shipped sample, with a bird's-eye-view figure in figures/demo/
bash demo.sh
```

`reproduce.sh` ends with `[reproduce] PASS: 20/20 Table 2 cells match the training records; 8/8 shipped batches
reproduced; shipped reference values agree with the training records (8/8)`.

Other forms:

```bash
bash demo.sh near_multiple_vehicles cpu 5        # fixed scenario and sample (position 0-7 in the shipped batch)
python demo_infer.py show --scenario high_lateral_acceleration --topology default --sample 3 --fig_dir figures/demo
python demo_infer.py verify --device cuda:0      # step [2] of reproduce.sh only
python check_table2.py                           # step [1] of reproduce.sh only
```

`show` prints seven steps: input polylines, ego branch, context Transformer, GNN, late fusion (norms of the three
128-d parts), output head (120 values -> 60 x 2 increments -> cumulative sum), and the sample's ADE / FDE, both as
computed by the original `validate()` (with the reference value of the complete-split run) and in the ego frame
(section 10). Example output: `figures/example_high_lateral_acceleration_nearby_only_val1155.png`.

## 3. Contents

```
reproduce.sh            whole check: check_table2.py + demo_infer.py verify, prints PASS / FAIL
demo.sh                 inference walk-through of a (random) shipped sample
demo_infer.py           modes: verify, show
check_table2.py         Table 2 against the training records (no computation)
requirements.txt        reference environment
code/                   original MAGNET code as used for training and evaluation (comments translated, section 7)
  train.py              original training script; its validate() and loss are what demo_infer.py calls
  transformer_predictor.py   original MAGNET model (TransformerPredictor)
  utils/dataset.py, config.py
  eval_inputs.py        rebuilds the shipped samples into the original GraphData objects (added for this repository)
checkpoints/magnet/     22 runs of Table 2 (5 scenarios x 4 run labels, plus 2 unused high-speed Default runs):
                          results/loss_record_*.json for all 22, models/best_model_*.pth for the 8 runs of the
                          two scenarios shipped in data/samples/
data/samples/           one evaluation batch (8 samples) per scenario, see section 8
reference/
  full_validation_reference.json   per-batch val_loss / ADE / FDE and per-sample ADE / FDE of the complete
                                   validation splits, for the 8 checkpoints
  full_package_results.txt         Tables 1 and 2 re-evaluated with the larger reproduction package
original_records/
  xiao.tex              the chapter
  table1_sources/       original result files behind Table 1
figures/                example output of demo.sh
provenance/             untranslated original code, translation check, script that built data/ and reference/
```

## 4. Results

### 4.1 Re-run here (`bash reproduce.sh`)

Shipped batches, original `validate()`, reproduced vs. reference values of the same batch (val_loss / ADE / FDE):

| Scenario (batch) | Run label | Reproduced | Reference |
|---|---|---|---|
| High lateral acceleration (batch 144 of 188) | Default | 0.2780 / 2.9121 / 5.1113 | 0.2780 / 2.9121 / 5.1113 |
| | Ego Only | 0.1903 / 1.8498 / 3.6998 | 0.1903 / 1.8498 / 3.6998 |
| | Nearby Only | 0.0860 / 1.0829 / 1.6575 | 0.0860 / 1.0829 / 1.6575 |
| | No Ego Road | 0.2741 / 2.9450 / 5.2440 | 0.2741 / 2.9450 / 5.2440 |
| Near multiple vehicles (batch 234 of 375) | Default | 0.1712 / 0.8724 / 1.5781 | 0.1712 / 0.8724 / 1.5781 |
| | Ego Only | 0.1158 / 1.5307 / 2.3906 | 0.1158 / 1.5307 / 2.3906 |
| | Nearby Only | 0.2488 / 2.9322 / 4.9609 | 0.2488 / 2.9322 / 4.9609 |
| | No Ego Road | 0.1253 / 2.0525 / 3.4531 | 0.1253 / 2.0525 / 3.4531 |

The last digit of val_loss can differ by 1 between machines (e.g. 0.2489 for near multiple vehicles / Nearby Only);
the check allows a difference of 1e-4 for each value.

Reference values of the complete validation split vs. the training record of the checkpoint epoch:

| Scenario (n) | Run label | All batches: val_loss / ADE / FDE | Training record (epoch) | Table 2 (min over epochs) |
|---|---|---|---|---|
| High lateral acceleration (1500) | Default | 0.3462 / 2.1726 / 4.2734 | same (89) | 2.13 / 4.20 |
| | Ego Only | 0.3009 / 2.0631 / 3.8697 | same (100) | 2.06 / 3.87 |
| | Nearby Only | 0.3176 / 1.6441 / 3.4241 | same (95) | 1.58 / 3.31 |
| | No Ego Road | 0.3673 / 2.3695 / 4.6415 | same (93) | 2.34 / 4.57 |
| Near multiple vehicles (3000) | Default | 0.2395 / 1.5440 / 2.8396 | same (97) | 1.51 / 2.77 |
| | Ego Only | 0.2197 / 1.3799 / 2.4396 | same (99) | 1.38 / 2.43 |
| | Nearby Only | 0.3137 / 1.8130 / 3.3663 | same (83) | 1.80 / 3.35 |
| | No Ego Road | 0.2403 / 1.6234 / 2.9448 | same (100) | 1.62 / 2.94 |

The checkpoint is saved at the epoch with the lowest validation loss, whereas Table 2 reports the per-epoch minima of
ADE and FDE (taken independently). Re-evaluating a checkpoint therefore gives its own epoch's record, which for most
cells is slightly above the table value.

### 4.2 Larger reproduction package (reference only, `reference/full_package_results.txt`)

- **Exact** (complete original validation splits): the 8 cells above; Transformer-only on high lateral acceleration,
  original 3000-sample manifest: 1.5276 / 3.0625, identical to the original result file.
- **Approximate** (medium speed, high speed and intersection, whose original data was replaced by a later
  re-processing): same code and checkpoints on 3000 currently available samples per scenario. Most values are
  higher than in the tables: MAGNET by up to 0.33 m ADE / 0.52 m FDE, Transformer-only by up to 0.39 m ADE /
  1.20 m FDE (intersection).
- **Table 1 recomputed** the same way: the presented method 2.214 / 4.682 (table: 2.09 / 4.47), Transformer-only
  2.412 / 5.063 (2.23 / 4.52), LSTM 2.208 / 6.435 (2.21 / 6.50), IDM 9.258 / 18.349 (9.13 / 18.04). 

## 5. Where every number in the chapter comes from

**Table 2** (5 scenarios x 4 run labels): `best_ade` / `best_fde` of each run's `results/loss_record_*.json` in
`checkpoints/magnet/results_type_<scenario>_<label>_adjacency/training_*/`, i.e. the per-epoch minimum on the
validation split (ADE and FDE taken independently, possibly at different epochs). `check_table2.py` prints 20 MATCH
lines plus 2 DIFF lines for two unused high-speed Default runs (450 / 50 and 4500 / 500 training / validation
samples).

**Table 1**:

| Row | Value | Source |
|---|---|---|
| The presented method | 2.09 / 4.47 | mean of the printed (rounded) Table 2 Nearby Only row over the 5 scenarios (2.086 / 4.466); from the unrounded training records the mean is 2.085 / 4.464 |
| Transformer-only | 2.23 / 4.52 | mean of 4 per-scenario evaluations on 3000-sample manifests (2.235 / 4.519): medium speed 1.653 / 3.624, high speed 2.938 / 5.782, high lateral acceleration 1.528 / 3.062, intersection 2.821 / 5.608 (`original_records/table1_sources/transformer_only_*`) |
| LSTM | 2.21 / 6.50 | `original_records/table1_sources/lstm_3000.json` (2.211 / 6.503), medium speed only |
| IDM | 9.13 / 18.04 | `original_records/table1_sources/idm_3000.json` (9.135 / 18.046), medium speed only |
| VectorNet | 2.19 / 4.25 | no result file, log or checkpoint containing these numbers was found; this row cannot be reproduced |

`original_records/table1_sources/gnn_nearby_only_3000.json` is a manifest evaluation of the medium-speed Nearby Only
checkpoint (1.497 / 3.674) made at the same time as the baselines; it is not used in either table.

## 6. Environment

Reference: Linux, Python 3.9.21, torch 2.0.1, torch_geometric 2.6.1 (versions of all packages in
`requirements.txt`). The check was run on an NVIDIA RTX A6000 (CUDA 11.8) and on CPU, and in a fresh environment
created from `requirements.txt` with the CPU build of torch. The nuPlan devkit is not needed: `code/utils/dataset.py`
imports it optionally and otherwise prints `Warning: nuPlan imports not available, some functions may not work`
(filtered out by the scripts); the functions that need it are not used.

## 7. Code provenance

`code/` contains the versions used for training:

| File | Git blob hash of the original | Note |
|---|---|---|
| `code/transformer_predictor.py` | af63073 | training-time version of the model; loads the shipped checkpoints with `strict=True` (779,850 parameters, as in the training records; all 22 runs were checked in the larger package) |
| `code/train.py` | 60fba56 | `validate()` and `ImprovedTrajectoryLoss` are what `demo_infer.py` calls |
| `code/utils/dataset.py` | 9c43c6a | `GraphData` / adjacency construction |
| `code/config.py` | f87bbd0 | the training-time copy differed only in `PAST_TIME_HORIZON` / `NUM_PAST_POSES` (2 / 20 instead of 4 / 40) and `TRAIN_PATH` (set per scenario); the evaluation reads only `BATCH_SIZE`, `SEED` and `VALIDATION_SPLIT` |

Changes made for this repository:
- **Translation.** Chinese comments, docstrings and log messages were translated to English. The translated files
  have the same abstract syntax tree as the originals except for string constants that contained Chinese
  (`provenance/check_translation_ast.py`, result in `provenance/translation_ast_check.txt`). The untranslated
  originals are in `provenance/original_code/`.
- **Added files.** `code/eval_inputs.py` (loader for the shipped samples) and an empty `code/utils/__init__.py`
  (so that `utils.dataset` always resolves to the shipped file).
- Absolute paths of the training machine remain in the original code (`DATA_PATH` / `MAP_PATH` in `code/config.py`,
  model folders added to `sys.path` in `code/train.py` and `code/transformer_predictor.py`) and in recorded metadata (`model_path` / `data_dir` in
  `original_records/table1_sources/*.json`, the `source` field of `data/samples/*.pt`). They are not used by the
  evaluation and were left unchanged to keep code and records identical to the originals.

## 8. Data

| Scenario | Original data | Shipped |
|---|---|---|
| High lateral acceleration | available (15000 samples; validation split 1500) | batch 144 of 188 (validation samples 1152-1159) |
| Near multiple vehicles | available (30000 samples; validation split 3000) | batch 234 of 375 (validation samples 1872-1879) |
| Medium speed, high speed, intersection | replaced in 2026-06 by a later re-processing of the same scenario type | none |

The validation split is the original one: `train_test_split(test_size=0.1, random_state=42)` over the sorted sample
files, batch size 8, no shuffling. The shipped batch of each scenario was drawn uniformly among the complete batches
with `numpy.random.default_rng(20261110)` (`provenance/build_from_full_package.py`, which also computed the reference
values). Each sample is the pre-processed graph used by the model (`x`, `y`, `gnn_x`, `gnn_edge_index`, ...), stored as
a dictionary of tensors and rebuilt into the original `GraphData` class. All samples come from the nuPlan
`train_pittsburgh` split.

The batch matters: the model's BatchNorm layers do not track running statistics, and the GNN feature is averaged over
the whole batch (section 10). The same sample evaluated in a different batch gives a different prediction, which is why
complete original batches are shipped.

## 9. Data source and license

The samples in `data/samples/` and the checkpoints are derived from the nuPlan dataset (Motional,
https://www.nuscenes.org/nuplan), `train_pittsburgh` split, and are subject to the nuPlan dataset terms of use
(CC BY-NC-SA 4.0, non-commercial). They are provided only to reproduce the numbers of the chapter.

## 10. Known issues to be aware of when reading the original results

- **All four topologies were trained on the same default graph.** The original `train.py` builds the dataset
  (line 630) before the topology is selected (line 696), and the selection only sets a variable inside `train.py`;
  `dataset.py` uses its own `CURRENT_ADJACENCY_CONFIG = "default"` (line 35). The topology name only ends up in the
  output folder name, and no torch random seed is set. The four columns of a scenario in Table 2 are therefore four
  independent trainings of the same model; their differences cannot be attributed to the topology. The
  re-evaluation also uses the default graph, as in training.
- **The ego branch is always zero.** The stored ego history is `[20, 7]`; `view(B, 20, 2)` fails and the exception is
  swallowed, so the branch output stays zero (`demo.sh` prints this).
- **The GNN feature is averaged over the whole batch** (no per-sample pooling), and BatchNorm uses batch statistics,
  so outputs depend on the batch composition.
- **ADE / FDE are computed at global coordinates in float32.** In `validate()` the ego history does not match any of
  the expected shapes, so the predicted and true trajectories are rebuilt from `ego_last_pos`, the global UTM
  position (about 588,000 / 4,475,000 m), by adding the 60 increments one by one in float32. At that magnitude float32
  resolves 0.0625 m in x and 0.5 m in y, and the rounding accumulates over the 60 steps. All ADE / FDE values of the
  training records, Table 2 and this repository are computed this way. Measured in the ego frame instead, the same
  predictions give ADE / FDE that are on average 0.02-0.08 / 0.05-0.12 m higher on the complete validation splits;
  for a single sample the difference can be several metres in either direction (per-sample values of both are in
  `reference/full_validation_reference.json`, keys `sample_ade` and `sample_ade_ego_frame`; `demo.sh` prints both).
- **Table 1 mixes protocols**: per-epoch validation minima over 5 scenarios (MAGNET), manifest evaluations over
  4 scenarios (Transformer-only) and over medium speed only (LSTM, IDM). The manifests were sampled from the whole
  scenario folder and therefore include training samples.
- **Split and test set.** Evaluation uses a random 10 % of the samples of each scenario type (`random_state=42`),
  split by sample rather than by log. This set is also used for checkpoint selection, early stopping and
  learning-rate scheduling, and Table 2 reports per-epoch minima on it; there is no separate test set. For high
  lateral acceleration, 1485 of the 1500 validation samples come from logs that also appear in training, and 1293
  have a training sample within 0.5 s of the same log; for near multiple vehicles the figures are 2997 / 3000 and
  2981 / 3000 (computed in the larger package from the nuPlan log names and timestamps, which are not shipped here).
- **High speed Default cell** (3.09 / 6.89) comes from a 2025-09-22 run on an older cache restricted to the first
  1000 samples (900 train / 100 validation, early stop at epoch 42); the other three high-speed cells use 4500 / 500
  samples.
- **Nearby-only threshold.** `dataset.py` uses 30 m (not used in training, see the first item).
