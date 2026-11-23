# Decisions outside the analysis plan

The analysis plan (`docs/analysis_plan_hubert.md`) covers the replication
with HuBERT Base and the padding check on WavLM Large. This file records
the definitions that the plan does not contain and that were fixed by the
first author before the code that uses them ran. Each section says what
was fixed and when; the details are in the notebook and the functions
named. The fixed scientific conventions are in `docs/conventions.md`, and
the places where the code interprets the plan are in `docs/deviations.md`.

## 1. Gate A5: the silence measurements (2026-11-21)

`notebooks/05_silence_measurements.ipynb` measures every file of the four
sets again and compares the result with the original tables. The gate is
not pre-registered: the plan names the silence measurements only as not
repeated for HuBERT Base and sets no tolerance for them. The definition
was fixed before any run of the notebook. The gate passes when every
check of `silence.GATE_CHECKS` passes; the checks are listed at the top of
the notebook.

- Tolerance: 1e-9 (`verify.SECONDS_TOLERANCE`) for seconds; a count must
  be equal. The original code lost the last digit of a value in seconds
  when it read its tables with the default parser.
- The printed numbers of the manuscript are compared after rounding once
  to the digits printed; none is typed into the notebook.
- No tolerance is loosened: a comparison that fails stops the notebook.

## 2. Gate A6: the acoustic probes (2026-11-22)

`notebooks/07_probes.ipynb` repeats the probes of WavLM Base and compares
the sample, the targets and the R² tables with the original ones. The
gate is not pre-registered: the plan names the probes only as not repeated
for HuBERT Base and sets no tolerance for them. The definition was fixed
before any run of the notebook. The gate passes when every check of
`probes.GATE_CHECKS` passes; the checks are listed at the top of the
notebook.

- The mean R², its standard deviation and the R² of the five folds may
  differ from the original tables by `probes.R2_TOLERANCE` (1e-9): the
  fits go through LAPACK and BLAS, whose last digits depend on the CPU and
  on the number of threads, and the machine that wrote the original tables
  is not recorded. The median alpha, the number of files and the number of
  speakers must be equal.
- The acoustic targets are compared with `verify.TARGET_TOLERANCE`.

## 3. Noise ceiling of the rank agreement (2026-11-22)

A step of `notebooks/08_analysis.ipynb` for WavLM Base
(`analysis.ScoreTables.rank_noise_ceiling`). It is not a table of the
plan and is not computed for HuBERT Base. The definition was fixed before
the code was written.

- Scores: the saved score files of WavLM Base, 12 layers, five seeds,
  both poolings, on the evaluation set and on In-the-Wild. Nothing is
  retrained and no encoder is loaded.
- Split: the speakers of a test set are divided at random into two
  halves, separately within each class, as the speaker bootstrap groups
  them (`bootstrap.class_groups`). Each half receives all files of its
  speakers in that class.
- Number and seed: 1,000 splits (`config.N_BOOT`), drawn from
  `config.BOOTSTRAP_SEED` with a stream of its own per test set and
  pooling.
- Curve: the seed-mean official EER per layer on each half.
- Ceiling: Kendall's τ between the curves of the two halves in the cell
  trained and tested on untrimmed audio.
- Cross-condition τ on the same splits: Kendall's τ between that curve on
  one half and the curve trained and tested on audio trimmed at 40 dB on
  the other half.
- Reported per test set and pooling: the median over the splits with the
  2.5th and 97.5th percentiles, for the ceiling and for the
  cross-condition τ; the τ of all files next to them; and the share of
  splits whose ceiling is at or below the τ of all files.

## 4. Supplementary condition `vad` (2026-11-23)

`notebooks/10_vad_condition.ipynb`: a sensitivity analysis of the
trimming rule for WavLM Base, layers 3 to 5. It is not in the plan and is
not pre-registered; no unit of the replication rule and no table of the
plan depends on it. The definition was fixed before the rule, the
configuration or the notebook were written and before any extraction.

- Detector: the WebRTC voice activity detector through the package
  `webrtcvad-wheels`, pinned in `requirements.txt`.
- Input: the mono 16 kHz waveform before peak normalisation, where the
  energy rule runs, as 16-bit PCM.
- Frame length: 10 ms (160 samples). Aggressiveness: 3.
- Onset and offset: the kept audio starts at the first frame of the first
  run of 3 consecutive speech frames (30 ms) and ends with the last frame
  of the last such run.
- What is removed: leading and trailing non-speech only. Pauses inside
  the utterance are kept, as with the energy rule.
- Fallback, as for the trimmed conditions: a file with no such run, or
  with fewer than 8,000 samples (0.5 s) left, keeps its untrimmed audio
  and is flagged.
- Extraction: the unpadded extraction of the trimmed conditions, all 12
  layers. The shards also store, for every file, its number of samples
  and the start and the end of the kept audio.
- Runs: 60. 30 models trained on the `vad` features (layers 3 to 5, mean
  and stats pooling, five seeds), scored on `vad` and on untrimmed audio;
  and 30 models trained on the untrimmed features with the same layers,
  poolings and seeds, scored on untrimmed and on `vad` audio. Model
  weights are not stored, so the models trained on untrimmed audio are
  trained again; their scores on untrimmed audio must equal those of the
  existing runs within 1e-4 (`verify.SCORE_TOLERANCE`), a hard stop. The
  cell trained and tested on untrimmed audio is read from the existing
  runs. The checkpoint is selected on the development set of the training
  condition, as for every other run.
- Folders: runs and tables of their own; the last step stops unless every
  file that existed before under `runs/`, `tables/` and `checks/` is
  unchanged.

Zhang et al. (2023, IEEE/ACM TASLP) report the WebRTC voice activity
detector with frames of 10 ms. They report no aggressiveness mode and do
not say whether pauses inside the utterance were removed. The
aggressiveness, the run of three frames, the removal of the edges only
and the fallback are choices of this study.

## 5. Original folders removed from Drive (2026-11-23)

The original unpadded feature shards of WavLM Base (312 shards) and the
original copy of `LA.zip` with the two official ASV score files were
deleted from Drive by the first author on 2026-11-23, to free space. The
original padded shards stay.

Before the deletion every file on both sides was read again and its
sha256 computed on that day:

- each of the 312 original shards had a copy with the same sha256 and the
  same number of bytes under `features/wavlm-base/unpadded/` of the Drive
  root, and both had the sha256 of the check table
  `shard_manifest_unpadded` (312 rows), written by
  `notebooks/02_migrate_wavlm_features.ipynb`;
- the 3 dataset files were identical to their copies under `datasets/` of
  the Drive root and to the check table `dataset_files`.

Every later notebook reads the copies. Two notebooks read the deleted
folders through the mapping file and cannot be run again as they are:
`notebooks/01_environment_check.ipynb` and
`notebooks/02_migrate_wavlm_features.ipynb`. Their check tables are kept.
