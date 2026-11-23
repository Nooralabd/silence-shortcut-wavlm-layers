# Analysis plan: replication with HuBERT Base and the layer-normalisation padding check

Status: DRAFT, not committed. To be committed unchanged before any HuBERT
Base feature, score or table exists and before any padding measurement on
WavLM Large exists.

## 1. Purpose

The WavLM Base results show that trimming leading and trailing silence
changes layer-wise results on ASVspoof 2019 LA and In-the-Wild, and that zero
padding alters the features of an encoder with group normalisation. This plan
fixes, before any new result is known, (i) how the analysis is repeated with
a second frozen encoder, HuBERT Base, and what counts as a replication, and
(ii) a padding check on an encoder with layer normalisation, WavLM Large,
with predictions stated in advance.

## 2. Configuration

- WavLM Base (reference): `microsoft/wavlm-base`, revision
  `efa81aae7ff777e464159e0f877d54eac5b84f81`.
- HuBERT Base: `facebook/hubert-base-ls960`, revision
  `dba3bb02fda4248b6e082697eee756de8fe8aa8a`, frozen, 12 transformer layers,
  hidden size 768, group normalisation in the convolutional encoder
  (`feat_extract_norm == "group"`, asserted).
- Everything else is identical to WavLM Base (`docs/conventions.md`):
  preprocessing, the trimming rule with the 0.5 s fallback, unpadded
  extraction, mean and population standard deviation, classifier, training,
  seeds 42–46, official EER and tandem t-DCF.
- Conditions: `clean`, `trim40`, `trim20`, `trim15`. The 40 dB threshold is
  the main analysis; 20 and 15 dB are the sensitivity analysis, as for WavLM
  Base.
- Runs: 12 layers × {mean, stats} × 4 training conditions × 5 seeds = 480
  models (240 for clean and `trim40`, 120 for each stricter threshold). Each
  model selects its checkpoint on the development set of its training
  condition.
- Scoring: every model scores the evaluation set and In-the-Wild in every
  condition it shares with clean. Clean-trained models score `clean`,
  `trim40`, `trim20` and `trim15`; models trained on `trim40`, `trim20` or
  `trim15` score their own condition and `clean`. The 2×2 design at each
  threshold therefore comes from scoring only, without extra training.
- The cross-condition cells of HuBERT Base come from the same models scored
  on the other test condition. For WavLM Base the cross-condition cells were
  obtained from separately trained models (same configuration and seed as
  the grid model). Each of those models also scored the test data of its own
  training condition, and these scores were compared with the scores of the
  grid model: check `cross_consistency` of the original analysis code,
  tables `cross_consistency.csv` (layers 3–5) and
  `cross_consistency_all_layers.csv` (the other layers), column
  `max_abs_score_diff`; `max_abs_score_diff` is 0 in all 480 score sets
  (120 + 360). The sha256 of the two tables is recorded here when they are
  migrated: `cross_consistency.csv` (sha256
  `68e1783e41fb2b042dabbd900a46e102ac76a6c3aae501e58a8f0e05dd99dae4`),
  `cross_consistency_all_layers.csv` (sha256
  `d937590e63f08ec7cbb3c1757d06cfcd08e05e66850f058f7e02535df8290de3`). The two designs
  therefore give the same scores. The WavLM Base reproduction in this
  repository uses the single-model design and compares every score set with
  the published runs (`results/checks/`).

## 3. Checks before extraction and training

- The model has 12 layers and group normalisation; kernel and stride lengths
  are read from the model configuration.
- Hidden-state extraction is tested on a small random model.
- For every file and every condition, the number of input samples and the
  fallback flag equal those of the WavLM Base features; the fallback counts
  per split and class are identical.

## 4. Selection

The dev-selected layer of each condition × pooling is the layer with the
lowest development-set EER averaged over seeds. A tie is equality of the
stored seed-mean EER values; on a tie the lower layer is selected. No layer,
checkpoint or configuration is selected on the evaluation set or on
In-the-Wild. There is no reference configuration for HuBERT Base. Layers
3–5 are used for the duration and per-attack analyses: they are the layers
analysed for WavLM Base, fixed for HuBERT Base before any HuBERT Base result
existed. If a HuBERT Base dev-selected layer lies outside 3–5, the duration
and per-attack analyses are additionally run for that layer and reported as
such.

## 5. Analyses

All quantities use the per-seed official EER averaged over the five seeds,
in the point estimate and inside every resample. Intervals are 95%
percentile intervals from 1,000 class-stratified resamples at the file level
and at the speaker level. Differences are paired. The resamples are the ones
used for WavLM Base (same files, same generator streams), so every quantity
is computed on identical resamples for both encoders.

1. The four cells E(C,C), E(C,T), E(T,C), E(T,T) per layer and pooling, on
   the evaluation set and on In-the-Wild, at each threshold.
2. Effects per layer and pooling with file- and speaker-level intervals:
   Δtest, Δtrain, Δtotal, Δtrain^C, Δtest^T and the interaction I, at each
   threshold.
3. Agreement between the clean-to-clean and trimmed-to-trimmed layer curves
   at each threshold: Kendall's τ and Spearman's ρ with speaker-level
   percentile and BCa intervals, and the exact two-sided Kendall p for 12
   layers.
4. The dev-selected layer per condition × pooling, with EER, tandem t-DCF and
   intervals.
5. In-the-Wild split by original duration (≤ 4 s and > 4 s) for layers 3–5
   at 40 dB: the four cells and the effects in each subset.
6. Per-attack change, trimmed minus clean, for the 13 evaluation attacks and
   layers 3–5 with both poolings.
7. Padding deviation: on 500 randomly sampled training files and on the 500
   most padded training files per condition, the relative deviation of padded
   features (with and without an attention mask) from unpadded features, per
   layer. Both samples are the ones of the WavLM Base check: the random
   sample is drawn once from the training list without replacement by
   `numpy.random.default_rng(0)` and is the same 500 files in both
   conditions; the most padded sample is, in each condition, the first 500
   training files when the files with padding are sorted by the amount of
   padding in that condition, largest first, and by file id in ascending
   order within equal amounts (files without padding are excluded). The
   files are identical for all encoders.
8. The best layer per test set, condition and pooling (lowest seed-mean EER),
   with the proportion of speaker-level resamples in which it is best.

### Intervals added

Three WavLM Base findings were reported without the intervals below. They
are added for both encoders. For WavLM Base they are an addition; no
previously reported number changes.

- **In-the-Wild by duration.** File level and speaker level, 1,000
  resamples, percentile intervals, for every effect of each subset. The
  resamples are those of the whole In-the-Wild set (class-stratified at the
  file and the speaker level, the same as for the effects); the EER of each
  subset is computed from the resampled files of that subset, so the number
  of files per class in a subset varies between resamples and the speaker
  structure across subsets is kept. No new generator stream. A resample in
  which a subset has no file of one class stops the analysis.
- **Change of the rank agreement with the threshold.** The paired difference
  τ(40 dB) − τ(threshold), for 20 and 15 dB, per test set and pooling, with
  its speaker-level percentile interval; both values of τ are computed in
  each of the same 1,000 speaker-level resamples.
- **Padding deviation.** The relative deviation (defined in Section 7) is
  reported with a file-level bootstrap interval: the files of the sample are
  resampled with replacement 1,000 times and the ratio is recomputed in each
  resample; percentile interval. The same resamples give a paired interval
  for the difference between the deviation with and without an attention
  mask. The intervals are computed for both samples. The samples themselves
  are not new: the WavLM Base check already used both (item 7 above); only
  the intervals are added.

## 6. Replication rule

A WavLM Base unit is **replicated** with HuBERT Base if the HuBERT Base
estimate has the same sign as the WavLM Base estimate and its speaker-level
95% interval excludes zero. For Kendall's τ the rule is that its
speaker-level interval excludes zero.

The interval used in the rule is the speaker-level **percentile** interval,
as in the paper. BCa intervals are descriptive.

The rule applies only to units in which the WavLM Base speaker-level
percentile interval itself excludes zero. All other units are descriptive.
The replication rule has no other thresholds and no new hypotheses. The only
numerical threshold this plan adds is the factor 10 in prediction (a) of
Section 7a.

No correction for multiple comparisons is applied. The outcome is reported
as the count of replicated units per finding, and all unit-level results are
shown. No single unit is interpreted individually.

### Units

The units are listed mechanically by `replication.replication_units` from
the WavLM Base tables of this repository (`effects_2x2.csv`,
`effects_matched_cells.csv`, `rank_agreement.csv`,
`rank_agreement_threshold_difference.csv`, `per_attack_effect.csv`,
`itw_by_duration_ci.csv`), after the tables that have a published
counterpart (`effects_2x2.csv`, `rank_agreement.csv`,
`per_attack_effect.csv`) have passed the reproduction checks against the
published WavLM Base results. `effects_2x2.csv` holds the six effects for
the layers with all four train/test cells (every layer at 40 dB, layers 3–5
at 20 and 15 dB). `effects_matched_cells.csv` holds Δtotal at 20 and 15 dB
for every layer, computed from the two matched cells with the same resamples
(`effects.total_effect_ci`); for layers 3–5 its rows must be identical to
the Δtotal rows of `effects_2x2.csv`, and it supplies the other layers.
`itw_by_duration_ci.csv` holds the six effects per duration subset with
paired speaker-level intervals (`effects.itw_by_duration_ci`);
`itw_by_duration.csv`, the table behind the published duration table, has
means and standard deviations over seeds and no intervals. The three tables
without a published counterpart (`effects_matched_cells.csv`,
`rank_agreement_threshold_difference.csv`, `itw_by_duration_ci.csv`) are
computed from the same score files and resamples as the reproduced tables.
The list is `docs/replication_units.csv`, committed together with this plan.
It has one row per unit with the estimate, the speaker-level percentile
interval, the sign and the flag `in_rule` (the interval excludes zero).

At 40 dB:

- effects: 6 effects (Δtest, Δtrain, Δtotal, Δtrain^C, Δtest^T, I) × 2 test
  sets × 12 layers × 2 poolings = 288 units;
- rank agreement: Kendall's τ, 2 test sets × 2 poolings = 4 units;
- per attack: trimmed minus clean on the evaluation set, 13 attacks × layers
  3–5 × 2 poolings = 78 units;
- duration: the 6 effects × 2 subsets (≤ 4 s, > 4 s) × layers 3–5 × 2
  poolings = 72 units on In-the-Wild.

At each of 20 and 15 dB:

- Δtotal at all 12 layers (it needs only the matched cells, which exist for
  WavLM Base at every layer): 2 test sets × 12 layers × 2 poolings = 48
  units;
- the other five effects at layers 3–5 (the WavLM Base cross-condition cells
  exist only there): 5 effects × 2 test sets × 3 layers × 2 poolings = 60
  units;
- rank agreement: Kendall's τ, 4 units;
- the paired difference τ(40 dB) − τ(threshold), 4 units;
- per attack: trimmed minus clean on the evaluation set, 13 attacks × layers
  3–5 × 2 poolings = 78 units (the WavLM Base per-attack tables at 20 and
  15 dB have speaker-level percentile intervals for these layers).

The five cross-cell effects of the other layers at 20 and 15 dB are
descriptive. The
identity of the best layer is descriptive at every threshold, with its
bootstrap proportion; there is no rule for it.

No unit is selected by hand: every unit of the file with `in_rule` true is
in the rule, and every other unit is descriptive.

### Units in the rule (summary of `docs/replication_units.csv`)

Written by `replication.summarise_units`
(`docs/replication_units_summary.csv`), one row per quantity × threshold ×
test set, and per duration subset. "k of n": units in the rule out of the
units of the row. "Positive" and "Negative" are the signs of the WavLM Base
estimates of the units in the rule. The file was generated at commit
`637b4b8` (sha256 of `docs/replication_units.csv`: `4adfaf95...`). It has
830 units: 442 at 40 dB, 194 at 20 dB and 194 at 15 dB. 608 units are in
the rule: 289 at 40 dB, 160 at 20 dB and 159 at 15 dB.

| Quantity | Threshold | Test set | In the rule | Positive | Negative |
|---|---|---|---|---|---|
| Δtest | 40 dB | evaluation set | 24 of 24 | 24 | 0 |
| Δtest | 40 dB | In-the-Wild | 3 of 24 | 2 | 1 |
| Δtrain | 40 dB | evaluation set | 24 of 24 | 0 | 24 |
| Δtrain | 40 dB | In-the-Wild | 9 of 24 | 0 | 9 |
| Δtotal | 40 dB | evaluation set | 24 of 24 | 24 | 0 |
| Δtotal | 40 dB | In-the-Wild | 11 of 24 | 0 | 11 |
| Δtrain^C | 40 dB | evaluation set | 24 of 24 | 24 | 0 |
| Δtrain^C | 40 dB | In-the-Wild | 6 of 24 | 0 | 6 |
| Δtest^T | 40 dB | evaluation set | 12 of 24 | 12 | 0 |
| Δtest^T | 40 dB | In-the-Wild | 17 of 24 | 3 | 14 |
| I | 40 dB | evaluation set | 24 of 24 | 0 | 24 |
| I | 40 dB | In-the-Wild | 12 of 24 | 1 | 11 |
| Kendall's τ | 40 dB | evaluation set | 2 of 2 | 2 | 0 |
| Kendall's τ | 40 dB | In-the-Wild | 2 of 2 | 2 | 0 |
| Per-attack trimmed minus clean | 40 dB | evaluation set | 71 of 78 | 65 | 6 |
| Δtest, ≤ 4 s | 40 dB | In-the-Wild | 0 of 6 | 0 | 0 |
| Δtest, > 4 s | 40 dB | In-the-Wild | 0 of 6 | 0 | 0 |
| Δtest^T, ≤ 4 s | 40 dB | In-the-Wild | 6 of 6 | 0 | 6 |
| Δtest^T, > 4 s | 40 dB | In-the-Wild | 2 of 6 | 0 | 2 |
| Δtotal, ≤ 4 s | 40 dB | In-the-Wild | 4 of 6 | 0 | 4 |
| Δtotal, > 4 s | 40 dB | In-the-Wild | 2 of 6 | 0 | 2 |
| Δtrain, ≤ 4 s | 40 dB | In-the-Wild | 2 of 6 | 0 | 2 |
| Δtrain, > 4 s | 40 dB | In-the-Wild | 1 of 6 | 0 | 1 |
| Δtrain^C, ≤ 4 s | 40 dB | In-the-Wild | 0 of 6 | 0 | 0 |
| Δtrain^C, > 4 s | 40 dB | In-the-Wild | 2 of 6 | 0 | 2 |
| I, ≤ 4 s | 40 dB | In-the-Wild | 4 of 6 | 0 | 4 |
| I, > 4 s | 40 dB | In-the-Wild | 1 of 6 | 0 | 1 |
| Δtotal, layers 1–12 | 20 dB | evaluation set | 24 of 24 | 24 | 0 |
| Δtotal, layers 1–12 | 20 dB | In-the-Wild | 12 of 24 | 0 | 12 |
| Δtest, layers 3–5 | 20 dB | evaluation set | 6 of 6 | 6 | 0 |
| Δtest, layers 3–5 | 20 dB | In-the-Wild | 1 of 6 | 1 | 0 |
| Δtrain, layers 3–5 | 20 dB | evaluation set | 6 of 6 | 0 | 6 |
| Δtrain, layers 3–5 | 20 dB | In-the-Wild | 3 of 6 | 0 | 3 |
| Δtrain^C, layers 3–5 | 20 dB | evaluation set | 6 of 6 | 6 | 0 |
| Δtrain^C, layers 3–5 | 20 dB | In-the-Wild | 0 of 6 | 0 | 0 |
| Δtest^T, layers 3–5 | 20 dB | evaluation set | 2 of 6 | 0 | 2 |
| Δtest^T, layers 3–5 | 20 dB | In-the-Wild | 4 of 6 | 0 | 4 |
| I, layers 3–5 | 20 dB | evaluation set | 6 of 6 | 0 | 6 |
| I, layers 3–5 | 20 dB | In-the-Wild | 6 of 6 | 0 | 6 |
| Kendall's τ | 20 dB | evaluation set | 2 of 2 | 2 | 0 |
| Kendall's τ | 20 dB | In-the-Wild | 2 of 2 | 2 | 0 |
| Per-attack trimmed minus clean | 20 dB | evaluation set | 78 of 78 | 66 | 12 |
| τ(40 dB) − τ(threshold) | 20 dB | evaluation set | 2 of 2 | 2 | 0 |
| τ(40 dB) − τ(threshold) | 20 dB | In-the-Wild | 0 of 2 | 0 | 0 |
| Δtotal, layers 1–12 | 15 dB | evaluation set | 24 of 24 | 24 | 0 |
| Δtotal, layers 1–12 | 15 dB | In-the-Wild | 10 of 24 | 0 | 10 |
| Δtest, layers 3–5 | 15 dB | evaluation set | 6 of 6 | 6 | 0 |
| Δtest, layers 3–5 | 15 dB | In-the-Wild | 2 of 6 | 2 | 0 |
| Δtrain, layers 3–5 | 15 dB | evaluation set | 6 of 6 | 0 | 6 |
| Δtrain, layers 3–5 | 15 dB | In-the-Wild | 5 of 6 | 0 | 5 |
| Δtrain^C, layers 3–5 | 15 dB | evaluation set | 6 of 6 | 6 | 0 |
| Δtrain^C, layers 3–5 | 15 dB | In-the-Wild | 0 of 6 | 0 | 0 |
| Δtest^T, layers 3–5 | 15 dB | evaluation set | 0 of 6 | 0 | 0 |
| Δtest^T, layers 3–5 | 15 dB | In-the-Wild | 4 of 6 | 0 | 4 |
| I, layers 3–5 | 15 dB | evaluation set | 6 of 6 | 0 | 6 |
| I, layers 3–5 | 15 dB | In-the-Wild | 6 of 6 | 0 | 6 |
| Kendall's τ | 15 dB | evaluation set | 2 of 2 | 2 | 0 |
| Kendall's τ | 15 dB | In-the-Wild | 2 of 2 | 2 | 0 |
| Per-attack trimmed minus clean | 15 dB | evaluation set | 78 of 78 | 66 | 12 |
| τ(40 dB) − τ(threshold) | 15 dB | evaluation set | 2 of 2 | 2 | 0 |
| τ(40 dB) − τ(threshold) | 15 dB | In-the-Wild | 0 of 2 | 0 | 0 |
| All units | 40 dB; 20 dB; 15 dB | both | 608 of 830 | 378 | 230 |

This plan is committed only after the file exists and this table is filled
in from it, and before any HuBERT Base result exists.

### Reporting

For each row of the summary, the outcome is a 2×2 table
(`replication.outcome_table`): WavLM Base interval excludes zero or includes
zero, against HuBERT Base interval excludes zero or includes zero, with sign
agreement where both exclude zero. It gives

- replicated: both exclude zero, same sign;
- both exclude zero, opposite sign;
- WavLM Base only;
- HuBERT Base only (the reverse count: HuBERT Base excludes zero where WavLM
  Base did not);
- neither.

The unit-level table behind the counts (`replication.compare_units`) is
published in full.

## 6a. Findings

Every finding stated in the abstract and the results of the manuscript on
classifier performance is listed with its units in
`docs/replication_units.csv`. The criterion is the count of replicated units
among the units in the rule; no thresholds are added. A finding stated for
WavLM Base as the absence of an effect is an **absence finding**: it is
reported descriptively with the 2×2 table and is not counted as replicated
or failed. Findings about the identity of a layer or the size of a range are
descriptive.

| # | Finding (manuscript) | Units (`quantity`, threshold, test set, layers) | Type |
|---|---|---|---|
| F1 | "Trimming at 40 dB raises the in-domain error at every layer" | Δtotal, 40 dB, evaluation set, layers 1–12 | effect |
| F2 | "At 40 dB, a model trained with silence loses 6.3–19.9 percentage points of equal error rate at every layer when only its test audio is trimmed." | Δtest, 40 dB, evaluation set, layers 1–12 | effect |
| F3 | "A model trained on trimmed audio changes much less when its test audio is trimmed (…; the interval excludes zero in 12 of the 24 configurations …)" | Δtest^T, 40 dB, evaluation set, layers 1–12 | effect |
| F4 | "Training on trimmed audio while testing on clean audio raises the EER by 2.03–8.91 points, with an interval excluding zero in all 24 configurations." | Δtrain^C, 40 dB, evaluation set, layers 1–12 | effect |
| F5 | "The interaction lies between −5.65 and −15.74 points and is negative with an interval excluding zero in all 24 configurations." | I, 40 dB, evaluation set, layers 1–12 | effect |
| F6 | (tables only) Δtrain on the evaluation set (effects table; negative at every layer) | Δtrain, 40 dB, evaluation set, layers 1–12 | effect |
| F7 | "On In-the-Wild, trimming the test audio of a model trained on clean audio changes the EER by at most 1.11 points in any of the 24 configurations; the speaker-level interval excludes zero in three of them, with mixed sign." | Δtest, 40 dB, In-the-Wild, layers 1–12 | absence finding |
| F8 | "Trimming both training and test audio lowers the EER with an interval excluding zero in 11 configurations" | Δtotal, 40 dB, In-the-Wild, layers 1–12 | effect |
| F9 | "The interaction is negative with an interval excluding zero in a different set of 11 configurations …, among them all six of layers 3–5"; "The interaction is positive for statistics pooling at layer 12" | I, 40 dB, In-the-Wild, layers 1–12 | effect |
| F10 | "in these six, trimming the test audio lowers the EER of the model trained on trimmed audio … but not that of the model trained on clean audio"; "At layer 12, trimming the test audio of the model trained on trimmed audio raises the EER …, as it does for statistics pooling at layer 8" | Δtest^T, 40 dB, In-the-Wild, layers 1–12 (the second half of the first sentence is F7) | effect |
| F11 | "At layers 6, 8, and 9, training on trimmed audio lowers the EER also when the test audio is clean …, with intervals excluding zero; at layer 7 the intervals include zero" | Δtrain^C, 40 dB, In-the-Wild, layers 1–12 | effect |
| F12 | (tables only) Δtrain on In-the-Wild (effects table) | Δtrain, 40 dB, In-the-Wild, layers 1–12 | effect |
| F13 | "The ranking is preserved only partly on the evaluation set (τ = 0.58) and more strongly on In-the-Wild (τ = 0.73–0.76)." | Kendall's τ, 40 dB, both test sets | effect (τ interval excludes zero); the comparison between the test sets is descriptive |
| F14 | "The best layer does not change: layer 5 on the evaluation set and layer 4 on In-the-Wild" | best layer with its bootstrap proportion | descriptive |
| F15 | "For files that fit in the window, it lowers the EER by 0.90 ± 0.08 points."; "For short files, the improvement requires trimming in both training and test." | duration units, ≤ 4 s subset, layers 3–5 | effect, for the units in the rule |
| F16 | "For files longer than the window, … trimming only the test audio has no effect" | Δtest, > 4 s subset, layers 3–5 | absence finding |
| F17 | "For long files, training on trimmed audio is enough" | Δtrain^C and the other duration units, > 4 s subset, layers 3–5 | effect, for the units in the rule |
| F18 | Per-attack changes: "A10 is the most affected …"; "A07, A08, A09, A12, A14, and A16 increase …, significantly in all six configurations"; "A17 is the only attack whose EER decreases after trimming …, significantly at both levels in all six configurations"; the small changes of A13, A18 and A19 | per-attack units, 40 dB, 13 attacks × layers 3–5 | effect, for the units in the rule; the ordering of the attacks is descriptive |
| F19 | "For these layers, the test-time effect on the evaluation set grows from 6.3–10.2 points at 40 dB to 10.6–18.0 points at 20 dB and 11.2–19.1 points at 15 dB" | Δtest, 20 and 15 dB, evaluation set, layers 3–5 | effect; the growth with the threshold is descriptive |
| F20 | "With statistics pooling, the EER after trimming is higher than on clean audio at every layer and every threshold" | Δtotal, 20 and 15 dB, evaluation set, layers 1–12 | effect |
| F21 | (tables only) Interaction on the evaluation set at 20 and 15 dB (threshold table) | I, 20 and 15 dB, evaluation set, layers 3–5 | effect |
| F22 | "On In-the-Wild, the interaction remains negative, with speaker-level intervals that exclude zero in all six configurations of layers 3–5 at both stricter thresholds." | I, 20 and 15 dB, In-the-Wild, layers 3–5 | effect |
| F23 | "Trimming the test audio alone, however, now raises the EER of the model trained on clean audio … The speaker-level intervals of this effect include zero except for mean pooling at layer 5 (20 and 15 dB) and at layer 3 (15 dB)." | Δtest, 20 and 15 dB, In-the-Wild, layers 3–5 | effect, for the units in the rule |
| F24 | "The layer rankings agree less at the stricter thresholds" | Kendall's τ at 20 and 15 dB; τ(40 dB) − τ(threshold) | effect |
| F25 | "the best layer on the evaluation set changes from layer 5 to layer 6 at 15 dB with both pooling methods … and at 20 dB with mean pooling"; "On In-the-Wild, the best layer remains layer 4" | best layer with its bootstrap proportion | descriptive |
| F26 | "A10 remains the attack most affected by trimming …"; "A17 is detected better after trimming at all three thresholds."; "A19 … is detected better after trimming at 20 and 15 dB in all six configurations, with speaker-level intervals that exclude zero." | per-attack units, 20 and 15 dB, 13 attacks × layers 3–5 | effect, for the units in the rule; the ordering of the attacks is descriptive |
| F27 | "Zero padding alters WavLM features through group normalization, which an attention mask does not undo." | padding deviation | descriptive for HuBERT Base (Section 7); tested by Section 7a |

The remaining effect units at 20 and 15 dB (Δtrain, Δtrain^C, Δtest^T at
layers 3–5, and Δtotal on In-the-Wild at layers 1–12) are in the file and in
the rule where the WavLM Base interval excludes zero.

Not repeated for HuBERT Base: the silence measurements and the silence-only
classifiers (they do not depend on the encoder) and the acoustic probes.

## 7. Padding deviation (descriptive)

The padding deviation is not part of the replication rule: it is a
non-negative quantity, so its interval excludes zero whenever padding changes
the features at all. It is reported descriptively for both encoders. The
HuBERT Base padding result is a check within the group-normalisation design;
it does not bear on the dependence on the normalisation, which is addressed
by Section 7a.

The definition is the one behind the published WavLM Base values at layer 4
on the random sample (0.31 on clean and 0.44 on trimmed audio without an
attention mask; 0.15 and 0.34 with one): the column `rel_mean_diff` for the
mean-pooled feature in the output of `attention_mask_check` of the original
analysis code, implemented here as `padding.padding_deviation`. Those values
are read from the per-layer table of that check (sha256
`0a266cbbd2271988becb6b6b86db5fcdb05bd3929207988692527f410610ba72`).

The WavLM Base reproduction must reproduce that table with
`padding.padding_deviation` on the same files, for both samples, both
conditions, both mask settings and all 12 layers, with an absolute
difference of at most 1e-4 in `rel_mean_diff`. If it does not, the
definition is corrected before this plan is committed.

For one layer and one sample of files:

- each file is preprocessed as in the padded extraction (zero-padded at the
  end to 64,000 samples) and encoded three ways: padded without an attention
  mask, padded with an attention mask, and unpadded (the reference: the real
  samples only);
- in all three the feature is the mean over the frames of the real samples,
  a vector of one value per hidden dimension and layer;
- per file, the absolute difference between the padded and the reference
  feature is averaged over the hidden dimensions; this is averaged over the
  files;
- the result is divided by the mean absolute value of the reference feature,
  averaged over the hidden dimensions and then over the files;
- files with fewer than 400 real samples are skipped.

Reported per layer, condition and sample (500 random and 500 most padded
training files), with file-level intervals:

- the relative deviation without an attention mask;
- the relative deviation with an attention mask;
- the paired difference between the two (with mask minus without mask).

"The mask reduces the deviation" is defined as: this paired difference is
negative and its interval excludes zero, on the random sample. On the random
sample this is the WavLM Base finding (the attention mask reduces the
deviation but does not remove it). On the most padded sample the difference
is reported as observed; for WavLM Base on trimmed audio at layer 4 the mask
increases the deviation (0.95 against 0.80).

In each bootstrap resample the numerator and the denominator are both
recomputed from the resampled files.

## 7a. Padding check with layer normalisation (WavLM Large)

This check, not the HuBERT Base padding result, bears on the claim that the
effect of padding depends on the normalisation of the convolutional encoder.
It involves feature extraction only; no classifier is trained.

- Encoder: `microsoft/wavlm-large`, revision
  `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c` (pinned in the encoder
  configuration), 24 transformer layers, hidden size 1024;
  `feat_extract_norm == "layer"` is asserted.
- Same preprocessing as for the other encoders, same two samples (500
  random files drawn with `numpy.random.default_rng(0)`, 500 most padded
  files per condition), same three extraction modes, same deviation
  definition and file-level intervals, both conditions (`clean`, `trim40`),
  all 24 layers.
- The feature-extractor normalisation of the checkpoint (zero mean and unit
  variance of the input) is not applied. It is computed over the whole
  input including the padding, so it would itself change with padding and
  mix its effect with the encoder's.

**Noise floor.** The random sample is extracted unpadded twice, once on the
GPU and once on the CPU (float32). The floor is the deviation between the
two, with the same definition (the GPU features as the reference) and the
same file-level interval, per layer and condition. It is computed for WavLM
Base, WavLM Large and HuBERT Base.

**Predictions for WavLM Large**, stated as a test and evaluated on the random
sample in each condition:

- (a) With an attention mask the deviation is at the noise floor: at every
  layer, the upper bound of its interval is at most 10 times the upper bound
  of the floor interval of that layer and condition, computed in float64 as
  upper <= 10.0 * floor_upper.
- (b) Without an attention mask the deviation stays above the floor: at
  every layer, the lower bound of its interval is above the upper bound of
  the floor interval of that layer and condition.

If the upper bound of a floor interval is 0, prediction (a) at that layer
requires the deviation with an attention mask to be 0.

Each prediction holds if it is met at all 24 layers in both conditions; the
result is reported per layer and condition either way. The most padded
sample is reported descriptively.

The factor 10 in (a) is the only numerical threshold this plan adds. It is
fixed before any padding measurement on WavLM Large exists.

## 8. Paired encoder comparison

With the same files and the same resamples, the difference WavLM Base minus
HuBERT Base is reported with file- and speaker-level 95% intervals for every
cell, every effect and Kendall's τ (per layer, pooling and test set where
applicable). These differences are descriptive; no replication decision
depends on them. An interval of the difference that includes zero is not
evidence of equal effects.

## 9. Reported regardless of the outcome

- All cells and effects for all 12 layers, both poolings, both test sets and
  all thresholds.
- τ and ρ with percentile and BCa intervals and the exact p, and the
  differences τ(40 dB) − τ(threshold).
- The dev-selected layers and their results.
- The best layer per test set, condition and pooling (lowest seed-mean EER)
  with the proportion of resamples in which it is best, for both encoders;
  descriptive.
- The duration split, the per-attack changes and the padding deviation.
- For every effect of the duration split, the difference between the two
  subsets (files longer than 4 s minus files of at most 4 s) with paired
  file- and speaker-level percentile intervals from the same resamples;
  descriptive, not a unit of the replication rule.
- The noise floor for the three encoders and both predictions of Section 7a,
  per layer and condition.
- The paired encoder differences.
- The 2×2 outcome table of every finding and the unit-level table, including
  every unit that is not replicated.

## 10. Deviations

Any deviation from this plan is listed in the paper with its reason.

Where a newly added interval includes zero for a WavLM Base effect the paper
describes as an effect, the wording is revised and listed as a deviation.

The paper cites the commit hash of this plan; any analysis not in it is
labelled exploratory.
