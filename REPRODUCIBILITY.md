# Reproducibility

Every number of the paper is computed from files by the functions below;
none is typed by hand. The fixed conventions are in
[docs/conventions.md](docs/conventions.md).

Paths are relative to `tables/<encoder>/` under the Drive root unless they
start with `measurements/`; the copies the paper cites are in `results/`
with `results/MANIFEST.csv` (path, rows, sha256).

## Tables and figures

| Paper item | CSV | Function | Notebook |
|---|---|---|---|
| Silence-only classifier (`tab:silence`) | `measurements/silence_only_baseline.csv` | `silence.silence_only_baselines` | `notebooks/05_silence_measurements.ipynb` |
| Leading-silence ECDF (`fig:ecdf`) | `measurements/silence_summary.csv` | `silence.summarise` | `notebooks/05_silence_measurements.ipynb` |
| Padding deviation (`fig:padding`) | `padding_deviation.csv` | `padding.padding_deviation` | `notebooks/06_padding_deviation.ipynb` |
| Padded against unpadded extraction (text) | `padded_vs_unpadded.csv` | `selection.padded_vs_unpadded` | `notebooks/08_analysis.ipynb` |
| EER per layer (`fig:layers`) | `metrics_per_run.csv` | `selection.seed_summary` | `notebooks/08_analysis.ipynb` |
| Selected configurations (`tab:main`) | `seed_summary.csv` (mean and sample SD over seeds of `metrics_per_run.csv`), `selected_configs_eer_ci.csv` | `selection.seed_summary`, `selection.select_on_dev`, `effects.selected_configs_eer_ci` | `notebooks/08_analysis.ipynb` |
| Four train/test cells (`fig:cross`) | `cells_2x2.csv` | `effects.cells_table` | `notebooks/08_analysis.ipynb` |
| Effects by layer (`fig:effects-layers`, `tab:effects`, `tab:effects-mean`, `tab:effects-other`, `fig:forest`) | `effects_2x2.csv` | `effects.effects_ci` | `notebooks/08_analysis.ipynb` |
| Rank agreement (`tab:rank`, `tab:bca`) | `rank_agreement.csv`, `rank_agreement_layers.csv` | `ranking.rank_agreement_ci` | `notebooks/08_analysis.ipynb` |
| EER per layer in the matched cells (`tab:rankdetail`) | `cells_2x2.csv` | `effects.cells_table` | `notebooks/08_analysis.ipynb` |
| In-the-Wild by duration (`tab:duration`) | `itw_by_duration.csv` | `effects.itw_by_duration` | `notebooks/08_analysis.ipynb` |
| Per-attack change (`fig:attack`, `tab:attack`) | `per_attack_effect.csv` | `effects.per_attack_effect_ci` | `notebooks/08_analysis.ipynb` |
| Trimming thresholds (`tab:threshold`) | `threshold_summary.csv`, `fallback_counts.csv`, `measurements/trim_amount.csv` | `ranking.threshold_summary`, `shards.fallback_counts`, `silence.summarise_removed` | `notebooks/05_silence_measurements.ipynb`, `notebooks/08_analysis.ipynb` |
| Residual silence (text) | `measurements/residual_silence_summary.csv`, `measurements/residual_silence_baseline.csv` | `silence.summarise`, `silence.silence_only_baselines` | `notebooks/05_silence_measurements.ipynb` |
| Probes (`fig:probes`, `tab:probes`, text) | `probe_r2.csv` (every layer, target and scheme, real and shuffled; one table per condition, clean and trim40), `probe_r2_table.csv` (all layers, mean ± SD as text), `probe_r2_selected.csv` (layers 1, 4, 8 and 12, speaker folds) | `probes.run_probes` | `notebooks/07_probes.ipynb` |

### Figures

Every figure is drawn by `ssl_silence.plots` from the tables above, without
computing a metric, in the style of the published figure. `plots.save_figure`
writes a 300 dpi PNG and a PDF and checks the palette; the figures drawn at
the text width are also checked to be 13 cm wide with no font below 7 pt.

| Figure | File | Function | Input table |
|---|---|---|---|
| `fig:ecdf` | `silence_ecdf` | `plots.silence_ecdf_figure` | the per-file measurement tables (`silence.silence_table`) |
| `fig:padding` | `padding_deviation` | `plots.padding_deviation_figure` | `padding_deviation.csv` |
| `fig:layers` | `layer_eer` | `plots.layer_eer_figure` | seed summary of `metrics_per_run.csv`, dev-selected layers |
| `fig:cross` | `cells_2x2` | `plots.cells_figure` | `cells_2x2.csv` |
| `fig:effects-layers` | `effects_by_layer` | `plots.effects_by_layer_figure` | `effects_2x2.csv` |
| `fig:attack` | `per_attack_effect` | `plots.per_attack_figure` | `per_attack_effect.csv` |
| `fig:probes` | `probe_r2` | `plots.probe_figure` | `probe_r2.csv` |
| `fig:forest` | `effects_forest` | `plots.effects_forest_figure` | `effects_2x2.csv` |

The silence distribution and the probe figure keep the size of the published
figures (wider than the text width; they are scaled when included). The
pipeline figure is drawn in the manuscript source.

The LaTeX bodies of the tables are written by `ssl_silence.paper.tables`.
`pytest --manuscript <file.tex> --tables results` rebuilds every table and
compares each number with the manuscript.

## In-text numbers

`ssl_silence.paper.intext` registers the numbers quoted in the running text:
each entry is a phrase of the manuscript with its numbers left open and a
function that computes them from the saved tables. The same
`pytest --manuscript <file.tex> --tables <folder>` run finds every phrase in
the manuscript (it must occur exactly once), reads its numbers and compares
them, rounded once to the digits printed. `intext.coverage` lists every
number of the abstract, the preprocessing subsection and the text from the
results to the references that is outside tables and figures, as checked,
cited (prior work), approximate ("about ..."), structural (layer, threshold,
year, reference) or unchecked; the test run fails on an unchecked number.

The table bodies are identified by their LaTeX label (`tab:effects` is
written to `effects.tex`).

## Library versions

`requirements.txt` pins the versions of the Colab runtime (Tesla T4) that
produced the features and results: torch 2.11.0, transformers 5.17.0,
numpy 2.1.3, scikit-learn 1.6.1, pandas 2.2.3, librosa 0.11.0. Every feature
shard records the installed versions, the GPU, the encoder checkpoint and its
Hugging Face revision.

The original WavLM Base feature shards record the commit of the original
analysis code that extracted them (read on Colab by
`notebooks/02_migrate_wavlm_features.ipynb`, check table
`original_shard_provenance_all`):

| Shards | Commit of the original code |
|---|---|
| unpadded, untrimmed | `efe002ad36fa` |
| unpadded, trimmed at 40 dB | `a113d691ce96` |
| unpadded, trimmed at 20 dB | `bb88cde9c2d3` |
| unpadded, trimmed at 15 dB | `bbab4a88cac5` |
| padded, untrimmed and trimmed at 40 dB | `208749d60fd4` |

All of them record torch 2.11.0+cu130, transformers 5.17.0, torchaudio as
the audio library and a Tesla T4. The five commits are on the main branch of
the original repository, before `646f5ae`, the commit that the parity tests
run against.

The original unpadded shards and the original copy of `LA.zip` with the two
ASV score files are removed from Drive by the first author (decision of
2026-11-23, to free space). Before that, every one of the 312 shards and of the 3 dataset
files was read again and had the sha256 of its copy under the Drive root
and of the check tables `shard_manifest_unpadded` and `dataset_files`
(`docs/decisions.md`, Section 5). The copies are
what every later notebook reads. `notebooks/01_environment_check.ipynb` and
`notebooks/02_migrate_wavlm_features.ipynb` read the original folders, so
they cannot be run again as they are; their check tables are kept.

### librosa

The trimming rule runs on `librosa.effects.trim`; the pin is librosa 0.11.0.
Evidence for the pin: in the original analysis, the trimmed length of every
file, recomputed on Colab with librosa 0.11.0 (2026-11-08), was compared
with the input length stored in the feature shards
(`min(trimmed length, 64,000) == n_real`). The table of that check,
`step0b_trim_length_check.csv`, has 12 rows (40, 20 and 15 dB × train, dev,
evaluation, In-the-Wild). In all 12 rows the number of equal files is the
number of files (train 25,380; dev 24,844; evaluation 71,237; In-the-Wild
31,779), the largest difference is 0 samples and `ok` is true. These values
were read from the table on Drive by the first author. Its sha256 is
recorded when `notebooks/02_migrate_wavlm_features.ipynb` copies it. The
re-measurement of every file with the code of this repository
(`notebooks/05_silence_measurements.ipynb`) repeats this comparison and is
the decisive check (`results/checks/`).

### Local test environment

The unit tests run on a CPU in a separate environment built from
`requirements.txt` (Python 3.13; CPU builds of torch 2.11.0 and torchaudio
2.11.0; the commands are in the README). `tests/test_environment.py` checks
that the pins of `requirements.txt` are the expected ones and that every
pinned package is installed in exactly that version. A version is pinned
only after it has been read from the Colab runtime. On Colab, torch,
torchaudio and torchcodec are the builds for CUDA 13.0 (`+cu130`), installed
from `https://download.pytorch.org/whl/cu130` by the install cell of the
notebooks; `torchcodec` is optional on a development machine.

### Audio libraries

Files are read with `torchaudio.load` and, when it cannot read a file, with
`soundfile.read`; the library that read each file is recorded in the feature
shards. Resampling to 16 kHz uses `torchaudio.transforms.Resample` with its
default settings. In torchaudio 2.11.0, `torchaudio.load` needs the
`torchcodec` package; without it every file is read with soundfile. For
16-bit PCM both give the same samples (integer value / 32768).

### Encoder checkpoints

| Encoder | Checkpoint | Revision |
|---|---|---|
| `wavlm-base` | `microsoft/wavlm-base` | `efa81aae7ff777e464159e0f877d54eac5b84f81` |
| `hubert-base` | `facebook/hubert-base-ls960` | `dba3bb02fda4248b6e082697eee756de8fe8aa8a` |

The WavLM Base features of the paper were extracted without a pinned
revision. That the pinned revision gives the same features is established by
re-extracting 200 files per split and condition and comparing them with the
stored features (`results/checks/`).

### Praat

The probe targets (mean F0, mean HNR, spectral tilt) are computed with
Praat 6.1.38 through `praat-parselmouth==0.4.7`. The source of this pin is
the saved Colab output of the original probe run (2026-10-07), which prints
`parselmouth 0.4.7 | Praat 6.1.38`. The probe run on the unpadded features
(2026-10-08), which produced the paper's probe table, installed
`praat-parselmouth` without a pin and did not print the version; it reused
the targets cached by the original run and did not recompute them. Release
0.4.7 (2025-11-27) was the latest on PyPI on both dates, so an unpinned
install resolved to it. Every probe output records the Parselmouth and Praat
versions. Recomputing the targets with 0.4.7 must reproduce the stored
targets exactly (`results/checks/`).
