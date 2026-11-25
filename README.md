# silence-shortcut-wavlm-layers

Code for the study of the silence shortcut in frozen self-supervised speech
representations for audio deepfake detection.

Nur Elabid and Mahmud Esad Arar, Fatih Sultan Mehmet Vakif University.

Paper: N. Elabid and M. E. Arar, "Do Layer Rankings Survive Silence Trimming?
The Silence Shortcut in Frozen WavLM Representations for Audio Deepfake
Detection", arXiv:2611.32457, 2026, <https://arxiv.org/abs/2611.32457>.
The code and the result tables of this arXiv version are the first release
of this repository, archived on Zenodo: <https://doi.org/10.5281/zenodo.24506386>.

A frozen encoder (WavLM Base, HuBERT Base) produces the hidden states of every
transformer layer once. They are pooled over time (mean, or mean and standard
deviation) and cached, and a small classifier is trained per layer. Each model
is trained on ASVspoof 2019 LA audio that is either unchanged (`clean`) or
stripped of leading and trailing silence (`trim40`, `trim20`, `trim15`), and
tested on the ASVspoof 2019 LA evaluation set and, zero-shot, on In-the-Wild in
all four train/test combinations.

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): every table, figure and in-text
  number of the paper with its CSV, function and notebook; library versions.
- [docs/conventions.md](docs/conventions.md): the fixed scientific conventions.
- [docs/analysis_plan_hubert.md](docs/analysis_plan_hubert.md): the
  pre-registered analysis plan for the replication with HuBERT Base and the
  padding check on WavLM Large; [docs/preregistration.md](docs/preregistration.md)
  records when and in which state it was committed.
- [docs/deviations.md](docs/deviations.md): the places where the code interprets the
  analysis plan, with the sentence of the plan and what the code does.
- [docs/decisions.md](docs/decisions.md): the definitions outside the plan that
  were fixed before the code that uses them ran.
- [CITATION.cff](CITATION.cff), [LICENSE](LICENSE) (MIT).

## Layout

```
src/ssl_silence/
  config.py       paths, constants, encoder and experiment configuration
  configs/        encoders.yaml, experiments.yaml
  encoders.py     encoder loading through transformers.AutoModel
  audio.py        loading, mono, 16 kHz, centre crop, peak normalisation
  trimming.py     the silence trimming rule
  silence.py      silence measurements and silence-only baselines
  extraction.py   unpadded and padded feature extraction, pooling
  shards.py       feature shards on disk
  classifier.py   the classifier
  training.py     the single training function, run records, score files
  metrics.py      official ASVspoof EER and tandem t-DCF
  selection.py    per-run metric table, seed summaries, dev-only selection
  bootstrap.py    stratified file- and speaker-level bootstrap
  effects.py      the 2x2 train/test design and its effects
  ranking.py      agreement between layer rankings
  padding.py      deviation of padded from unpadded features
  probes.py       acoustic probes
  plots.py        figures
  verify.py       comparison of tensors, score files and tables
  paper/          paper tables, manuscript parser, in-text numbers
notebooks/        Colab runners: mount, clone, install, call functions
tests/            pytest, CPU, synthetic data
results/          the CSV tables behind the paper, with MANIFEST.csv (path, rows, sha256)
```

## What can be run from this repository

- **The tests.** They run on a CPU with synthetic data and need no dataset
  (see "Installation"). The tests that compare this package with the
  original analysis code are skipped: that code is not public.
- **The numbers of the paper.** `results/` holds the tables that the paper
  tables and the in-text numbers are computed from.
  `pytest --manuscript <file.tex> --tables results` rebuilds every table of
  the paper and every registered in-text number and compares them with the
  manuscript source (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).
- **The functions.** Extraction, training, scoring, the bootstrap and every
  table are functions of `src/ssl_silence`. They read the root folder from
  `SSL_SILENCE_ROOT`, and the notebooks show the order of the calls. Outside
  Colab they were run on synthetic data only (the tests).

What cannot be run as it is:

- **The notebooks.** They are the record of how the results were produced
  on Colab. Each of them reads a mapping file on the Drive root of the
  authors that names the folders of the original outputs, and compares its
  results with those outputs; the mapping file and the original outputs are
  not released. `notebooks/01_environment_check.ipynb` also clones the
  original analysis code, and it and
  `notebooks/02_migrate_wavlm_features.ipynb` read original folders that no
  longer exist ([docs/decisions.md](docs/decisions.md), Section 5).
- **Anything that needs the stored features, score files or check tables.**
  They are on the Drive root of the authors and are not released; the check
  tables that `REPRODUCIBILITY.md` names under `results/checks/` are among
  them. Features and scores are recomputed by the functions from the
  datasets.

The repository holds the tables behind the paper only. Tables of the
analyses of the plan are added with the version of the paper that reports
them.

## Installation (local, CPU, tests)

The tests run in an environment of their own with the pinned versions of
`requirements.txt` (Python 3.13, CPU builds of torch and torchaudio).
`tests/test_environment.py` fails if an installed version differs from its
pin. `torchcodec` is optional on a development machine (it needs FFmpeg;
without it every file is read with soundfile) and is checked where it is
installed.

```bash
python -m venv .venv
```

```bash
.venv/bin/python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

```bash
.venv/bin/python -m pip install --no-deps -e .
```

```bash
.venv/bin/python -m pytest
```

On Windows the interpreter is `.venv/Scripts/python.exe`. With `uv`, the
same four steps are `uv venv --python 3.13 .venv` and
`uv pip install --python .venv ...` with the arguments above.

## Data

The datasets are not redistributed.

- ASVspoof 2019 Logical Access (`LA.zip`, with the CM protocols and the
  official ASV scores): https://datashare.ed.ac.uk/handle/10283/3336
- In-the-Wild (`release_in_the_wild.zip`): https://deepfake-total.com/in_the_wild

The two archives are stored once on the Drive root (`datasets/`) and read
from there. In-the-Wild is downloaded from a pinned revision of its Hugging
Face repository (`mueller91/In-The-Wild`), and its sha256 is checked before
every use. Audio, features, score files and archives are never committed.

## Colab workflow

All outputs are written under `ROOT = /content/drive/MyDrive/ssl_silence`
(environment variable `SSL_SILENCE_ROOT`):

```
features/<encoder>/<extraction>/<condition>/<split>_shard_0000.pt
runs/<encoder>/<extraction>/runs.csv, scores/<run_id>.npz
tables/<encoder>/         analysis tables
measurements/             per-file audio measurements (encoder independent)
figures/  checks/
```

Encoders are `wavlm-base` and `hubert-base`; extractions are `unpadded`
(primary) and `padded` (comparison). The notebooks are numbered in execution
order and take the encoder as a parameter where it applies. Long jobs are
resumable. Every metric is computed from the saved scores; layers and
checkpoints are selected on the development set only.

| Notebook | Runtime | Purpose |
|---|---|---|
| `notebooks/01_environment_check.ipynb` | T4 | library versions, tests, reference checks on real files |
| `notebooks/02_migrate_wavlm_features.ipynb` | T4 | copy and verify the existing WavLM Base features |
| `notebooks/03_extract_features.ipynb` | T4 | feature extraction of HuBERT Base, one condition per run, after a short check; re-extraction check of WavLM Base |
| `notebooks/04_train.ipynb` | T4 | classifier training and scoring of one encoder, a pilot of two runs and then the grid; HuBERT Base: checks of the runs, score sets and run records; WavLM Base: comparison with the original scores |
| `notebooks/05_silence_measurements.ipynb` | CPU | silence and trimming measurements of every file, silence-only classifiers, compared with the original tables (gate A5) |
| `notebooks/06_padding_deviation.ipynb` | T4 | deviation of padded from unpadded features of one encoder, with file-level intervals and the noise floor, after a check of the pre-registered files; WavLM Base: comparison with the original table; WavLM Large: the two predictions about the noise floor |
| `notebooks/07_probes.ipynb` | CPU (high RAM, all cores) | acoustic probes of WavLM Base, compared with the original tables (gate A6) |
| `notebooks/08_analysis.ipynb` | T4 (GPU not used) | metrics, intervals, effects, rank agreement and units of one encoder; WavLM Base: comparison with the original tables; HuBERT Base: the tables of the analysis plan, the paired differences to WavLM Base and the replication rule, after a check of the pre-registered files |
| `notebooks/09_paper_tables_and_figures.ipynb` | CPU | tables and figures of the paper (planned; not written yet: the tables are built by `ssl_silence.paper.tables` and the figures by `ssl_silence.plots`) |
| `notebooks/10_vad_condition.ipynb` | T4 | supplementary condition of WavLM Base, not in the analysis plan: leading and trailing non-speech removed by the WebRTC voice activity detector; extraction, 60 runs for layers 3 to 5, effects next to the condition trimmed at 40 dB |

## Citation

```bibtex
@misc{elabid2026silence,
  title         = {Do Layer Rankings Survive Silence Trimming? The Silence Shortcut in Frozen {WavLM} Representations for Audio Deepfake Detection},
  author        = {Elabid, Nur and Arar, Mahmud Esad},
  year          = {2026},
  eprint        = {2611.32457},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  doi           = {10.48550/arXiv.2611.32457},
  url           = {https://arxiv.org/abs/2611.32457}
}
```
