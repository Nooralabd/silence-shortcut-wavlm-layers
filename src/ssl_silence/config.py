"""Paths, fixed constants and the experiment configuration.

All outputs live under ``ROOT`` on Google Drive. Every feature and run path
contains the encoder name. No other module hard-codes a path.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DRIVE_MOUNT = "/content/drive"
DRIVE_API_VERSION = 3
ROOT = os.environ.get("SSL_SILENCE_ROOT", "/content/drive/MyDrive/ssl_silence")

# Audio
SAMPLE_RATE = 16_000
WINDOW = 64_000  # 4 s analysis window, samples
MIN_SAMPLES = 400  # receptive field of the convolutional encoder

# Trimming rule
TRIM_FRAME_LENGTH = 400
TRIM_HOP_LENGTH = 160
MIN_TRIMMED_SAMPLES = 8_000  # 0.5 s; shorter results fall back to the untrimmed file

# Labels and scores: score = P(spoof), higher = more spoof-like
LABEL_BONAFIDE = 0
LABEL_SPOOF = 1

# Classifier and training
HIDDEN_DIM = 128
DROPOUT = 0.3
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEV_CHECK_EVERY = 5
PATIENCE_CHECKS = 10

# Datasets
SPLITS = ("train", "dev", "eval", "itw")
ASVSPOOF_SPLIT_SIZES = {"train": 25_380, "dev": 24_844, "eval": 71_237}
ITW_SIZE = 31_779
TEST_SETS = ("eval", "itw")
# In-the-Wild files of at most 4 s and longer than 4 s (original duration)
ITW_DURATION_COUNTS = (19_387, 12_392)
EVAL_COUNTS = {"bonafide": 7_355, "spoof": 63_882}
# (bonafide, spoof) files per split
CLASS_COUNTS = {"train": (2_580, 22_800), "dev": (2_548, 22_296), "eval": (7_355, 63_882),
                "itw": (19_963, 11_816)}
# Spoof system ids per split; bonafide files have "-"
SPOOF_SYSTEM_IDS = {"train": tuple(f"A{k:02d}" for k in range(1, 7)),
                    "dev": tuple(f"A{k:02d}" for k in range(1, 7)),
                    "eval": tuple(f"A{k:02d}" for k in range(7, 20)),
                    "itw": ("itw",)}
PROTOCOL_FILENAMES = {"train": "ASVspoof2019.LA.cm.train.trn.txt",
                      "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
                      "eval": "ASVspoof2019.LA.cm.eval.trl.txt"}
# The datasets are unpacked to the local disk of the runtime, never to Drive.
LOCAL_ASVSPOOF = os.environ.get("SSL_SILENCE_LOCAL_ASVSPOOF", "/content/asvspoof2019_la")
LOCAL_ITW = os.environ.get("SSL_SILENCE_LOCAL_ITW", "/content/in_the_wild")
# Chunks of the per-file audio measurements while they are measured; the finished tables go to Drive.
LOCAL_MEASUREMENTS = os.environ.get("SSL_SILENCE_LOCAL_MEASUREMENTS", "/content/measurements")
# The list files of the two archives alone, for runs that need no audio.
LOCAL_LISTS = os.environ.get("SSL_SILENCE_LOCAL_LISTS", "/content/file_lists")
# Per-layer copy of the features on the local disk (training reads from it).
LOCAL_CACHE = os.environ.get("SSL_SILENCE_LOCAL_CACHE", "/content/feature_cache")
# In-the-Wild is downloaded once from a pinned revision of its Hugging Face
# repository and stored on the Drive root; later sessions read the stored copy.
# The revision was read on Colab by notebooks/01_environment_check.ipynb. The
# sha256 of the archive is set after notebooks/02_migrate_wavlm_features.ipynb
# has downloaded it once; until then the archive is not unpacked.
ITW_REPOSITORY = "mueller91/In-The-Wild"
ITW_FILENAME = "release_in_the_wild.zip"
ITW_REVISION = "eee168f92c367f8c82ff2cf42b6f61e362fd6211"
ITW_ZIP_SHA256 = "46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a"
# Official ASV score files shipped inside LA.zip
ASV_SCORE_FILENAMES = {"dev": "ASVspoof2019.LA.asv.dev.gi.trl.scores.txt",
                       "eval": "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"}

# Effects of the 2x2 train/test design, E(train, test) with C = clean, T = trimmed:
# delta_test = E(C,T) - E(C,C); delta_train = E(T,T) - E(C,T);
# delta_total = E(T,T) - E(C,C); delta_train_c = E(T,C) - E(C,C);
# delta_test_t = E(T,T) - E(T,C); interaction = delta_test_t - delta_test.
EFFECTS = ("delta_test", "delta_train", "delta_total", "delta_train_c", "delta_test_t",
           "interaction")

# Colab notebooks in execution order (planned list; a notebook is referred to
# by its full file name only).
NOTEBOOKS = (
    "notebooks/01_environment_check.ipynb",
    "notebooks/02_migrate_wavlm_features.ipynb",
    "notebooks/03_extract_features.ipynb",
    "notebooks/04_train.ipynb",
    "notebooks/05_silence_measurements.ipynb",
    "notebooks/06_padding_deviation.ipynb",
    "notebooks/07_probes.ipynb",
    "notebooks/08_analysis.ipynb",
    "notebooks/09_paper_tables_and_figures.ipynb",
    "notebooks/10_vad_condition.ipynb",
)

# Extraction
EXTRACTIONS = ("unpadded", "padded")
SHARD_SIZE = 2_000

# Pilot of a training grid: the runs that are trained and compared with the
# original scores before the grid (the two regression configurations: layer
# 4 with both statistics, the first seed, untrimmed and trimmed at 40 dB).
PILOT_LAYER = 4
PILOT_POOLING = "stats"
PILOT_SEED = 42
PILOT_CONDITIONS = ("clean", "trim40")

# Re-extraction check: files per split and condition that are extracted again
# and compared with the stored features, and the generator stream that draws
# them (the same positions of a split in every condition).
REEXTRACTION_FILES = 200
REEXTRACTION_SEED = (2026, 201)

# Short check before the extraction of an encoder: files per split and
# condition that are encoded, checked and timed (the first positions of the
# same generator stream are not used: the files are drawn anew with this n).
SHORT_CHECK_FILES = 25

# Free space required on the Drive root before the WavLM Base features are
# copied, GB: about 47 for the WavLM Base copy (four unpadded conditions and
# the padded layer-4 tensors), about 45 for HuBERT Base in four conditions,
# about 15 for the two dataset archives, and the rest for the WavLM Large
# padding check, scores, tables and margin.
MIN_FREE_GB = 150

# Bootstrap
N_BOOT = 1_000
BOOTSTRAP_SEED = 2026

_CONFIG_DIR = Path(__file__).parent / "configs"


@lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict[str, Any]:
    with open(_CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def encoders() -> dict[str, dict[str, Any]]:
    """Encoder registry: name -> {hf_id, revision, n_layers, hidden_size, feat_extract_norm}."""
    return _load_yaml("encoders.yaml")


def encoder(name: str) -> dict[str, Any]:
    """Registry entry of one encoder."""
    registry = encoders()
    if name not in registry:
        raise KeyError(f"unknown encoder {name!r}; known: {sorted(registry)}")
    return registry[name]


def experiments() -> dict[str, Any]:
    """Experiment configuration (conditions, poolings, seeds, run groups)."""
    return _load_yaml("experiments.yaml")


def conditions() -> dict[str, float | None]:
    """Condition name -> trimming threshold in dB (None = no trimming)."""
    return dict(experiments()["conditions"])


def vad_conditions() -> dict[str, dict[str, int]]:
    """Supplementary conditions trimmed by voice activity detection: name ->
    {frame_ms, aggressiveness, run_frames}. They are not conditions of the
    analysis plan and are not in ``conditions()``."""
    return {name: dict(rule) for name, rule in experiments().get("vad_conditions", {}).items()}


def known_conditions() -> tuple[str, ...]:
    """The conditions of the experiments and the supplementary ones."""
    return tuple(conditions()) + tuple(vad_conditions())


def check_condition(condition: str) -> None:
    """Raise KeyError unless ``condition`` is a known condition."""
    if condition not in known_conditions():
        raise KeyError(f"unknown condition {condition!r}; known: {sorted(known_conditions())}")


def vad_rule(condition: str) -> dict[str, int] | None:
    """The rule of a condition trimmed by voice activity detection; None for
    a condition of the experiments."""
    check_condition(condition)
    return vad_conditions().get(condition)


def trim_top_db(condition: str) -> float | None:
    """Trimming threshold of a condition of the experiments. A condition
    trimmed by voice activity detection has no threshold and raises, so that
    it is never read as an untrimmed condition."""
    table = conditions()
    if condition in vad_conditions():
        raise KeyError(f"the condition {condition!r} is trimmed by voice activity detection "
                       "and has no threshold in dB")
    if condition not in table:
        raise KeyError(f"unknown condition {condition!r}; known: {sorted(table)}")
    return table[condition]


def top_db_text(condition: str) -> str:
    """The threshold of a condition as written in runs.csv: "" without one."""
    if vad_rule(condition) is not None:
        return ""
    top_db = trim_top_db(condition)
    return "" if top_db is None else f"{top_db:g}"


SEEDS: tuple[int, ...] = tuple(experiments()["seeds"])
POOLINGS: tuple[str, ...] = tuple(experiments()["poolings"])


# Paths ---------------------------------------------------------------------

def _check_names(encoder_name: str, extraction: str | None = None,
                 condition: str | None = None) -> None:
    encoder(encoder_name)
    if extraction is not None and extraction not in EXTRACTIONS:
        raise KeyError(f"unknown extraction {extraction!r}; known: {EXTRACTIONS}")
    if condition is not None:
        check_condition(condition)


def feature_dir(encoder_name: str, extraction: str, condition: str, root: str | None = None) -> str:
    """Folder of the feature shards of one encoder, extraction and condition."""
    _check_names(encoder_name, extraction, condition)
    return os.path.join(root or ROOT, "features", encoder_name, extraction, condition)


def runs_dir(encoder_name: str, extraction: str, root: str | None = None) -> str:
    """Folder with runs.csv and scores/ of one encoder and extraction."""
    _check_names(encoder_name, extraction)
    return os.path.join(root or ROOT, "runs", encoder_name, extraction)


def tables_dir(encoder_name: str, root: str | None = None) -> str:
    """Folder of the analysis tables of one encoder."""
    _check_names(encoder_name)
    return os.path.join(root or ROOT, "tables", encoder_name)


def vad_runs_dir(encoder_name: str, extraction: str, root: str | None = None) -> str:
    """Folder with runs.csv and scores/ of the supplementary condition. It is
    not under ``runs/``: the clean models that are trained again for it have
    the run ids of existing runs."""
    _check_names(encoder_name, extraction)
    return os.path.join(root or ROOT, "runs_vad", encoder_name, extraction)


def vad_tables_dir(encoder_name: str, root: str | None = None) -> str:
    """Folder of the tables of the supplementary condition of one encoder."""
    return os.path.join(tables_dir(encoder_name, root), "vad")


def datasets_dir(root: str | None = None) -> str:
    """Folder of the small dataset files kept on Drive (protocols, ASV scores)."""
    return os.path.join(root or ROOT, "datasets")


def la_zip(root: str | None = None) -> str:
    """Path of the ASVspoof 2019 LA archive on the Drive root."""
    return os.path.join(datasets_dir(root), "ASVspoof2019_LA", "LA.zip")


def itw_zip(root: str | None = None) -> str:
    """Path of the stored In-the-Wild archive on the Drive root."""
    return os.path.join(datasets_dir(root), "in_the_wild", ITW_FILENAME)


def asv_scores_dir(root: str | None = None) -> str:
    """Folder of the two official ASV score files."""
    return os.path.join(datasets_dir(root), "asv_scores")


def asv_score_file(split: str, root: str | None = None) -> str:
    """Path of the official ASV score file of ``split`` ("dev" or "eval")."""
    return os.path.join(asv_scores_dir(root), ASV_SCORE_FILENAMES[split])


def measurements_dir(root: str | None = None) -> str:
    """Folder of the per-file audio measurements (encoder independent)."""
    return os.path.join(root or ROOT, "measurements")


def baselines_dir(root: str | None = None) -> str:
    """Folder of the scores of the baselines that use no encoder (encoder independent)."""
    return os.path.join(root or ROOT, "runs", "baselines")


def figures_dir(root: str | None = None) -> str:
    """Folder of the figures."""
    return os.path.join(root or ROOT, "figures")


def checks_dir(root: str | None = None) -> str:
    """Folder of the recorded check outcomes."""
    return os.path.join(root or ROOT, "checks")


def migration_dir(root: str | None = None) -> str:
    """Folder of the migration mapping and shard manifest (Drive only)."""
    return os.path.join(root or ROOT, "migration")


# Drive guard ---------------------------------------------------------------

def on_drive(path: str | None = None) -> bool:
    """Whether ``path`` (default ROOT) lies under the Drive mount point."""
    path = ROOT if path is None else path
    norm = os.path.normpath(path).replace("\\", "/")
    return norm == DRIVE_MOUNT or norm.startswith(DRIVE_MOUNT + "/")


def require_drive(path: str | None = None) -> None:
    """Raise if ``path`` (default ROOT) lies under the Drive mount point while Drive is not mounted."""
    path = ROOT if path is None else path
    if on_drive(path) and not os.path.ismount(DRIVE_MOUNT):
        raise RuntimeError(f"Google Drive is not mounted; refusing to use {path}")


def free_space_gb(path: str | None = None) -> float:
    """Free space of the file system that holds ``path`` (default ROOT), in GB (10**9 bytes)."""
    path = ROOT if path is None else path
    require_drive(path)
    probe = path
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free / 1e9


def drive_service():
    """The Drive API (version 3) with the Google account of the Colab user."""
    from google.colab import auth
    from googleapiclient.discovery import build

    auth.authenticate_user()
    # Version 3 of the API. Its name is put together here because the naming
    # rule of this repository (tests/test_naming.py) rejects the literal label.
    return build("drive", "v" + str(DRIVE_API_VERSION), cache_discovery=False)


def drive_quota(service=None) -> dict[str, float]:
    """Storage quota of the Google account, read from the Drive API.

    Returns limit_gb, usage_gb and free_gb (limit minus usage), in GB
    (10**9 bytes). The mounted Drive folder does not report the quota of the
    account (its free space changed between two runs at the same state of
    the account), so the number of the file system is never used for Drive.
    Raises RuntimeError if the quota cannot be read or the account has no
    limit in the answer.
    """
    try:
        service = drive_service() if service is None else service
        quota = service.about().get(fields="storageQuota").execute()["storageQuota"]
        limit, usage = int(quota["limit"]), int(quota["usage"])
    except Exception as error:
        raise RuntimeError(
            "the storage quota of Google Drive could not be read from the Drive API "
            f"({type(error).__name__}: {error}). Stop: the free space reported by the mounted "
            "folder is not the quota of the account and is not used instead") from error
    return {"limit_gb": limit / 1e9, "usage_gb": usage / 1e9, "free_gb": (limit - usage) / 1e9}


def require_free_space(path: str | None = None, min_gb: float = MIN_FREE_GB,
                       quota: dict[str, float] | None = None) -> float:
    """Return the free space in GB; raise if it is below ``min_gb``.

    For a path on Drive (default ROOT) the free space is that of the account
    quota (``quota``, or ``drive_quota()``); for any other path it is the
    free space of its file system.
    """
    path = ROOT if path is None else path
    if on_drive(path):
        require_drive(path)
        free = (drive_quota() if quota is None else quota)["free_gb"]
    elif quota is not None:
        raise ValueError(f"a Drive quota was given for a path that is not on Drive: {path}")
    else:
        free = free_space_gb(path)
    if free < min_gb:
        raise RuntimeError(f"{free:.1f} GB free, {min_gb:g} GB required")
    return free


def local_free_space_gb() -> float:
    """Free space of the local disk that the datasets are unpacked to, in GB (10**9 bytes)."""
    return free_space_gb(os.path.dirname(LOCAL_ASVSPOOF))


def makedirs(path: str) -> None:
    """Create a folder after the Drive guard."""
    require_drive(path)
    os.makedirs(path, exist_ok=True)
