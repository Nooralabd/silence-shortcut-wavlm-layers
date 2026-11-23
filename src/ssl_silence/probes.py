"""Linear probes: what can be read from the mean-pooled features of each layer?

Sample: 5,000 ASVspoof 2019 LA training files, stratified by label and
attack (proportional allocation, fixed seed).

Acoustic targets, computed with Praat through Parselmouth on the audio that
enters the extraction (mono 16 kHz, trimmed in a trimmed condition, centre
crop to 4 s, no padding); all settings are in PRAAT:

    f0_mean_hz       mean F0 over the voiced frames of "To Pitch (ac)"
    hnr_mean_db      mean harmonics-to-noise ratio of "To Harmonicity (cc)"
                     over the frames in which it is defined
    tilt_db_per_khz  spectral tilt: per voiced frame, the slope of a
                     least-squares line of the log-magnitude spectrum (dB)
                     against frequency (kHz) over 0-8 kHz of a 25 ms
                     Hamming-windowed FFT; mean over the voiced frames

A file with fewer than MIN_VOICED_FRAMES usable frames has no target (NaN)
and is left out of the probe of that target.

Silence and length targets come from the silence measurements of the same
files (``silence.silence_table``).

Probe, per layer and target: StandardScaler -> ridge regression, R2 on five
held-out folds, in two schemes:

    random    KFold (shuffled); alpha by the leave-one-out of RidgeCV on the
              training fold. Files of one speaker are on both sides.
    speaker   GroupKFold by speaker, for the outer folds and for the choice
              of alpha inside every training fold.

The control repeats the probe with the targets permuted.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as cfg

N_SAMPLE = 5000
SAMPLE_SEED = 0
MIN_VOICED_FRAMES = 10
ACOUSTIC_TARGETS = ("f0_mean_hz", "hnr_mean_db", "tilt_db_per_khz")
PARSELMOUTH_VERSION = "0.4.7"
PRAAT_VERSION = "6.1.38"
PRAAT = {
    "time_step_s": 0.01,
    "pitch_floor_hz": 75.0,
    "pitch_ceiling_hz": 600.0,
    "hnr_silence_threshold": 0.1,
    "hnr_periods_per_window": 1.0,
    "hnr_undefined_db": -200.0,   # value of Praat for a frame without harmonicity
    "tilt_window_s": 0.025,
    "tilt_n_fft": 512,
    "tilt_f_max_hz": 8000.0,
    "tilt_amin": 1e-10,
}
TARGETS_FILE = "probe_targets.csv"


# Sample ------------------------------------------------------------------------------

def stratified_sample(file_list: pd.DataFrame, n: int = N_SAMPLE,
                      seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """``n`` rows of ``file_list``, stratified by (label, system_id).

    Proportional allocation with largest-remainder rounding; every stratum is
    sampled without replacement by one generator seeded with ``seed``, strata
    in sorted order. The rows keep their index and their order.
    """
    if n > len(file_list):
        raise ValueError(f"cannot sample {n} of {len(file_list)} files")
    rng = np.random.default_rng(seed)
    strata = file_list.groupby(["label", "system_id"], sort=True).indices
    keys = sorted(strata)
    sizes = np.array([len(strata[k]) for k in keys], dtype=float)
    exact = sizes / sizes.sum() * n
    allocation = np.floor(exact).astype(int)
    remainder = n - allocation.sum()
    for i in np.argsort(-(exact - allocation), kind="mergesort")[:remainder]:
        allocation[i] += 1
    chosen = []
    for k, a in zip(keys, allocation):
        chosen.extend(rng.choice(strata[k], size=int(a), replace=False))
    return file_list.iloc[np.sort(np.array(chosen))]


# Acoustic targets ----------------------------------------------------------------------

def praat_versions() -> dict[str, str]:
    """Versions of Parselmouth and of the Praat it contains."""
    import parselmouth

    return {"parselmouth": str(parselmouth.__version__), "praat": str(parselmouth.PRAAT_VERSION)}


def require_praat() -> dict[str, str]:
    """Raise unless the targets are computed with the pinned Parselmouth and Praat."""
    found = praat_versions()
    expected = {"parselmouth": PARSELMOUTH_VERSION, "praat": PRAAT_VERSION}
    if found != expected:
        raise RuntimeError(f"probe targets need {expected}, found {found}")
    return found


def real_window(y: np.ndarray, window: int = cfg.WINDOW) -> np.ndarray:
    """The samples inside the window: centre crop if longer, unchanged if shorter."""
    n = y.shape[0]
    if n > window:
        start = (n - window) // 2
        return y[start:start + window]
    return y


def spectral_tilt(y: np.ndarray, voiced_times_s: np.ndarray,
                  sr: int = cfg.SAMPLE_RATE) -> float:
    """Mean slope (dB per kHz) of the log-magnitude spectrum over 0 to
    ``tilt_f_max_hz``, one 25 ms Hamming-windowed FFT around every voiced
    frame centre. A frame that does not fit inside the signal is left out;
    NaN if none fits."""
    half = int(round(PRAAT["tilt_window_s"] * sr / 2))
    n_fft = PRAAT["tilt_n_fft"]
    freqs_khz = np.fft.rfftfreq(n_fft, 1 / sr) / 1000.0
    keep = freqs_khz <= PRAAT["tilt_f_max_hz"] / 1000.0
    window = np.hamming(2 * half)
    x = freqs_khz[keep]
    x_c = x - x.mean()
    slopes = []
    for t in voiced_times_s:
        c = int(round(t * sr))
        lo, hi = c - half, c + half
        if lo < 0 or hi > y.shape[0]:
            continue
        magnitude = np.abs(np.fft.rfft(y[lo:hi] * window, n=n_fft))[keep]
        db = 20 * np.log10(np.maximum(magnitude, PRAAT["tilt_amin"]))
        slopes.append(float(np.dot(x_c, db - db.mean()) / np.dot(x_c, x_c)))
    return float(np.mean(slopes)) if slopes else float("nan")


def acoustic_targets(y: np.ndarray, sr: int = cfg.SAMPLE_RATE) -> dict[str, float]:
    """F0, HNR and spectral tilt of one waveform y (N,).

    Returns the three targets, n_voiced_frames and n_hnr_frames. F0 and the
    tilt are NaN with fewer than MIN_VOICED_FRAMES voiced frames, the HNR
    with fewer than MIN_VOICED_FRAMES frames in which it is defined.
    """
    import parselmouth

    require_praat()
    sound = parselmouth.Sound(np.asarray(y, dtype=np.float64), sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=PRAAT["time_step_s"], pitch_floor=PRAAT["pitch_floor_hz"],
                           pitch_ceiling=PRAAT["pitch_ceiling_hz"])
    f0 = pitch.selected_array["frequency"]
    voiced = f0 > 0
    n_voiced = int(voiced.sum())
    times = pitch.xs()[voiced]
    harmonicity = sound.to_harmonicity(time_step=PRAAT["time_step_s"],
                                       minimum_pitch=PRAAT["pitch_floor_hz"],
                                       silence_threshold=PRAAT["hnr_silence_threshold"],
                                       periods_per_window=PRAAT["hnr_periods_per_window"])
    hnr = harmonicity.values[0]
    defined = hnr > PRAAT["hnr_undefined_db"]
    n_hnr = int(defined.sum())
    enough = n_voiced >= MIN_VOICED_FRAMES
    return {
        "f0_mean_hz": float(f0[voiced].mean()) if enough else float("nan"),
        "hnr_mean_db": (float(hnr[defined].mean()) if n_hnr >= MIN_VOICED_FRAMES
                        else float("nan")),
        "tilt_db_per_khz": (spectral_tilt(np.asarray(y, dtype=np.float64), times, sr)
                            if enough else float("nan")),
        "n_voiced_frames": n_voiced,
        "n_hnr_frames": n_hnr,
    }


def targets_for_file(path: str, condition: str = "clean") -> dict[str, float]:
    """The acoustic targets of one file on the audio of ``condition``: loaded
    as the extraction loads it, trimmed with the fallback rule in a trimmed
    condition, then the samples inside the 4 s window."""
    from .audio import load_audio, to_mono_16k
    from .trimming import apply_trim

    top_db = cfg.trim_top_db(condition)
    wav, sr, _ = load_audio(path)
    wav = to_mono_16k(wav, sr)
    if top_db is not None:
        wav, _ = apply_trim(wav, top_db)
    return acoustic_targets(real_window(wav[0].numpy()))


def read_targets(path: str) -> pd.DataFrame:
    """Read a target table; ids stay strings and numbers are read as written."""
    return pd.read_csv(path, dtype={"audio_id": str, "system_id": str},
                       keep_default_na=False, na_values=[""], float_precision="round_trip")


def compute_targets(sample: pd.DataFrame, out_dir: str, condition: str = "clean",
                    chunk_size: int = 500, progress: bool = True) -> pd.DataFrame:
    """Acoustic targets of every file of ``sample`` on the audio of ``condition``.

    ``sample`` has the columns audio_id, path, label and system_id. Resumable
    chunks are written under ``<out_dir>/target_chunks``; the whole table
    (audio_id, label, system_id and the keys of ``acoustic_targets``) is
    written to ``<out_dir>/probe_targets.csv`` with a meta file that records
    the condition, the Praat settings and the Parselmouth and Praat versions.
    One folder per condition.
    """
    from .provenance import write_table

    chunk_dir = os.path.join(out_dir, "target_chunks")
    cfg.makedirs(chunk_dir)
    n = len(sample)
    n_chunks = (n + chunk_size - 1) // chunk_size
    parts = []
    for k in range(n_chunks):
        path = os.path.join(chunk_dir, f"targets_chunk_{k:04d}.csv")
        if not os.path.exists(path):
            rows = []
            for _, item in sample.iloc[k * chunk_size:(k + 1) * chunk_size].iterrows():
                row = {"audio_id": item["audio_id"], "label": int(item["label"]),
                       "system_id": item["system_id"]}
                row.update(targets_for_file(item["path"], condition))
                rows.append(row)
            tmp = path + ".tmp"
            pd.DataFrame(rows).to_csv(tmp, index=False)
            os.replace(tmp, path)
            if progress:
                print(f"[probe targets, {condition}] chunk {k + 1}/{n_chunks} done")
        parts.append(read_targets(path))
    full = pd.concat(parts, ignore_index=True)
    if list(full["audio_id"]) != [str(a) for a in sample["audio_id"]]:
        raise RuntimeError(f"the target chunks do not match the sample; delete {chunk_dir}")
    write_table(full, os.path.join(out_dir, TARGETS_FILE), condition=condition,
                praat_settings=PRAAT, min_voiced_frames=MIN_VOICED_FRAMES, **require_praat())
    return full


# Silence and length targets ---------------------------------------------------------------

SILENCE_TOP_DB = 40      # threshold at which the silence targets are measured
LEAD_TARGET = f"win_lead_{SILENCE_TOP_DB}"
SILENCE_TARGETS = (LEAD_TARGET, "pad", "length_in_window")
# Targets probed in each condition. After trimming, the leading silence is zero
# in every file that was trimmed, so it is not a target of a trimmed condition.
TARGETS = {"clean": ACOUSTIC_TARGETS + SILENCE_TARGETS,
           "trim40": ACOUSTIC_TARGETS + ("pad", "length_in_window")}
ALL_TARGETS = ACOUSTIC_TARGETS + SILENCE_TARGETS
CV_SCHEMES = ("random", "speaker")
ALPHAS = np.logspace(-2, 4, 13)
N_FOLDS = 5
CV_SEED = 0
FEATURE_KEY = "mean"     # the probes use the mean-pooled features of the unpadded extraction
SELECTED_LAYERS = (1, 4, 8, 12)


def condition_targets(measurements: pd.DataFrame, condition: str = "clean") -> pd.DataFrame:
    """Silence and length targets of the window in ``condition``, indexed by audio id.

    ``measurements`` is a ``silence.silence_table``. Columns (samples):
    win_lead_40 (leading silence inside the window at 40 dB), pad (zeros a
    padded window would add) and length_in_window (window minus pad, the
    samples that enter the encoder). In a trimmed condition pad comes from
    the length after trimming, and the leading silence is 0 except in files
    that kept their untrimmed audio.
    """
    top_db = cfg.trim_top_db(condition)
    m = measurements.set_index(measurements["audio_id"].astype(str))
    lead = m[LEAD_TARGET].astype(float)
    if top_db is None:
        pad = m["pad"].astype(float)
    else:
        pad = m[f"pad_trimmed_{top_db:g}"].astype(float)
        fallback = m[f"trim_fallback_{top_db:g}"].astype(int) == 1
        lead = lead.where(fallback, 0.0)
    out = pd.DataFrame({LEAD_TARGET: lead, "pad": pad, "length_in_window": cfg.WINDOW - pad})
    out.index.name = "audio_id"
    return out


def attach_silence_targets(targets: pd.DataFrame, measurements: pd.DataFrame,
                           condition: str = "clean",
                           columns: Sequence[str] = SILENCE_TARGETS) -> pd.DataFrame:
    """Add the silence and length targets of ``condition`` to a target table
    by audio id. Every file of the table must be in the measurements."""
    by_id = condition_targets(measurements, condition)
    out = targets.copy()
    ids = out["audio_id"].astype(str)
    missing = ids[~ids.isin(by_id.index)]
    if len(missing):
        raise KeyError(f"{len(missing)} sampled files are missing from the measurement table")
    for column in columns:
        out[column] = ids.map(by_id[column]).to_numpy()
    return out


# Features ------------------------------------------------------------------------------------

def load_sample_features(cache_dir: str, split: str, audio_ids: Sequence[str],
                         key: str = FEATURE_KEY) -> tuple[np.ndarray, list[int]]:
    """Features of the given files from the layer cache of ``shards.build_layer_cache``.

    Returns ((n_files, n_layers, D) float32 in the order of ``audio_ids``,
    the stored layer numbers).
    """
    import torch

    index = torch.load(os.path.join(cache_dir, f"{split}_index.pt"), map_location="cpu",
                       weights_only=False)
    position = {a: i for i, a in enumerate(index["audio_ids"])}
    try:
        rows = np.array([position[str(a)] for a in audio_ids])
    except KeyError as error:
        raise KeyError(f"audio id {error} is not in the {split} layer cache") from None
    array = np.load(os.path.join(cache_dir, f"{split}_{key}.npy"), mmap_mode="r")
    order = np.argsort(rows)
    picked = np.asarray(array[:, rows[order], :])          # (L, n, D), read in file order
    out = np.empty_like(picked)
    out[:, order, :] = picked
    return np.ascontiguousarray(out.transpose(1, 0, 2)), [int(k) for k in index["layers"]]


# Probe -----------------------------------------------------------------------------------------

def _folds(n_folds: int, seed: int, groups: np.ndarray | None):
    from sklearn.model_selection import GroupKFold, KFold

    if groups is None:
        return KFold(n_folds, shuffle=True, random_state=seed)
    return GroupKFold(n_folds, shuffle=True, random_state=seed)


def _fit_grouped(x_train: np.ndarray, y_train: np.ndarray, g_train: np.ndarray,
                 n_folds: int, seed: int, alphas: np.ndarray):
    """StandardScaler + ridge with the alpha chosen by a GroupKFold over the
    speakers of the training fold. Returns (fitted pipeline, alpha)."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    inner = _folds(min(n_folds, len(np.unique(g_train))), seed, g_train)
    means = []
    for alpha in alphas:
        scores = []
        for i_train, i_test in inner.split(x_train, y_train, g_train):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x_train[i_train], y_train[i_train])
            scores.append(model.score(x_train[i_test], y_train[i_test]))
        means.append(np.mean(scores))
    best = float(alphas[int(np.argmax(means))])
    model = make_pipeline(StandardScaler(), Ridge(alpha=best))
    model.fit(x_train, y_train)
    return model, best


def ridge_probe_r2(x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
                   n_folds: int = N_FOLDS, seed: int = CV_SEED,
                   alphas: np.ndarray = ALPHAS) -> dict[str, object]:
    """Cross-validated R2 of a ridge probe. x (n, D), y (n,).

    Without ``groups``: the random scheme. With ``groups`` (n,) speaker ids:
    the speaker scheme (see the module docstring).

    Returns r2_mean, r2_sd (sample standard deviation over the folds),
    r2_folds (list) and alpha_median.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    outer = _folds(n_folds, seed, groups)
    scores, chosen = [], []
    for train, test in outer.split(x, y, groups):
        if groups is None:
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
            model.fit(x[train], y[train])
            alpha = float(model[-1].alpha_)
        else:
            model, alpha = _fit_grouped(x[train], y[train], groups[train], n_folds, seed, alphas)
        scores.append(float(model.score(x[test], y[test])))
        chosen.append(alpha)
    return {"r2_mean": float(np.mean(scores)), "r2_sd": float(np.std(scores, ddof=1)),
            "r2_folds": scores, "alpha_median": float(np.median(chosen))}


def probe_all_layers(features: np.ndarray, targets: pd.DataFrame,
                     target_names: Sequence[str] = ALL_TARGETS,
                     groups: np.ndarray | None = None,
                     cv_schemes: Sequence[str] = CV_SCHEMES, layers: Sequence[int] | None = None,
                     shuffle_seed: int = CV_SEED, n_jobs: int = 1,
                     progress: bool = True) -> pd.DataFrame:
    """Ridge probes for every layer, target and scheme, with real and with permuted targets.

    ``features`` is (n_files, n_layers, D) in the file order of ``targets``;
    ``layers`` are the layer numbers of its second axis (default 1, 2, ...).
    Files without a target (NaN) are left out per target. Every name of
    ``target_names`` must be a column of ``targets``. ``groups`` (n_files,)
    are the speaker ids, needed for the speaker scheme. The permutation of
    the control is drawn once per target with ``shuffle_seed``. Every fit is
    seeded, so the result does not depend on ``n_jobs``.

    Returns one row per scheme, layer, target and control ("real" or
    "shuffled"): cv, layer, target, control, n_files, n_groups, r2_mean,
    r2_sd, alpha_median, r2_fold_1 ... r2_fold_5.
    """
    from joblib import Parallel, delayed

    n_files, n_layers, _ = features.shape
    layers = list(range(1, n_layers + 1)) if layers is None else [int(k) for k in layers]
    if len(layers) != n_layers:
        raise ValueError("layers and features have different numbers of layers")
    if len(targets) != n_files:
        raise ValueError("features and targets have different lengths")
    if "speaker" in cv_schemes and groups is None:
        raise ValueError("the speaker scheme needs the speaker ids (groups)")
    if groups is not None and len(groups) != n_files:
        raise ValueError("groups and features have different lengths")
    unknown = [c for c in cv_schemes if c not in CV_SCHEMES]
    if unknown:
        raise ValueError(f"cv schemes must be among {CV_SCHEMES}, got {unknown}")
    missing = [name for name in target_names if name not in targets.columns]
    if missing:
        raise KeyError(f"targets missing from the table: {missing}")
    rows = []
    for name in target_names:
        y_all = targets[name].to_numpy(dtype=np.float64)
        ok = np.isfinite(y_all)
        y = y_all[ok]
        y_shuffled = np.random.default_rng(shuffle_seed).permutation(y)
        g = None if groups is None else np.asarray(groups)[ok]
        x_ok = features[ok].astype(np.float64)
        for cv in cv_schemes:
            g_cv = g if cv == "speaker" else None
            jobs = [(k, layer, control, yy) for k, layer in enumerate(layers)
                    for control, yy in (("real", y), ("shuffled", y_shuffled))]
            results = Parallel(n_jobs=n_jobs)(
                delayed(ridge_probe_r2)(x_ok[:, k, :], yy, g_cv) for k, _, _, yy in jobs)
            block = []
            for (_, layer, control, _), r in zip(jobs, results):
                row = {"cv": cv, "layer": layer, "target": name, "control": control,
                       "n_files": int(ok.sum()),
                       "n_groups": int(len(np.unique(g))) if g is not None else np.nan,
                       "r2_mean": r["r2_mean"], "r2_sd": r["r2_sd"],
                       "alpha_median": r["alpha_median"]}
                row.update({f"r2_fold_{i + 1}": v for i, v in enumerate(r["r2_folds"])})
                block.append(row)
            rows += block
            if progress:
                real = [row for row in block if row["control"] == "real"]
                best = max(real, key=lambda row: row["r2_mean"])
                shuffled = max(row["r2_mean"] for row in block if row["control"] == "shuffled")
                print(f"[probe/{cv}] {name}: best layer {best['layer']} "
                      f"R2 = {best['r2_mean']:.3f} +- {best['r2_sd']:.3f}; "
                      f"shuffled max {shuffled:.3f}")
    return pd.DataFrame(rows)


# Tables ------------------------------------------------------------------------------------------

def probe_table(r2: pd.DataFrame) -> pd.DataFrame:
    """The real-target results side by side: one row per layer, columns
    ``<target>|random`` and ``<target>|speaker`` holding "mean +- sd"."""
    real = r2[r2["control"] == "real"].copy()
    real["cell"] = real.apply(lambda r: f"{r['r2_mean']:.3f} +- {r['r2_sd']:.3f}", axis=1)
    wide = real.pivot_table(index="layer", columns=["target", "cv"], values="cell",
                            aggfunc="first")
    targets = [name for name in ALL_TARGETS if name in set(real["target"])]
    wide = wide.reindex(columns=pd.MultiIndex.from_product([targets, list(CV_SCHEMES)]))
    wide.columns = [f"{a}|{b}" for a, b in wide.columns]
    return wide.reset_index()


def selected_layers_table(r2: pd.DataFrame, layers: Sequence[int] = SELECTED_LAYERS,
                          cv: str = "speaker") -> pd.DataFrame:
    """Mean R2 of the real targets at the selected layers in one scheme: one
    row per target (in the order of ALL_TARGETS), columns target, cv and
    ``L<k>`` for every layer."""
    real = r2[(r2["control"] == "real") & (r2["cv"] == cv)
              & r2["layer"].isin([int(k) for k in layers])]
    wide = real.pivot_table(index="target", columns="layer", values="r2_mean", aggfunc="first")
    missing = [int(k) for k in layers if int(k) not in wide.columns]
    if missing:
        raise KeyError(f"layers missing from the probe table: {missing}")
    wide = wide.reindex([name for name in ALL_TARGETS if name in wide.index])
    out = pd.DataFrame({"target": wide.index, "cv": cv})
    for k in layers:
        out[f"L{int(k)}"] = wide[int(k)].to_numpy()
    return out


def run_probes(sample: pd.DataFrame, cache_dir: str, out_dir: str, measurements: pd.DataFrame,
               encoder: str = "wavlm-base", condition: str = "clean", split: str = "train",
               targets_dir: str | None = None, n_jobs: int = 1,
               progress: bool = True) -> pd.DataFrame:
    """The probes of one encoder and condition for a sample of files.

    ``sample`` has audio_id, path, label, system_id and speaker.
    ``cache_dir`` is the layer cache of the unpadded features of ``encoder``
    in ``condition``. ``measurements`` is the ``silence.silence_table`` of
    ``split``. The acoustic targets are computed in ``targets_dir`` (default
    ``out_dir``; resumable, so targets computed before are read, not computed
    again) on the audio of ``condition``.

    Writes ``probe_r2.csv`` (the rows of ``probe_all_layers`` with the
    leading columns encoder, extraction and condition), ``probe_r2_table.csv``
    (``probe_table``) and ``probe_r2_selected.csv`` (``selected_layers_table``)
    to ``out_dir``. Returns the first.
    """
    from .provenance import write_table

    cfg.encoder(encoder)
    names = TARGETS[condition]
    cfg.makedirs(out_dir)
    targets = compute_targets(sample, targets_dir or out_dir, condition, progress=progress)
    targets = attach_silence_targets(targets, measurements, condition,
                                     [name for name in names if name in SILENCE_TARGETS])
    speakers = sample.set_index(sample["audio_id"].astype(str))["speaker"]
    groups = targets["audio_id"].astype(str).map(speakers).to_numpy()
    if pd.isna(groups).any():
        raise KeyError("sampled files without a speaker id")
    x, layers = load_sample_features(cache_dir, split, targets["audio_id"].tolist())
    r2 = probe_all_layers(x, targets, names, groups=groups, layers=layers, n_jobs=n_jobs,
                          progress=progress)
    r2.insert(0, "encoder", encoder)
    r2.insert(1, "extraction", "unpadded")
    r2.insert(2, "condition", condition)
    meta = {"encoder": encoder, "condition": condition, "split": split, "n_sample": len(sample),
            "alphas": [float(a) for a in ALPHAS], "n_folds": N_FOLDS, "cv_seed": CV_SEED,
            **praat_versions()}
    write_table(r2, os.path.join(out_dir, "probe_r2.csv"), **meta)
    write_table(probe_table(r2), os.path.join(out_dir, "probe_r2_table.csv"), **meta)
    selected = [k for k in SELECTED_LAYERS if k in layers]
    write_table(selected_layers_table(r2, selected), os.path.join(out_dir, "probe_r2_selected.csv"),
                **meta)
    return r2


# Gate A6: the probes against the original sample, targets and tables ---------------------------

GATE_CONDITIONS = ("clean", "trim40")
# Copies made by notebooks/02_migrate_wavlm_features.ipynb under <migration>/original/files.
ORIGINAL_SAMPLE = "probe_sample.csv"
ORIGINAL_TARGETS = {"clean": "probe_targets_clean.csv", "trim40": "probe_targets_trim40.csv"}
# Names of the original R2 tables in the mapping file.
ORIGINAL_TABLES = {"clean": "probe_r2_clean", "trim40": "probe_r2_trim40"}
# Speakers of the sample: n_groups of the original tables.
N_SPEAKERS = 20
# Columns of an R2 table that are compared with R2_TOLERANCE, and those that must be equal.
R2_COLUMNS = ("r2_mean", "r2_sd") + tuple(f"r2_fold_{i + 1}" for i in range(N_FOLDS))
EXACT_COLUMNS = ("alpha_median", "n_files", "n_groups")
# The fits go through LAPACK and BLAS, whose last digits depend on the CPU and
# on the number of threads; the machine that wrote the original tables is not
# recorded. Decided by the first author before any run (docs/decisions.md).
R2_TOLERANCE = 1e-9
NOT_PORTED = "not ported, not in the manuscript"
# The check tables that decide gate A6, in the order in which they are written.
GATE_CHECKS = ("probe_sample", "probe_speakers", "probe_targets", "probe_r2_original_targets",
               "probe_r2_new_targets", "probe_table_cells", "probe_printed_numbers")


def compare_sample(sample: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """The sample of this package against the original sample (audio_id, label,
    system_id), file by file and in the same order. One row."""
    def same(column: str) -> bool:
        return (len(sample) == len(original)
                and list(sample[column].astype(str)) == list(original[column].astype(str)))

    row = {"n_files": len(sample), "n_original": len(original),
           "same_files_in_order": same("audio_id"), "same_labels": same("label"),
           "same_attacks": same("system_id")}
    row["ok"] = bool(row["same_files_in_order"] and row["same_labels"] and row["same_attacks"])
    return pd.DataFrame([row])


def speaker_check(sample: pd.DataFrame, expected: int = N_SPEAKERS) -> pd.DataFrame:
    """Every sampled file has a speaker id and the sample has ``expected`` speakers. One row."""
    speakers = sample["speaker"]
    without = int((speakers.isna() | (speakers.astype(str).str.strip() == "")).sum())
    n_speakers = int(speakers[speakers.notna()].astype(str).nunique())
    return pd.DataFrame([{"n_files": len(sample), "n_without_speaker": without,
                          "n_speakers": n_speakers, "expected": int(expected),
                          "ok": bool(without == 0 and n_speakers == expected)}])


def compare_targets(new: pd.DataFrame, original: pd.DataFrame, condition: str,
                    tolerance: float | None = None) -> pd.DataFrame:
    """The acoustic targets of this package against an original target table, on audio_id.

    One row per column of the original table (``verify.compare_tables``,
    tolerance ``verify.TARGET_TOLERANCE``), with the leading column
    ``condition`` and the column ``identical``: no row is missing, no value
    differs and the largest difference is exactly 0.0.
    """
    from . import verify

    result = verify.compare_tables(new, original, ["audio_id"],
                                   tolerance=verify.TARGET_TOLERANCE if tolerance is None else tolerance)
    largest = result["max_abs_diff"].fillna(0.0)                 # NaN: a column of text, compared by value
    exact = verify.compare_tables(new, original, ["audio_id"], tolerance=0.0)["ok"]
    result["identical"] = (exact & (largest == 0.0)).to_numpy()
    result.insert(0, "condition", condition)
    return result


def in_sample_order(targets: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """``targets`` in the file order of ``sample``; both must hold the same files, each once."""
    ids = [str(a) for a in sample["audio_id"]]
    by_id = targets.set_index(targets["audio_id"].astype(str), drop=False)
    if by_id.index.has_duplicates or sorted(by_id.index) != sorted(ids):
        raise KeyError("the target table and the sample do not hold the same files")
    return by_id.loc[ids].reset_index(drop=True)


def rerun_table(target_check: pd.DataFrame) -> pd.DataFrame:
    """Which targets are probed again with the new values (check 4 of the gate).

    The rule: a target whose new values are identical (0.0) to the original
    target table is the same computation in both checks and is not probed
    again; a target with any difference is. The silence and length targets are
    attached to the original and to the new acoustic targets from the same
    measurement table, so their values are the same in both checks.

    ``target_check``: the rows of ``compare_targets`` of the conditions. One
    row per condition and target of ``TARGETS``: condition, target, source,
    max_abs_diff, identical, probed_again.
    """
    rows = []
    for condition in dict.fromkeys(target_check["condition"]):
        part = target_check[target_check["condition"] == condition].set_index("column")
        for name in TARGETS[condition]:
            if name in ACOUSTIC_TARGETS:
                identical = bool(part.loc[name, "identical"])
                rows.append({"condition": condition, "target": name, "source": "original target table",
                             "max_abs_diff": float(part.loc[name, "max_abs_diff"]),
                             "identical": identical, "probed_again": not identical})
            else:
                rows.append({"condition": condition, "target": name,
                             "source": "silence measurements, the same values in both checks",
                             "max_abs_diff": 0.0, "identical": True, "probed_again": False})
    return pd.DataFrame(rows)


def _sha256(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def read_block(path: str) -> pd.DataFrame:
    """A block of ``probe_blocks``, every number read back exactly as written."""
    return pd.read_csv(path, float_precision="round_trip")


def probe_blocks(features: np.ndarray, targets: pd.DataFrame, target_names: Sequence[str],
                 groups: np.ndarray, layers: Sequence[int], block_dir: str, tag: str,
                 n_jobs: int = 1, progress: bool = True) -> pd.DataFrame:
    """``probe_all_layers`` in resumable blocks, one per target and scheme.

    A finished block is written to ``<block_dir>/<tag>__<target>__<scheme>.csv``
    with a ``.meta.json`` that holds the sha256 of the features, of the target
    values and of the speaker ids. A block that exists is read instead of
    computed; it must have been computed from the same inputs (RuntimeError
    otherwise). The meta file is written last, so an interrupted block is
    computed again. Returns the rows of ``probe_all_layers`` for
    ``target_names`` and both schemes, in its order.
    """
    import json

    cfg.makedirs(block_dir)
    inputs = {"n_files": int(features.shape[0]), "layers": [int(k) for k in layers],
              "features_sha256": _sha256(features),
              "groups_sha256": _sha256(np.array([str(g) for g in groups], dtype="U"))}
    blocks = []
    for name in target_names:
        stamp = dict(inputs, target=name,
                     target_sha256=_sha256(targets[name].to_numpy(dtype=np.float64)))
        for cv in CV_SCHEMES:
            path = os.path.join(block_dir, f"{tag}__{name}__{cv}.csv")
            meta = path[:-len(".csv")] + ".meta.json"
            if os.path.exists(path) and os.path.exists(meta):
                with open(meta, encoding="utf-8") as fh:
                    found = json.load(fh)
                if found != dict(stamp, cv=cv):
                    raise RuntimeError(f"{path} was computed from other inputs; delete it and its meta file")
                if progress:
                    print(f"[probe/{cv}] {name}: read from {os.path.basename(path)}")
                blocks.append(read_block(path))
                continue
            block = probe_all_layers(features, targets, [name], groups=groups, cv_schemes=[cv],
                                     layers=layers, n_jobs=n_jobs, progress=progress)
            for target_path, write in ((path, lambda p: block.to_csv(p, index=False)),
                                       (meta, lambda p: _write_json(p, dict(stamp, cv=cv)))):
                tmp = target_path + ".tmp"
                write(tmp)
                os.replace(tmp, target_path)
            blocks.append(read_block(path))
    return pd.concat(blocks, ignore_index=True)


def _write_json(path: str, value: Mapping[str, object]) -> None:
    import json

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=1)


def merge_reruns(first: pd.DataFrame, again: pd.DataFrame | None,
                 target_names: Sequence[str]) -> pd.DataFrame:
    """The R2 table of the new targets: the rows of ``again`` for the targets
    that were probed again and the rows of ``first`` for every other target,
    in the order of ``target_names``."""
    redone = set() if again is None else set(again["target"])
    parts = []
    for name in target_names:
        source = again if name in redone else first
        parts.append(source[source["target"] == name])
    return pd.concat(parts, ignore_index=True)


def assemble(r2: Mapping[str, pd.DataFrame], encoder: str = "wavlm-base") -> pd.DataFrame:
    """One table of the conditions of ``r2`` with the leading columns encoder,
    extraction and condition (the columns of ``run_probes``)."""
    parts = []
    for condition, table in r2.items():
        part = table.drop(columns=[c for c in ("encoder", "extraction", "condition")
                                   if c in table.columns])
        part.insert(0, "encoder", encoder)
        part.insert(1, "extraction", "unpadded")
        part.insert(2, "condition", condition)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def compare_r2(r2: pd.DataFrame, condition: str, legacy: Mapping[str, object],
               copies: str | None = None) -> pd.DataFrame:
    """An R2 table of this package against the original table of ``condition``.

    ``verify.compare_with_legacy`` on the keys of the mapping file. The
    columns of R2_COLUMNS may differ by R2_TOLERANCE; those of EXACT_COLUMNS
    must be equal. Added columns: ``identical`` (no row missing and the
    largest difference exactly 0.0) and ``note``. The targets that only the
    original table has are listed in further rows with the note NOT_PORTED;
    they are not compared.
    """
    from . import verify

    name = ORIGINAL_TABLES[condition]
    new = r2.drop(columns=[c for c in ("condition",) if c in r2.columns]).assign(condition=condition)
    result = verify.compare_with_legacy(new, name, legacy, copies)
    present = result["n_only_new"].eq(0) & result["n_only_old"].eq(0) & result["n_matched"].gt(0)
    largest = result["max_abs_diff"].fillna(0.0)
    r2_column = result["column"].isin(R2_COLUMNS)
    exact = result["column"].isin(EXACT_COLUMNS)
    unknown = sorted(set(result["column"]) - set(R2_COLUMNS) - set(EXACT_COLUMNS))
    if unknown:
        raise KeyError(f"columns of {name} without a rule: {unknown}")
    result["tolerance"] = np.where(exact, 0.0, R2_TOLERANCE)
    result["identical"] = (present & (largest == 0.0)).to_numpy()
    result["ok"] = (present & np.where(r2_column, largest <= R2_TOLERANCE, largest == 0.0)).to_numpy()
    result["note"] = ""
    path = (verify.legacy_path(legacy, legacy["tables"][name]["file"]) if copies is None
            else os.path.join(copies, "tables", f"{name}.csv"))
    original = pd.read_csv(path, float_precision="round_trip")
    others = [t for t in dict.fromkeys(original["target"]) if t not in TARGETS[condition]]
    rows = [{"table": name, "column": f"target {t}", "n_old": int((original["target"] == t).sum()),
             "identical": False, "ok": True, "note": NOT_PORTED} for t in others]
    return pd.concat([result, pd.DataFrame(rows)], ignore_index=True) if rows else result


def largest_difference(check: pd.DataFrame) -> float:
    """The largest difference of the compared columns of a ``compare_r2`` table."""
    compared = check[check["note"] == ""]
    return float(compared["max_abs_diff"].max())


def selected_cells_check(new: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """The cells of the probe table of the manuscript (``paper.tables.probes``:
    clean audio, speaker folds, selected layers, two decimals) built from the
    new R2 table and from the original one. One row per cell: target, layer,
    cell, original and ok (the two texts are equal)."""
    from .paper import tables as T

    header, built = T.probes({T.PROBE_R2: new})
    _, expected = T.probes({T.PROBE_R2: original})
    rows = [{"target": a[0], "layer": layer, "cell": x, "original": y, "ok": bool(x == y and x != "--")}
            for a, b in zip(built, expected) for layer, x, y in zip(header[1:], a[1:], b[1:])]
    return pd.DataFrame(rows)


def time_probes(features: np.ndarray, targets: pd.DataFrame, name: str, groups: np.ndarray,
                n_probes: int, n_jobs: int = 1) -> pd.DataFrame:
    """Time ``n_jobs`` probes per scheme, run side by side as the long part
    runs them, and scale to ``n_probes`` probes per scheme.

    One row per scheme: cv, probes_timed, seconds, n_probes, n_jobs and
    estimated_minutes. The fitted probes are discarded.
    """
    import time

    from joblib import Parallel, delayed

    y_all = targets[name].to_numpy(dtype=np.float64)
    ok = np.isfinite(y_all)
    x, y, g = features[ok].astype(np.float64), y_all[ok], np.asarray(groups)[ok]
    n_layers = x.shape[1]
    rows = []
    for cv in CV_SCHEMES:
        start = time.perf_counter()
        Parallel(n_jobs=n_jobs)(delayed(ridge_probe_r2)(x[:, k % n_layers, :], y,
                                                       g if cv == "speaker" else None)
                                for k in range(n_jobs))
        seconds = time.perf_counter() - start
        rows.append({"cv": cv, "probes_timed": int(n_jobs), "seconds": seconds,
                     "n_probes": int(n_probes), "n_jobs": int(n_jobs),
                     "estimated_minutes": seconds / n_jobs * n_probes / 60})
    return pd.DataFrame(rows)
