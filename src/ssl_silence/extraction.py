"""Frozen feature extraction: hidden states of every transformer layer, pooled over time.

Two extractions:

    unpadded (primary)   each file is encoded alone at its own length inside
                         the 4 s window, without padding or attention mask
    padded (comparison)  the file is zero-padded at the end to the window;
                         pooled over all frames and over the frames of the
                         real samples

The standard deviation is the population standard deviation (division by the
number of frames). The output of the convolutional encoder
(``hidden_states[0]``) is not used.
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from . import config as cfg
from . import encoders, provenance, shards, verify
from .audio import load_audio, padded_window, to_mono_16k, unpadded_window
from .trimming import apply_trim, apply_vad


def pool_frames(h: torch.Tensor, n_real_frames: int | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and population standard deviation over time.

    ``h`` is (T, D) or (L, T, D). With ``n_real_frames`` only the first
    ``n_real_frames`` frames are pooled.

    Returns (mean, std), each (D,) or (L, D).
    """
    if n_real_frames is not None:
        if not 1 <= n_real_frames <= h.shape[-2]:
            raise ValueError(f"n_real_frames={n_real_frames} outside 1..{h.shape[-2]}")
        h = h[..., :n_real_frames, :]
    return h.mean(dim=-2), h.std(dim=-2, unbiased=False)


@torch.no_grad()
def hidden_states(model, x: torch.Tensor, n_layers: int, device: str | torch.device,
                  attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Hidden states of the transformer layers for one waveform x (n,).

    ``attention_mask`` is (n,), 1 for real samples; None passes no mask.

    Returns (L, T, D) on ``device``; index 0 is transformer layer 1.
    """
    inputs = x.unsqueeze(0).to(device)
    if attention_mask is None:
        out = model(inputs, output_hidden_states=True)
    else:
        out = model(inputs, attention_mask=attention_mask.unsqueeze(0).to(device),
                    output_hidden_states=True)
    if len(out.hidden_states) != n_layers + 1:
        raise ValueError(f"{len(out.hidden_states) - 1} layers returned, expected {n_layers}")
    return torch.stack(out.hidden_states[1:1 + n_layers], 0).squeeze(1)


@torch.no_grad()
def encode_unpadded(model, x: torch.Tensor, n_layers: int,
                    device: str | torch.device) -> dict[str, torch.Tensor | int]:
    """Pooled features of an unpadded waveform x (n,): batch of one, no mask.

    Returns mean, std (L, D) float32 on the CPU and n_frames.
    """
    h = hidden_states(model, x, n_layers, device)
    mean, std = pool_frames(h)
    return {"mean": mean.float().cpu(), "std": std.float().cpu(), "n_frames": int(h.shape[1])}


@torch.no_grad()
def encode_padded(model, wav: torch.Tensor, n_real: int, n_layers: int,
                  device: str | torch.device, use_attention_mask: bool = False
                  ) -> dict[str, torch.Tensor | int]:
    """Pooled features of a zero-padded waveform (window,) with ``n_real`` real samples.

    With ``use_attention_mask`` the mask is 1 for the real samples and 0 for
    the padding.

    Returns all_frames_mean, all_frames_std, real_frames_mean,
    real_frames_std (L, D) float32 on the CPU, n_frames and n_real_frames.
    """
    mask = None
    if use_attention_mask:
        mask = torch.zeros(wav.shape[0], dtype=torch.long)
        mask[:n_real] = 1
    h = hidden_states(model, wav, n_layers, device, attention_mask=mask)
    n_frames = int(h.shape[1])
    n_real_frames = encoders.real_frame_count(n_frames, n_real, *encoders.conv_layers(model))
    all_mean, all_std = pool_frames(h)
    real_mean, real_std = pool_frames(h, n_real_frames)
    return {"all_frames_mean": all_mean.float().cpu(), "all_frames_std": all_std.float().cpu(),
            "real_frames_mean": real_mean.float().cpu(), "real_frames_std": real_std.float().cpu(),
            "n_frames": n_frames, "n_real_frames": n_real_frames}


def trim_rule(condition: str) -> tuple[float | None, dict | None]:
    """(threshold in dB, rule of the voice activity detection) of a condition:
    one of them at most is not None."""
    vad = cfg.vad_rule(condition)
    return (None if vad is not None else cfg.trim_top_db(condition)), vad


def prepare_file(path: str, top_db: float | None, extraction: str,
                 vad: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Load a file and prepare the encoder input of ``extraction``.

    load -> mono 16 kHz -> trim at ``top_db`` (None: no trimming), or remove
    the leading and trailing non-speech found with the rule ``vad`` -> window.

    Returns x (the waveform), n_real, trim_fallback, too_short and backend;
    with ``vad`` also n_samples (the samples before trimming) and kept_start,
    kept_end (the interval that was kept).
    """
    wav, sr, backend = load_audio(path)
    wav = to_mono_16k(wav, sr)
    fallback, kept = False, {}
    if vad is not None:
        if top_db is not None:
            raise ValueError("a file is trimmed at a threshold or by voice activity detection, not both")
        n_samples = int(wav.shape[1])
        wav, fallback, start, end = apply_vad(wav, dict(vad))
        kept = {"n_samples": n_samples, "kept_start": int(start), "kept_end": int(end)}
    elif top_db is not None:
        wav, fallback = apply_trim(wav, top_db)
    if extraction == "unpadded":
        x, n_real, too_short = unpadded_window(wav)
    elif extraction == "padded":
        x, n_real = padded_window(wav)
        too_short = n_real < cfg.MIN_SAMPLES
    else:
        raise KeyError(f"unknown extraction {extraction!r}; known: {cfg.EXTRACTIONS}")
    return {"x": x, "n_real": int(n_real), "trim_fallback": bool(fallback),
            "too_short": bool(too_short), "backend": backend, **kept}


def extraction_meta(encoder: str, entry: Mapping[str, Any], extraction: str, condition: str,
                    split: str, device: str | torch.device, backends: list[str]) -> dict[str, Any]:
    """Provenance stored in every shard."""
    top_db, vad = trim_rule(condition)
    rule = {} if vad is None else {"vad": {
        **vad, "package": "webrtcvad-wheels",
        "package_version": provenance.installed_version("webrtcvad-wheels")}}
    return {**rule,
            "encoder": encoder, "hf_id": entry["hf_id"], "revision": entry["revision"],
            "model_class": entry["model_class"],
            "attn_implementation": entry["attn_implementation"],
            "dtype": str(encoders.DTYPE).replace("torch.", ""),
            "extraction": extraction, "condition": condition,
            "trim_top_db": top_db, "split": split,
            "window": cfg.WINDOW, "sample_rate": cfg.SAMPLE_RATE, "min_samples": cfg.MIN_SAMPLES,
            "audio_backends": backends, "device": str(device),
            "gpu_name": provenance.gpu_name(device), "versions": provenance.versions(),
            "git_commit": provenance.git_commit()}


def extract_shards(items: pd.DataFrame, feature_dir: str, split: str, *, encoder: str,
                   extraction: str, condition: str, shard_size: int = cfg.SHARD_SIZE,
                   device: str | torch.device | None = None, model=None,
                   entry: Mapping[str, Any] | None = None, max_new_shards: int | None = None,
                   progress: bool = True) -> list[str]:
    """Extract the pooled features of ``items`` into resumable shards.

    ``items`` has the columns path, audio_id, label, system_id; its order
    defines the shards. Files are encoded one at a time. Existing shards are
    kept, and a shard is written atomically when complete. ``model`` is
    loaded from the registry when not given; a given model is checked against
    ``entry`` (default: the registry entry of ``encoder``).

    Returns the paths of all shards that exist when the function returns.
    """
    top_db, vad = trim_rule(condition)
    kept_keys = shards.KEPT_KEYS if vad is not None else ()
    feature_keys = shards.FEATURE_KEYS[extraction]
    entry = cfg.encoder(encoder) if entry is None else entry
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.makedirs(feature_dir)

    n_total = len(items)
    count = shards.n_shards(n_total, shard_size)
    written = 0
    for k in range(count):
        path = shards.shard_path(feature_dir, split, k)
        if os.path.exists(path):
            continue
        if max_new_shards is not None and written >= max_new_shards:
            break
        if model is None:
            model = encoders.load_encoder(encoder, device)
        else:
            encoders.check_encoder(model, entry)
        n_layers = int(model.config.num_hidden_layers)
        lo, hi = k * shard_size, min((k + 1) * shard_size, n_total)
        keys = feature_keys + tuple(key for key in shards.PER_FILE_KEYS[extraction]
                                    if key != "labels") + kept_keys
        buffer: dict[str, list] = {key: [] for key in keys}
        backends = set()
        for _, item in items.iloc[lo:hi].iterrows():
            prepared = prepare_file(item["path"], top_db, extraction, vad)
            backends.add(prepared["backend"])
            if extraction == "unpadded":
                encoded = encode_unpadded(model, prepared["x"], n_layers, device)
            else:
                encoded = encode_padded(model, prepared["x"], prepared["n_real"], n_layers, device)
            encoded.update(n_real=prepared["n_real"], trim_fallback=prepared["trim_fallback"],
                           too_short=prepared["too_short"],
                           **{key: prepared[key] for key in kept_keys})
            for key in buffer:
                buffer[key].append(encoded[key])
        part = items.iloc[lo:hi]
        payload: dict[str, Any] = {key: torch.stack(buffer[key]) for key in feature_keys}
        for key in keys[len(feature_keys):]:
            dtype = torch.bool if key in ("trim_fallback", "too_short") else torch.long
            payload[key] = torch.tensor(buffer[key], dtype=dtype)
        payload.update(
            labels=torch.tensor(part["label"].to_numpy(), dtype=torch.long),
            audio_ids=[str(a) for a in part["audio_id"]],
            system_ids=[str(s) for s in part["system_id"]],
            lo=lo, hi=hi, n_total=n_total, shard_size=shard_size,
            meta=extraction_meta(encoder, entry, extraction, condition, split, device,
                                 sorted(backends)))
        shards.write_shard(path, payload, extraction)
        written += 1
        if progress:
            print(f"[{encoder}/{extraction}/{condition}/{split}] shard {k + 1}/{count} "
                  f"(files {lo}-{hi}) saved")
    return [shards.shard_path(feature_dir, split, k) for k in range(count)
            if os.path.exists(shards.shard_path(feature_dir, split, k))]


# Re-extraction check ---------------------------------------------------------

def sample_positions(n_total: int, n: int = cfg.REEXTRACTION_FILES,
                     seed=cfg.REEXTRACTION_SEED) -> list[int]:
    """``n`` positions of a file list of ``n_total`` files, drawn without
    replacement from a fixed generator stream, in ascending order. The same
    ``n_total`` gives the same positions, so a split has the same files in
    every condition."""
    if not 1 <= n <= n_total:
        raise ValueError(f"cannot draw {n} of {n_total} files")
    drawn = np.random.default_rng(seed).choice(n_total, size=n, replace=False)
    return [int(k) for k in np.sort(drawn)]


def stored_files(feature_dir: str, split: str, positions: Sequence[int],
                 extraction: str = "unpadded") -> dict[str, Any]:
    """What the shards of a split store for the files at ``positions`` of the file list.

    Returns every feature key (n, L, D), every per-file key (n,), the lists
    of ids, ``layers`` (the 1-based layers stored, None for all) and
    ``audio_backends`` (the audio libraries recorded in the shards read).
    Only the shards that hold one of the positions are read.
    """
    positions = [int(p) for p in positions]
    if positions != sorted(set(positions)):
        raise ValueError("positions must be ascending and unique")
    tensor_keys = shards.FEATURE_KEYS[extraction] + shards.PER_FILE_KEYS[extraction]
    parts: dict[str, list] = {key: [] for key in tensor_keys + shards.LIST_KEYS}
    layers: list[list[int] | None] = []
    backends: set[str] = set()
    found = 0
    for shard in shards.split_shards(feature_dir, split, extraction):
        rows = [p - shard["lo"] for p in positions if shard["lo"] <= p < shard["hi"]]
        if not rows:
            continue
        found += len(rows)
        for key in tensor_keys:
            parts[key].append(shard[key][rows].clone())
        for key in shards.LIST_KEYS:
            parts[key] += [str(shard[key][row]) for row in rows]
        layers.append(shard["layers"])
        meta = shard["meta"]
        backends.update(meta.get("audio_backends") or meta.get("source", {}).get("audio_backends") or [])
    if found != len(positions):
        raise IndexError(f"{len(positions) - found} positions are outside split {split!r}")
    if any(entry != layers[0] for entry in layers):
        raise ValueError(f"the shards of split {split!r} store different layers")
    out: dict[str, Any] = {key: torch.cat(parts[key]) for key in tensor_keys}
    out.update({key: parts[key] for key in shards.LIST_KEYS})
    out.update(layers=layers[0], audio_backends=sorted(backends))
    return out


def reextract(items: pd.DataFrame, positions: Sequence[int], *, model, extraction: str,
              condition: str, device: str | torch.device,
              layers: Sequence[int] | None = None) -> dict[str, Any]:
    """Extract the files at ``positions`` of the file list ``items`` again.

    The files go through the functions that ``extract_shards`` uses, one at a
    time. With ``layers`` (1-based) the feature tensors keep those layers
    only. Returns the keys of ``stored_files``; nothing is written.
    """
    top_db, vad = trim_rule(condition)
    n_layers = int(model.config.num_hidden_layers)
    feature_keys = shards.FEATURE_KEYS[extraction]
    per_file = tuple(key for key in shards.PER_FILE_KEYS[extraction] if key != "labels")
    buffer: dict[str, list] = {key: [] for key in feature_keys + per_file}
    backends: set[str] = set()
    part = items.iloc[[int(p) for p in positions]]
    for path in part["path"]:
        prepared = prepare_file(path, top_db, extraction, vad)
        backends.add(prepared["backend"])
        if extraction == "unpadded":
            encoded = encode_unpadded(model, prepared["x"], n_layers, device)
        else:
            encoded = encode_padded(model, prepared["x"], prepared["n_real"], n_layers, device)
        encoded.update(n_real=prepared["n_real"], trim_fallback=prepared["trim_fallback"],
                       too_short=prepared["too_short"])
        for key in buffer:
            buffer[key].append(encoded[key])
    out: dict[str, Any] = {key: torch.stack(buffer[key]) for key in feature_keys}
    if layers is not None:
        index = [int(layer) - 1 for layer in layers]
        out = {key: value[:, index, :] for key, value in out.items()}
    for key in per_file:
        dtype = torch.bool if key in ("trim_fallback", "too_short") else torch.long
        out[key] = torch.tensor(buffer[key], dtype=dtype)
    out.update(labels=torch.tensor(part["label"].to_numpy(), dtype=torch.long),
               audio_ids=[str(a) for a in part["audio_id"]],
               system_ids=[str(s) for s in part["system_id"]],
               layers=None if layers is None else [int(layer) for layer in layers],
               audio_backends=sorted(backends))
    return out


def reextraction_check(items: pd.DataFrame, feature_dir: str, split: str, *, model,
                       extraction: str, condition: str, device: str | torch.device,
                       n: int = cfg.REEXTRACTION_FILES, seed=cfg.REEXTRACTION_SEED,
                       tolerance: float = verify.FEATURE_TOLERANCE) -> pd.DataFrame:
    """Re-extract ``n`` files of a split and compare them with the stored features.

    ``items`` is the file list of the split in the order of the shards. The
    feature tensors pass up to ``tolerance``; lengths, frame counts, flags,
    labels and ids must have the stored values (an integer or boolean tensor
    is compared in the type of the stored one, so only its values count).
    When the shards store some layers only, those layers are compared.

    One row per key (``verify.compare_reextraction``), with split, n_files,
    layers ("all" or the stored layers) and the audio libraries of both sides.
    """
    positions = sample_positions(len(items), n, seed)
    stored = stored_files(feature_dir, split, positions, extraction)
    new = reextract(items, positions, model=model, extraction=extraction, condition=condition,
                    device=device, layers=stored["layers"])
    exact = shards.PER_FILE_KEYS[extraction] + shards.LIST_KEYS
    for key in shards.PER_FILE_KEYS[extraction]:
        if stored[key].dtype != new[key].dtype and not stored[key].dtype.is_floating_point:
            new[key] = new[key].to(stored[key].dtype)
    table = verify.compare_reextraction(new, stored, shards.FEATURE_KEYS[extraction], exact,
                                        tolerance)
    layers = "all" if stored["layers"] is None else " ".join(str(k) for k in stored["layers"])
    table.insert(0, "split", split)
    table.insert(1, "n_files", len(positions))
    table.insert(2, "layers", layers)
    return table.assign(audio_backends=" ".join(new["audio_backends"]),
                        stored_audio_backends=" ".join(stored["audio_backends"]))


# Checks around the extraction of another encoder ------------------------------

def file_list_check(items: pd.DataFrame, reference: pd.DataFrame, split: str) -> pd.DataFrame:
    """The file list of a split against the files of the reference shards.

    ``reference`` is ``shards.shard_lengths`` of the reference features (all
    files in shard order). One row: the number of files on both sides, and
    whether the ids are the same in the same order and the labels are equal.
    """
    stored = reference[reference["split"] == split]
    ids = [str(a) for a in items["audio_id"]]
    same_ids = ids == stored["audio_id"].astype(str).tolist()
    same_labels = bool(same_ids and np.array_equal(items["label"].to_numpy().astype(int),
                                                   stored["label"].to_numpy().astype(int)))
    return pd.DataFrame([{"split": split, "n_files": len(items), "n_reference": len(stored),
                          "same_ids_in_order": same_ids, "same_labels": same_labels,
                          "ok": bool(same_ids and same_labels and len(items) > 0)}])


def short_check(items: pd.DataFrame, reference: pd.DataFrame, split: str, *, model,
                condition: str, device: str | torch.device, n: int = cfg.SHORT_CHECK_FILES,
                seed=cfg.REEXTRACTION_SEED) -> pd.DataFrame:
    """Encode ``n`` files of a split unpadded and check what the full extraction relies on.

    The files are those of ``sample_positions`` (the same positions of a
    split in every condition) and go through ``prepare_file`` and
    ``encode_unpadded``, one at a time. ``reference`` is
    ``shards.shard_lengths`` of the reference features of the same condition.

    One row per file: the number of frames against the output length of the
    convolutional encoder for the encoder input, the shape and finiteness of
    the pooled features, and the number of real samples and the two flags
    against the reference. ``seconds`` is the time of loading, preparing and
    encoding the file; one file is encoded before the timing starts.
    """
    top_db = cfg.trim_top_db(condition)
    n_layers = int(model.config.num_hidden_layers)
    shape = (n_layers, int(model.config.hidden_size))
    kernels, strides = encoders.conv_layers(model)
    stored = reference[reference["split"] == split].reset_index(drop=True)
    positions = sample_positions(len(items), min(n, len(items)), seed)
    cuda = torch.device(device).type == "cuda"
    encode_unpadded(model, prepare_file(items.iloc[positions[0]]["path"], top_db, "unpadded")["x"],
                    n_layers, device)
    rows = []
    for position in positions:
        item = items.iloc[position]
        start = time.perf_counter()
        prepared = prepare_file(item["path"], top_db, "unpadded")
        encoded = encode_unpadded(model, prepared["x"], n_layers, device)
        if cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        conv_frames = encoders.conv_output_length(len(prepared["x"]), kernels, strides)
        known = position < len(stored) and str(stored.loc[position, "audio_id"]) == str(item["audio_id"])
        row = {"split": split, "condition": condition, "position": int(position),
               "audio_id": str(item["audio_id"]), "n_input": int(len(prepared["x"])),
               "n_frames": int(encoded["n_frames"]), "conv_frames": int(conv_frames),
               "n_real": prepared["n_real"],
               "reference_n_real": int(stored.loc[position, "n_real"]) if known else -1,
               "trim_fallback": prepared["trim_fallback"],
               "reference_trim_fallback": bool(stored.loc[position, "trim_fallback"]) if known else None,
               "too_short": prepared["too_short"],
               "reference_too_short": bool(stored.loc[position, "too_short"]) if known else None,
               "shape_ok": tuple(encoded["mean"].shape) == shape == tuple(encoded["std"].shape),
               "finite": bool(torch.isfinite(encoded["mean"]).all() and torch.isfinite(encoded["std"]).all()),
               "seconds": seconds}
        row["ok"] = bool(known and row["n_frames"] == row["conv_frames"] and row["shape_ok"]
                         and row["finite"] and row["n_real"] == row["reference_n_real"]
                         and row["trim_fallback"] == row["reference_trim_fallback"]
                         and row["too_short"] == row["reference_too_short"])
        rows.append(row)
    return pd.DataFrame(rows)


def extraction_estimate(timing: pd.DataFrame, sizes: Mapping[str, int], n_layers: int,
                        hidden_size: int, shard_size: int = cfg.SHARD_SIZE) -> pd.DataFrame:
    """Expected time and size of the unpadded extraction, per condition.

    ``timing`` has the columns condition, split and seconds (``short_check``);
    ``sizes`` is the number of files per split. The time of a split is the
    mean seconds per timed file times its files. The size is that of the two
    float32 tensors (mean and standard deviation, layers x hidden size) of
    every file, in GB of 10^9 bytes; ids, flags and the meta of a shard are
    not counted.
    """
    rows = []
    for condition, part in timing.groupby("condition", sort=False):
        per_file = part.groupby("split")["seconds"].mean()
        missing = sorted(set(sizes) - set(per_file.index))
        if missing:
            raise KeyError(f"{condition}: no timed file of {missing}")
        seconds = float(sum(per_file[split] * n for split, n in sizes.items()))
        n_files = int(sum(sizes.values()))
        rows.append({"condition": condition, "n_timed_files": len(part), "n_files": n_files,
                     "n_shards": int(sum(shards.n_shards(n, shard_size) for n in sizes.values())),
                     "seconds_per_file": seconds / n_files, "hours": seconds / 3600,
                     "feature_gb": n_files * n_layers * hidden_size * 2 * 4 / 1e9})
    return pd.DataFrame(rows)


LENGTH_KEYS = ("label", "n_real", "trim_fallback", "too_short")


def lengths_check(new: pd.DataFrame, reference: pd.DataFrame,
                  splits: Sequence[str] = cfg.SPLITS) -> pd.DataFrame:
    """The files of extracted shards against those of the reference shards.

    Both tables are ``shards.shard_lengths``. One row per split: whether the
    ids are the same in the same order, and the number of files whose label,
    number of real samples, fallback flag or too-short flag differs (-1 when
    the ids differ). With equal labels and flags the fallback counts per
    split and class are equal too.
    """
    rows = []
    for split in splits:
        a = new[new["split"] == split].reset_index(drop=True)
        b = reference[reference["split"] == split].reset_index(drop=True)
        same_ids = len(a) > 0 and a["audio_id"].astype(str).tolist() == b["audio_id"].astype(str).tolist()
        row = {"split": split, "n_files": len(a), "n_reference": len(b), "same_ids_in_order": bool(same_ids)}
        for key in LENGTH_KEYS:
            row[f"n_different_{key}"] = int((a[key].to_numpy() != b[key].to_numpy()).sum()) if same_ids else -1
        row["n_fallback"] = int(a["trim_fallback"].sum())
        row["ok"] = bool(same_ids and all(row[f"n_different_{key}"] == 0 for key in LENGTH_KEYS))
        rows.append(row)
    return pd.DataFrame(rows)


META_COLUMNS = ("encoder", "hf_id", "revision", "model_class", "attn_implementation", "dtype",
                "extraction", "condition", "git_commit", "gpu_name")
META_VERSIONS = ("python", "torch", "torchaudio", "torchcodec", "transformers", "librosa", "numpy",
                 "soundfile")


def shard_meta_table(feature_dir: str, splits: Sequence[str] = cfg.SPLITS,
                     extraction: str = "unpadded") -> pd.DataFrame:
    """What the shards of a feature folder record about their extraction.

    One row per distinct combination of encoder, checkpoint revision, model
    class, attention implementation, condition, commit, GPU, library versions
    and audio libraries, with the number of shards and files that have it.
    """
    rows = []
    for split in splits:
        for shard in shards.split_shards(feature_dir, split, extraction):
            meta = shard["meta"]
            row = {key: str(meta.get(key)) for key in META_COLUMNS}
            row.update({key: str(meta.get("versions", {}).get(key)) for key in META_VERSIONS})
            row["audio_backends"] = " ".join(meta.get("audio_backends") or [])
            row["n_files"] = shard["hi"] - shard["lo"]
            rows.append(row)
    table = pd.DataFrame(rows)
    keys = [column for column in table.columns if column != "n_files"]
    grouped = table.groupby(keys, sort=False, dropna=False)["n_files"].agg(n_shards="size", n_files="sum")
    return grouped.reset_index()


def shard_meta_check(table: pd.DataFrame, encoder: str, condition: str,
                     extraction: str = "unpadded",
                     versions: Mapping[str, str] | None = None) -> pd.DataFrame:
    """``shard_meta_table`` with ``ok``: every shard records the registry entry
    of ``encoder``, the condition, a commit without uncommitted changes, and
    the library versions of this runtime (``versions``; default
    ``provenance.versions()``), so that shards written in several sessions
    were written with the same software."""
    entry = cfg.encoder(encoder)
    versions = provenance.versions() if versions is None else versions
    wanted = {"encoder": encoder, "hf_id": entry["hf_id"], "revision": entry["revision"],
              "model_class": entry["model_class"],
              "attn_implementation": entry["attn_implementation"], "extraction": extraction,
              "condition": condition}
    wanted.update({key: versions[key] for key in META_VERSIONS})
    ok = pd.Series(True, index=table.index)
    for key, value in wanted.items():
        ok &= table[key] == str(value)
    ok &= ~table["git_commit"].str.endswith("-dirty") & (table["git_commit"] != "unknown")
    return table.assign(ok=ok)
