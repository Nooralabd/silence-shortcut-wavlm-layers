"""The supplementary condition trimmed by voice activity detection: the rule,
its configuration, the shards, the runs and the tables."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

import world
from models import tiny_entry, tiny_model
from signals import voice
from ssl_silence import analysis, bootstrap
from ssl_silence import config as cfg
from ssl_silence import extraction as X
from ssl_silence import shards, training, trimming as T, vad, verify

RULE = {"frame_ms": 10, "aggressiveness": 3, "run_frames": 3}
FRAME = 160


def clip(lead: int, body: int, trail: int, floor: float = 1e-3, seed: int = 0) -> torch.Tensor:
    """(1, N) float32: a voiced segment between two stretches of low noise."""
    y = np.concatenate([np.zeros(lead), voice(body, noise=0.02, seed=seed), np.zeros(trail)])
    y = y + np.random.default_rng(seed + 100).normal(0.0, floor, y.size)
    return torch.from_numpy(y.astype(np.float32)).unsqueeze(0)


# Configuration -----------------------------------------------------------------------------

def test_the_condition_is_supplementary_and_changes_no_run_of_the_experiments():
    assert cfg.vad_conditions() == {"vad": RULE} and cfg.vad_rule("vad") == RULE
    # The conditions of the analysis plan are unchanged, and so are its runs.
    assert list(cfg.conditions()) == ["clean", "trim40", "trim20", "trim15"]
    assert cfg.known_conditions() == ("clean", "trim40", "trim20", "trim15", "vad")
    assert [len(training.plan_runs(encoder, "unpadded")) for encoder in ("wavlm-base", "hubert-base")] \
        == [480, 480]
    assert len(training.plan_runs("wavlm-base", "padded")) == 10
    assert not any("vad" in p.test_conditions or p.spec.condition == "vad"
                   for encoder in ("wavlm-base", "hubert-base")
                   for p in training.plan_runs(encoder, "unpadded"))
    assert analysis.trained_conditions("wavlm-base") == ("trim40", "trim20", "trim15")
    # It has no threshold: it is never read as an untrimmed condition.
    with pytest.raises(KeyError, match="voice activity detection"):
        cfg.trim_top_db("vad")
    assert cfg.vad_rule("trim40") is None and cfg.vad_rule("clean") is None
    assert (cfg.top_db_text("vad"), cfg.top_db_text("clean"), cfg.top_db_text("trim40")) == ("", "", "40")
    for unknown in (cfg.vad_rule, cfg.check_condition, cfg.top_db_text):
        with pytest.raises(KeyError, match="unknown condition"):
            unknown("trim30")
    assert X.trim_rule("vad") == (None, RULE) and X.trim_rule("trim20") == (20, None)
    assert X.trim_rule("clean") == (None, None)
    # Its folders: the features with the other conditions, the runs and the tables apart.
    root = os.path.join("some", "root")
    assert cfg.feature_dir("wavlm-base", "unpadded", "vad", root) == os.path.join(
        root, "features", "wavlm-base", "unpadded", "vad")
    assert cfg.vad_runs_dir("wavlm-base", "unpadded", root) == os.path.join(
        root, "runs_vad", "wavlm-base", "unpadded")
    assert cfg.vad_tables_dir("wavlm-base", root) == os.path.join(root, "tables", "wavlm-base", "vad")
    assert not cfg.vad_runs_dir("wavlm-base", "unpadded", root).startswith(
        cfg.runs_dir("wavlm-base", "unpadded", root))


def test_the_sixty_runs_of_the_condition():
    planned = vad.planned_runs()
    assert len(planned) == 60 and vad.layers(planned) == (3, 4, 5) == vad.layers()
    clean, new = vad.runs_of(planned, "clean"), vad.runs_of(planned, "vad")
    assert planned == clean + new and len(clean) == len(new) == 30      # the clean models come first
    for runs, condition, other in ((clean, "clean", "vad"), (new, "vad", "clean")):
        assert {(p.spec.layer, p.spec.pooling, p.spec.seed) for p in runs} == {
            (layer, pooling, seed) for layer in (3, 4, 5) for pooling in ("mean", "stats")
            for seed in cfg.SEEDS}
        for p in runs:
            assert p.test_conditions == (condition, other)
            assert p.spec.selection_set == f"dev@{condition}"            # the dev set of the training condition
            assert p.score_set_names == (f"eval@{condition}", f"itw@{condition}", f"dev@{other}",
                                         f"eval@{other}", f"itw@{other}")
    assert planned[0].spec.run_id == "wavlm-base_unpadded_clean_L03_mean_s42"
    assert planned[-1].spec.run_id == "wavlm-base_unpadded_vad_L05_stats_s46"
    # The clean models have the run ids of existing runs: that is why they have their own folder.
    existing = {p.spec.run_id for p in training.plan_runs("wavlm-base", "unpadded")}
    assert {p.spec.run_id for p in clean} <= existing
    assert not {p.spec.run_id for p in new} & existing


# The rule ------------------------------------------------------------------------------------

def test_pcm_conversion_gives_the_integers_of_16_bit_audio_back():
    integers = np.array([-32768, -1, 0, 1, 12345, 32767], dtype=np.int16)
    assert np.array_equal(T.to_pcm16(integers / 32768.0), integers)
    assert np.array_equal(T.to_pcm16(np.float32(integers / 32768.0)), integers)
    assert T.to_pcm16(np.array([1.0, -1.5, 2.0])).tolist() == [32767, -32768, 32767]     # clipped
    assert T.to_pcm16(np.zeros(3)).dtype == np.dtype("<i2")


def test_bounds_are_the_first_and_the_last_run_of_speech_frames():
    def bounds(text, run=3):
        return T.bounds_from_speech(np.array([c == "s" for c in text]), run, FRAME)

    assert bounds("...sss....") == (3 * FRAME, 6 * FRAME)
    assert bounds("s..ssss..ss.sss..s") == (3 * FRAME, 15 * FRAME)      # single and double frames at the edges do not count
    assert bounds("sssss") == (0, 5 * FRAME)
    assert bounds("ss.ss.ss") == (0, 0) and bounds("") == (0, 0) and bounds("ss") == (0, 0)
    assert bounds(".s.ss.", run=1) == (FRAME, 5 * FRAME)                 # with a run of one: first and last speech frame
    assert bounds("ss.sss.ss", run=2) == (0, 9 * FRAME)
    # Pauses inside are kept: the interval is one piece from the first to the last run.
    assert bounds("sss......sss") == (0, 12 * FRAME)
    with pytest.raises(ValueError):
        bounds("sss", run=0)


def test_the_detector_judges_whole_frames_of_one_file_and_nothing_else():
    wav = clip(16_000, 24_000, 16_005)
    speech = T.vad_speech_frames(wav[0].numpy(), 10, 3)
    assert speech.dtype == bool and speech.shape == (350,)               # the last 5 samples are not judged
    assert np.array_equal(speech, T.vad_speech_frames(wav[0].numpy(), 10, 3))       # the same file, the same decisions
    # The decisions of a file do not depend on the files judged before it.
    T.vad_speech_frames(np.random.default_rng(0).normal(0, 0.3, 48_000), 10, 3)
    assert np.array_equal(speech, T.vad_speech_frames(wav[0].numpy(), 10, 3))
    assert not speech[:95].any() and speech[105:240].all() and not speech[275:].any()
    assert T.vad_speech_frames(np.zeros(159), 10, 3).shape == (0,)
    for bad in (dict(frame_ms=15, aggressiveness=3), dict(frame_ms=10, aggressiveness=4)):
        with pytest.raises(ValueError):
            T.vad_speech_frames(wav[0].numpy(), **bad)
    # A longer frame and a lower aggressiveness are other rules.
    assert T.vad_speech_frames(wav[0].numpy(), 30, 3).shape == (116,)
    assert T.vad_speech_frames(wav[0].numpy(), 10, 0).sum() >= speech.sum()


@pytest.mark.parametrize("lead, body, trail", [(16_000, 24_000, 16_000), (12_345, 20_000, 7_777),
                                               (0, 30_000, 0), (40_000, 16_000, 0)])
def test_leading_and_trailing_non_speech_is_removed(lead, body, trail):
    wav = clip(lead, body, trail)
    n = wav.shape[1]
    start, end, fallback = T.vad_interval(wav, RULE)
    assert not fallback and start % FRAME == 0 and end % FRAME == 0
    # The kept audio starts in the frame in which the voice starts (or up to two frames later) and
    # ends after the voice, within the hangover of the detector.
    assert lead - FRAME < start <= lead + 2 * FRAME
    assert min(n - n % FRAME, lead + body - 2 * FRAME) <= end <= min(n, lead + body + 20 * FRAME)
    out, flag, a, b = T.apply_vad(wav, RULE)
    assert (a, b, flag) == (start, end, False) and torch.equal(out, wav[:, start:end])
    # Only the edges: the interval is the same when a pause is put inside the voice.
    paused = torch.cat([wav[:, :lead + body // 2], wav[:, :4_800] * 0 + 1e-3 * torch.randn(1, 4_800),
                        wav[:, lead + body // 2:]], dim=1) if lead >= 4_800 else None
    if paused is not None:
        start_p, end_p, fallback_p = T.vad_interval(paused, RULE)
        assert not fallback_p and start_p == start and end - 3 * FRAME <= end_p - 4_800 <= end + 3 * FRAME


def test_the_fallback_is_that_of_the_energy_rule():
    zeros = torch.zeros(1, 20_000)
    assert T.vad_interval(zeros, RULE) == (0, 20_000, True)
    out, fallback, start, end = T.apply_vad(zeros, RULE)
    assert fallback and (start, end) == (0, 20_000) and out is not None and out.shape == (1, 20_000)
    noise = torch.from_numpy(np.random.default_rng(0).normal(0, 1e-3, 32_000).astype(np.float32))[None]
    assert T.vad_interval(noise, RULE) == (0, 32_000, True)             # no speech: the untrimmed audio
    short = clip(8_000, 4_000, 8_000)                                    # 0.25 s of voice: fewer than 0.5 s left
    speech = T.vad_speech_frames(short[0].numpy(), 10, 3)
    start, end = T.bounds_from_speech(speech, 3, FRAME)
    assert 0 < end - start < cfg.MIN_TRIMMED_SAMPLES
    assert T.vad_interval(short, RULE) == (0, 20_000, True)
    assert T.vad_interval(torch.zeros(1, 0), RULE) == (0, 0, True)
    assert T.resolve_trim(20_000, 0, 7_999, False) == (0, 20_000, True)
    assert T.resolve_trim(20_000, 0, 8_000, False) == (0, 8_000, False)  # exactly 0.5 s is kept
    with pytest.raises(ValueError):
        T.vad_interval(torch.zeros(20_000), RULE)


# Files and shards ------------------------------------------------------------------------------

def write_items(folder) -> pd.DataFrame:
    """Nine files in the layout of ``datasets.items``: voices between low noise, one of them
    longer than the window, a file of zeros and a file of noise."""
    folder.mkdir(parents=True, exist_ok=True)
    waves = {f"voice_{k}": clip(8_000 + 1_000 * k, 16_000 + 2_000 * k, 4_000 + 500 * k, seed=k)
             for k in range(6)}
    waves["long_voice"] = clip(9_600, 80_000, 6_400, seed=7)
    waves["zeros"] = torch.zeros(1, 20_000)
    waves["noise"] = torch.from_numpy(np.random.default_rng(1).normal(0, 1e-3, 24_000).astype(np.float32))[None]
    rows = []
    for k, (name, wav) in enumerate(waves.items()):
        path = folder / f"{name}.flac"
        sf.write(str(path), wav[0].numpy(), 16_000, subtype="PCM_16")
        rows.append({"path": str(path), "audio_id": name, "label": k % 2,
                     "system_id": "-" if k % 2 == 0 else "A07", "speaker": f"spk{k % 3}"})
    return pd.DataFrame(rows)


def test_a_file_is_prepared_with_the_rule_and_its_interval(tmp_path):
    items = write_items(tmp_path).set_index("audio_id")
    prepared = X.prepare_file(items.loc["voice_2", "path"], None, "unpadded", RULE)
    wav = X.to_mono_16k(*X.load_audio(items.loc["voice_2", "path"])[:2])
    assert prepared["n_samples"] == wav.shape[1] == 10_000 + 20_000 + 5_000
    start, end, fallback = T.vad_interval(wav, RULE)
    assert (prepared["kept_start"], prepared["kept_end"], prepared["trim_fallback"]) == (start, end, False)
    assert 10_000 - FRAME < start <= 10_000 + 2 * FRAME and prepared["n_real"] == end - start
    assert prepared["x"].shape == (end - start,) and not prepared["too_short"]
    assert torch.allclose(prepared["x"], wav[0, start:end] / (wav[0, start:end].abs().max() + 1e-8))
    # Longer than the window after trimming: the centre of the kept audio.
    long = X.prepare_file(items.loc["long_voice", "path"], None, "unpadded", RULE)
    assert long["kept_end"] - long["kept_start"] > cfg.WINDOW == long["n_real"] == len(long["x"])
    for name in ("zeros", "noise"):
        whole = X.prepare_file(items.loc[name, "path"], None, "unpadded", RULE)
        assert whole["trim_fallback"] and (whole["kept_start"], whole["kept_end"]) == (0, whole["n_samples"])
        assert whole["n_real"] == whole["n_samples"]
    # Without the rule nothing about an interval is returned, and the two rules exclude each other.
    assert "kept_start" not in X.prepare_file(items.loc["voice_2", "path"], 40, "unpadded")
    with pytest.raises(ValueError, match="not both"):
        X.prepare_file(items.loc["voice_2", "path"], 40, "unpadded", RULE)


def test_the_shards_of_the_condition_store_the_kept_interval(tmp_path):
    items = write_items(tmp_path / "audio")
    model, entry = tiny_model("wavlm", n_layers=3), tiny_entry("wavlm", n_layers=3)
    out = str(tmp_path / "features")
    kwargs = dict(encoder="wavlm-base", extraction="unpadded", condition="vad", shard_size=4,
                  device="cpu", model=model, entry=entry, progress=False)
    first = X.extract_shards(items, out, "dev", max_new_shards=1, **kwargs)
    stamp = os.path.getmtime(first[0])
    paths = X.extract_shards(items, out, "dev", **kwargs)
    assert len(paths) == 3 and os.path.getmtime(paths[0]) == stamp       # resumable
    loaded = shards.split_shards(out, "dev", mmap=False)
    for shard in loaded:
        n = shard["hi"] - shard["lo"]
        for key in shards.KEPT_KEYS:
            assert shard[key].shape == (n,) and shard[key].dtype == torch.long
        assert shard["mean"].shape == (n, 3, 32)
        meta = shard["meta"]
        assert meta["condition"] == "vad" and meta["trim_top_db"] is None
        assert meta["vad"] == {**RULE, "package": "webrtcvad-wheels", "package_version": "2.0.14.post1"}
    kept = shards.kept_intervals(out, splits=("dev",))
    assert kept["audio_id"].tolist() == items["audio_id"].tolist()
    assert list(kept.columns) == ["split", "audio_id", "system_id", "label", "n_samples", "kept_start",
                                  "kept_end", "n_real", "trim_fallback"]
    for row, (_, item) in zip(kept.itertuples(), items.iterrows()):
        prepared = X.prepare_file(item["path"], None, "unpadded", RULE)
        assert (row.n_samples, row.kept_start, row.kept_end, row.n_real, row.trim_fallback) == (
            prepared["n_samples"], prepared["kept_start"], prepared["kept_end"], prepared["n_real"],
            prepared["trim_fallback"])
    assert kept.set_index("audio_id")["trim_fallback"].to_dict() == {
        **{f"voice_{k}": False for k in range(6)}, "long_voice": False, "zeros": True, "noise": True}
    assert not vad.interval_problems(kept, RULE).any(axis=None)
    # The features are those of the kept audio, encoded alone.
    item = items.iloc[1]
    prepared = X.prepare_file(item["path"], None, "unpadded", RULE)
    direct = X.encode_unpadded(model, prepared["x"], 3, "cpu")
    assert torch.equal(loaded[0]["mean"][1], direct["mean"]) and torch.equal(loaded[0]["std"][1], direct["std"])
    # A shard of a condition of the experiments stores no interval, and such shards are refused here.
    energy = str(tmp_path / "energy")
    X.extract_shards(items, energy, "dev", **{**kwargs, "condition": "trim40"})
    assert not set(shards.KEPT_KEYS) & set(shards.read_shard(shards.shard_path(energy, "dev", 0)))
    assert "vad" not in shards.read_shard(shards.shard_path(energy, "dev", 0))["meta"]
    with pytest.raises(KeyError, match="do not store"):
        shards.kept_intervals(energy, splits=("dev",))
    # The re-extraction of the same files gives the stored features.
    again = X.reextraction_check(items, out, "dev", model=model, extraction="unpadded", condition="vad",
                                 device="cpu", n=5)
    assert again["ok"].all()


def test_a_shard_holds_all_three_keys_of_the_interval_or_none(tmp_path):
    n = 2
    payload = {"mean": torch.zeros(n, 1, 4), "std": torch.zeros(n, 1, 4), "n_real": torch.ones(n, dtype=torch.long),
               "n_frames": torch.ones(n, dtype=torch.long), "trim_fallback": torch.zeros(n, dtype=torch.bool),
               "too_short": torch.zeros(n, dtype=torch.bool), "labels": torch.zeros(n, dtype=torch.long),
               "audio_ids": ["a", "b"], "system_ids": ["-", "-"], "lo": 0, "hi": n, "n_total": n,
               "shard_size": 2, "meta": {}}
    path = str(tmp_path / "dev_shard_0000.pt")
    with pytest.raises(KeyError, match="without the others"):
        shards.write_shard(path, {**payload, "kept_start": torch.zeros(n, dtype=torch.long)}, "unpadded")
    whole = {**payload, **{key: torch.zeros(n, dtype=torch.long) for key in shards.KEPT_KEYS}}
    with pytest.raises(ValueError, match="kept_end holds 1 files"):
        shards.write_shard(path, {**whole, "kept_end": torch.zeros(1, dtype=torch.long)}, "unpadded")
    shards.write_shard(path, whole, "unpadded")
    assert set(shards.KEPT_KEYS) <= set(shards.read_shard(path))


def test_the_short_check_times_the_files_and_checks_their_intervals(tmp_path):
    items = write_items(tmp_path)
    model = tiny_model("wavlm", n_layers=3)
    table = vad.short_check(items, "dev", model=model, device="cpu", n=6)
    assert len(table) == 6 and table["ok"].all() and (table["seconds"] > 0).all()
    assert table["position"].tolist() == X.sample_positions(len(items), 6)
    assert (table["n_frames"] == table["conv_frames"]).all() and table["interval_ok"].all()
    assert set(table["condition"]) == {"vad"} and set(table["split"]) == {"dev"}
    with pytest.raises(ValueError, match="not trimmed by voice activity detection"):
        vad.short_check(items, "dev", model=model, device="cpu", n=2, condition="trim40")
    estimate = X.extraction_estimate(table, {"dev": 1_000}, 12, 768)
    assert estimate["condition"].tolist() == ["vad"] and estimate["n_files"].tolist() == [1_000]


def kept_table(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["split", "audio_id", "system_id", "label", "n_samples", "kept_start",
                                       "kept_end", "n_real", "trim_fallback"])


def test_the_problems_of_an_interval_are_named():
    good = [("dev", "a", "-", 0, 100_000, 16_000, 96_000, 64_000, False),      # longer than the window
            ("dev", "b", "A07", 1, 20_000, 1_600, 9_600, 8_000, False),         # exactly 0.5 s kept
            ("dev", "c", "-", 0, 7_001, 0, 7_001, 7_001, True)]                 # fallback: the whole file
    assert not vad.interval_problems(kept_table(good), RULE).any(axis=None)
    bad = {"outside": ("dev", "a", "-", 0, 20_000, 1_600, 20_160, 18_560, False),
           "off_frame": ("dev", "a", "-", 0, 20_000, 1_601, 17_600, 15_999, False),
           "short_without_fallback": ("dev", "a", "-", 0, 20_000, 1_600, 9_440, 7_840, False),
           "fallback_not_whole": ("dev", "a", "-", 0, 20_000, 160, 20_000, 19_840, True),
           "n_real_differs": ("dev", "a", "-", 0, 20_000, 1_600, 17_600, 16_001, False)}
    for name, row in bad.items():
        problems = vad.interval_problems(kept_table([row]), RULE)
        assert problems.columns[problems.iloc[0].to_numpy()].tolist() == [name], name


def test_the_shards_are_checked_against_the_untrimmed_shards_and_the_rule():
    rows = [("train", "a", "-", 0, 30_000, 1_600, 25_600, 24_000, False),
            ("train", "b", "A01", 1, 20_000, 0, 20_000, 20_000, True),
            ("dev", "c", "-", 0, 90_000, 3_200, 80_000, 64_000, False)]
    kept = kept_table(rows)
    reference = pd.DataFrame({"split": ["train", "train", "dev"], "audio_id": ["a", "b", "c"],
                              "label": [0, 1, 0], "n_real": [30_000, 20_000, 64_000],
                              "trim_fallback": False, "too_short": False})
    table = vad.shards_check(kept, reference, RULE, splits=("train", "dev"))
    assert table["ok"].all() and table["n_files"].tolist() == [2, 1] and table["n_fallback"].tolist() == [1, 0]
    assert table["n_longer_than_untrimmed"].tolist() == [0, 0]
    # Another order, another label, a file that is longer than its untrimmed file, a wrong interval.
    assert not vad.shards_check(kept, reference.iloc[[1, 0, 2]], RULE, ("train", "dev"))["ok"].iloc[0]
    assert vad.shards_check(kept, reference.assign(label=[0, 0, 0]), RULE, ("train", "dev"))["ok"].tolist() \
        == [False, True]
    longer = vad.shards_check(kept, reference.assign(n_real=[23_999, 20_000, 64_000]), RULE, ("train", "dev"))
    assert longer["n_longer_than_untrimmed"].tolist() == [1, 0] and longer["ok"].tolist() == [False, True]
    wrong = kept.copy()
    wrong.loc[2, "kept_start"] = 3_201
    assert vad.shards_check(wrong, reference, RULE, ("train", "dev"))["n_off_frame"].tolist() == [0, 1]
    assert not vad.shards_check(kept.iloc[:0], reference, RULE, ("train",))["ok"].any()   # nothing extracted
    assert vad.shards_check(kept, reference, splits=("train", "dev"))["ok"].all()         # the rule of the configuration


def test_the_removed_amount_per_file_and_per_class():
    rows = [("eval", "a", "-", 0, 48_000, 8_000, 40_000, 32_000, False),       # 0.5 s lead, 0.5 s trail
            ("eval", "b", "-", 0, 32_000, 16_000, 32_000, 16_000, False),      # 1.0 s lead
            ("eval", "c", "-", 0, 16_000, 0, 16_000, 16_000, True),            # fallback: nothing removed
            ("eval", "d", "A07", 1, 16_000, 1_600, 16_000, 14_400, False),
            ("itw", "e", "itw", 1, 80_000, 0, 72_000, 64_000, False)]
    per_file = vad.removed_per_file(kept_table(rows))
    assert per_file["removed_lead_s"].tolist() == [0.5, 1.0, 0.0, 0.1, 0.0]
    assert per_file["removed_trail_s"].tolist() == [0.5, 0.0, 0.0, 0.0, 0.5]
    assert per_file["removed_total_s"].tolist() == [1.0, 1.0, 0.0, 0.1, 0.5]
    assert per_file["kept_s"].tolist() == [2.0, 1.0, 1.0, 0.9, 4.5] and per_file["orig_s"].tolist() == [3, 2, 1, 1, 5]
    assert per_file["fallback"].tolist() == [False, False, True, False, False]
    summary = vad.summarise_removed(per_file)
    assert summary[["split", "label", "n", "n_fallback"]].to_numpy().tolist() == [
        ["eval", 0, 3, 1], ["eval", 1, 1, 0], ["itw", 1, 1, 0]]
    first = summary.iloc[0]
    assert first["removed_total_median_s"] == 1.0 and first["removed_total_mean_s"] == pytest.approx(2 / 3)
    assert first["removed_lead_median_s"] == 0.5 and first["removed_trail_mean_s"] == pytest.approx(0.5 / 3)
    assert first["removed_frac_median"] == pytest.approx(1 / 3) and set(summary["condition"]) == {"vad"}
    assert summary.iloc[2]["removed_frac_median"] == pytest.approx(0.1)


# Runs and tables ---------------------------------------------------------------------------------

SEEDS = (42, 43, 44)


def scores(condition: str, test_condition: str, pooling: str, layer: int, seed: int, split: str,
           shift: float = 0.0) -> np.ndarray:
    key = [int(condition == "vad"), int(test_condition == "vad"), int(pooling == "stats"), layer, seed,
           list(world.SPLITS).index(split)]
    y = world.labels_of(split)
    separation = 0.5 + 0.3 * layer - (0.6 if condition != test_condition else 0.0) - 0.4 * (condition == "vad")
    logits = np.random.default_rng(key).normal(0.0, 1.0, y.size) + separation * y
    return (1.0 / (1.0 + np.exp(-logits)) + shift).astype(np.float32)


def write_runs(folder: str, planned, shift: float = 0.0, best_epoch: int = 5, sets_of=None) -> None:
    for run in planned:
        spec = run.spec
        names = (spec.selection_set,) + run.score_set_names if sets_of is None else sets_of(run)
        sets = {name: training.ScoreSet(
            scores(spec.condition, name.partition("@")[2], spec.pooling, spec.layer, spec.seed,
                   name.partition("@")[0], shift), world.labels_of(name.partition("@")[0]),
            world.ids_of(name.partition("@")[0])) for name in names}
        training.record_run(spec, sets, runs_dir=folder, overwrite=True,
                            fit_info={"n_train": 0, "in_dim": 0, "best_dev_eer_roc_corners": "nan",
                                      "best_epoch": best_epoch, "epochs_run": 50, "device": "cpu",
                                      "gpu_name": "cpu"})


def existing_runs(folder: str, shift: float = 0.0, best_epoch: int = 5) -> str:
    """The clean runs of the same layers as they exist: scored on clean audio only."""
    clean = vad.runs_of(vad.planned_runs(SEEDS), "clean")
    write_runs(folder, clean, shift, best_epoch,
               sets_of=lambda run: ("dev@clean", "eval@clean", "itw@clean"))
    return folder


def test_the_clean_models_trained_again_are_compared_with_the_existing_runs(tmp_path):
    planned = vad.planned_runs(SEEDS)
    runs, base = str(tmp_path / "runs_vad"), existing_runs(str(tmp_path / "runs"))
    write_runs(runs, planned)
    table = vad.clean_scores_check(planned, runs, base)
    # 18 clean runs, three score sets each on clean audio; the runs trained on the new condition are not compared.
    assert len(table) == 18 * 3 and table["ok"].all() and table["exactly_zero"].all()
    assert set(table["score_set"]) == {"dev@clean", "eval@clean", "itw@clean"}
    assert not table["run_id"].str.contains("_vad_").any() and (table["max_abs_diff"] == 0).all()
    assert table["n"].tolist()[:3] == [world.SPLITS[split] for split in ("dev", "eval", "itw")]
    verify.require_ok(table, "clean scores")

    # A difference at the tolerance passes, a larger one fails; so does another selected epoch.
    for shift, ok in ((0.0000625, True), (0.00025, False)):
        other = existing_runs(str(tmp_path / f"runs_{ok}"), shift=shift)
        table = vad.clean_scores_check(planned, runs, other)
        assert table["ok"].all() == ok and not table["exactly_zero"].any()
        assert table["max_abs_diff"].max() == pytest.approx(shift, rel=1e-3)
    epoch = vad.clean_scores_check(planned, runs, existing_runs(str(tmp_path / "runs_epoch"), best_epoch=10))
    assert not epoch["ok"].any() and epoch["within_tolerance"].all() and not epoch["best_epoch_equal"].any()
    assert set(epoch["best_epoch"]) == {5} and set(epoch["existing_best_epoch"]) == {10}

    # A clean run that was not trained yet, or that has no existing run: failing rows, not an error.
    partial = str(tmp_path / "partial")
    write_runs(partial, planned[:2])
    table = vad.clean_scores_check(planned, partial, base)
    assert len(table) == 54 and int(table["ok"].sum()) == 6
    assert not vad.clean_scores_check(planned, str(tmp_path / "nothing"), base)["ok"].any()
    few = str(tmp_path / "few")
    write_runs(few, vad.runs_of(planned, "clean")[:3],
               sets_of=lambda run: ("dev@clean", "eval@clean", "itw@clean"))
    assert int(vad.clean_scores_check(planned, runs, few)["ok"].sum()) == 9


def test_the_cell_of_clean_models_on_clean_audio_is_read_from_the_existing_runs(tmp_path):
    planned = vad.planned_runs(SEEDS)
    runs, base = str(tmp_path / "runs_vad"), existing_runs(str(tmp_path / "runs"), shift=0.01)
    write_runs(runs, planned)
    store = bootstrap.ScoreStore(runs, "wavlm-base", "unpadded", SEEDS, base_runs_dir=base)
    own = bootstrap.ScoreStore(runs, "wavlm-base", "unpadded", SEEDS)
    assert store.folder_of("clean", "clean") == base
    assert [store.folder_of(*pair) for pair in (("clean", "vad"), ("vad", "clean"), ("vad", "vad"))] == [runs] * 3
    assert own.folder_of("clean", "clean") == runs
    for split in ("eval", "itw"):
        matrix, labels, ids = store.cell("clean", "clean", "stats", 4, split)
        expected = np.stack([scores("clean", "clean", "stats", 4, seed, split, 0.01) for seed in SEEDS])
        assert np.array_equal(matrix, expected.astype(np.float64))        # the existing runs
        assert not np.array_equal(matrix, own.cell("clean", "clean", "stats", 4, split)[0])
        # The same run id in both folders: the other cells of the clean models come from the new runs.
        for pair in (("clean", "vad"), ("vad", "clean"), ("vad", "vad")):
            assert np.array_equal(store.cell(*pair, "stats", 4, split)[0], own.cell(*pair, "stats", 4, split)[0])
        assert np.array_equal(store.cell("clean", "clean", "stats", 4, split)[0], matrix)   # again, from the cache


def test_the_tables_of_the_condition_from_the_two_runs_folders(tmp_path):
    planned = vad.planned_runs(SEEDS)
    runs, base = str(tmp_path / "runs_vad"), existing_runs(str(tmp_path / "runs"))
    write_runs(runs, planned)
    out = str(tmp_path / "tables" / "wavlm-base" / "vad")
    tables = analysis.ScoreTables(
        "wavlm-base", runs, out, speaker_maps={name: world.speakers_of(name) for name in ("eval", "itw")},
        attack_map=world.attacks_of_eval(), conditions=("vad",), layers=(3, 4, 5),
        cross_layers={"vad": (3, 4, 5)}, detail_layers=(3, 4, 5), seeds=SEEDS, n_boot=40,
        base_runs_dir=base, log=lambda text: None)
    assert tables.inputs["runs_csv_sha256"].keys() == {"unpadded", "base"}
    effects = tables.effects()
    assert len(effects) == 2 * 2 * 3 * 6 and set(effects["condition"]) == {"vad"}
    assert sorted(set(effects["layer"])) == [3, 4, 5]
    # The cells are the seed-mean official EER of the four score sets; cc is that of the existing runs.
    from ssl_silence import metrics

    row = effects[(effects["test_set"] == "eval") & (effects["pooling"] == "stats") & (effects["layer"] == 4)
                  & (effects["effect"] == "interaction")].iloc[0]
    y = world.labels_of("eval")
    eer = {cell: float(np.mean([metrics.eer(y, scores(train, test, "stats", 4, seed, "eval"))
                                for seed in SEEDS]))
           for cell, (train, test) in {"cc": ("clean", "clean"), "ct": ("clean", "vad"),
                                       "tc": ("vad", "clean"), "tt": ("vad", "vad")}.items()}
    for cell, value in eer.items():
        assert row[f"eer_{cell}"] == pytest.approx(value, abs=1e-9), cell
    assert row["estimate"] == pytest.approx(eer["tt"] - eer["tc"] - eer["ct"] + eer["cc"], abs=1e-9)

    cells = vad.cells_from_effects(effects)
    assert len(cells) == 12 and cells["eer_cc"].notna().all()
    assert list(cells.columns) == ["encoder", "condition", "test_set", "pooling", "layer", "eer_cc", "eer_ct",
                                   "eer_tc", "eer_tt", "n_files"]
    broken = effects.copy()
    broken.loc[0, "eer_cc"] += 1.0
    with pytest.raises(ValueError, match="different cells"):
        vad.cells_from_effects(broken)

    # Side by side with the condition of the analysis: every row of the new condition has its row there.
    reference = effects.assign(condition="trim40", estimate=-effects["estimate"],
                               spk_excludes_zero=True)
    extra = reference.assign(condition="trim20")
    side = vad.side_by_side(effects, pd.concat([reference, extra], ignore_index=True))
    assert len(side) == len(effects) and not side["same_sign"][effects["estimate"].to_numpy() != 0].any()
    assert np.array_equal(side["estimate_vad"], effects["estimate"])
    assert np.array_equal(side["estimate_trim40"], -effects["estimate"])
    assert np.array_equal(side["both_spk_exclude_zero"], effects["spk_excludes_zero"].astype(bool))
    assert {"eer_cc_vad", "eer_cc_trim40", "spk_ci_low_vad", "spk_ci_high_trim40"} <= set(side.columns)
    with pytest.raises(KeyError, match="have no row of trim40"):
        vad.side_by_side(effects, reference[reference["layer"] != 5])
    with pytest.raises(ValueError, match="no row of vad"):
        vad.side_by_side(reference, reference)

    # The per-attack effect and the written files: only under the folder of the condition.
    attacks = tables.per_attack_effect()
    assert set(attacks["condition"]) == {"vad"} and sorted(set(attacks["layer"])) == [3, 4, 5]
    written = sorted(name for name in os.listdir(out) if name.endswith(".csv"))
    assert written == ["effects_2x2.csv", "per_attack_effect.csv"]
    assert sorted(os.listdir(os.path.join(out, "parts"))) == [
        "effects_2x2_vad.csv", "effects_2x2_vad.meta.json", "per_attack_effect_vad.csv",
        "per_attack_effect_vad.meta.json"]
    assert sorted(os.listdir(str(tmp_path / "tables" / "wavlm-base"))) == ["vad"]


def test_folders_are_left_out_of_the_comparison_of_files(tmp_path):
    tables = tmp_path / "tables" / "wavlm-base"
    (tables / "vad" / "parts").mkdir(parents=True)
    for file in (tables / "effects_2x2.csv", tables / "vad" / "effects_2x2.csv",
                 tables / "vad" / "parts" / "effects_2x2_vad.csv", tmp_path / "tables" / "vad_note.csv"):
        file.write_text(file.name, encoding="utf-8")
    folders = [str(tmp_path / "tables")]
    before = verify.file_hashes(folders, skip=("vad_",), skip_folders=(str(tables / "vad"),))
    assert [os.path.relpath(file, tmp_path).replace(os.sep, "/") for file in before] == [
        "tables/wavlm-base/effects_2x2.csv"]
    assert len(verify.file_hashes(folders)) == 4
    # A table of the new condition is written again and one is added: nothing that existed changed.
    (tables / "vad" / "effects_2x2.csv").write_text("again", encoding="utf-8")
    (tables / "vad" / "new.csv").write_text("new", encoding="utf-8")
    same = verify.unchanged_check(before, folders, skip=("vad_",), skip_folders=(str(tables / "vad"),))
    assert same["ok"].all() and len(same) == 1
    # A table with the same name outside the folder is not left out.
    (tables / "effects_2x2.csv").write_text("changed", encoding="utf-8")
    changed = verify.unchanged_check(before, folders, skip=("vad_",), skip_folders=(str(tables / "vad"),))
    assert changed["status"].tolist() == ["changed"] and not changed["ok"].any()
