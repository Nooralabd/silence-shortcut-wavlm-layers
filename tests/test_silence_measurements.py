"""Silence measurements per file and their summaries, with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from signals import (BLIP_CENTRE, BLIP_GAP, MS, blip_cases, clip, residual_cases, trimming_cases,
                     write_file_list)
from ssl_silence import config as cfg
from ssl_silence import silence as S
from ssl_silence import trimming as T

HOP, FRAME = cfg.TRIM_HOP_LENGTH, cfg.TRIM_FRAME_LENGTH


def test_thresholds_follow_the_conditions():
    assert S.PRE_TRIM_TOP_DB == cfg.trim_top_db("trim40")
    assert set(S.REMOVED_TOP_DBS) == {d for d in cfg.conditions().values() if d is not None}
    assert set(S.RESIDUAL_TOP_DBS) <= set(S.SILENCE_TOP_DBS) and S.PRE_TRIM_TOP_DB in S.RESIDUAL_TOP_DBS


def test_window_view_by_hand():
    # A file shorter than the window is padded at the end.
    assert S.window_view(40_000, 8_000, 30_000) == {"win_lead": 8_000, "win_trail": 10_000,
                                                    "pad": 24_000, "crop": 0}
    assert S.window_view(64_000, 0, 64_000) == {"win_lead": 0, "win_trail": 0, "pad": 0, "crop": 0}
    # A file of 100,000 samples keeps its centre [18,000, 82,000).
    assert S.window_view(100_000, 20_000, 70_000) == {"win_lead": 2_000, "win_trail": 12_000,
                                                      "pad": 0, "crop": 36_000}
    assert S.window_view(100_000, 10_000, 90_000) == {"win_lead": 0, "win_trail": 0,
                                                      "pad": 0, "crop": 36_000}
    # Sound only before the window: the whole window is silence.
    assert S.window_view(100_000, 0, 10_000)["win_trail"] == 64_000
    assert S.window_view(100_000, 90_000, 100_000)["win_lead"] == 64_000
    # Another window: 100 samples keep [10, 90).
    assert S.window_view(100, 10, 60, window=80) == {"win_lead": 0, "win_trail": 30, "pad": 0,
                                                     "crop": 20}


def test_silence_of_known_length():
    m = S.measure_silence(clip(8_000, 24_000, 12_000)[0].numpy())
    assert m["n_samples"] == 44_000 and m["all_zero"] == 0
    assert m["pad"] == 20_000 and m["crop"] == 0
    for d in S.SILENCE_TOP_DBS:
        assert 8_000 - FRAME <= m[f"lead_{d}"] <= 8_000 and m[f"lead_{d}"] % HOP == 0
        assert 12_000 - FRAME <= m[f"trail_{d}"] <= 12_000
        assert m[f"win_lead_{d}"] == m[f"lead_{d}"] and m[f"win_trail_{d}"] == m[f"trail_{d}"]
        assert m[f"trim_fallback_{d}"] == 0
        assert m[f"trimmed_len_{d}"] == 44_000 - m[f"lead_{d}"] - m[f"trail_{d}"]
        assert m[f"pad_trimmed_{d}"] == 64_000 - m[f"trimmed_len_{d}"]
    assert m["lead_15"] >= m["lead_50"]                      # a stricter threshold removes more
    assert m["peak_frame_rms"] == pytest.approx(0.5 / np.sqrt(2), rel=0.02)
    assert list(m)[:4] == ["n_samples", "all_zero", "pad", "crop"] and list(m)[-1] == "peak_frame_rms"


def test_edge_cases_of_one_waveform():
    zeros = S.measure_silence(np.zeros(20_000, dtype=np.float32), top_dbs=(40,))
    assert zeros == {"n_samples": 20_000, "all_zero": 1, "pad": 44_000, "crop": 0, "lead_40": 0,
                     "trail_40": 0, "win_lead_40": 0, "win_trail_40": 0, "trim_fallback_40": 1,
                     "trimmed_len_40": 20_000, "pad_trimmed_40": 44_000, "peak_frame_rms": 0.0}
    empty = S.measure_silence(np.zeros(0, dtype=np.float32), top_dbs=(40,))
    assert empty["n_samples"] == 0 and empty["all_zero"] == 1 and empty["pad"] == 64_000
    assert empty["trim_fallback_40"] == 1 and empty["trimmed_len_40"] == 0
    # Fewer than 0.5 s would remain: the file is kept, but the detected silence is reported.
    short = S.measure_silence(clip(16_000, 4_800, 16_000)[0].numpy(), top_dbs=(40,))
    assert short["trim_fallback_40"] == 1 and short["trimmed_len_40"] == 36_800
    assert short["lead_40"] > 15_000 and short["pad_trimmed_40"] == 64_000 - 36_800
    long = S.measure_silence(clip(30_000, 60_000, 10_000)[0].numpy(), top_dbs=(40,))
    assert long["crop"] == 36_000 and long["pad"] == 0
    assert 30_000 - 18_000 - FRAME <= long["win_lead_40"] <= 12_000 and long["win_trail_40"] == 0
    assert long["pad_trimmed_40"] == 64_000 - long["trimmed_len_40"] > 0


def test_a_quiet_file_cannot_be_trimmed():
    """The RMS floor of librosa: a file whose loudest frame is below 1e-3 keeps its silence at 40 dB."""
    assert S.untrimmable_below(40) == pytest.approx(1e-3) and S.untrimmable_below(20) == pytest.approx(1e-4)
    quiet = S.measure_silence(clip(8_000, 24_000, 12_000, amplitude=1e-3)[0].numpy(), top_dbs=(40, 20))
    assert S.untrimmable_below(20) < quiet["peak_frame_rms"] < S.untrimmable_below(40)
    assert quiet["lead_40"] == 0 and quiet["trail_40"] == 0          # nothing detected at 40 dB
    assert quiet["lead_20"] > 7_000                                  # but at 20 dB
    loud = S.measure_silence(clip(8_000, 24_000, 12_000, amplitude=1e-2)[0].numpy(), top_dbs=(40,))
    assert loud["peak_frame_rms"] > S.untrimmable_below(40) and loud["lead_40"] > 7_000


def test_the_three_measurements_of_one_waveform_agree():
    """The length after trimming is the same in the measurement of the untrimmed
    audio, in the residual measurement and in the removed amount."""
    for name, wav in trimming_cases().items():
        m = S.measure_waveform(wav)
        n = int(wav.shape[1])
        assert m["n_samples"] == m["residual_n_samples_orig"] == m["amount_n_samples_orig"] == n, name
        assert m["residual_fallback_trim40"] == m["trim_fallback_40"] == m["amount_fallback_40"], name
        assert m["residual_n_samples"] == m["trimmed_len_40"] == m["amount_n_samples_trimmed_40"], name
        for d in (20, 15):
            assert m[f"trimmed_len_{d}"] == m[f"amount_n_samples_trimmed_{d}"], (name, d)
            assert m[f"trim_fallback_{d}"] == m[f"amount_fallback_{d}"], (name, d)
        trimmed, _ = T.apply_trim(wav, 40)
        assert m["residual_peak_frame_rms"] == S.measure_silence(trimmed[0].numpy())["peak_frame_rms"]
        if not m["trim_fallback_40"]:
            removed = m["amount_removed_lead_s_40"] * cfg.SAMPLE_RATE
            assert removed == m["lead_40"], name
    keys = list(S.measure_waveform(torch.zeros(1, 500)))
    assert sum(k.startswith("residual_") for k in keys) == 4 + 7 * len(S.RESIDUAL_TOP_DBS) + 3
    assert sum(k.startswith("amount_") for k in keys) == 2 + 6 * len(S.REMOVED_TOP_DBS)


# Tables ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    folder = tmp_path_factory.mktemp("audio")
    file_list = write_file_list(folder)
    out_dir = tmp_path_factory.mktemp("measurements")
    return file_list, str(out_dir), S.measure_items(file_list, str(out_dir), "eval", chunk_size=7,
                                                    progress=False)


def test_measure_items_writes_resumable_chunks(measured, tmp_path):
    import os

    file_list, out_dir, wide = measured
    assert list(wide.columns[:6]) == S.ITEM_COLUMNS and "path" not in wide.columns
    assert list(wide["audio_id"]) == list(file_list["audio_id"]) and len(wide) == len(file_list)
    assert len(os.listdir(os.path.join(out_dir, "chunks"))) == -(-len(file_list) // 7)
    assert S.read_measurements(os.path.join(out_dir, "eval.csv")).equals(wide)
    first = S.measure_file(file_list["path"].iloc[0])
    assert {k: wide[k].iloc[0] for k in first} == first
    assert set(wide["sr_orig"]) == {16_000, 22_050, 48_000} and set(wide["n_channels"]) == {1, 2}

    def never(path):
        raise AssertionError("a finished chunk was measured again")

    again = S.measure_items(file_list, out_dir, "eval", chunk_size=7, progress=False, measure=never)
    assert again.equals(wide)
    with pytest.raises(RuntimeError, match="do not match"):
        S.measure_items(file_list.iloc[::-1], out_dir, "eval", chunk_size=7, progress=False)
    with pytest.raises(RuntimeError, match="do not match"):
        S.measure_items(file_list.iloc[:-1], out_dir, "eval", chunk_size=7, progress=False)
    assert len(S.measure_items(file_list.iloc[:0], str(tmp_path), "none", progress=False)) == 0


def test_strings_that_look_like_missing_values_survive(tmp_path):
    file_list = pd.DataFrame({"dataset": "x", "split": "itw", "audio_id": ["1", "NA"],
                              "path": ["a", "b"], "label": [0, 1], "system_id": ["-", "itw"],
                              "speaker": ["None", "null"]})
    table = S.measure_items(file_list, str(tmp_path), "itw", progress=False,
                            measure=lambda path: {"n_samples": 3})
    assert list(table["audio_id"]) == ["1", "NA"] and list(table["speaker"]) == ["None", "null"]


def test_the_three_parts_of_the_wide_table(measured):
    _, _, wide = measured
    silence, residual, removed = S.silence_table(wide), S.residual_table(wide), S.removed_table(wide)
    first = S.ITEM_COLUMNS
    assert list(silence.columns[:6]) == list(residual.columns[:6]) == list(removed.columns[:6]) == first
    assert not any(c.startswith(("residual_", "amount_")) for c in silence.columns)
    assert {"sr_orig", "n_channels", "peak_frame_rms", "lead_50", "pad_trimmed_15"} <= set(silence.columns)
    assert {"sr_orig", "n_channels", "fallback_trim40", "n_samples_orig", "lead_15",
            "n_samples"} <= set(residual.columns) and "lead_50" not in residual.columns
    assert list(removed.columns[6:8]) == ["n_samples_orig", "orig_s"]
    assert {f"{k}_{d}" for k in T.REMOVED_PER_DB_KEYS for d in (40, 20, 15)} == set(removed.columns[8:])
    assert (residual["n_samples"] == silence["trimmed_len_40"]).all()
    assert len(silence.columns) + len(residual.columns) + len(removed.columns) == len(wide.columns) + 14
    with pytest.raises(KeyError):
        S.residual_table(silence)


def test_summary_in_seconds():
    table = pd.DataFrame({"label": [0, 0, 0, 1], "lead_40": [0, 16_000, 32_000, 8_000],
                          "trail_40": 1_600, "win_lead_40": 0, "win_trail_40": 0,
                          "n_samples": 64_000, "pad": [0, 0, 0, 16_000]})
    summary = S.summarise(table, 40).set_index(["label", "measure"])
    assert len(summary) == 12
    row = summary.loc[(0, "lead_40")]
    assert (row["n"], row["mean_s"], row["median_s"], row["q25_s"], row["q75_s"], row["max_s"]) == (
        3, 1.0, 1.0, 0.5, 1.5, 2.0)
    assert summary.loc[(1, "pad"), "median_s"] == 1.0 and summary.loc[(0, "trail_40"), "max_s"] == 0.1


def test_share_of_files_with_residual_silence():
    table = pd.DataFrame({"fallback_trim40": [0, 0, 0, 1, 0],
                          "lead_40": [0, 800, 801, 9_000, 1_600],     # 800 samples = 50 ms
                          "trail_40": [900, 0, 0, 9_000, 0]})
    row = S.residual_silence_share({"eval": table}).iloc[0]
    assert (row["n_files"], row["n_fallback"], row["n_lead_over"], row["n_trail_over"]) == (5, 1, 2, 1)
    assert row["pct_lead_over"] == 40.0 and row["pct_trail_over"] == 20.0


def test_weak_sound_at_an_edge_is_residual_silence_at_the_stricter_thresholds():
    """A segment 30 dB below the tone survives trimming at 40 dB. Measured on
    the trimmed audio it is not silence at 40 dB, but it is at 20 and 15 dB."""
    limit = 0.05 * cfg.SAMPLE_RATE                                   # 50 ms = 800 samples
    weak = {"weak_lead_80ms": (80 * MS, 0), "weak_trail_200ms": (0, 200 * MS),
            "weak_both_edges": (200 * MS, 80 * MS)}
    rows = []
    for name, wav in residual_cases().items():
        m = S.measure_waveform(wav)
        weak_lead, weak_trail = weak[name]
        assert m["residual_fallback_trim40"] == 0
        # Trimming at 40 dB removed the silence and kept the weak segments.
        assert abs(m["residual_n_samples"] - (24_000 + weak_lead + weak_trail)) <= 2 * FRAME, name
        assert m["residual_lead_40"] <= HOP and m["residual_trail_40"] <= HOP, name
        for d in (20, 15):
            for side, length in (("lead", weak_lead), ("trail", weak_trail)):
                found = m[f"residual_{side}_{d}"]
                assert abs(found - length) <= FRAME, (name, side, d, found)
                assert (found > limit) == (length > 0), (name, side, d)
        rows.append({"fallback_trim40": m["residual_fallback_trim40"],
                     **{f"{side}_{d}": m[f"residual_{side}_{d}"]
                        for side in ("lead", "trail") for d in (40, 20, 15)}})
    table = pd.DataFrame(rows)
    at40 = S.residual_silence_share({"x": table}).iloc[0]
    assert (at40["top_db"], at40["n_files"], at40["n_lead_over"], at40["n_trail_over"]) == (40, 3, 0, 0)
    for d in (20, 15):
        row = S.residual_silence_share({"x": table}, top_db=d).iloc[0]
        assert (row["top_db"], row["n_lead_over"], row["n_trail_over"]) == (d, 2, 2)
        assert row["pct_lead_over"] == pytest.approx(200 / 3)


def test_silence_defined_by_40_db_remains_when_the_cut_falls_inside_a_marginal_frame():
    """How more than 50 ms of silence, as defined by 40 dB, can remain after
    trimming at 40 dB.

    The trim starts at the centre of the first frame above the threshold. The
    frames of the trimmed audio lie on the same grid; the first one keeps only
    the second half of its samples (the first half is now before the cut and
    is replaced by zeros). A blip 39 dB below the tone fills one frame: that
    frame is 1 dB above the threshold, its neighbours hold 60 % of the blip
    and are below it. After the cut the first frame holds half of the blip,
    3 dB less, and is below the threshold too: the blip, the gap and the
    zeros up to the tone are leading silence of the trimmed file.
    """
    limit = 0.05 * cfg.SAMPLE_RATE
    wav = blip_cases()["blip_39db_before_tone"]
    y = wav[0].numpy()
    before = T.frame_db(y)
    k = BLIP_CENTRE // HOP
    assert before[k] == pytest.approx(-39.0, abs=0.05)                  # the only frame above -40
    assert before[k - 1] == pytest.approx(-39.0 + 10 * np.log10(0.6), abs=0.05) and before[k - 1] < -40
    assert before[k + 1] < -40 and (before[:k] < -40).all()
    start, end, fallback = T.trim_interval(wav, 40)
    assert (start, fallback) == (BLIP_CENTRE, False)                    # the cut is at the frame centre
    after = T.frame_db(y[start:end])
    assert after[0] == pytest.approx(-39.0 + 10 * np.log10(0.5), abs=0.05) and after[0] < -40
    first_tone_frame = int(np.flatnonzero(after > -40)[0])
    # The tone starts 200 + 1,600 samples after the cut; the first frame that reaches it is frame 11.
    assert first_tone_frame == (200 + BLIP_GAP - 200) // HOP + 1 == 11
    m = S.measure_waveform(wav)
    assert m["lead_40"] == BLIP_CENTRE and m["amount_removed_lead_s_40"] == BLIP_CENTRE / 16_000
    assert m["residual_lead_40"] == first_tone_frame * HOP == 1_760 > limit       # 110 ms remain
    assert m["residual_win_lead_40"] == 1_760 and m["residual_trail_40"] <= HOP
    # Trimming the trimmed audio again at 40 dB removes it: the rule is not idempotent here.
    again = T.trim_interval(wav[:, start:end], 40)
    assert again[0] == 1_760

    # A blip 35 dB below the tone: three frames are above the threshold, the cut
    # falls one frame earlier, and the first frame stays above it.
    control = S.measure_waveform(blip_cases()["blip_35db_before_tone"])
    assert control["lead_40"] == BLIP_CENTRE - HOP and control["residual_lead_40"] == 0
    assert control["residual_lead_20"] > limit                           # silence only at 20 dB

    rows = [{"fallback_trim40": x["residual_fallback_trim40"], "lead_40": x["residual_lead_40"],
             "trail_40": x["residual_trail_40"]} for x in (m, control)]
    share = S.residual_silence_share({"x": pd.DataFrame(rows)}).iloc[0]
    assert (share["n_files"], share["n_lead_over"], share["pct_lead_over"]) == (2, 1, 50.0)


def test_quiet_file_counts():
    table = pd.DataFrame({"label": [0, 0, 1, 1, 1],
                          "peak_frame_rms": [0.0, 5e-4, 5e-5, 9.99e-4, 1e-3],
                          "all_zero": [1, 0, 0, 0, 0]})
    counts = S.quiet_file_counts({"train": table}).set_index("label")
    assert list(counts.columns) == ["table", "n_files", "n_all_zero", "n_below_40", "n_below_20",
                                    "n_below_15"]
    assert counts.loc["all", ["n_files", "n_all_zero", "n_below_40", "n_below_20", "n_below_15"]
                      ].tolist() == [5, 1, 4, 2, 2]
    assert counts.loc["0", "n_below_40"] == 2 and counts.loc["1", "n_below_40"] == 2
    assert S.untrimmable_below(15) == pytest.approx(10 ** (-5 + 15 / 20))
    assert 5e-5 < S.untrimmable_below(15) < 1e-4


def test_duration_map_and_subsets():
    table = pd.DataFrame({"audio_id": ["a", "b", "c"], "sr_orig": 16_000,
                          "n_samples": [64_000, 64_001, 8_000]})
    durations = S.duration_map(table)
    assert durations == {"a": 4.0, "b": 64_001 / 16_000, "c": 0.5}
    assert S.check_duration_subsets(durations, expected=(2, 1)) == (2, 1)
    with pytest.raises(ValueError, match="duration subsets"):
        S.check_duration_subsets(durations)
    with pytest.raises(ValueError, match="not at 16000 Hz"):
        S.duration_map(table.assign(sr_orig=[16_000, 44_100, 16_000]))
    assert cfg.ITW_DURATION_COUNTS == (19_387, 12_392) and sum(cfg.ITW_DURATION_COUNTS) == cfg.ITW_SIZE



# Gate A5 -----------------------------------------------------------------------------------------

def test_remaining_silence_counts_every_file_per_label():
    table = pd.DataFrame({"label": [0, 0, 0, 1, 1, 1, 1, 1],
                          "fallback_trim40": [0, 0, 1, 0, 0, 0, 1, 0],
                          "lead_40": [0, 801, 9_600, 800, 1_600, 3_200, 6_400, 0],     # 800 samples = 50 ms
                          "win_lead_40": [0, 0, 9_600, 800, 1_600, 0, 6_400, 0]})
    share = S.remaining_silence({"eval": table})
    assert list(share.columns) == [
        "table", "label", "n_files", "n_fallback", "n_lead_over", "pct_lead_over", "n_win_lead_over",
        "pct_win_lead_over", "n_lead_over_trimmed", "pct_lead_over_trimmed", "lead_median_s", "lead_q25_s",
        "lead_q75_s"]
    assert share["label"].tolist() == ["0", "1", "all"] and set(share["table"]) == {"eval"}
    rows = share.set_index("label")
    counts = ["n_files", "n_fallback", "n_lead_over", "n_win_lead_over", "n_lead_over_trimmed"]
    assert rows.loc["0", counts].tolist() == [3, 1, 2, 1, 1]
    assert rows.loc["1", counts].tolist() == [5, 1, 3, 2, 2]
    assert rows.loc["all", counts].tolist() == [8, 2, 5, 3, 3]
    # The percentages are of all files of the row; a file kept untrimmed by the fallback rule counts.
    assert rows.loc["all", ["pct_lead_over", "pct_win_lead_over", "pct_lead_over_trimmed"]].tolist() == [
        62.5, 37.5, 37.5]
    assert rows.loc["1", "pct_lead_over"] == 60.0
    # The count without the fallback files is that of residual_silence_share.
    assert rows.loc["all", "n_lead_over_trimmed"] == S.residual_silence_share(
        {"eval": table.assign(trail_40=0)}).iloc[0]["n_lead_over"]
    # Median and quartiles of the whole-file length of the files that count, in seconds.
    assert rows.loc["all", ["lead_median_s", "lead_q25_s", "lead_q75_s"]].tolist() == [0.2, 0.1, 0.4]
    assert rows.loc["0", "lead_median_s"] == pytest.approx((801 + 9_600) / 2 / 16_000)
    none = S.remaining_silence({"x": table.assign(lead_40=0, win_lead_40=0)}).set_index("label")
    assert none.loc["all", "n_lead_over"] == 0 and np.isnan(none.loc["all", "lead_median_s"])


def test_quiet_files_are_the_files_that_are_counted():
    table = pd.DataFrame({"audio_id": ["a", "b", "c", "d", "e"], "label": [0, 0, 1, 1, 1],
                          "n_samples": [10, 20, 30, 40, 50], "all_zero": [1, 0, 0, 0, 0],
                          "peak_frame_rms": [0.0, 5e-4, 5e-5, 9.99e-4, 1e-3]})
    quiet = S.quiet_files({"train": table, "itw": table.iloc[4:]})
    assert list(quiet.columns) == ["table", "audio_id", "label", "n_samples", "all_zero", "peak_frame_rms",
                                   "top_db", "below"]
    assert quiet["audio_id"].tolist() == ["a", "b", "c", "d"] and set(quiet["table"]) == {"train"}
    assert set(quiet["below"]) == {S.untrimmable_below(40)} and S.untrimmable_below(40) == pytest.approx(1e-3)
    counts = S.quiet_file_counts({"train": table}).set_index("label")
    assert len(quiet) == counts.loc["all", "n_below_40"]
    assert S.quiet_files({"train": table}, top_db=20)["audio_id"].tolist() == ["a", "c"]


def test_the_wide_table_is_compared_with_the_three_original_per_file_tables(measured, tmp_path):
    _, _, wide = measured
    for folder, part in S.ORIGINAL_PARTS.items():
        (tmp_path / folder).mkdir()
        # The original tables have fewer columns than this package measures.
        table = part(wide).drop(columns=["peak_frame_rms", "lead_15"], errors="ignore")
        table.to_csv(tmp_path / folder / "eval.csv", index=False)
    same = S.compare_with_original_files(wide, "eval", str(tmp_path))
    assert same["ok"].all() and list(same.columns[:3]) == ["part", "split", "column"]
    assert same["part"].unique().tolist() == list(S.ORIGINAL_PARTS) and set(same["split"]) == {"eval"}
    assert set(same["tolerance"]) == {1e-9} and (same["n_matched"] == len(wide)).all()
    compared = set(same.loc[same["part"] == "measurements_dir", "column"])
    assert "peak_frame_rms" not in compared and {"sr_orig", "lead_40", "speaker", "label"} <= compared
    assert "orig_s" in set(same.loc[same["part"] == "removed_measurements_dir", "column"])
    # One sample more in one file, or seconds beyond the tolerance, fail in their column only.
    other = wide.copy()
    other.loc[3, "residual_lead_40"] += 1
    other.loc[5, "amount_removed_lead_s_40"] += 2e-9
    other.loc[6, "amount_removed_trail_s_40"] += 5e-10
    different = S.compare_with_original_files(other, "eval", str(tmp_path))
    failed = different[~different["ok"]]
    assert sorted(zip(failed["part"], failed["column"])) == [
        ("removed_measurements_dir", "removed_lead_s_40"), ("residual_measurements_dir", "lead_40")]
    assert failed["n_different"].tolist() == [1, 1]
    # A file that only one of the two tables has fails every column.
    assert not S.compare_with_original_files(wide.iloc[1:], "eval", str(tmp_path))["ok"].any()


def test_summary_tables_have_the_names_and_keys_of_the_original_tables(measured):
    _, _, wide = measured
    halves = {"dev": wide.iloc[::2].reset_index(drop=True), "eval": wide}
    tables = S.summary_tables(halves)
    assert tuple(tables) == S.SUMMARY_TABLES
    raw = {split: S.silence_table(table) for split, table in halves.items()}
    residual = {split: S.residual_table(table) for split, table in halves.items()}
    n_labels = sum(table["label"].nunique() for table in halves.values())
    summary = tables["silence_summary"]
    assert len(summary) == 6 * n_labels and not summary.duplicated(["table", "label", "measure"]).any()
    assert set(summary["measure"]) == set(S.silence_columns(40, with_duration=True))
    pd.testing.assert_frame_equal(summary[summary["table"] == "eval"].drop(columns="table").reset_index(drop=True),
                                  S.summarise(raw["eval"], 40))
    residual_summary = tables["residual_silence_summary"]
    assert len(residual_summary) == 6 * n_labels * len(S.RESIDUAL_TOP_DBS)
    assert not residual_summary.duplicated(["table", "top_db", "label", "measure"]).any()
    part = residual_summary[(residual_summary["table"] == "dev") & (residual_summary["top_db"] == 20)]
    pd.testing.assert_frame_equal(part.drop(columns=["table", "top_db"]).reset_index(drop=True),
                                  S.summarise(residual["dev"], 20))
    amount = tables["trim_amount"]
    assert set(amount["top_db"]) == set(S.REMOVED_TOP_DBS) and set(amount["split"]) == set(wide["split"])
    pd.testing.assert_frame_equal(tables["padding_only_baseline"], S.padding_only_baseline(raw))
    pd.testing.assert_frame_equal(tables["padding_by_label"], S.padding_by_label(raw))
    assert S.BASELINE_TOP_DBS == {"clean": (30, 40, 50), "trim40": (15, 20, 30)}


def test_the_earlier_counts_are_reported_next_to_the_measured_ones():
    share = pd.DataFrame({"table": ["train", "train", "dev", "eval", "itw"], "label": ["0", "all", "all", "all", "all"],
                          "n_files": [5, 10, 10, 10, 10], "n_fallback": [0, 2, 0, 1, 3],
                          "n_lead_over": [9, 1_054, 972, 2_382, 985],
                          "n_lead_over_trimmed": [9, 1_046, 972, 2_300, 900]})
    report = S.earlier_counts_report(share)
    assert report["table"].tolist() == ["train", "dev", "eval", "itw"] and "ok" not in report.columns
    assert report["earlier"].tolist() == [1_046, 972, 2_382, 938]
    assert report["reproduced_by"].tolist() == [
        "files without fallback", "every file and files without fallback", "every file", "neither"]
    assert S.EARLIER_COUNTS == {"train": 1_046, "dev": 972, "eval": 2_382, "itw": 938}


def test_duration_checks_of_in_the_wild():
    table = pd.DataFrame({"sr_orig": 16_000, "n_samples": [64_000, 64_001, 8_000]})
    checks = S.duration_checks(table, expected=(2, 1))
    assert list(checks.columns) == ["check", "found", "expected", "ok"] and checks["ok"].all()
    assert checks["check"].tolist() == ["files not at 16000 Hz", "files of at most 4 s", "files longer than 4 s"]
    assert checks["found"].tolist() == [0, 2, 1]
    assert S.duration_checks(table)["ok"].tolist() == [True, False, False]
    other = S.duration_checks(table.assign(sr_orig=[16_000, 44_100, 16_000]), expected=(2, 1))
    assert other["ok"].tolist() == [False, True, True] and other.loc[0, "found"] == 1
    assert S.TIMING_FILES == 100
