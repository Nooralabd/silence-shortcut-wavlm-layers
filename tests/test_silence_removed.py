"""Removed amounts, fallback counts and the trimmed-length check, with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals import write_file_list
from ssl_silence import audio as A
from ssl_silence import config as cfg
from ssl_silence import silence as S
from ssl_silence import trimming as T


def removed_rows() -> pd.DataFrame:
    """Four files with invented amounts at 40 and 20 dB (seconds)."""
    rows = []
    for audio_id, split, label, orig, at40, at20 in (
            ("a", "eval", 0, 4.0, (0.75, 0.25, 0, 0), (1.5, 0.5, 0, 0)),
            ("b", "eval", 0, 6.0, (2.0, 1.0, 0, 0), (0.0, 0.0, 1, 0)),
            ("c", "eval", 1, 2.0, (0.0, 0.5, 0, 0), (0.25, 0.75, 0, 0)),
            ("d", "itw", 1, 0.0, (0.0, 0.0, 1, 1), (0.0, 0.0, 1, 1))):
        row = {"dataset": "x", "split": split, "audio_id": audio_id, "label": label,
               "system_id": "-", "speaker": "s", "n_samples_orig": int(orig * 16_000),
               "orig_s": orig}
        for d, (lead, trail, fallback, short) in ((40, at40), (20, at20)):
            row.update({f"n_samples_trimmed_{d}": int((orig - lead - trail) * 16_000),
                        f"removed_lead_s_{d}": lead, f"removed_trail_s_{d}": trail,
                        f"removed_total_s_{d}": lead + trail, f"fallback_{d}": fallback,
                        f"too_short_{d}": short})
        rows.append(row)
    return pd.DataFrame(rows)


def test_removed_long_has_one_row_per_file_and_threshold():
    long = S.removed_long(removed_rows(), top_dbs=(40, 20))
    assert list(long.columns) == S.ITEM_COLUMNS + ["top_db", "n_samples_orig", "orig_s",
                                                   *T.REMOVED_PER_DB_KEYS]
    assert list(long["top_db"]) == [40] * 4 + [20] * 4 and list(long["audio_id"]) == list("abcd") * 2
    row = long[(long["audio_id"] == "a") & (long["top_db"] == 20)].iloc[0]
    assert (row["removed_lead_s"], row["removed_total_s"], row["n_samples_trimmed"]) == (1.5, 2.0, 32_000)
    with pytest.raises(KeyError, match="15 dB"):
        S.removed_long(removed_rows(), top_dbs=(40, 15))


def test_summary_of_the_removed_seconds():
    summary = S.summarise_removed(S.removed_long(removed_rows(), top_dbs=(40, 20)))
    assert list(summary["top_db"]) == [40, 40, 40, 20, 20, 20]
    assert summary[["split", "label"]].values.tolist()[:3] == [["eval", 0], ["eval", 1], ["itw", 1]]
    row = summary.iloc[0]                                        # 40 dB, eval, bonafide: a and b
    assert (row["n"], row["n_fallback"]) == (2, 0)
    assert (row["removed_total_median_s"], row["removed_total_q25_s"], row["removed_total_q75_s"],
            row["removed_total_mean_s"]) == (2.0, 1.5, 2.5, 2.0)
    assert row["removed_lead_median_s"] == 1.375 and row["removed_trail_mean_s"] == 0.625
    assert row["removed_frac_median"] == pytest.approx((1.0 / 4.0 + 3.0 / 6.0) / 2)
    at20 = summary.iloc[3]                                       # b falls back: 0 removed
    assert at20["n_fallback"] == 1 and at20["removed_total_median_s"] == 1.0
    empty = summary.iloc[2]                                      # the empty file counts as 0
    assert empty["removed_frac_median"] == 0.0 and empty["n_fallback"] == 1


def test_fallback_counts_and_their_comparison_with_the_shards():
    long = S.removed_long(removed_rows(), top_dbs=(40, 20))
    counts = S.fallback_counts_from_removed(long)
    assert list(counts.columns) == ["top_db", "split", "label", "n_files", "n_fallback",
                                    "n_too_short", "pct_fallback"]
    at20 = counts[(counts["top_db"] == 20) & (counts["split"] == "eval")].set_index("label")
    assert at20["n_files"].to_dict() == {"0": 2, "1": 1, "all": 3}
    assert at20["n_fallback"].to_dict() == {"0": 1, "1": 0, "all": 1}
    assert at20.loc["all", "pct_fallback"] == pytest.approx(100 / 3)
    itw = counts[(counts["top_db"] == 40) & (counts["split"] == "itw")].set_index("label")
    assert itw.loc["all", "n_too_short"] == 1 and np.isnan(itw.loc["0", "pct_fallback"])

    measured = counts[counts["top_db"] == 20]
    reference = measured.drop(columns="top_db").reset_index(drop=True)
    check = S.compare_fallback_counts(measured, reference)
    assert check["ok"].all() and len(check) == len(reference)
    assert list(check.columns) == ["split", "label", "n_files", "n_fallback", "n_too_short",
                                   "n_files_measured", "n_fallback_measured",
                                   "n_too_short_measured", "ok"]
    wrong = reference.copy()
    wrong.loc[0, "n_fallback"] += 1
    assert S.compare_fallback_counts(measured, wrong)["ok"].tolist() == [False] + [True] * 5
    with pytest.raises(ValueError, match="one threshold"):
        S.compare_fallback_counts(counts, reference)
    extra = pd.concat([reference, reference.iloc[:1].assign(split="dev")], ignore_index=True)
    with pytest.raises(KeyError, match="no measured row"):
        S.compare_fallback_counts(measured, extra)


def test_trimmed_length_against_the_shards():
    long = pd.DataFrame({"split": "eval", "audio_id": ["a", "b", "c"], "top_db": 40,
                         "n_samples_trimmed": [30_000, 70_000, 64_000]})
    lengths = pd.DataFrame({"split": "eval", "audio_id": ["c", "a", "b"], "label": [1, 0, 0],
                            "n_real": [64_000, 30_000, 64_000]})
    diffs = S.trim_length_diffs(long, lengths, 40)
    assert list(diffs["audio_id"]) == ["c", "a", "b"]                    # shard order
    assert list(diffs["n_expected"]) == [64_000, 30_000, 64_000]         # cropped to the window
    assert (diffs["abs_diff_samples"] == 0).all()
    assert S.summarise_trim_lengths(diffs).iloc[0].tolist() == ["eval", 3, 3, 0, True]
    off = S.trim_length_diffs(long, lengths.assign(n_real=[64_000, 30_160, 64_000]), 40)
    summary = S.summarise_trim_lengths(off).iloc[0]
    assert (summary["n_equal"], summary["max_abs_diff_samples"], summary["ok"]) == (2, 160, False)
    with pytest.raises(KeyError, match="no rows at 20 dB"):
        S.trim_length_diffs(long, lengths, 20)
    with pytest.raises(KeyError, match="only in the shards"):
        S.trim_length_diffs(long.iloc[:2], lengths, 40)
    with pytest.raises(KeyError, match="duplicate"):
        S.trim_length_diffs(pd.concat([long, long.iloc[:1]]), lengths, 40)


def test_measured_files_give_the_lengths_the_extraction_uses(tmp_path):
    """From audio files to the two checks: the measured counts and lengths equal
    what trimming the same files gives, at every threshold."""
    file_list = pd.concat([write_file_list(tmp_path, "dev", seed=1),
                           write_file_list(tmp_path, "eval", seed=2)], ignore_index=True)
    wide = pd.concat([S.measure_items(file_list[file_list["split"] == split],
                                      str(tmp_path / "m"), split, progress=False)
                      for split in ("dev", "eval")], ignore_index=True)
    long = S.removed_long(S.removed_table(wide))
    assert len(long) == 3 * len(file_list)
    for top_db in S.REMOVED_TOP_DBS:
        rows = []
        for item in file_list.itertuples():
            wav, sr, _ = A.load_audio(item.path)
            trimmed, fallback = T.apply_trim(A.to_mono_16k(wav, sr), top_db)
            n = int(trimmed.shape[1])
            rows.append({"split": item.split, "audio_id": item.audio_id, "label": item.label,
                         "n_real": min(n, cfg.WINDOW), "trim_fallback": fallback,
                         "too_short": n < cfg.MIN_SAMPLES})
        lengths = pd.DataFrame(rows)
        summary = S.summarise_trim_lengths(S.trim_length_diffs(long, lengths, top_db))
        assert summary["ok"].all() and summary["n_files"].sum() == len(file_list)
        reference = pd.DataFrame([
            {"split": split, "label": label, "n_files": int(mask.sum()),
             "n_fallback": int(part["trim_fallback"][mask].sum()),
             "n_too_short": int(part["too_short"][mask].sum())}
            for split, part in lengths.groupby("split", sort=False)
            for label, mask in (("0", part["label"] == 0), ("1", part["label"] == 1),
                                ("all", part["label"] >= 0))])
        measured = S.fallback_counts_from_removed(long[long["top_db"] == top_db])
        check = S.compare_fallback_counts(measured, reference)
        assert check["ok"].all() and check["n_fallback"].sum() > 0


def test_the_measured_trimming_is_compared_with_the_shards_of_one_threshold():
    removed = removed_rows()
    long = S.removed_long(removed, top_dbs=(40, 20))
    lengths = pd.DataFrame({
        "split": removed["split"], "audio_id": removed["audio_id"], "label": removed["label"],
        "n_real": np.minimum(removed["n_samples_trimmed_20"], cfg.WINDOW),
        "trim_fallback": removed["fallback_20"].astype(bool), "too_short": removed["too_short_20"].astype(bool)})
    counts, same = S.compare_with_shards(long, lengths, 20)
    assert counts["ok"].all() and same["ok"].all()
    assert list(counts.columns[:3]) == ["top_db", "split", "label"] and set(counts["top_db"]) == {20}
    assert same["split"].tolist() == ["eval", "itw"] and same["n_files"].tolist() == [3, 1]
    assert counts.set_index(["split", "label"]).loc[("eval", "all"), "n_fallback_measured"] == 1
    # The shards of 20 dB are not those of 40 dB: one fallback and two lengths differ.
    other_counts, other_same = S.compare_with_shards(long, lengths, 40)
    assert not other_counts["ok"].all() and not other_same["ok"].all()
    assert S.GATE_TABLES == ("silence_summary", "residual_silence_summary", "trim_amount",
                             "padding_only_baseline", "padding_by_label", "silence_only_baseline",
                             "residual_silence_baseline")
    assert S.GATE_CHECKS == ("silence_per_file", *(f"table_{name}" for name in S.GATE_TABLES),
                             "silence_fallback_counts", "silence_trim_lengths", "silence_itw_durations",
                             "silence_printed_numbers", "silence_quiet_files")
