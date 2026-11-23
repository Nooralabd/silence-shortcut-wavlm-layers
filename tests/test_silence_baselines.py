"""The silence-only classifier and the padding-only score, with known answers."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ssl_silence import metrics as M
from ssl_silence import silence as S
from ssl_silence import training as T


def separable_table(n: int, seed: int, top_db: int = 40, reverse: bool = False) -> pd.DataFrame:
    """Bonafide files have long leading silence, spoof files short (or the reverse)."""
    rng = np.random.default_rng(seed)
    label = (np.arange(n) % 3 > 0).astype(int)
    long_lead = (label == 0) != reverse
    lead = np.where(long_lead, rng.integers(9_000, 20_000, n), rng.integers(0, 900, n))
    return pd.DataFrame({"audio_id": [f"f{seed}_{k}" for k in range(n)], "label": label,
                         f"lead_{top_db}": lead, f"trail_{top_db}": rng.integers(0, 4_000, n),
                         f"win_lead_{top_db}": lead, f"win_trail_{top_db}": rng.integers(0, 4_000, n),
                         "n_samples": rng.integers(30_000, 90_000, n),
                         "pad": rng.integers(0, 30_000, n)})


def test_mannwhitney_auc_by_hand():
    separated = S.mannwhitney_auc(np.array([1, 2, 3]), np.array([4, 5]))
    assert (separated["auc"], separated["u"], separated["n_bonafide"], separated["n_spoof"]) == (1.0, 6.0, 3, 2)
    assert S.mannwhitney_auc(np.array([4, 5]), np.array([1, 2, 3]))["auc"] == 0.0
    tied = S.mannwhitney_auc(np.array([1, 1]), np.array([1, 1]))
    # All values equal: U is at its mean. The pinned scipy (1.16.3) gives p = 1; a
    # later scipy gives nan. The package returns what scipy returns, as the original code did.
    assert tied["auc"] == 0.5 and tied["p"] == 1.0
    # One of four pairs has the spoof value above, one is a tie: (1 + 0.5) / 4.
    assert S.mannwhitney_auc(np.array([2, 3]), np.array([1, 3]))["auc"] == pytest.approx(0.375)
    with pytest.raises(ValueError):
        S.mannwhitney_auc(np.array([]), np.array([1.0]))


def test_silence_only_classifier_on_separable_silence(tmp_path):
    tables = {"train": separable_table(300, 0), "dev": separable_table(90, 1),
              "eval": separable_table(120, 2), "itw": separable_table(120, 3, reverse=True)}
    path = S.silence_only_baseline(tables, 40, scores_dir=str(tmp_path))
    assert os.path.basename(path) == "silence-only_sil_40db_clean.npz"
    sets, meta = T.load_scores(path)
    assert set(sets) == {"dev@clean", "eval@clean", "itw@clean"}
    assert meta["columns"] == ["lead_40", "trail_40", "win_lead_40", "win_trail_40"]
    assert (meta["model"], meta["features"], meta["top_db"], meta["condition"], meta["n_train"]) == (
        "silence-only", "sil", 40, "clean", 300)
    scores = sets["eval@clean"]
    assert scores.scores.dtype == np.float32 and scores.scores.min() >= 0 and scores.scores.max() <= 1
    assert list(scores.ids) == list(tables["eval"]["audio_id"])
    assert np.array_equal(scores.labels, tables["eval"]["label"])
    assert M.eer(scores.labels, scores.scores) == 0.0
    # Where the cue runs the other way, the classifier is wrong on every file.
    assert M.eer(sets["itw@clean"].labels, sets["itw@clean"].scores) == 100.0
    with pytest.raises(FileExistsError):
        S.silence_only_baseline(tables, 40, scores_dir=str(tmp_path))
    again = T.load_scores(S.silence_only_baseline(tables, 40, scores_dir=str(tmp_path),
                                                  overwrite=True))[0]
    assert np.array_equal(again["dev@clean"].scores, sets["dev@clean"].scores)   # deterministic


def test_silence_only_table_and_names(tmp_path):
    tables = {"train": separable_table(300, 0, 30), "dev": separable_table(90, 1, 30),
              "eval": separable_table(120, 2, 30)}
    for table in tables.values():
        for column in S.silence_columns(30):
            table[column.replace("30", "15")] = table[column] // 2
    result = S.silence_only_baselines(tables, top_dbs=(30, 15), condition="trim40",
                                      scores_dir=str(tmp_path))
    assert list(result.columns) == ["condition", "features", "top_db", "dev", "eval"]
    assert result[["features", "top_db"]].values.tolist() == [["sil", 15], ["sil", 30],
                                                               ["sil+dur", 15], ["sil+dur", 30]]
    assert set(result["condition"]) == {"trim40"} and (result["eval"] == 0.0).all()
    assert sorted(os.listdir(tmp_path)) == [
        "silence-only_sil-dur_15db_trim40.npz", "silence-only_sil-dur_30db_trim40.npz",
        "silence-only_sil_15db_trim40.npz", "silence-only_sil_30db_trim40.npz"]
    sets, meta = T.load_scores(os.path.join(tmp_path, "silence-only_sil-dur_30db_trim40.npz"))
    assert meta["columns"][-2:] == ["n_samples", "pad"] and set(sets) == {"dev@trim40", "eval@trim40"}
    row = result[(result["features"] == "sil+dur") & (result["top_db"] == 30)].iloc[0]
    assert row["dev"] == M.eer(sets["dev@trim40"].labels, sets["dev@trim40"].scores)
    with pytest.raises(KeyError, match="train"):
        S.silence_only_baseline({"dev": tables["dev"]}, 30, scores_dir=str(tmp_path))
    with pytest.raises(KeyError):
        S.silence_only_baseline(tables, 30, condition="trimmed", scores_dir=str(tmp_path))
    with pytest.raises(KeyError):
        S.silence_only_baseline(tables, 30, features="all", scores_dir=str(tmp_path))


def padding_table() -> pd.DataFrame:
    return pd.DataFrame({
        "label": [0, 0, 0, 0, 1, 1, 1, 1],
        "system_id": ["-", "-", "-", "-", "A07", "A07", "A08", "A08"],
        "pad": [48_000, 40_000, 36_000, 33_600, 32_000, 16_000, 0, 0],
        "pad_trimmed_40": [56_000, 48_000, 40_000, 36_000, 32_000, 32_000, 16_000, 0],
        "pad_trimmed_15": 0})


def test_pad_column_per_condition():
    assert S.pad_column("clean") == "pad" and S.pad_column("trim40") == "pad_trimmed_40"
    assert S.pad_column("trim15") == "pad_trimmed_15"
    with pytest.raises(KeyError):
        S.pad_column("trimmed")


def test_padding_by_label():
    table = S.padding_by_label({"eval": padding_table(), "train": padding_table()})
    assert len(table) == 2 * 2 * 2 + 2 * 2                    # all x label, and two attacks on eval
    keys = ["table", "condition", "group", "label"]
    table = table.set_index(keys)
    row = table.loc[("eval", "clean", "all", 0)]
    assert (row["n"], row["median_s"], row["q25_s"], row["q75_s"]) == (4, 2.375, 2.2125, 2.625)
    assert row["iqr_s"] == pytest.approx(0.4125) and row["frac_padded"] == 1.0
    assert row["frac_over_2s"] == 1.0
    spoof = table.loc[("eval", "clean", "all", 1)]
    assert (spoof["median_s"], spoof["frac_padded"], spoof["frac_over_2s"]) == (0.5, 0.5, 0.0)
    assert table.loc[("eval", "trim40", "A08", 1), "median_s"] == 0.5
    assert ("train", "clean", "A07", 1) not in table.index and ("eval", "clean", "A07", 0) not in table.index
    assert S.padding_by_label({"eval": padding_table()}, conditions=("trim15",))["median_s"].eq(0).all()


def test_padding_only_score():
    result = S.padding_only_baseline({"eval": padding_table()}).set_index("condition")
    clean = result.loc["clean"]
    # Every bonafide file has more padding than every spoof file.
    assert (clean["eer_plus_pad"], clean["eer_minus_pad"], clean["eer_best"]) == (100.0, 0.0, 0.0)
    assert clean["sign_best"] == "-pad" and clean["auc"] == 0.0
    assert (clean["n_bonafide"], clean["n_spoof"]) == (4, 4)
    y = padding_table()["label"].to_numpy()
    pad = padding_table()["pad_trimmed_40"].to_numpy() / 16_000
    trimmed = result.loc["trim40"]
    assert trimmed["eer_plus_pad"] == M.eer(y, pad) and trimmed["eer_minus_pad"] == M.eer(y, -pad)
    assert trimmed["eer_best"] == min(trimmed["eer_plus_pad"], trimmed["eer_minus_pad"])
    uninformative = S.padding_only_baseline({"x": padding_table()}, conditions=("trim15",)).iloc[0]
    assert uninformative["auc"] == 0.5 and uninformative["sign_best"] == "+pad"
