"""Padding deviation, its intervals and the floor predictions, with known answers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from models import tiny_model
from signals import write_dataset
from ssl_silence import config as cfg
from ssl_silence import extraction, padding as P


# Samples ---------------------------------------------------------------------------

def test_random_items_are_fixed_by_the_seed_and_keep_the_list_order():
    items = pd.DataFrame({"audio_id": [f"f{k}" for k in range(1_000)]})
    chosen = P.random_items(items, n=50)
    expected = np.sort(np.random.default_rng(0).choice(1_000, size=50, replace=False))
    assert chosen.index.tolist() == expected.tolist()
    assert chosen.index.is_monotonic_increasing and chosen["audio_id"].is_unique
    assert P.random_items(items, n=50).equals(chosen)
    assert not P.random_items(items, n=50, seed=1).equals(chosen)
    assert (P.SAMPLE_SIZE, P.SAMPLE_SEED) == (500, 0)
    with pytest.raises(ValueError):
        P.random_items(items, n=1_001)


def test_most_padded_items_order_and_exclusions():
    items = pd.DataFrame({"audio_id": ["e", "d", "c", "b", "a"]})
    padding = {"a": 100, "b": 300, "c": 0, "d": 300, "e": 200}
    chosen = P.most_padded_items(items, padding, n=3)
    assert chosen["audio_id"].tolist() == ["b", "d", "e"]  # largest first, then ascending id
    assert P.most_padded_items(items, padding, n=10)["audio_id"].tolist() == ["b", "d", "e", "a"]
    with pytest.raises(KeyError):
        P.most_padded_items(items, {"a": 1}, n=3)


def test_padding_from_stored_lengths():
    lengths = pd.DataFrame({"audio_id": ["a", "b", "c"], "n_real": [64_000, 20_000, 300]})
    assert P.padding_from_lengths(lengths) == {"a": 0, "b": 44_000, "c": 63_700}


def test_pad_bins():
    assert [P.pad_bin(n) for n in (0, 1, 15_999, 16_000, 31_999, 32_000, 48_000, 64_000)] == \
        ["0", "<1 s", "<1 s", "1-2 s", "1-2 s", "2-3 s", ">3 s", ">3 s"]
    with pytest.raises(ValueError):
        P.pad_bin(64_001)


# Deviation -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    items = write_dataset(tmp_path_factory.mktemp("audio"))
    model = tiny_model("hubert", n_layers=3)
    per_layer, per_file = P.padding_deviation(model, items, "cpu", "clean", "random")
    return items, model, per_layer, per_file


def test_deviation_tables_layout(measured):
    items, _, per_layer, per_file = measured
    assert len(per_layer) == 3 * 2 * 3  # comparisons x statistics x layers
    assert set(per_layer["comparison"]) == {"no_mask", "mask", "mask_vs_no_mask"}
    assert set(per_layer["statistic"]) == {"mean", "std"} and set(per_layer["layer"]) == {1, 2, 3}
    assert set(per_layer["n_skipped"]) == {1} and set(per_layer["n_files"]) == {len(items) - 1}
    assert "tiny_file" not in per_file["audio_id"].tolist()  # fewer than 400 real samples
    assert (per_file["pad"] == cfg.WINDOW - per_file["n_real"]).all()
    assert (per_file["pad_s"] == per_file["pad"] / 16_000).all()
    assert per_file.loc[per_file["audio_id"] == "long_file", "pad_bin"].item() == "0"


def test_deviation_is_the_documented_ratio(measured):
    items, model, per_layer, per_file = measured
    numerator, denominator, largest = np.zeros(3), np.zeros(3), np.zeros(3)
    n = 0
    for item in items.itertuples():
        prepared = extraction.prepare_file(item.path, None, "padded")
        if prepared["n_real"] < cfg.MIN_SAMPLES:
            continue
        padded = extraction.encode_padded(model, prepared["x"], prepared["n_real"], 3, "cpu")
        reference = extraction.encode_unpadded(model, prepared["x"][:prepared["n_real"]], 3, "cpu")
        diff = (padded["real_frames_mean"] - reference["mean"]).abs()
        numerator += diff.mean(1).numpy()
        largest = np.maximum(largest, diff.amax(1).numpy())
        denominator += reference["mean"].abs().mean(1).numpy()
        n += 1
    rows = per_layer[(per_layer["comparison"] == "no_mask") & (per_layer["statistic"] == "mean")]
    np.testing.assert_allclose(rows["rel_mean_diff"], numerator / denominator, rtol=1e-12)
    np.testing.assert_allclose(rows["mean_abs_diff"], numerator / n, rtol=1e-12)
    np.testing.assert_allclose(rows["mean_abs_value"], denominator / n, rtol=1e-12)
    np.testing.assert_allclose(rows["max_abs_diff"], largest, rtol=1e-12)
    # The per-file columns carry the same numerators and denominators.
    np.testing.assert_allclose(per_file["no_mask_mean_L2"].sum(), numerator[1], rtol=1e-12)
    np.testing.assert_allclose(per_file["reference_mean_L2"].sum(), denominator[1], rtol=1e-12)


def test_a_file_that_fills_the_window_has_no_deviation(measured):
    _, _, _, per_file = measured
    full = per_file[per_file["audio_id"] == "long_file"].iloc[0]
    for column in ("no_mask_mean_L1", "no_mask_std_L3", "mask_vs_no_mask_mean_L2"):
        assert full[column] == 0.0
    padded = per_file[per_file["audio_id"] == "padded_tone"].iloc[0]
    assert padded["no_mask_mean_L1"] > 0 and padded["mask_vs_no_mask_mean_L1"] > 0


@pytest.mark.parametrize("norm", ["group", "layer"])
def test_three_extraction_modes_run_for_both_normalisations(tmp_path, norm):
    items = write_dataset(tmp_path)
    model = tiny_model("wavlm", n_layers=2, norm=norm)
    per_layer, per_file = P.padding_deviation(model, items, "cpu", "clean", "most_padded")
    assert len(per_layer) == 3 * 2 * 2 and len(per_file) == len(items) - 1
    assert np.isfinite(per_layer["rel_mean_diff"]).all()


def test_deviation_by_padding(measured):
    _, _, per_layer, per_file = measured
    binned = P.deviation_by_padding(per_file, per_layer, layers=(2,))
    assert set(binned["comparison"]) == set(P.COMPARISONS)
    assert binned.groupby("comparison")["n_files"].sum().tolist() == [len(per_file)] * 3
    scale = per_layer[(per_layer["comparison"] == "no_mask") & (per_layer["statistic"] == "mean")
                      & (per_layer["layer"] == 2)]["mean_abs_value"].item()
    for row in binned[binned["comparison"] == "no_mask"].itertuples():
        part = per_file[per_file["pad_bin"] == row.pad_bin]
        assert row.mean_rel_L2 == pytest.approx(part["no_mask_mean_L2"].mean() / scale, rel=1e-12)
    zero = binned[(binned["pad_bin"] == "0") & (binned["comparison"] == "no_mask")]
    assert zero["mean_rel_all"].item() == 0.0


# Intervals -------------------------------------------------------------------------

def per_file_table(n: int = 40, floor_only: bool = False) -> pd.DataFrame:
    """Two layers. Layer 1: numerator 0.2 * (1 + file index), reference 1 + file index, so
    every ratio is 0.2 (0.1 with mask). Layer 2: numerators with noise."""
    rng = np.random.default_rng(5)
    i = np.arange(n, dtype=float)
    table = pd.DataFrame({"audio_id": [f"f{k}" for k in range(n)], "condition": "clean",
                          "sample": "random"})
    for statistic in P.STATISTICS:
        table[f"reference_{statistic}_L1"] = 1 + i
        table[f"reference_{statistic}_L2"] = 2.0
        if floor_only:
            table[f"floor_{statistic}_L1"] = 1e-6 * (1 + i)
            table[f"floor_{statistic}_L2"] = 0.0
        else:
            table[f"no_mask_{statistic}_L1"] = 0.2 * (1 + i)
            table[f"mask_{statistic}_L1"] = 0.1 * (1 + i)
            table[f"no_mask_{statistic}_L2"] = rng.uniform(0.5, 1.5, n)
            table[f"mask_{statistic}_L2"] = rng.uniform(0.5, 1.5, n)
    return table


def test_intervals_of_a_constant_ratio_are_degenerate():
    table = per_file_table()
    out = P.deviation_intervals(table, n_boot=200).set_index(["statistic", "layer", "quantity"])
    row = out.loc[("mean", 1, "no_mask")]
    assert row["estimate"] == pytest.approx(0.2) and row["ci_low"] == pytest.approx(0.2)
    assert row["ci_high"] == pytest.approx(0.2) and row["n_files"] == 40 and row["n_boot"] == 200
    difference = out.loc[("mean", 1, "mask_minus_no_mask")]
    assert difference["estimate"] == pytest.approx(-0.1)
    assert difference["ci_low"] == pytest.approx(-0.1) and difference["ci_high"] == pytest.approx(-0.1)
    assert len(out) == 2 * 2 * 3


def test_intervals_are_the_percentiles_of_the_resampled_ratio():
    table = per_file_table()
    out = P.deviation_intervals(table, n_boot=300).set_index(["statistic", "layer", "quantity"])
    index = np.random.default_rng(P.INTERVAL_SEED).integers(0, 40, size=(300, 40))
    numerator = table["no_mask_mean_L2"].to_numpy()
    masked = table["mask_mean_L2"].to_numpy()
    boot = numerator[index].mean(1) / 2.0
    row = out.loc[("mean", 2, "no_mask")]
    assert row["estimate"] == pytest.approx(numerator.mean() / 2.0)
    assert row["ci_low"] == pytest.approx(np.percentile(boot, 2.5))
    assert row["ci_high"] == pytest.approx(np.percentile(boot, 97.5))
    assert row["ci_low"] < row["estimate"] < row["ci_high"]
    paired = masked[index].mean(1) / 2.0 - boot
    difference = out.loc[("mean", 2, "mask_minus_no_mask")]
    assert difference["ci_low"] == pytest.approx(np.percentile(paired, 2.5))
    assert difference["ci_high"] == pytest.approx(np.percentile(paired, 97.5))
    assert P.deviation_intervals(table, n_boot=300).equals(P.deviation_intervals(table, n_boot=300))
    assert P.INTERVAL_SEED == (2026, 101)


def test_one_file_and_zero_deviation_give_point_intervals():
    one = per_file_table(n=1)
    out = P.deviation_intervals(one, n_boot=50)
    assert (out["ci_low"] == out["ci_high"]).all() and (out["ci_low"] == out["estimate"]).all()
    zero = per_file_table()
    zero["no_mask_mean_L1"] = 0.0
    row = P.deviation_intervals(zero, n_boot=50).set_index(["statistic", "layer", "quantity"]).loc[
        ("mean", 1, "no_mask")]
    assert (row["estimate"], row["ci_low"], row["ci_high"]) == (0.0, 0.0, 0.0)


def test_mask_reduces_deviation_definition():
    intervals = pd.DataFrame({
        "condition": "clean", "sample": "random", "statistic": "mean", "layer": [1, 2, 3, 4],
        "quantity": "mask_minus_no_mask", "estimate": [-0.2, -0.1, 0.1, -0.1],
        "ci_low": [-0.3, -0.3, 0.05, -0.2], "ci_high": [-0.1, 0.02, 0.2, 0.0]})
    out = P.mask_reduces_deviation(intervals)
    # Negative with an interval below zero; an interval touching zero does not count.
    assert out["mask_reduces"].tolist() == [True, False, False, False]
    assert P.mask_reduces_deviation(intervals, sample="most_padded").empty


# Noise floor and predictions -----------------------------------------------------

def test_noise_floor_is_zero_for_identical_extractions_and_positive_otherwise(tmp_path):
    items = write_dataset(tmp_path)
    model = tiny_model("hubert", n_layers=2)
    per_layer, per_file = P.noise_floor(model, "cpu", model, "cpu", items, "clean", "random")
    assert set(per_layer["comparison"]) == {"floor"} and len(per_layer) == 2 * 2
    assert (per_layer["rel_mean_diff"] == 0).all() and (per_layer["max_abs_diff"] == 0).all()
    assert set(per_layer["n_skipped"]) == {1}
    shifted = tiny_model("hubert", n_layers=2)
    with torch.no_grad():
        for parameter in shifted.parameters():
            parameter.mul_(1 + 1e-4)
    per_layer, per_file = P.noise_floor(model, "cpu", shifted, "cpu", items, "clean", "random")
    assert (per_layer["rel_mean_diff"] > 0).all() and (per_layer["rel_mean_diff"] < 0.05).all()
    floor = P.deviation_intervals(per_file, comparisons=("floor",), difference=None, n_boot=100)
    assert set(floor["quantity"]) == {"floor"} and (floor["ci_low"] > 0).all()
    mean_rows = per_layer[per_layer["statistic"] == "mean"]
    np.testing.assert_allclose(floor[floor["statistic"] == "mean"]["estimate"],
                               mean_rows["rel_mean_diff"], rtol=1e-9)


def interval_rows(quantity, values) -> pd.DataFrame:
    """values: layer -> (estimate, ci_low, ci_high)."""
    return pd.DataFrame([{"condition": "clean", "sample": "random", "statistic": "mean",
                          "layer": layer, "quantity": quantity, "estimate": e, "ci_low": lo,
                          "ci_high": hi} for layer, (e, lo, hi) in values.items()])


def test_floor_predictions():
    floor = interval_rows("floor", {1: (1e-6, 5e-7, 2e-6), 2: (1e-6, 5e-7, 2e-6),
                                    3: (0.0, 0.0, 0.0), 4: (0.0, 0.0, 0.0)})
    intervals = pd.concat([
        interval_rows("mask", {1: (1e-5, 5e-6, 1.9e-5),    # at most 10 x the floor: (a) holds
                               2: (1e-5, 5e-6, 2.1e-5),    # above 10 x the floor: (a) fails
                               3: (0.0, 0.0, 0.0),         # zero floor, zero deviation: holds
                               4: (1e-9, 0.0, 2e-9)}),     # zero floor, deviation not 0: fails
        interval_rows("no_mask", {1: (0.3, 0.25, 0.35),    # above the floor: (b) holds
                                  2: (1e-6, 1e-6, 3e-6),   # lower bound inside the floor: fails
                                  3: (0.2, 0.1, 0.3),
                                  4: (0.0, 0.0, 0.0)})])   # not above a zero floor: fails
    out = P.floor_predictions(intervals, floor).set_index("layer")
    assert out["a_holds"].tolist() == [True, False, True, False]
    assert out["b_holds"].tolist() == [True, False, True, False]
    assert P.FLOOR_FACTOR == 10.0 and isinstance(P.FLOOR_FACTOR, float)
    stricter = P.floor_predictions(intervals, floor, factor=5).set_index("layer")
    assert stricter["a_holds"].tolist() == [False, False, True, False]
    with pytest.raises(KeyError):
        P.floor_predictions(intervals, floor[floor["layer"] < 4])


def test_prediction_a_at_the_boundary_is_the_exact_float64_expression():
    # (a) holds iff upper <= 10.0 * floor_upper in float64, with no margin.
    # 10.0 * 2e-6 is 1.9999999999999998e-05 in float64, just below 2e-5, so an
    # upper bound written as 2e-5 does not satisfy the expression, while the
    # product itself does.
    floor_upper = 2e-6
    product = 10.0 * floor_upper
    assert product == 1.9999999999999998e-05 and product < 2e-5
    floor = interval_rows("floor", {1: (1e-6, 5e-7, floor_upper), 2: (1e-6, 5e-7, floor_upper),
                                    3: (1e-6, 5e-7, floor_upper)})
    no_mask = interval_rows("no_mask", {1: (0.3, 0.2, 0.4), 2: (0.3, 0.2, 0.4), 3: (0.3, 0.2, 0.4)})
    mask = interval_rows("mask", {1: (1e-5, 5e-6, product),                     # equal: holds
                                  2: (1e-5, 5e-6, 2e-5),                        # one step above
                                  3: (1e-5, 5e-6, np.nextafter(product, 0.0))})  # one step below
    out = P.floor_predictions(pd.concat([mask, no_mask]), floor).set_index("layer")
    assert out["a_holds"].tolist() == [True, False, True]
    assert np.nextafter(product, 1.0) == 2e-5
    for layer, upper in ((1, product), (2, 2e-5), (3, np.nextafter(product, 0.0))):
        assert out.loc[layer, "a_holds"] == (np.float64(upper) <= 10.0 * np.float64(floor_upper))


# The files of an earlier run ---------------------------------------------------------

def test_sample_ids_and_items_of_an_earlier_run():
    per_file = pd.DataFrame({"audio_id": ["c", "a", "b", "a", 7], "condition": ["x", "x", "y", "y", "y"],
                             "sample": ["random", "random", "random", "most_padded", "most_padded"]})
    assert P.sample_ids(per_file, "x", "random") == ["c", "a"]
    assert P.sample_ids(per_file, "y", "most_padded") == ["a", "7"]
    with pytest.raises(KeyError, match="no rows"):
        P.sample_ids(per_file, "x", "most_padded")
    with pytest.raises(ValueError, match="repeats"):
        P.sample_ids(pd.concat([per_file, per_file.iloc[:1]]), "x", "random")
    items = pd.DataFrame({"audio_id": ["a", "b", "c", 7], "path": ["pa", "pb", "pc", "p7"]})
    chosen = P.items_with_ids(items, ["c", "7", "a"])
    assert chosen["path"].tolist() == ["pc", "p7", "pa"] and chosen["audio_id"].tolist() == ["c", "7", "a"]
    with pytest.raises(KeyError, match="1 audio ids are not in the file list"):
        P.items_with_ids(items, ["a", "z"])
    assert P.SAMPLES == ("random", "most_padded")


def test_a_sample_is_compared_as_a_set_of_files():
    chosen = pd.DataFrame({"audio_id": ["a", "b", "c"]})
    assert P.compare_samples(chosen, ["a", "b", "c"]) == {
        "n_files": 3, "n_reference": 3, "n_common": 3, "same_order": True, "ok": True}
    other_order = P.compare_samples(chosen, ["c", "a", "b"])
    assert other_order["ok"] and not other_order["same_order"]
    assert not P.compare_samples(chosen, ["a", "b", "d"])["ok"]
    assert not P.compare_samples(chosen, ["a", "b"])["ok"]
    assert not P.compare_samples(chosen.iloc[:0], [])["ok"]


def test_published_rows(measured):
    _, _, per_layer, per_file = measured
    rows = P.published_rows(per_layer)
    assert set(rows["comparison"]) == {"no_mask", "mask"} and set(rows["statistic"]) == {"mean"}
    assert len(rows) == 2 * 3
    by_amount = P.published_rows(P.deviation_by_padding(per_file, per_layer))
    assert set(by_amount["comparison"]) == {"no_mask", "mask"} and "mean_rel_all" in by_amount.columns


# The samples of the reference encoder, for another encoder ----------------------------

def reference_table() -> pd.DataFrame:
    return pd.DataFrame({"audio_id": ["a", "b", "c", "c", "d"],
                         "condition": ["clean"] * 3 + ["trim40"] * 2,
                         "sample": ["random"] * 3 + ["most_padded"] * 2,
                         "n_real": [100, 200, 64_000, 50_000, 40_000]})


def test_stored_lengths_are_the_real_samples_inside_the_window():
    lengths = pd.DataFrame({"audio_id": ["a", 7, "long"], "n_real": [100, 64_000, 70_000]})
    assert P.stored_lengths(lengths) == {"a": 100, "7": 64_000, "long": 64_000}
    assert P.stored_lengths(lengths, window=150) == {"a": 100, "7": 150, "long": 150}


def test_lengths_from_audio_are_those_of_the_padded_extraction(measured):
    items, _, _, per_file = measured
    lengths = P.lengths_from_audio(items, "clean")
    assert set(lengths) == set(items["audio_id"].astype(str))
    # Every measured file has the length the deviation recorded for it; the skipped ones are short.
    recorded = dict(zip(per_file["audio_id"].astype(str), per_file["n_real"]))
    assert recorded and all(lengths[a] == n for a, n in recorded.items())
    assert all(lengths[a] < cfg.MIN_SAMPLES for a in set(lengths) - set(recorded))
    assert max(lengths.values()) == cfg.WINDOW
    # Trimming changes the real length of a file with silence, and never makes it longer.
    trimmed = P.lengths_from_audio(items, "trim40")
    assert all(trimmed[a] <= lengths[a] for a in lengths) and trimmed != lengths


def test_a_reference_sample_is_checked_by_length_and_by_the_files_drawn_again():
    reference = reference_table()
    lengths = {"a": 100, "b": 200, "c": 64_000}
    drawn = pd.DataFrame({"audio_id": ["a", "b", "c"]})
    row = P.check_reference_sample(reference, "clean", "random", lengths, "audio", drawn)
    assert row == {"condition": "clean", "sample": "random", "n_files": 3, "lengths_from": "audio",
                   "n_lengths_equal": 3, "redrawn": "same files in the same order", "ok": True}
    # Another order, another file, or one file less: not the same sample.
    for other in (["b", "a", "c"], ["a", "b", "x"], ["a", "b"]):
        row = P.check_reference_sample(reference, "clean", "random", lengths, "audio",
                                       pd.DataFrame({"audio_id": other}))
        assert (row["redrawn"], row["ok"], row["n_lengths_equal"]) == ("other files", False, 3)
    # A length that differs, or a file without a length, fails whatever was drawn.
    row = P.check_reference_sample(reference, "clean", "random", {**lengths, "b": 201}, "shards", drawn)
    assert (row["n_lengths_equal"], row["ok"], row["lengths_from"]) == (2, False, "shards")
    row = P.check_reference_sample(reference, "clean", "random", {"a": 100, "c": 64_000}, "audio", drawn)
    assert (row["n_lengths_equal"], row["ok"]) == (2, False)
    # A sample that cannot be drawn again is checked by its lengths alone, and says so.
    row = P.check_reference_sample(reference, "trim40", "most_padded", {"c": 50_000, "d": 40_000}, "audio")
    assert (row["redrawn"], row["ok"], row["n_files"]) == (P.NOT_REDRAWN, True, 2)
    # The length of a file is that of its condition: "c" has another length when untrimmed.
    row = P.check_reference_sample(reference, "trim40", "most_padded", lengths | {"d": 40_000}, "audio")
    assert (row["n_lengths_equal"], row["ok"]) == (1, False)
    with pytest.raises(KeyError, match="no rows"):
        P.check_reference_sample(reference, "clean", "most_padded", lengths, "audio")
    assert (P.REFERENCE_ENCODER, P.PREDICTED_ENCODER, P.TIMING_FILES) == ("wavlm-base", "wavlm-large", 25)


def prediction_rows(n_layers=24, conditions=("clean", "trim40"), a_fails=(), b_fails=()) -> pd.DataFrame:
    return pd.DataFrame([{"condition": c, "sample": "random", "layer": layer,
                          "a_holds": (c, layer) not in a_fails, "b_holds": (c, layer) not in b_fails}
                         for c in conditions for layer in range(1, n_layers + 1)])


def test_a_prediction_holds_only_if_it_is_met_at_every_layer_in_every_condition():
    verdict = P.prediction_verdict(prediction_rows(), 24, ("clean", "trim40")).set_index("prediction")
    assert list(verdict.index) == ["a", "b"]
    assert verdict["n_cells"].tolist() == [48, 48] and verdict["n_met"].tolist() == [48, 48]
    assert verdict["holds"].tolist() == [True, True]
    assert "10.0 x" in verdict.loc["a", "statement"] and "lower bound" in verdict.loc["b", "statement"]
    # One layer of one condition is enough for a prediction not to hold; the other is not affected.
    verdict = P.prediction_verdict(prediction_rows(a_fails={("trim40", 24)}), 24,
                                   ("clean", "trim40")).set_index("prediction")
    assert verdict["n_met"].tolist() == [47, 48] and verdict["holds"].tolist() == [False, True]
    verdict = P.prediction_verdict(prediction_rows(b_fails={("clean", 1), ("clean", 2)}), 24,
                                   ("clean", "trim40")).set_index("prediction")
    assert verdict["n_met"].tolist() == [48, 46] and verdict["holds"].tolist() == [True, False]
    # A missing layer, a missing condition or a repeated row is not a verdict.
    with pytest.raises(ValueError, match="2 conditions x 24 layers"):
        P.prediction_verdict(prediction_rows(n_layers=23), 24, ("clean", "trim40"))
    with pytest.raises(ValueError):
        P.prediction_verdict(prediction_rows(conditions=("clean",)), 24, ("clean", "trim40"))
    rows = prediction_rows()
    with pytest.raises(ValueError):
        P.prediction_verdict(pd.concat([rows.iloc[1:], rows.iloc[1:2]]), 24, ("clean", "trim40"))


def test_the_checkpoint_cache_is_a_local_folder_here():
    from ssl_silence import encoders

    folder = encoders.cache_dir()
    assert folder and not cfg.on_drive(folder)
    assert cfg.on_drive("/content/drive/MyDrive/cache/hub")


def test_the_tables_of_the_gate_are_checked_against_their_meta_files(tmp_path):
    from ssl_silence import provenance

    def table(name, commit, same_sha256=True, meta=True):
        path = tmp_path / f"{name}.csv"
        path.write_text(f"a\n{name}\n", encoding="utf-8")
        digest = provenance.sha256_file(path)
        if meta:
            record = {"git_commit": commit, "sha256": digest if same_sha256 else "0" * 64}
            (tmp_path / f"{name}.meta.json").write_text(json.dumps(record), encoding="utf-8")
        return str(path), digest

    full = "637b4b89b1fa8eb551f178c90a520df58a65a1ac"
    files = dict([table("full_hash", full), table("short_hash", "637b4b8"),
                  table("other_commit", "a7b341d252b6e3bf24ec5a82a046db89d7e21176"),
                  table("dirty", full + "-dirty"), table("changed_since", full, same_sha256=False),
                  table("no_meta", full, meta=False)])
    out = P.gate_files_check(files)
    assert list(out.columns) == ["file", "git_commit", "recorded_sha256", "sha256", "ok"]
    assert out["file"].tolist() == list(files) and out["sha256"].tolist() == list(files.values())
    assert out["ok"].tolist() == [True, True, False, False, False, False]
    assert out["git_commit"].tolist()[2:] == ["a7b341d252b6e3bf24ec5a82a046db89d7e21176", full + "-dirty", full,
                                              "missing"]
    assert out["recorded_sha256"].tolist()[-2:] == ["0" * 64, "missing"]
    # The commit is the one at which gate A4 passed; another commit can be asked for.
    assert P.GATE_COMMIT == "637b4b8"
    assert P.gate_files_check(files, commit="a7b341d")["ok"].tolist() == [False, False, True, False, False, False]
    assert P.gate_files_check({}).empty
