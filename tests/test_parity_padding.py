"""padding against the original attention-mask check, on a small random WavLM.
Skipped without the original code."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models import tiny_model
from signals import write_dataset
from ssl_silence import config as cfg
from ssl_silence import extraction, padding as P

N_LAYERS = 12
# Conditions, comparisons and statistics of this repository -> names of the original code.
CONDITIONS = {"clean": "clean", "trim40": "trimmed"}
COMPARISONS = {"no_mask": "nomask_vs_ref", "mask": "mask_vs_ref", "mask_vs_no_mask": "mask_vs_nomask"}
STATISTICS = {"mean": "masked_mean", "std": "masked_std"}
PAD_COLUMNS = {"clean": "pad", "trim40": "pad_trimmed_40"}
LAYER_COLUMNS = ("max_abs_diff", "mean_abs_diff", "mean_abs_value", "rel_mean_diff",
                 "rel_max_diff", "max_abs_value", "n_files", "n_skipped")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("audio")), tiny_model("wavlm", n_layers=N_LAYERS)


def identical(new, old) -> float:
    new, old = np.asarray(new), np.asarray(old)
    if new.shape != old.shape:
        return np.inf
    if new.dtype.kind in "OUS" or old.dtype.kind in "OUS":
        return 0.0 if (new == old).all() else np.inf
    return float(np.max(np.abs(new.astype(float) - old.astype(float)))) if new.size else 0.0


def record(parity_record, name, differences):
    worst = max(differences)
    parity_record("padding", name, len(differences), worst)
    assert worst == 0.0, f"{name}: max abs diff {worst}"


def paddings(items: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Padding of every file per condition, as the stored lengths give it."""
    out = {}
    for condition in CONDITIONS:
        top_db = cfg.trim_top_db(condition)
        n_real = [extraction.prepare_file(path, top_db, "padded")["n_real"] for path in items["path"]]
        out[condition] = {a: cfg.WINDOW - n for a, n in zip(items["audio_id"], n_real)}
    return out


def test_samples_and_bins(setup, original, parity_record):
    items, _ = setup
    L = original("latency")
    padding = paddings(items)
    measurements = pd.DataFrame({"audio_id": items["audio_id"],
                                 **{PAD_COLUMNS[c]: [padding[c][a] for a in items["audio_id"]]
                                    for c in CONDITIONS}})
    differences = []
    for condition, old_condition in CONDITIONS.items():
        for n in (1, 3, 5, 100):
            new = P.most_padded_items(items, padding[condition], n=n)
            old = L.most_padded_items(items, measurements, n=n, condition=old_condition)
            differences.append(identical(new["audio_id"], old["audio_id"]))
    record(parity_record, "most_padded_items", differences)
    big = pd.DataFrame({"audio_id": [f"f{k}" for k in range(25_380)]})
    differences = [identical(P.random_items(big, n, seed).index, L.random_items(big, n, seed).index)
                   for n, seed in ((500, 0), (500, 1), (7, 0), (25_380, 0))]
    record(parity_record, "random_items", differences)
    values = list(range(0, 64_001, 97)) + [15_999, 16_000, 31_999, 32_000, 47_999, 48_000, 64_000]
    record(parity_record, "pad_bin", [0.0 if P.pad_bin(v) == L.pad_bin(v) else np.inf for v in values])


def test_padding_deviation_and_bins(setup, original, parity_record):
    items, model = setup
    L = original("latency")
    layer_diffs, file_diffs, bin_diffs = [], [], []
    for condition, old_condition in CONDITIONS.items():
        new_layer, new_file = P.padding_deviation(model, items, "cpu", condition, "random")
        old_layer, old_file = L.attention_mask_check(model, items, "cpu", n_layers=N_LAYERS,
                                                     condition=old_condition, sample="random",
                                                     progress=False)
        assert len(new_layer) == len(old_layer) == 3 * 2 * N_LAYERS
        for comparison, old_comparison in COMPARISONS.items():
            for statistic, old_statistic in STATISTICS.items():
                new = new_layer[(new_layer["comparison"] == comparison)
                                & (new_layer["statistic"] == statistic)].sort_values("layer")
                old = old_layer[(old_layer["comparison"] == old_comparison)
                                & (old_layer["statistic"] == old_statistic)].sort_values("layer")
                layer_diffs += [identical(new[c], old[c]) for c in LAYER_COLUMNS]
                for layer in range(1, N_LAYERS + 1):
                    file_diffs.append(identical(
                        new_file[f"{comparison}_{statistic}_L{layer}"],
                        old_file[f"{old_comparison}_{old_statistic}_L{layer}"]))
                file_diffs.append(identical(new_file[f"{comparison}_{statistic}_max"],
                                            old_file[f"{old_comparison}_{old_statistic}_max_all"]))
        for column in ("audio_id", "n_real", "n_real_frames", "pad", "pad_s", "pad_bin"):
            file_diffs.append(identical(new_file[column], old_file[column]))

        new_bins = P.deviation_by_padding(new_file, new_layer)
        old_bins = L.attention_mask_binned(old_file, old_layer)
        assert len(new_bins) == len(old_bins)
        for comparison, old_comparison in COMPARISONS.items():
            new = new_bins[new_bins["comparison"] == comparison].reset_index(drop=True)
            old = old_bins[old_bins["comparison"] == old_comparison].reset_index(drop=True)
            bin_diffs += [identical(new["pad_bin"], old["pad_bin"]),
                          identical(new["n_files"], old["n_files"])]
            for statistic, old_statistic in STATISTICS.items():
                for suffix in ("rel_L3", "rel_L4", "rel_L5", "rel_all"):
                    bin_diffs.append(identical(new[f"{statistic}_{suffix}"],
                                               old[f"{old_statistic}_{suffix}"]))
    record(parity_record, "padding_deviation (per layer)", layer_diffs)
    record(parity_record, "padding_deviation (per file)", file_diffs)
    record(parity_record, "deviation_by_padding", bin_diffs)
