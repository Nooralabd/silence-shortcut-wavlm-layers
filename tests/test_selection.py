"""Per-run metrics, seed summaries and dev-only selection, with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import world
from ssl_silence import metrics as M
from ssl_silence import selection as S
from ssl_silence import training as T


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    base = tmp_path_factory.mktemp("world")
    runs_dir = world.write_runs(str(base / "runs"))
    asv = S.load_asv(world.write_asv_scores(str(base / "asv")))
    table = S.metrics_per_run(runs_dir, asv)
    return runs_dir, asv, table, S.seed_summary(table)


def test_metrics_table_layout(tables):
    runs_dir, asv, table, _ = tables
    assert list(table.columns) == S.METRIC_COLUMNS and set(asv) == {"dev", "eval"}
    assert set(table["metric"]) == {"eer", "tandem_min_tdcf"}
    assert set(table[table["metric"] == "tandem_min_tdcf"]["split"]) == {"dev", "eval"}
    n_sets = sum(1 + len(run.score_set_names) for run in world.planned_runs())
    assert (table["metric"] == "eer").sum() == n_sets
    assert table["value"].between(0, 100).all()


def test_values_are_the_metrics_of_the_saved_scores(tables):
    runs_dir, asv, table, _ = tables
    run_id = "wavlm-base_unpadded_clean_L03_stats_s43"
    sets, _ = T.load_scores(f"{runs_dir}/scores/{run_id}.npz")
    part = table[table["run_id"] == run_id].set_index(["score_set", "metric"])["value"]
    assert set(sets) == {"dev@clean", "eval@clean", "itw@clean", "dev@trim40", "eval@trim40",
                         "itw@trim40", "dev@trim20", "eval@trim20", "itw@trim20"}
    for name, s in sets.items():
        assert part[(name, "eer")] == M.eer(s.labels, s.scores)
    s = sets["eval@trim40"]
    assert part[("eval@trim40", "tandem_min_tdcf")] == M.tandem_min_tdcf(s.labels, s.scores, asv["eval"])
    assert ("itw@clean", "tandem_min_tdcf") not in part.index
    row = table[(table["run_id"] == run_id) & (table["score_set"] == "itw@trim20")].iloc[0]
    assert (row["split"], row["test_condition"], row["condition"]) == ("itw", "trim20", "clean")
    assert row["n_bonafide"] + row["n_spoof"] == world.SPLITS["itw"]


def test_seed_summary_is_mean_and_sample_sd_over_seeds(tables):
    _, _, table, summary = tables
    assert set(summary["n_seeds"]) == {3}
    cell = summary[(summary["condition"] == "clean") & (summary["test_condition"] == "trim40")
                   & (summary["pooling"] == "stats") & (summary["layer"] == 2)].iloc[0]
    values = table[(table["condition"] == "clean") & (table["score_set"] == "eval@trim40")
                   & (table["pooling"] == "stats") & (table["layer"] == "2")
                   & (table["metric"] == "eer")]["value"].to_numpy(dtype=float)
    assert len(values) == 3
    assert cell["eval_eer_mean"] == pytest.approx(values.mean(), rel=1e-12)
    assert cell["eval_eer_sd"] == pytest.approx(values.std(ddof=1), rel=1e-12)
    # 3 conditions x 4 layers x 2 poolings matched cells; cross cells as designed.
    assert len(S.matched_cells(summary)) == 24
    assert len(summary) == 24 + 2 * 8 + 2 * 4


def frame(rows):
    """Seed summary with the given (condition, pooling, layer, dev EER) matched cells."""
    return pd.DataFrame([{"encoder": "wavlm-base", "extraction": "unpadded", "frames": "",
                          "condition": c, "test_condition": c, "pooling": p, "layer": layer,
                          "n_seeds": 5, "dev_eer_mean": dev} for c, p, layer, dev in rows])


def test_select_on_dev_takes_the_lowest_dev_eer_and_the_lower_layer_on_a_tie():
    summary = frame([("clean", "stats", 3, 0.43), ("clean", "stats", 4, 0.54),
                     ("clean", "stats", 5, 0.50), ("clean", "mean", 3, 0.44),
                     ("clean", "mean", 5, 0.44), ("clean", "mean", 4, 0.60),
                     ("trim40", "mean", 5, 1.60), ("trim40", "mean", 4, 1.89)])
    best = S.select_on_dev(summary).set_index(["condition", "pooling"])
    assert best.loc[("clean", "stats"), "layer"] == 3
    assert best.loc[("clean", "mean"), "layer"] == 3   # tie of the stored means: lower layer
    assert best.loc[("trim40", "mean"), "layer"] == 5
    assert best.loc[("clean", "stats"), "best_in_condition"]
    assert not best.loc[("clean", "mean"), "best_in_condition"]


def test_selection_ignores_every_cell_that_is_not_the_matched_dev_cell():
    summary = frame([("clean", "stats", 3, 0.6), ("clean", "stats", 4, 0.5)])
    cross = summary.assign(test_condition="trim40", dev_eer_mean=[0.1, 9.0])
    summary = pd.concat([summary, cross], ignore_index=True)
    summary["eval_eer_mean"] = [0.1, 9.0, 0.1, 9.0]   # evaluation results never select
    summary["itw_eer_mean"] = [0.1, 9.0, 0.1, 9.0]
    assert S.select_on_dev(summary)["layer"].tolist() == [4]


def test_selected_configs_labels():
    summary = frame([("clean", "stats", 3, 0.43), ("clean", "stats", 4, 0.54),
                     ("clean", "mean", 3, 0.44), ("clean", "mean", 4, 0.60),
                     ("trim40", "mean", 4, 1.5), ("trim40", "mean", 5, 1.6),
                     ("trim40", "stats", 5, 1.6), ("trim40", "stats", 4, 1.9),
                     ("trim20", "stats", 2, 0.1)])
    configs = S.selected_configs(summary, "wavlm-base").set_index("config")["selection"].to_dict()
    assert configs == {"clean/mean/L3": "dev-selected", "clean/stats/L3": "dev-selected",
                       "clean/stats/L4": "reference", "trim40/mean/L4": "reference + dev-selected",
                       "trim40/stats/L5": "dev-selected"}
    hubert = summary.assign(encoder="hubert-base")
    assert set(S.selected_configs(hubert, "hubert-base")["selection"]) == {"dev-selected"}
    assert S.REFERENCE_CONFIGS == {"wavlm-base": (("trim40", "mean", 4), ("clean", "stats", 4))}


def test_padded_vs_unpadded():
    def cell(extraction, frames, condition, pooling, eval_eer):
        return {"encoder": "wavlm-base", "extraction": extraction, "frames": frames,
                "condition": condition, "test_condition": condition, "pooling": pooling,
                "layer": 4, "n_seeds": 5, "eval_eer_mean": eval_eer, "eval_eer_sd": 0.02}

    summary = pd.DataFrame([cell("padded", "all-frames", "clean", "stats", 1.2),
                            cell("unpadded", "", "clean", "stats", 1.5),
                            cell("padded", "real-frames", "trim40", "mean", 6.4)])
    out = S.padded_vs_unpadded(summary)
    assert out["extraction"].tolist() == ["padded", "unpadded", "unpadded - padded"] * 2
    clean = out[out["config"] == "clean/stats/L4"].set_index("extraction")
    assert clean.loc["unpadded - padded", "eval_eer_mean"] == pytest.approx(0.3)
    assert np.isnan(clean.loc["unpadded - padded", "eval_eer_sd"])
    trimmed = out[out["config"] == "trim40/mean/L4"].set_index("extraction")
    assert trimmed.loc["unpadded", "n_seeds"] == 0 and np.isnan(trimmed.loc["unpadded", "eval_eer_mean"])
    assert trimmed.loc["padded", "frames"] == "real-frames"
