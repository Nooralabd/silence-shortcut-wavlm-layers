"""Holm and Benjamini-Hochberg adjustments of the effects, with known answers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control, norm

from ssl_silence import multiplicity as Mu
from ssl_silence import provenance
from ssl_silence import replication

REPO = Path(__file__).resolve().parents[1]
ENCODER = "wavlm-base"


def effects_table(estimates, errors, test_set="itw", effect="delta_test", condition="trim40"):
    n = len(estimates)
    estimates, errors = np.asarray(estimates, dtype=float), np.asarray(errors, dtype=float)
    return pd.DataFrame({
        "encoder": ENCODER, "condition": condition, "test_set": test_set, "pooling": "stats",
        "layer": np.arange(1, n + 1), "effect": effect, "estimate": estimates,
        "spk_ci_low": estimates - 1.96 * errors, "spk_ci_high": estimates + 1.96 * errors,
        "spk_boot_se": errors, "spk_excludes_zero": np.abs(estimates) > 1.96 * errors})


def test_normal_p_is_two_sided():
    assert Mu.normal_p(norm.ppf(0.975), 1.0) == pytest.approx(0.05)
    assert Mu.normal_p(-norm.ppf(0.975) * 3, 3.0) == pytest.approx(0.05)
    assert Mu.normal_p(0.0, 2.0) == 1.0
    assert np.allclose(Mu.normal_p([1.0, -2.0], [1.0, 1.0]), [2 * norm.sf(1), 2 * norm.sf(2)])


def test_holm_by_hand():
    # Sorted: 0.01, 0.02, 0.03, 0.04 -> x4, x3, x2, x1 = 0.04, 0.06, 0.06, 0.04 -> monotone 0.04, 0.06, 0.06, 0.06
    assert np.allclose(Mu.holm([0.04, 0.01, 0.03, 0.02]), [0.06, 0.04, 0.06, 0.06])
    assert np.allclose(Mu.holm([0.01, 0.04, 0.03, 0.2, 0.5]), [0.05, 0.12, 0.12, 0.4, 0.5])
    assert np.allclose(Mu.holm([0.6, 0.7]), [1.0, 1.0])              # capped at 1
    assert np.allclose(Mu.holm([0.03]), [0.03])                       # one comparison: unchanged
    # The first step is Bonferroni: the smallest of 24 needs p < 0.05 / 24.
    p = np.r_[0.0020, np.full(23, 0.9)]
    assert Mu.holm(p)[0] == pytest.approx(0.048) and Mu.holm(np.r_[0.0021, p[1:]])[0] > 0.05


def test_benjamini_hochberg_equals_scipy():
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 24, 100):
        p = rng.random(n) ** 3
        assert np.allclose(Mu.benjamini_hochberg(p), false_discovery_control(p, method="bh"))
    assert np.allclose(Mu.benjamini_hochberg([0.01, 0.04, 0.03, 0.2, 0.5]),
                       [0.05, 0.2 / 3, 0.2 / 3, 0.25, 0.5])
    p = rng.random(24)
    assert (Mu.holm(p) >= Mu.benjamini_hochberg(p) - 1e-15).all()     # Holm is never the smaller one
    assert (Mu.benjamini_hochberg(p) >= p).all() and (Mu.holm(p) <= 1).all()


def test_each_effect_on_each_test_set_is_its_own_family():
    strong = effects_table([5.0, 4.0, -6.0], [1.0, 1.0, 1.0], effect="interaction")
    mixed = effects_table([2.2, 0.1, -2.6, 3.5], [1.0, 1.0, 1.0, 1.0], test_set="eval")
    other = effects_table([9.0], [1.0], condition="trim20")           # another condition: left out
    adjusted = Mu.adjusted_effects(pd.concat([strong, mixed, other], ignore_index=True))
    assert len(adjusted) == 7 and set(adjusted["condition"]) == {"trim40"}
    assert adjusted.groupby(["test_set", "effect"], sort=False)["n_family"].first().tolist() == [3, 4]
    part = adjusted[adjusted["test_set"] == "eval"]
    p = 2 * norm.sf(np.abs([2.2, 0.1, 2.6, 3.5]))
    assert np.allclose(part["p_normal"], p) and np.allclose(part["z"], [2.2, 0.1, -2.6, 3.5])
    assert np.allclose(part["p_holm"], Mu.holm(p)) and np.allclose(part["p_bh"], Mu.benjamini_hochberg(p))
    assert part["significant_unadjusted"].tolist() == [True, False, True, True]
    assert part["significant_holm"].tolist() == [False, False, True, True]     # 0.0278 x 2 = 0.056
    assert part["significant_bh"].tolist() == [True, False, True, True]
    # The family of three is adjusted by three, not by seven.
    first = adjusted[adjusted["effect"] == "interaction"]
    assert np.allclose(first["p_holm"], Mu.holm(2 * norm.sf([5.0, 4.0, 6.0])))
    summary = Mu.summarise(adjusted)
    assert summary[["test_set", "effect", "n_family", "n_interval_excludes_zero", "n_unadjusted", "n_holm",
                    "n_bh"]].to_numpy().tolist() == [["itw", "interaction", 3, 3, 3, 3, 3],
                                                     ["eval", "delta_test", 4, 3, 3, 2, 3]]
    assert summary["interval_but_not_holm"].tolist() == ["", "stats/L1"]
    assert summary["interval_but_not_bh"].tolist() == ["", ""]
    assert summary["interval_but_not_unadjusted"].tolist() == ["", ""]


def test_an_interval_and_the_normal_p_can_disagree_and_the_summary_names_it():
    table = effects_table([1.0, 3.0], [1.0, 1.0])
    table.loc[0, "spk_excludes_zero"] = True          # a skewed percentile interval that excludes zero
    summary = Mu.summarise(Mu.adjusted_effects(table))
    assert summary["interval_but_not_unadjusted"].tolist() == ["stats/L1"]
    assert summary["n_interval_excludes_zero"].tolist() == [2] and summary["n_unadjusted"].tolist() == [1]


def test_bad_tables_raise():
    table = effects_table([1.0, 2.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="no row of trim15"):
        Mu.adjusted_effects(table, condition="trim15")
    with pytest.raises(ValueError, match="twice"):
        Mu.adjusted_effects(pd.concat([table, table], ignore_index=True))
    for bad in (0.0, np.nan):
        broken = table.copy()
        broken.loc[1, "spk_boot_se"] = bad
        with pytest.raises(ValueError, match="standard error"):
            Mu.adjusted_effects(broken)


def test_the_tables_are_written_with_the_sha256_of_the_effects_table(tmp_path):
    effects = pd.concat([effects_table([2.2, 0.1, -2.6, 3.5], [1.0, 1.0, 1.0, 1.0]),
                         effects_table([9.0], [1.0], condition="trim20")], ignore_index=True)
    source = tmp_path / "effects_2x2.csv"
    effects.to_csv(source, index=False, lineterminator="\n")
    table_path, summary_path = Mu.write_tables(source, tmp_path / "out")
    assert (table_path.name, summary_path.name) == (
        "multiple_comparisons_wavlm-base.csv", "multiple_comparisons_summary_wavlm-base.csv")
    written = pd.read_csv(table_path, float_precision="round_trip")
    direct = Mu.adjusted_effects(effects)
    assert list(written.columns) == list(direct.columns) and len(written) == 4
    assert np.array_equal(written["p_holm"], direct["p_holm"])
    for path in (table_path, summary_path):
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta["effects_sha256"] == provenance.sha256_file(source) and meta["approximation"] is True
        assert meta["method"] == Mu.METHOD and meta["alpha"] == 0.05 and meta["condition"] == "trim40"
        assert meta["sha256"] == provenance.sha256_file(path)
    two = pd.concat([effects, effects.assign(encoder="hubert-base")], ignore_index=True)
    two.to_csv(source, index=False, lineterminator="\n")
    with pytest.raises(ValueError, match="2 encoders"):
        Mu.write_tables(source, tmp_path / "out")



def test_the_committed_tables_follow_from_their_own_columns():
    """The tables of WavLM Base in docs/ were computed from the effects table
    on Drive, which the tests cannot read: the adjusted p-values are computed
    again from the estimates and standard errors the committed table holds."""
    table_path = REPO / "docs" / f"{Mu.TABLE}_{ENCODER}.csv"
    summary_path = REPO / "docs" / f"{Mu.SUMMARY}_{ENCODER}.csv"
    if not table_path.exists() and not summary_path.exists():
        pytest.skip("the tables of the multiple comparisons are not in this copy of the repository")
    table = pd.read_csv(table_path, float_precision="round_trip")
    assert len(table) == 288 and set(table["n_family"]) == {24} and set(table["condition"]) == {"trim40"}
    assert table.groupby(["test_set", "effect"]).ngroups == 12
    again = Mu.adjusted_effects(table)
    for column in ("z", "p_normal", "p_holm", "p_bh"):
        assert np.allclose(table[column], again[column], rtol=1e-12, atol=0), column
    for column in ("significant_unadjusted", "significant_holm", "significant_bh"):
        assert table[column].tolist() == again[column].tolist()
    summary = pd.read_csv(summary_path, keep_default_na=False)
    assert summary.astype(str).equals(Mu.summarise(table).astype(str))
    for path in (table_path, summary_path):
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta["sha256"] == replication.sha256_lf(str(path))
        assert meta["approximation"] is True and meta["method"] == Mu.METHOD
        assert len(meta["effects_sha256"]) == 64 and not meta["git_commit"].endswith("-dirty")
    # On the evaluation set nothing changes except for the test-time effect of the model trained on trimmed audio.
    eval_rows = summary[summary["test_set"] == "eval"]
    unchanged = eval_rows[eval_rows["effect"] != "delta_test_t"]
    assert (unchanged[["n_interval_excludes_zero", "n_unadjusted", "n_holm", "n_bh"]] == 24).all().all()
