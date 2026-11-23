"""Metrics on synthetic data with known answers."""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

from ssl_silence import config as cfg
from ssl_silence import metrics as M

ASV = M.AsvErrorRates(Pfa_asv=0.02, Pmiss_asv=0.05, Pmiss_spoof_asv=0.4,
                      eer_asv=0.03, threshold=0.0)
# With the default cost model:
#   C1 = 0.9405 * (1 - 0.05) - 0.0095 * 10 * 0.02 = 0.891575
#   C2 = 10 * 0.05 * (1 - 0.4)                    = 0.3
C1, C2 = 0.891575, 0.3


# EER -------------------------------------------------------------------------

def test_eer_perfect_separation_is_zero():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert M.eer(labels, scores) == pytest.approx(0.0)
    assert M.eer_roc_corners(labels, scores) == pytest.approx(0.0)


def test_eer_inverted_scores_is_hundred():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.7, 0.8, 0.9, 0.1, 0.2, 0.3])
    assert M.eer(labels, scores) == pytest.approx(100.0)
    assert M.eer_roc_corners(labels, scores) == pytest.approx(100.0)


def test_eer_known_value():
    # Threshold 0.6: one of four bonafide (0.6) is flagged and one of four
    # spoof (0.4) is missed, so both error rates are 25 %.
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])
    assert M.eer(labels, scores) == pytest.approx(25.0)


def test_roc_corner_eer_differs_from_the_official_eer():
    # The crossing at threshold 0.6 (both error rates 25 %) lies on a vertical
    # ROC segment that the corner-point curve drops; the nearest corner gives
    # 37.5 %.
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.6, 0.8, 0.9])
    assert M.eer_roc_corners(labels, scores) == pytest.approx(37.5)
    assert M.eer(labels, scores) == pytest.approx(25.0)


def test_eer_is_the_official_function_in_percent():
    rng = np.random.default_rng(0)
    bona = rng.normal(0.0, 1.0, 4000)
    spoof = rng.normal(2.0, 1.0, 6000)
    labels = np.r_[np.zeros(4000, int), np.ones(6000, int)]
    scores = np.r_[bona, spoof]
    value = M.eer(labels, scores)
    official, _ = M.compute_eer(-bona, -spoof)
    assert value == official * 100
    assert M.eer_roc_corners(labels, scores) == pytest.approx(value, abs=0.05)
    # Two unit-variance Gaussians two standard deviations apart: Phi(-1) = 15.87 %.
    assert value == pytest.approx(15.87, abs=0.6)


def test_eer_ignores_the_order_of_the_trials_and_accepts_float32():
    rng = np.random.default_rng(3)
    labels = (rng.random(2000) < 0.9).astype(int)
    scores = (rng.normal(0, 1, 2000) + labels).astype(np.float32)
    order = rng.permutation(2000)
    assert M.eer(labels, scores) == M.eer(labels[order], scores[order])
    assert M.eer(labels, scores) == M.eer(labels, scores.astype(np.float64))


def test_eer_is_invariant_to_a_monotone_transform():
    rng = np.random.default_rng(4)
    labels = np.r_[np.zeros(300, int), np.ones(700, int)]
    logits = np.r_[rng.normal(-1, 1, 300), rng.normal(1, 1, 700)]
    assert M.eer(labels, logits) == M.eer(labels, 3.0 * logits + 5.0)


# Official DET curve and ASV error rates ----------------------------------------

def test_det_curve_small_example():
    frr, far, thr = M.compute_det_curve(np.array([2.0, 3.0]), np.array([0.0, 1.0]))
    np.testing.assert_allclose(frr, [0, 0, 0, 0.5, 1.0])
    np.testing.assert_allclose(far, [1, 0.5, 0, 0, 0])
    np.testing.assert_allclose(thr, [-0.001, 0, 1, 2, 3])


def test_det_curve_puts_targets_first_among_tied_scores():
    # Stable sort of (targets, nontargets): at the tied score the target is
    # rejected before the nontarget.
    frr, far, _ = M.compute_det_curve(np.array([1.0]), np.array([1.0]))
    np.testing.assert_allclose(frr, [0, 1, 1])
    np.testing.assert_allclose(far, [1, 1, 0])


def test_asv_error_rates_at_a_threshold():
    tar = np.array([1.0, 2.0, 3.0, 4.0])
    non = np.array([0.0, 1.0, 2.5, 0.5])
    spoof = np.array([0.0, 3.0, 3.5, 5.0, 1.0])
    pfa, pmiss, pmiss_spoof, pfa_spoof = M.obtain_asv_error_rates(tar, non, spoof, 2.0)
    assert pfa == pytest.approx(0.25)
    assert pmiss == pytest.approx(0.25)
    assert pmiss_spoof == pytest.approx(0.4)
    assert pfa_spoof == pytest.approx(0.6)
    assert M.obtain_asv_error_rates(tar, non, np.array([]), 2.0)[2:] == (None, None)


def test_asv_error_rates_from_file(tmp_path):
    lines = (["LA_0001 target 5.0", "LA_0001 target 4.0", "LA_0001 target 3.0",
              "LA_0001 target 2.0"]
             + ["LA_0002 nontarget -1.0", "LA_0002 nontarget -2.0",
                "LA_0002 nontarget 0.0", "LA_0002 nontarget 1.0"]
             + ["A07 spoof 4.5", "A07 spoof 2.5", "A08 spoof 1.5", "A08 spoof -3.0"])
    path = tmp_path / "asv.txt"
    path.write_text("\n".join(lines) + "\n")
    asv = M.load_asv_error_rates(str(path))
    # Separable ASV: EER 0 at threshold 1.0, the highest nontarget score, which
    # is accepted by the ">= threshold" rule of the error-rate function.
    assert asv.eer_asv == pytest.approx(0.0)
    assert asv.threshold == pytest.approx(1.0)
    assert asv.Pmiss_asv == pytest.approx(0.0)
    assert asv.Pfa_asv == pytest.approx(0.25)
    assert asv.Pmiss_spoof_asv == pytest.approx(0.25)


def test_asv_error_rates_need_spoof_trials():
    with pytest.raises(ValueError):
        M.asv_error_rates(np.array([1.0, 2.0]), np.array([0.0, 0.5]), np.array([]))


# Tandem t-DCF ------------------------------------------------------------------

def test_default_cost_model():
    c = M.default_cost_model()
    assert c["Pspoof"] == 0.05
    assert c["Ptar"] == pytest.approx(0.9405)
    assert c["Pnon"] == pytest.approx(0.0095)
    assert (c["Cmiss_asv"], c["Cfa_asv"], c["Cmiss_cm"], c["Cfa_cm"]) == (1, 10, 1, 10)
    assert c["Ptar"] + c["Pnon"] + c["Pspoof"] == pytest.approx(1.0)


def test_tandem_tdcf_hand_computed():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([-3.0, -4.0, -5.0, -6.0, 0.0, -1.0, -2.0, -3.5])
    # The best threshold accepts every bonafide and one spoof (-3.5):
    # Pmiss_cm = 0, Pfa_cm = 0.25, cost C2 * 0.25 / min(C1, C2) = 0.25.
    assert M.tandem_min_tdcf(labels, scores, ASV) == pytest.approx(0.25)


def test_tandem_tdcf_perfect_and_inverted():
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert M.tandem_min_tdcf(labels, np.array([.1, .2, .3, .7, .8, .9]), ASV) == pytest.approx(0.0)
    # An inverted CM can do no better than accepting everything: C2 / min(C1, C2) = 1.
    assert M.tandem_min_tdcf(labels, np.array([.7, .8, .9, .1, .2, .3]), ASV) == pytest.approx(1.0)


def test_tandem_tdcf_matches_brute_force():
    rng = np.random.default_rng(1)
    bona = rng.normal(0.0, 1.0, 300)
    spoof = rng.normal(1.5, 1.0, 500)
    labels = np.r_[np.zeros(300, int), np.ones(500, int)]
    scores = np.r_[bona, spoof]
    best = np.inf
    for t in np.r_[scores, scores.max() + 1.0]:
        pmiss_cm = np.mean(bona >= t)
        pfa_cm = np.mean(spoof < t)
        best = min(best, (C1 * pmiss_cm + C2 * pfa_cm) / min(C1, C2))
    assert M.tandem_min_tdcf(labels, scores, ASV) == pytest.approx(best, abs=1e-12)


def test_tandem_tdcf_is_invariant_to_a_monotone_transform():
    rng = np.random.default_rng(2)
    labels = np.r_[np.zeros(200, int), np.ones(300, int)]
    logits = np.r_[rng.normal(-1, 1, 200), rng.normal(1, 1, 300)]
    probs = 1 / (1 + np.exp(-logits))
    assert M.tandem_min_tdcf(labels, logits, ASV) == pytest.approx(
        M.tandem_min_tdcf(labels, probs, ASV), abs=1e-12)


def test_tdcf_rejects_hard_decisions_bad_priors_and_bad_scores():
    model = M.default_cost_model()
    with pytest.raises(ValueError):
        M.compute_tDCF_legacy(np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0.02, 0.05, 0.4,
                              model, False)
    with pytest.raises(ValueError):
        M.compute_tDCF_legacy(np.array([1.0, 2.0]), np.array([0.0, 0.5]), 0.02, 0.05, 0.4,
                              dict(model, Ptar=0.5), False)
    with pytest.raises(ValueError):
        M.compute_tDCF_legacy(np.array([1.0, np.nan]), np.array([0.0, 0.5]), 0.02, 0.05, 0.4,
                              model, False)
    with pytest.raises(ValueError):
        M.compute_tDCF_legacy(np.array([1.0, 2.0]), np.array([0.0, 0.5]), 0.02, 0.05, None,
                              model, False)


def test_tdcf_prints_the_cost_model(capsys):
    M.compute_tDCF_legacy(np.array([1.0, 2.0]), np.array([0.0, 0.5]), 0.02, 0.05, 0.4,
                          M.default_cost_model(), True)
    assert "t-DCF MODEL" in capsys.readouterr().out


# ASV score files -----------------------------------------------------------------

def test_extract_asv_scores(tmp_path):
    archive = tmp_path / "LA.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for split, name in cfg.ASV_SCORE_FILENAMES.items():
            z.writestr(f"LA/ASVspoof2019_LA_asv_scores/{name}", f"x target 1.0 {split}\n")
        z.writestr("LA/other.txt", "ignored")
    out = M.extract_asv_scores(str(archive), str(tmp_path / "asv"))
    assert set(out) == {"dev", "eval"}
    assert open(out["eval"]).read() == "x target 1.0 eval\n"
    open(out["eval"], "w").write("kept")
    assert M.extract_asv_scores(str(archive), str(tmp_path / "asv")) == out
    assert open(out["eval"]).read() == "kept"
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("readme.txt", "no scores")
    with pytest.raises(FileNotFoundError):
        M.extract_asv_scores(str(empty), str(tmp_path / "none"))


def test_asv_score_paths():
    assert cfg.asv_score_file("eval", "/data").replace("\\", "/") == \
        "/data/datasets/asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
