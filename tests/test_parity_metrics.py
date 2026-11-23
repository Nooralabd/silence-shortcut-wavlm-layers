"""metrics against the original analysis code, and against the reference values
of a saved set of real evaluation scores. Skipped without the original code."""

from __future__ import annotations

import os

import numpy as np
import pytest

from ssl_silence import config as cfg
from ssl_silence import metrics as M

REFERENCE_SCORES = "cm_scores_L4.npz"
# Reference values of the saved evaluation scores (layer 4, statistics pooling,
# seed 42 of the earliest pipeline), verified on the Colab runtime.
REFERENCE = {"eer": 1.2100524, "eer_roc_corners": 1.2108350, "tandem_min_tdcf": 0.0354752,
             "asv_eer_percent": 2.4578, "Pfa_asv": 0.024605, "Pmiss_asv": 0.024581,
             "Pmiss_spoof_asv": 0.239348}
ASV = M.AsvErrorRates(Pfa_asv=0.02, Pmiss_asv=0.05, Pmiss_spoof_asv=0.4, eer_asv=0.03,
                      threshold=0.0)


def cases():
    """Synthetic (name, labels, scores) cases: sizes, class balances, dtypes and ties."""
    rng = np.random.default_rng(20260)
    out = []
    sizes = [50, 200, 1_000, 5_000, 20_000, 80_000]
    for k in range(192):
        n = sizes[k % len(sizes)]
        p_spoof = (0.5, 0.7, 0.9)[k % 3]
        labels = (rng.random(n) < p_spoof).astype(np.int64)
        labels[0], labels[1] = 0, 1
        separation = (0.0, 0.5, 1.5, 3.0)[k % 4]
        logits = rng.normal(0.0, 1.0, n) + separation * labels
        scores = 1.0 / (1.0 + np.exp(-logits))
        kind = k % 4
        if kind == 0:
            scores = scores.astype(np.float32)
        elif kind == 1:
            scores = np.round(scores, 2)
        elif kind == 2:
            scores = np.round(scores.astype(np.float32), 1)
        out.append((f"random{k}", labels, scores))
    labels = np.r_[np.zeros(40, np.int64), np.ones(60, np.int64)]
    ramp = np.linspace(0.0, 1.0, 100)
    out += [("separable", labels, ramp), ("inverted", labels, ramp[::-1].copy()),
            ("constant", labels, np.full(100, 0.5)),
            ("two_values", labels, labels.astype(np.float64)),
            ("float32_saturated", labels, np.r_[np.zeros(40), np.ones(60)].astype(np.float32)),
            ("single_pair", np.array([0, 1]), np.array([0.2, 0.9])),
            ("tie_pair", np.array([0, 1]), np.array([0.5, 0.5])),
            ("imbalanced", np.r_[0, np.ones(999, np.int64)], np.linspace(0.0, 1.0, 1000))]
    return out


def outcome(function, *args):
    """Value of a call, or the type of the exception it raises."""
    try:
        return function(*args)
    except Exception as error:  # compared by type with the original code
        return type(error)


def difference(new, old) -> float:
    """Largest absolute difference between two outcomes (inf if they are not comparable)."""
    if isinstance(new, type) or isinstance(old, type):
        return 0.0 if new is old else np.inf
    if isinstance(new, tuple):
        if not isinstance(old, tuple) or len(new) != len(old):
            return np.inf
        return max(difference(a, b) for a, b in zip(new, old))
    if new is None or old is None:
        return 0.0 if new is old else np.inf
    a, b = np.asarray(new, dtype=np.float64), np.asarray(old, dtype=np.float64)
    if a.shape != b.shape:
        return np.inf
    if a.size == 0:
        return 0.0
    equal_nan = np.isnan(a) & np.isnan(b)
    return float(np.max(np.where(equal_nan, 0.0, np.abs(a - b))))


def compare(parity_record, name, pairs):
    worst = max(difference(new, old) for new, old in pairs)
    parity_record("metrics", name, len(pairs), worst)
    assert worst == 0.0, f"{name}: max abs diff {worst}"


def split(labels, scores):
    cm = -np.asarray(scores, dtype=np.float64)
    return cm[labels == 0], cm[labels == 1]


def test_eer(original, parity_record):
    O = original("metrics")
    compare(parity_record, "eer",
            [(outcome(M.eer, y, s), outcome(O.eer, y, s)) for _, y, s in cases()])


def test_eer_roc_corners(original_function, parity_record):
    old = original_function("metrics", "eer_roc_corners")
    compare(parity_record, "eer_roc_corners",
            [(outcome(M.eer_roc_corners, y, s), outcome(old, y, s)) for _, y, s in cases()])


def test_det_curve_and_compute_eer(original, parity_record):
    O = original("metrics")
    det, eer = [], []
    for _, y, s in cases():
        bona, spoof = split(y, s)
        det.append((outcome(M.compute_det_curve, bona, spoof),
                    outcome(O.compute_det_curve, bona, spoof)))
        eer.append((outcome(M.compute_eer, bona, spoof), outcome(O.compute_eer, bona, spoof)))
    compare(parity_record, "compute_det_curve", det)
    compare(parity_record, "compute_eer", eer)


def test_tandem_tdcf(original, parity_record):
    O = original("metrics")
    old_asv = O.AsvErrorRates(**ASV.__dict__)
    minimum, curve = [], []
    model = M.default_cost_model()
    assert model == O.default_cost_model()
    for _, y, s in cases():
        minimum.append((outcome(M.tandem_min_tdcf, y, s, ASV),
                        outcome(O.tandem_min_tdcf, y, s, old_asv)))
        bona, spoof = split(y, s)
        args = (bona, spoof, ASV.Pfa_asv, ASV.Pmiss_asv, ASV.Pmiss_spoof_asv, model, False)
        curve.append((outcome(M.compute_tDCF_legacy, *args),
                      outcome(O.compute_tDCF_legacy, *args)))
    compare(parity_record, "tandem_min_tdcf", minimum)
    compare(parity_record, "compute_tDCF_legacy", curve)
    parity_record("metrics", "default_cost_model", 1, 0.0)


def test_asv_error_rates(original, parity_record, tmp_path):
    O = original("metrics")
    rng = np.random.default_rng(7)
    fixed, at_eer, from_file = [], [], []
    for k in range(40):
        n = (30, 300, 3_000, 30_000)[k % 4]
        tar = rng.normal(2.0, 1.0, n)
        non = rng.normal(0.0, 1.0, n // 2 + 1)
        spoof = rng.normal(1.0, 1.5, n)
        if k % 2:
            tar, non, spoof = np.round(tar, 1), np.round(non, 1), np.round(spoof, 1)
        threshold = float(rng.normal(1.0, 0.5))
        fixed.append((M.obtain_asv_error_rates(tar, non, spoof, threshold),
                      O.obtain_asv_error_rates(tar, non, spoof, threshold)))
        new, old = M.asv_error_rates(tar, non, spoof), O.asv_error_rates(tar, non, spoof)
        at_eer.append((tuple(new.__dict__.values()), tuple(old.__dict__.values())))
        if n <= 3_000:
            path = tmp_path / f"asv_{k}.txt"
            with open(path, "w") as fh:
                for key, values in (("target", tar), ("nontarget", non), ("spoof", spoof)):
                    for value in values:
                        fh.write(f"src {key} {float(value)!r}\n")
            new, old = M.load_asv_error_rates(str(path)), O.load_asv_error_rates(str(path))
            from_file.append((tuple(new.__dict__.values()), tuple(old.__dict__.values())))
    compare(parity_record, "obtain_asv_error_rates", fixed)
    compare(parity_record, "asv_error_rates", at_eer)
    compare(parity_record, "load_asv_error_rates", from_file)


# Saved real evaluation scores ----------------------------------------------------

def unavailable(message: str) -> None:
    """Skip, or fail when SSL_SILENCE_REQUIRE_REFERENCE=1 makes the reference check mandatory."""
    if os.environ.get("SSL_SILENCE_REQUIRE_REFERENCE") == "1":
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module")
def reference_scores(request):
    """The saved evaluation scores of the original code (one file in its checkout).

    With SSL_SILENCE_REQUIRE_REFERENCE=1 a missing original code or score
    file fails the tests that use it; otherwise they are skipped.
    """
    if not os.environ.get("SSL_SILENCE_ORIGINAL_SRC"):
        unavailable("SSL_SILENCE_ORIGINAL_SRC is not set")
    original_src = request.getfixturevalue("original_src")
    found = sorted(original_src.parent.rglob(REFERENCE_SCORES))
    if len(found) != 1:
        unavailable(f"{REFERENCE_SCORES}: {len(found)} files found in the checkout of the original code, "
                    "expected 1")
    with np.load(found[0], allow_pickle=False) as data:
        return data["labels"], data["scores"]


def test_reference_scores_are_the_full_evaluation_set(reference_scores):
    labels, scores = reference_scores
    assert labels.shape == scores.shape == (cfg.ASVSPOOF_SPLIT_SIZES["eval"],)
    assert int((labels == 0).sum()) == cfg.EVAL_COUNTS["bonafide"]
    assert int((labels == 1).sum()) == cfg.EVAL_COUNTS["spoof"]


def test_reference_eer(reference_scores, original, original_function, parity_record):
    labels, scores = reference_scores
    O = original("metrics")
    value = M.eer(labels, scores)
    assert value == pytest.approx(REFERENCE["eer"], abs=5e-8)
    corners = M.eer_roc_corners(labels, scores)
    assert corners == pytest.approx(REFERENCE["eer_roc_corners"], abs=5e-8)
    compare(parity_record, "eer (saved evaluation scores)", [(value, O.eer(labels, scores))])
    compare(parity_record, "eer_roc_corners (saved evaluation scores)",
            [(corners, original_function("metrics", "eer_roc_corners")(labels, scores))])


def test_reference_tandem_tdcf(reference_scores, original, parity_record):
    asv_dir = os.environ.get("SSL_SILENCE_ASV_SCORES")
    if not asv_dir:
        unavailable("SSL_SILENCE_ASV_SCORES is not set (folder of the official ASV score files)")
    path = os.path.join(asv_dir, cfg.ASV_SCORE_FILENAMES["eval"])
    if not os.path.exists(path):
        unavailable(f"official ASV score file not found: {path}")
    labels, scores = reference_scores
    O = original("metrics")
    asv = M.load_asv_error_rates(path)
    assert asv.eer_asv * 100 == pytest.approx(REFERENCE["asv_eer_percent"], abs=5e-5)
    assert asv.Pfa_asv == pytest.approx(REFERENCE["Pfa_asv"], abs=5e-7)
    assert asv.Pmiss_asv == pytest.approx(REFERENCE["Pmiss_asv"], abs=5e-7)
    assert asv.Pmiss_spoof_asv == pytest.approx(REFERENCE["Pmiss_spoof_asv"], abs=5e-7)
    value = M.tandem_min_tdcf(labels, scores, asv)
    assert value == pytest.approx(REFERENCE["tandem_min_tdcf"], abs=5e-8)
    old = O.tandem_min_tdcf(labels, scores, O.load_asv_error_rates(path))
    compare(parity_record, "tandem_min_tdcf (saved evaluation scores)", [(value, old)])


def test_a_required_reference_fails_instead_of_skipping(monkeypatch):
    monkeypatch.setenv("SSL_SILENCE_REQUIRE_REFERENCE", "1")
    with pytest.raises(pytest.fail.Exception, match="not found"):
        unavailable("score file not found")
    monkeypatch.delenv("SSL_SILENCE_REQUIRE_REFERENCE")
    with pytest.raises(pytest.skip.Exception):
        unavailable("score file not found")
