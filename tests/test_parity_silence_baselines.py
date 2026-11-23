"""The silence-only classifier and the padding-only score against the original
analysis code. Skipped without the original code."""

from __future__ import annotations

import functools

import numpy as np
import pytest

from signals import write_file_list
from ssl_silence import silence as S
from ssl_silence import training as T
from test_parity_silence_measurements import frame_difference

SPLITS = ("train", "dev", "eval", "itw")
OLD_TOP_DBS = (30, 40, 50)
OLD_CONDITIONS = {"clean": "clean", "trim40": "trimmed"}


@pytest.fixture(scope="module")
def measured(tmp_path_factory, original):
    """Four splits measured by both codes (untrimmed audio and residual silence)."""
    O = original("trim")
    folder = tmp_path_factory.mktemp("audio")
    new_dir, old_dir = (str(tmp_path_factory.mktemp(n)) for n in ("new", "old"))
    new = {"silence": {}, "residual": {}}
    old = {"silence": {}, "residual": {}}
    for seed, split in enumerate(SPLITS):
        file_list = write_file_list(folder, split, seed=10 + seed)
        wide = S.measure_items(file_list, new_dir, split, progress=False)
        new["silence"][split], new["residual"][split] = S.silence_table(wide), S.residual_table(wide)
        old["silence"][split] = O.measure_items(file_list, old_dir, split, progress=False)
        old["residual"][split] = O.measure_items(
            file_list, old_dir, f"residual_{split}", progress=False,
            measure=functools.partial(O.measure_file, pre_trim_top_db=S.PRE_TRIM_TOP_DB,
                                      top_dbs=S.RESIDUAL_TOP_DBS))
    return O, new, old


def record(parity_record, name, differences):
    parity_record("silence", name, len(differences), max(differences))
    assert max(differences) == 0.0, f"{name}: max abs diff {max(differences)}"


def test_mannwhitney_auc(original, parity_record):
    O = original("stats")
    rng = np.random.default_rng(6)
    differences = []
    for n0, n1, digits in ((5, 7, 3), (40, 300, 1), (200, 200, 0)):
        bonafide, spoof = np.round(rng.normal(0, 1, n0), digits), np.round(rng.normal(0.4, 1, n1), digits)
        new, old = S.mannwhitney_auc(bonafide, spoof), O.mannwhitney_auc(bonafide, spoof)
        assert list(new) == list(old)
        differences += [abs(new[k] - old[k]) for k in old]
    record(parity_record, "mannwhitney_auc", differences)


def test_silence_only_classifier(measured, original, parity_record, tmp_path):
    O, new, old = measured
    old_metrics = original("metrics")
    scores, eers = [], []
    for part, condition, top_dbs in (("silence", "clean", OLD_TOP_DBS),
                                     ("residual", "trim40", (15, 20, 30))):
        table = S.silence_only_baselines(new[part], top_dbs, condition,
                                         scores_dir=str(tmp_path / "new")
                                         ).set_index(["features", "top_db"])
        tests = {k: v for k, v in old[part].items() if k not in ("train", "dev")}
        for features, with_duration in S.FEATURE_SETS.items():
            for d in top_dbs:
                result = O.silence_baseline(old[part]["train"], old[part]["dev"], tests, top_db=d,
                                            with_duration=with_duration,
                                            results_dir=str(tmp_path / "old" / part),
                                            overwrite=True)
                sets, _ = T.load_scores(S.baseline_score_path(str(tmp_path / "new"), features, d,
                                                              condition))
                assert set(sets) == {T.set_name(split, condition) for split in result.scores}
                for split, reference in result.scores.items():
                    mine = sets[T.set_name(split, condition)]
                    assert mine.scores.dtype == reference.scores.dtype
                    assert np.array_equal(mine.labels, reference.labels)
                    assert list(mine.ids) == list(reference.ids)
                    scores.append(float(np.max(np.abs(mine.scores.astype(np.float64)
                                                      - reference.scores.astype(np.float64)))))
                    eers.append(abs(table.loc[(features, d), split]
                                    - old_metrics.eer(reference.labels, reference.scores)))
    assert len(scores) == 2 * 2 * 3 * 3
    record(parity_record, "silence_only_baseline (scores; untrimmed audio and residual silence)",
           scores)
    record(parity_record, "silence_only_baselines (EER table)", eers)


def test_padding_by_label_and_padding_only_score(measured, original, parity_record):
    _, new, old = measured
    A = original("analysis")
    for name, new_function, old_function in (
            ("padding_by_label", S.padding_by_label, A.padding_by_label),
            ("padding_only_baseline", S.padding_only_baseline, A.padding_baseline)):
        mine, reference = new_function(new["silence"]), old_function(old["silence"])
        assert list(mine.columns) == list(reference.columns) and len(mine) == len(reference)
        mine = mine.assign(condition=mine["condition"].map(OLD_CONDITIONS))
        record(parity_record, name, [frame_difference(mine, reference)])
    assert (S.padding_by_label(new["silence"])["group"] != "all").any()
