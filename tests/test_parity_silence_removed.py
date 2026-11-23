"""Removed amounts and the trimmed-length check against the original analysis code.
Skipped without the original code."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from signals import write_file_list
from ssl_silence import config as cfg
from ssl_silence import silence as S
from test_parity_silence_measurements import frame_difference


@pytest.fixture(scope="module")
def removed(tmp_path_factory, original):
    """Removed amounts of two splits: the part of this package's wide table and
    the table of the original measurement pass."""
    O = original("trim")
    folder = tmp_path_factory.mktemp("audio")
    new_dir, old_dir = (str(tmp_path_factory.mktemp(n)) for n in ("new", "old"))
    new, old = [], []
    for seed, split in enumerate(("dev", "eval")):
        file_list = write_file_list(folder, split, seed=seed)
        new.append(S.removed_table(S.measure_items(file_list, new_dir, split, progress=False)))
        old.append(O.measure_items(file_list, old_dir, split, progress=False,
                                   measure=functools.partial(O.removed_amounts,
                                                             top_dbs=S.REMOVED_TOP_DBS)))
    return O, pd.concat(new, ignore_index=True), pd.concat(old, ignore_index=True)


def record(parity_record, name, differences):
    parity_record("silence", name, len(differences), max(differences))
    assert max(differences) == 0.0, f"{name}: max abs diff {max(differences)}"


def test_removed_long_and_summary(removed, parity_record):
    O, new, old = removed
    new_long, old_long = S.removed_long(new), O.removed_long(old)
    assert list(new_long.columns) == list(old_long.columns)
    record(parity_record, "removed_long", [frame_difference(new_long, old_long)])
    new_summary, old_summary = S.summarise_removed(new_long), O.summarise_removed(old_long)
    assert list(new_summary.columns) == list(old_summary.columns) and len(new_summary) == 12
    record(parity_record, "summarise_removed", [frame_difference(new_summary, old_summary)])


def test_fallback_counts_and_length_check(removed, parity_record):
    O, new, old = removed
    new_long, old_long = S.removed_long(new), O.removed_long(old)
    new_counts, old_counts = S.fallback_counts_from_removed(new_long), O.fallback_counts_from_removed(old_long)
    assert new_counts["n_fallback"].sum() > 0
    record(parity_record, "fallback_counts_from_removed", [frame_difference(new_counts, old_counts)])

    rng = np.random.default_rng(5)
    differences, lengths_differences = [], []
    for top_db in S.REMOVED_TOP_DBS:
        measured_new = new_counts[new_counts["top_db"] == top_db]
        measured_old = old_counts[old_counts["top_db"] == top_db]
        reference = measured_old.drop(columns="top_db").reset_index(drop=True)
        reference.loc[1, "n_fallback"] += 1                          # one count disagrees
        new_check = S.compare_fallback_counts(measured_new, reference)
        old_check = O.compare_fallback_counts(measured_old, reference)
        assert list(new_check.columns) == list(old_check.columns) and not new_check["ok"].all()
        differences.append(frame_difference(new_check, old_check))

        at = old_long[old_long["top_db"] == top_db]
        n_real = np.minimum(at["n_samples_trimmed"].to_numpy(), cfg.WINDOW)
        n_real = n_real + (rng.random(n_real.size) < 0.1) * 160      # some lengths disagree
        lengths = pd.DataFrame({"split": at["split"].to_numpy(), "audio_id": at["audio_id"].to_numpy(),
                                "label": at["label"].to_numpy(), "n_real": n_real})
        lengths = lengths.sample(frac=1.0, random_state=1).sort_values("split", kind="mergesort")
        new_diffs = S.trim_length_diffs(new_long, lengths, top_db)
        old_diffs = O.trim_length_diffs(old_long, lengths, top_db)
        lengths_differences += [frame_difference(new_diffs, old_diffs),
                                frame_difference(S.summarise_trim_lengths(new_diffs),
                                                 O.summarise_trim_lengths(old_diffs))]
        assert new_diffs["abs_diff_samples"].max() == 160
    record(parity_record, "compare_fallback_counts", differences)
    record(parity_record, "trim_length_diffs and summarise_trim_lengths", lengths_differences)
