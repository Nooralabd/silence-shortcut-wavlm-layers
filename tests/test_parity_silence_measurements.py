"""Silence measurements against the original analysis code.
Skipped without the original code."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from signals import trimming_cases, write_file_list
from ssl_silence import silence as S
from test_parity_trimming import same

OLD_TOP_DBS = (30, 40, 50)          # the thresholds of the original measurement
NEW_ONLY = ("peak_frame_rms",)      # measured by this package only


def compare(parity_record, name, pairs):
    worst = max(same(new, old) for new, old in pairs)
    parity_record("silence", name, len(pairs), worst)
    assert worst == 0.0, f"{name}: max abs diff {worst}"


def without(measurement: dict, old: dict) -> dict:
    """The keys of the original measurement, in its order; the rest must be new columns."""
    extra = set(measurement) - set(old)
    assert extra <= set(NEW_ONLY), extra
    return {k: measurement[k] for k in old if k in measurement}


def frame_difference(new: pd.DataFrame, old: pd.DataFrame) -> float:
    """0.0 if the two tables hold the same rows and values in the columns of ``old``."""
    if len(new) != len(old) or not set(old.columns) <= set(new.columns):
        return np.inf
    worst = 0.0
    for column in old.columns:
        a, b = new[column].to_numpy(), old[column].to_numpy()
        if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
            worst = max(worst, 0.0 if list(map(str, a)) == list(map(str, b)) else np.inf)
        else:
            a, b = a.astype(np.float64), b.astype(np.float64)
            both_nan = np.isnan(a) & np.isnan(b)
            worst = max(worst, float(np.max(np.abs(np.where(both_nan, 0.0, a - b)), initial=0.0)))
    return worst


@pytest.fixture(scope="module")
def tables(tmp_path_factory, original):
    """The same files measured by both codes: the one wide table of this
    package and the three tables of the original measurement passes."""
    O = original("trim")
    file_list = write_file_list(tmp_path_factory.mktemp("audio"))
    new_dir, old_dir = (str(tmp_path_factory.mktemp(n)) for n in ("new", "old"))
    wide = S.measure_items(file_list, new_dir, "eval", chunk_size=9, progress=False,
                           measure=functools.partial(S.measure_file, top_dbs=OLD_TOP_DBS))
    old = {
        "silence": O.measure_items(file_list, old_dir, "eval", chunk_size=9, progress=False),
        "residual": O.measure_items(
            file_list, old_dir, "residual", chunk_size=9, progress=False,
            measure=functools.partial(O.measure_file, pre_trim_top_db=S.PRE_TRIM_TOP_DB,
                                      top_dbs=S.RESIDUAL_TOP_DBS)),
        "removed": O.measure_items(
            file_list, old_dir, "removed", chunk_size=9, progress=False,
            measure=functools.partial(O.removed_amounts, top_dbs=S.REMOVED_TOP_DBS))}
    return {"O": O, "file_list": file_list, "wide": wide, "old": old, "new_dir": new_dir}


def test_window_view_and_one_waveform(original, parity_record):
    O = original("trim")
    views = []
    for n in (0, 399, 30_000, 64_000, 64_001, 100_000):
        for start, end in ((0, 0), (0, n), (n // 4, n // 2), (n // 2, n), (n, n)):
            views.append((S.window_view(n, start, end), O.window_view(n, start, end)))
    views.append((S.window_view(100, 10, 60, window=80), O.window_view(100, 10, 60, window=80)))
    compare(parity_record, "window_view", views)
    waveforms = []
    for wav in trimming_cases().values():
        y = wav[0].numpy()
        for top_dbs in (OLD_TOP_DBS, (15, 20, 30, 40)):
            old = O.measure_silence(y, top_dbs)
            waveforms.append((without(S.measure_silence(y, top_dbs), old), old))
    compare(parity_record, "measure_silence", waveforms)


def test_one_file_measured_once_equals_three_original_passes(tables, parity_record):
    O = tables["O"]
    untrimmed, residual, removed = [], [], []
    for path in tables["file_list"]["path"]:
        new = S.measure_file(path, top_dbs=OLD_TOP_DBS)
        old = O.measure_file(path, top_dbs=OLD_TOP_DBS)
        plain = {k: v for k, v in new.items()
                 if not k.startswith((S.RESIDUAL_PREFIX, S.AMOUNT_PREFIX))}
        untrimmed.append((without(plain, old), old))
        old = O.measure_file(path, top_dbs=S.RESIDUAL_TOP_DBS, pre_trim_top_db=S.PRE_TRIM_TOP_DB)
        part = {"sr_orig": new["sr_orig"], "n_channels": new["n_channels"],
                **{k[len(S.RESIDUAL_PREFIX):]: v for k, v in new.items()
                   if k.startswith(S.RESIDUAL_PREFIX)}}
        residual.append((without(part, old), old))
        old = O.removed_amounts(path, S.REMOVED_TOP_DBS)
        part = {k[len(S.AMOUNT_PREFIX):]: v for k, v in new.items()
                if k.startswith(S.AMOUNT_PREFIX)}
        removed.append((part, old))
    compare(parity_record, "measure_file (untrimmed audio)", untrimmed)
    compare(parity_record, "measure_file (residual silence after trimming at 40 dB)", residual)
    compare(parity_record, "measure_file (removed amounts at 40, 20 and 15 dB)", removed)


def test_tables_of_a_split(tables, parity_record):
    """The tables as the original code returns them (read back from its CSV files
    with the default parser) against the parts of the wide table read the same way."""
    wide = pd.read_csv(f"{tables['new_dir']}/eval.csv",
                       dtype={"audio_id": str, "system_id": str, "speaker": str})
    parts = {"silence": S.silence_table(wide), "residual": S.residual_table(wide),
             "removed": S.removed_table(wide)}
    differences = []
    for name, old in tables["old"].items():
        new = parts[name]
        assert set(new.columns) - set(old.columns) <= set(NEW_ONLY)
        differences.append(frame_difference(new, old))
    # Read exactly, the numbers differ from the default parser by rounding at most.
    exact = S.silence_table(tables["wide"])
    assert frame_difference(exact, tables["old"]["silence"]) < 1e-12
    parity_record("silence", "measure_items (untrimmed, residual and removed tables)",
                  sum(len(t) * len(t.columns) for t in tables["old"].values()), max(differences))
    assert max(differences) == 0.0


def test_summary_share_and_duration(tables, original, parity_record):
    O = tables["O"]
    silence = S.silence_table(tables["wide"])
    residual = S.residual_table(tables["wide"])
    differences = [frame_difference(S.summarise(silence, d), O.summarise(tables["old"]["silence"], d))
                   for d in (30, 40)]
    differences.append(frame_difference(S.summarise(residual, 15),
                                        O.summarise(tables["old"]["residual"], 15)))
    parity_record("silence", "summarise", len(differences), max(differences))
    assert max(differences) == 0.0

    old_residual = tables["old"]["residual"]

    def share_differences(new_table, old_table, top_db=None):
        """This package's counts against the row of the original feasibility table."""
        halves = {"first": new_table.iloc[::2], "second": new_table.iloc[1::2], "all": new_table}
        old_halves = {"first": old_table.iloc[::2], "second": old_table.iloc[1::2],
                      "all": old_table}
        new = S.residual_silence_share(halves, top_db=top_db).set_index("table")
        old = O.strict_trim_feasibility(old_halves, top_dbs=(15, 20))
        out = []
        for name in halves:
            out += [abs(new.loc[name, "n_files"] - old.loc["n", name]),
                    abs(new.loc[name, "n_fallback"] - old.loc["fallback_40", name]),
                    abs(new.loc[name, "n_lead_over"] - old.loc["lead40_gt50ms", name]),
                    abs(new.loc[name, "n_trail_over"] - old.loc["trail40_gt50ms", name])]
        return out, int(new.loc["all", "n_lead_over"]), int(new.loc["all", "n_trail_over"])

    # 1. The measured tables at the trimming threshold. The weak segments are
    #    sound at 40 dB; the file whose trim starts inside a marginal frame keeps
    #    more than 50 ms of silence as defined by 40 dB.
    differences, lead, trail = share_differences(residual, old_residual)
    kept = residual[(residual["lead_40"] > 800)
                    & (residual["fallback_trim40"] == 0)]["audio_id"].tolist()
    assert kept == ["eval_blip_39db_before_tone"] and (lead, trail) == (1, 0)
    # 2. The measured silence at 20 and 15 dB (the files with a weak segment at
    #    an edge keep more than 50 ms). The original function counts the columns
    #    of the trimming threshold, so it is given the measured 20 or 15 dB
    #    columns under those names.
    for d in (20, 15):
        relabelled = old_residual.assign(lead_40=old_residual[f"lead_{d}"],
                                         trail_40=old_residual[f"trail_{d}"])
        part, lead, trail = share_differences(residual, relabelled, top_db=d)
        assert lead >= 2 and trail >= 2
        differences += part
    # 3. Invented lengths, the same in both tables, for a wider range of counts.
    rng = np.random.default_rng(3)
    invented = {"lead_40": rng.integers(0, 2_000, len(residual)),
                "trail_40": rng.integers(0, 1_200, len(residual))}
    part, lead, _ = share_differences(residual.assign(**invented), old_residual.assign(**invented))
    assert lead > 5
    differences += part
    parity_record("silence", "residual_silence_share (40 dB; measured 20 and 15 dB; invented lengths)",
                  len(differences), float(max(differences)))
    assert max(differences) == 0.0


def test_duration_map(tmp_path, original, parity_record):
    A = original("analysis")
    file_list = write_file_list(tmp_path, split="itw", other_rates=False)
    table = S.silence_table(S.measure_items(file_list, str(tmp_path / "m"), "itw", progress=False))
    new, old = S.duration_map(table), A.load_duration_map("itw", str(tmp_path / "m"))
    assert list(new) == list(old) and len(new) == len(file_list)
    differences = [abs(new[k] - old[k]) for k in new]
    parity_record("silence", "duration_map", len(differences), max(differences))
    assert max(differences) == 0.0
