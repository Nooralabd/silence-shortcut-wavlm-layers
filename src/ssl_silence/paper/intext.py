"""Registry of the numbers that the manuscript states in its running text.

An entry ties one phrase of the manuscript to the saved tables: a template
of the phrase in which every number is written ``{}``, and a function that
computes those numbers from the tables. ``check`` finds the phrase in the
source (it must occur exactly once), reads its numbers and compares each
with the computed value, rounded once to the digits the phrase prints.

``coverage`` lists every number of the checked sections that is outside the
tables and figures, with one of these states:

    checked      read and compared by an entry
    cited        a number of prior work, named by an entry of kind "cited"
    approximate  a rounded statement ("about ...") named by an entry of kind
                 "approximate"; it is found but not compared
    structural   a layer, threshold, year, table or equation number, etc.
    unchecked    none of the above: to be covered or looked at by hand

Templates are matched on the source without comments and with every run of
white space read as one space.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .. import config as cfg
from . import tables as T

SILENCE_SUMMARY = "measurements/silence_summary.csv"
RESIDUAL_SUMMARY = "measurements/residual_silence_summary.csv"
RESIDUAL_BASELINE = "measurements/residual_silence_baseline.csv"
RESIDUAL_SHARE = "measurements/residual_silence_share.csv"
PADDING_DEVIATION = "padding_deviation.csv"
PADDING_BY_AMOUNT = "padding_deviation_by_amount.csv"
PADDING_PER_FILE = "padding_deviation_files.csv"
PADDING_BASELINE = "measurements/padding_only_baseline.csv"
PADDING_BY_LABEL = "measurements/padding_by_label.csv"
PADDED_VS_UNPADDED = "padded_vs_unpadded.csv"
BY_DURATION_CI = "itw_by_duration_ci.csv"
SOURCES = (T.SILENCE_BASELINE, T.SEED_SUMMARY, T.SELECTED_CI, T.EFFECTS, T.RANK, T.RANK_LAYERS,
           T.CELLS, T.BY_DURATION, T.PER_ATTACK, T.THRESHOLDS, T.FALLBACK, T.TRIM_AMOUNT,
           T.PROBE_R2, SILENCE_SUMMARY, RESIDUAL_SUMMARY, RESIDUAL_BASELINE, RESIDUAL_SHARE,
           PADDING_DEVIATION, PADDING_BY_AMOUNT, PADDING_PER_FILE, PADDING_BASELINE,
           PADDING_BY_LABEL, PADDED_VS_UNPADDED, BY_DURATION_CI)
NUMBER = r"\$?([-+]?\d+(?:\{,\}\d{3})*(?:\.\d+)?)\$?"
ENCODER = "wavlm-base"
SIX = dict(layer=[3, 4, 5])
ASV = ["train", "dev", "eval"]
STRICT = ["trim20", "trim15"]

Value = tuple[float, int]          # (value, digits printed)
KINDS = ("result", "config", "cited", "approximate", "statement")
UNCOMPARED = ("cited", "approximate")
STATES = ("checked", "cited", "approximate", "structural", "unchecked")


class SourceMissing(LookupError):
    """A table, or the rows an entry needs, are not among the saved tables."""


class Tables:
    """The saved tables of a folder, read when first used."""

    def __init__(self, tables_dir: str):
        self.tables_dir = tables_dir
        self._cache: dict[str, pd.DataFrame] = {}

    @classmethod
    def of(cls, frames: dict[str, pd.DataFrame]) -> "Tables":
        """Tables given as data frames, by their file name; no other table exists."""
        tables = cls(os.devnull)
        for name, table in frames.items():
            if "encoder" in table.columns:
                table = table[table["encoder"] == ENCODER]
            tables._cache[name] = table
        return tables

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            path = os.path.join(self.tables_dir, *name.split("/"))
            if not os.path.exists(path):
                raise SourceMissing(f"table {name}")
            table = pd.read_csv(path, float_precision="round_trip")
            if "encoder" in table.columns:
                table = table[table["encoder"] == ENCODER]
            self._cache[name] = table
        return self._cache[name]

    def rows(self, name: str, **where) -> pd.DataFrame:
        """The rows of a table with the given values (a list selects several)."""
        table = self[name]
        mask = pd.Series(True, index=table.index)
        for column, value in where.items():
            if column not in table.columns:
                raise SourceMissing(f"column {column} of {name}")
            wanted = value if isinstance(value, (list, tuple)) else [value]
            mask &= table[column].isin(wanted)
        return table[mask]

    def col(self, name: str, column: str, **where) -> np.ndarray:
        """The values of a column in the selected rows; at least one row."""
        rows = self.rows(name, **where)
        if rows.empty or column not in rows.columns:
            raise SourceMissing(f"{column} of {name} for {where}")
        values = rows[column].to_numpy(dtype=np.float64)
        if np.isnan(values).all():
            raise SourceMissing(f"{column} of {name} for {where}")
        return values

    def val(self, name: str, column: str, **where) -> float:
        """The value of a column in exactly one row."""
        values = self.col(name, column, **where)
        if values.size != 1:
            raise SourceMissing(f"{values.size} rows of {name} for {where}")
        return float(values[0])


@dataclass(frozen=True)
class Entry:
    """One phrase of the manuscript and the numbers it states."""

    key: str
    section: str
    template: str
    values: Callable[[Tables], Sequence[Value]] | None = None
    # "result", "config" (a constant of the package), "cited", "approximate" or
    # "statement": a phrase without a number that states something the tables
    # decide. Its phrase must be present, and ``values`` must give ``expected``.
    kind: str = "result"
    expected: tuple[float, ...] = ()
    note: str = ""
    # Set when the first author has decided that the manuscript, not the table,
    # is to be changed: a difference is then reported as "pending correction".
    pending: str = ""


def span(values, digits: int) -> list[Value]:
    """The smallest and the largest value."""
    return [(float(np.min(values)), digits), (float(np.max(values)), digits)]


def magnitudes(values, digits: int) -> list[Value]:
    """The smallest and the largest magnitude."""
    return span(np.abs(values), digits)


def by_magnitude(values, digits: int) -> list[Value]:
    """The value of smallest magnitude, then the value of largest magnitude, signs kept."""
    values = np.asarray(values, dtype=np.float64)
    return [(float(values[np.argmin(np.abs(values))]), digits),
            (float(values[np.argmax(np.abs(values))]), digits)]


def _fallback(t: Tables, column: str, **where) -> np.ndarray:
    return t.col(T.FALLBACK, column, **where)


def _effect(t: Tables, effect: str, test_set: str, column: str = "estimate", **where) -> np.ndarray:
    where.setdefault("condition", "trim40")
    return t.col(T.EFFECTS, column, effect=effect, test_set=test_set, **where)


def _excluding_zero(t: Tables, effect: str, test_set: str, sign: int = 0, **where) -> int:
    """Number of speaker-level intervals that exclude zero: all of them
    (``sign`` 0), those below zero (-1) or those above zero (+1)."""
    low, high = (_effect(t, effect, test_set, c, **where) for c in ("spk_ci_low", "spk_ci_high"))
    below, above = high < 0, low > 0
    return int({0: below | above, -1: below, 1: above}[sign].sum())


def _subset_excluding_zero(t: Tables, effect: str, subset: str) -> list[Value]:
    """In a duration subset of In-the-Wild at 40 dB: the number of
    configurations of layers 3 to 5 whose speaker-level interval of ``effect``
    excludes zero, and the number of configurations."""
    excludes = t.col(BY_DURATION_CI, "spk_excludes_zero", condition="trim40", test_set="itw",
                     subset=subset, effect=effect, **SIX)
    return [(int(excludes.astype(bool).sum()), 0), (int(excludes.size), 0)]


def _matched(t: Tables, column: str, **where) -> np.ndarray:
    rows = t.rows(T.SEED_SUMMARY, **where)
    rows = rows[rows["condition"] == rows["test_condition"]]
    if "extraction" in rows.columns:
        rows = rows[rows["extraction"] == "unpadded"]
    if rows.empty:
        raise SourceMissing(f"{column} of the seed summary for {where}")
    return rows[column].to_numpy(dtype=np.float64)


def _selected(t: Tables, column: str, condition: str, selection: Sequence[str] | None = None
              ) -> np.ndarray:
    """A seed-summary column for the selected configurations of a condition."""
    configs = t.rows(T.SELECTED_CI, condition=condition).drop_duplicates(["pooling", "layer"])
    if selection is not None:
        configs = configs[configs["selection"].isin(selection)]
    return np.array([_matched(t, column, condition=condition, pooling=c.pooling, layer=c.layer)[0]
                     for c in configs.itertuples()])


def _attack(t: Tables, attacks: Sequence[str], condition: str = "trim40", **where) -> np.ndarray:
    return t.col(T.PER_ATTACK, "estimate", condition=condition, attack=list(attacks), **where)


def _cells(t: Tables, test_set: str, pooling: str, layer: int) -> list[Value]:
    return [(t.val(T.CELLS, f"{test_set}_eer_mean", condition="trim40", pooling=pooling,
                   layer=layer, cell=cell), 2) for cell in ("cc", "ct", "tc", "tt")]


def _median(t: Tables, name: str, table: str, measure: str, label: int, **where) -> Value:
    return (t.val(name, "median_s", table=table, measure=measure, label=label, **where), 2)


def _removed(t: Tables, split: str, label: int, top_db: int, column: str = "removed_total_median_s"
             ) -> float:
    return t.val(T.TRIM_AMOUNT, column, split=split, label=label, top_db=top_db)


def _pvu(t: Tables, config: str, extraction: str, column: str) -> Value:
    return (t.val(PADDED_VS_UNPADDED, column, config=config, extraction=extraction), 2)


def _rank_detail(t: Tables) -> list[Value]:
    """Increase of layers 1 and 2 after trimming and their ranks before and after."""
    cells = t.rows(T.CELLS, condition="trim40", pooling="stats")
    cc = cells[cells["cell"] == "cc"].set_index("layer")["eval_eer_mean"]
    tt = cells[cells["cell"] == "tt"].set_index("layer")["eval_eer_mean"]
    if cc.empty or tt.empty:
        raise SourceMissing("cells of statistics pooling")
    rank_cc, rank_tt = cc.rank(method="min"), tt.rank(method="min")
    return [(tt[1] - cc[1], 1), (tt[2] - cc[2], 1), (rank_cc[1], 0), (rank_cc[2], 0),
            (rank_tt[1], 0), (rank_tt[2], 0)]


def _padding_sample_size() -> int:
    from ..padding import SAMPLE_SIZE

    return SAMPLE_SIZE


def _ratio_span(t: Tables, layers: Sequence[int] = T.RANK_DETAIL_RANGE, cell: str = "tt"
                ) -> list[Value]:
    """Smallest and largest ratio of a cell of the models trained on trimmed
    audio to the clean matched cell, E(T, .) / E(C, C), on the evaluation set
    (statistics pooling, 40 dB). ``cell`` is "tt" (trimmed test audio) or
    "tc" (clean test audio). The ratio is taken from the unrounded cells."""
    cells = t.rows(T.CELLS, condition="trim40", pooling="stats", layer=list(layers))
    cc = cells[cells["cell"] == "cc"].set_index("layer")["eval_eer_mean"]
    top = cells[cells["cell"] == cell].set_index("layer")["eval_eer_mean"]
    if cc.empty or top.empty:
        raise SourceMissing(f"cells of layers {list(layers)}")
    return span((top / cc).to_numpy(), 1)


def lower_bound(value: float, digits: int) -> Value:
    """``value`` cut off after ``digits`` decimals, not rounded: the number a
    statement "at or above x" may print (99.883385 is at or above 99.8, not
    at or above 99.9). Used only by an entry whose phrase states a lower
    bound; every other number is rounded once."""
    from decimal import ROUND_FLOOR, Decimal

    cut = Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_FLOOR)
    return (float(cut), digits)


def _common(values) -> float:
    """The value that all ``values`` share; NaN, which equals no printed
    number, when they differ."""
    values = np.unique(np.asarray(values, dtype=np.float64))
    return float(values[0]) if values.size == 1 else float("nan")


def _share_below(t: Tables, **where) -> np.ndarray:
    """Per cent of the resamples with a smaller tau than the observed one,
    ties counted one half: the normal distribution function of ``z0_tau``
    (``ranking.bca_interval`` defines z0 as the quantile of that share)."""
    from scipy.stats import norm

    return 100 * norm.cdf(t.col(T.RANK, "z0_tau", **where))


def _bca_upper_level(t: Tables, alpha: float = 0.05, **where) -> np.ndarray:
    """Percentile of the resamples at which the upper BCa limit of tau lies:
    the corrected level of ``ranking.bca_interval``,
    Phi(z0 + (z0 + z) / (1 - a (z0 + z))) with z the 1 - alpha/2 quantile,
    from ``z0_tau`` and ``a_tau``."""
    from scipy.stats import norm

    z0, a = t.col(T.RANK, "z0_tau", **where), t.col(T.RANK, "a_tau", **where)
    num = z0 + norm.ppf(1 - alpha / 2)
    return 100 * norm.cdf(z0 + num / (1 - a * num))


def _best_layers(t: Tables, test_set: str, configs: Sequence[tuple[str, str]],
                 columns: Sequence[str] = ("best_layer_trimmed",)) -> float:
    """The best layer shared by the given (condition, pooling) pairs."""
    return _common([t.val(T.RANK, column, condition=c, pooling=p, test_set=test_set)
                    for c, p in configs for column in columns])


def _best_shares(t: Tables) -> list[Value]:
    """Smallest and largest share of resamples in which the best layer is best (40 dB)."""
    shares = []
    for r in t.rows(T.RANK, condition="trim40").itertuples():
        for name in ("clean", "trimmed"):
            shares.append(t.val(T.RANK_LAYERS, f"p_best_{name}", condition="trim40",
                                test_set=r.test_set, pooling=r.pooling,
                                layer=int(getattr(r, f"best_layer_{name}"))))
    return [(100 * min(shares), 0), (100 * max(shares), 0)]


E = Entry
ENTRIES: tuple[Entry, ...] = (
    # Abstract ---------------------------------------------------------------------------------
    E("abstract_tau", "abstract",
      r"(Kendall's $\tau$ of {} in domain, {}--{} on In-the-Wild)",
      lambda t: [(t.col(T.RANK, "kendall_tau", condition="trim40", test_set="eval").max(), 2)]
      + span(t.col(T.RANK, "kendall_tau", condition="trim40", test_set="itw"), 2)),
    E("abstract_delta_test", "abstract",
      r"rises by {}--{} percentage points at every layer",
      lambda t: span(_effect(t, "delta_test", "eval"), 1)),

    # 1 Introduction (the contributions repeat results of Section 4) ------------------------------
    E("intro_tau", "1",
      r"rankings is {} on the ASVspoof 2019 LA evaluation set and {}--{} on In-the-Wild, and it "
      r"decreases at the stricter thresholds of 20 and 15~dB ({}--{} on the evaluation set and "
      r"{}--{} on In-the-Wild)",
      lambda t: [(t.col(T.RANK, "kendall_tau", condition="trim40", test_set="eval").max(), 2)]
      + span(t.col(T.RANK, "kendall_tau", condition="trim40", test_set="itw"), 2)
      + span(t.col(T.RANK, "kendall_tau", condition=STRICT, test_set="eval"), 2)
      + span(t.col(T.RANK, "kendall_tau", condition=STRICT, test_set="itw"), 2)),
    # "does not change": the best layer of the clean and of the trimmed ranking, both poolings.
    E("intro_best_layer", "1", r"(layer~{} on the evaluation set, layer~{} on In-the-Wild)",
      lambda t: [(_best_layers(t, s, [("trim40", "stats"), ("trim40", "mean")],
                               ("best_layer_clean", "best_layer_trimmed")), 0)
                 for s in ("eval", "itw")]),
    E("intro_best_layer_strict", "1",
      r"the best layer on the evaluation set becomes layer~{} (with statistics pooling at 20~dB, "
      r"layers 5 and 6 are nearly equal, {}\% and {}\%), while layer~{} remains best on In-the-Wild",
      lambda t: [(_best_layers(t, "eval", [("trim15", "stats"), ("trim15", "mean"),
                                           ("trim20", "mean")]), 0)]
      + [(_matched(t, "eval_eer_mean", condition="trim20", pooling="stats", layer=k)[0], 2)
         for k in (5, 6)]
      + [(_best_layers(t, "itw", [(c, p) for c in STRICT for p in T.POOLINGS]), 0)]),
    E("intro_delta_test", "1",
      r"rises by {}--{} percentage points when only the test audio is trimmed, with an interval "
      r"excluding zero at every layer (40~dB). For layers 3--5, the increase is {}--{} points at "
      r"20 and 15~dB",
      lambda t: span(_effect(t, "delta_test", "eval"), 1)
      + span(_effect(t, "delta_test", "eval", condition=STRICT, **SIX), 1)),
    E("intro_attack_a10", "1", r"an increase of {}--{} percentage points of EER for A10",
      lambda t: span(_attack(t, ["A10"]), 1)),

    # 3.1 Data ----------------------------------------------------------------------------------------
    E("data_speakers", "3.1",
      r"The evaluation partition contains {} speakers with bonafide files and {} with spoofed "
      r"files; In-the-Wild contains {} speakers in each class",
      lambda t: [(_common(t.col(T.SELECTED_CI, "n_speakers_bonafide", test_set="eval")), 0),
                 (_common(t.col(T.SELECTED_CI, "n_speakers_spoof", test_set="eval")), 0),
                 (_common(np.concatenate([t.col(T.SELECTED_CI, c, test_set="itw")
                                          for c in ("n_speakers_bonafide", "n_speakers_spoof")])), 0)]),

    # 3.2 Preprocessing and silence trimming --------------------------------------------------------
    E("sampling_rate", "3.2", r"mono waveform at {}~kHz",
      lambda t: [(cfg.SAMPLE_RATE / 1000, 0)], kind="config"),
    E("trim_frames", "3.2", r"(frame length {} samples, hop {} samples)",
      lambda t: [(cfg.TRIM_FRAME_LENGTH, 0), (cfg.TRIM_HOP_LENGTH, 0)], kind="config"),
    E("window_samples", "3.2", r"({} samples); shorter files keep",
      lambda t: [(cfg.WINDOW, 0)], kind="config"),
    E("minimum_samples", "3.2",
      r"Files shorter than {} samples, the receptive field of the convolutional encoder, are "
      r"padded to {} samples",
      lambda t: [(cfg.MIN_SAMPLES, 0), (cfg.MIN_SAMPLES, 0)], kind="config"),
    E("no_short_file", "3.2", r"No file was shorter than {} samples at either threshold",
      lambda t: [(cfg.MIN_SAMPLES, 0)], kind="config"),
    E("fallback_40", "3.2",
      r"this applied to {} training, {} development, {} evaluation, and {} In-the-Wild files",
      lambda t: [(_fallback(t, "n_fallback", condition="trim40", split=s, label="all")[0], 0)
                 for s in cfg.SPLITS]),
    E("fallback_40_share", "3.2", r"applies to at most {}\% of the files in any set",
      lambda t: [(_fallback(t, "pct_fallback", condition="trim40", label="all").max(), 2)]),
    E("fallback_strict", "3.2",
      r"applied to {}, {}, {}, and {} files at 20~dB and to {}, {}, {}, and {} files at 15~dB",
      lambda t: [(_fallback(t, "n_fallback", condition=c, split=s, label="all")[0], 0)
                 for c in STRICT for s in cfg.SPLITS]),
    E("fallback_strict_share", "3.2",
      r"to at most {}\% of the ASVspoof 2019 LA files and {}\% of the In-the-Wild files",
      lambda t: [(_fallback(t, "pct_fallback", condition=STRICT, split=ASV, label="all").max(), 2),
                 (_fallback(t, "pct_fallback", condition=STRICT, split="itw", label="all").max(), 1)]),
    E("fallback_per_class_itw", "3.2",
      r"({}\% against {}\% at 20~dB and {}\% against {}\% at 15~dB)",
      lambda t: [(_fallback(t, "pct_fallback", condition=c, split="itw", label=label)[0], 2)
                 for c in STRICT for label in ("0", "1")]),
    E("fallback_per_class_asvspoof", "3.2",
      r"it applies to {}--{}\% of the files of either class",
      lambda t: span(_fallback(t, "pct_fallback", condition=STRICT, split=ASV, label=["0", "1"]), 2)),

    # 3.6 Statistical analysis (the intervals of the rank agreement) ----------------------------------
    E("statistics_tau_below", "3.6", r"{}--{}\% of the resamples give a smaller $\tau$",
      lambda t: span(_share_below(t, condition="trim40"), 0)),
    E("statistics_bca_z0", "3.6", r"ranges from {} to {} over the twelve configurations",
      lambda t: span(t.col(T.RANK, "z0_tau"), 2)),
    E("statistics_tau_ties", "3.6", r"{}--{}\% of the resamples equal the observed $\tau$",
      lambda t: span(100 * t.col(T.RANK, "tie_share_tau"), 0)),
    # The smallest level over the four configurations of 40 dB. The phrase
    # states a lower bound ("at or above"), so the level is cut off after one
    # decimal instead of rounded (decided by the first author, 2026-11-21).
    E("statistics_bca_upper_level", "3.6", r"at or above the {}th percentile of the resamples",
      lambda t: [lower_bound(_bca_upper_level(t, condition="trim40").min(), 1)]),

    # 4.1 Silence as a cue in the data ----------------------------------------------------------------
    E("lead_medians_eval", "4.1",
      r"is {}~s for bonafide and {}~s for spoofed files over the whole file, and {}~s and {}~s within",
      lambda t: [_median(t, SILENCE_SUMMARY, "eval", m, label) for m in ("lead_40", "win_lead_40")
                 for label in (0, 1)]),
    E("silence_only_eval", "4.1",
      r"reaches {}\% EER on the evaluation set when silence is measured at 30~dB and {}\% at 40~dB",
      lambda t: [(t.val(T.SILENCE_BASELINE, "eval", condition="clean", features="sil", top_db=d), 2)
                 for d in (30, 40)]),
    E("silence_only_itw", "4.1",
      r"give EERs between {}\% and {}\%; there, the median leading silence is {}~s for bonafide "
      r"and {}~s for spoofed files",
      lambda t: span(t.col(T.SILENCE_BASELINE, "itw", condition="clean", top_db=[30, 40]), 2)
      + [_median(t, SILENCE_SUMMARY, "itw", "lead_40", label) for label in (0, 1)]),

    # 4.2 Effect of zero padding ------------------------------------------------------------------------
    E("padding_samples", "4.2",
      r"features of {} randomly sampled training files and of the {} most heavily padded",
      lambda t: [(_padding_sample_size(), 0), (_padding_sample_size(), 0)], kind="config"),
    E("padding_deviation_layer4", "4.2",
      r"padding changes the features by {} of their own magnitude on clean audio and by {} on "
      r"trimmed audio. The attention mask reduces these values to {} and {}",
      lambda t: [(t.val(PADDING_DEVIATION, "rel_mean_diff", condition=c, sample="random",
                        comparison=m, statistic="mean", layer=4), 2)
                 for m in ("no_mask", "mask") for c in ("clean", "trim40")]),
    E("padding_only_classifier", "4.2",
      r"gives {}--{}\% EER across the training, development, evaluation, and In-the-Wild sets",
      lambda t: span(t.col(PADDING_BASELINE, "eer_best", condition=["clean", "trim40"]), 1)),
    E("padded_against_unpadded", "4.2",
      r"gives an evaluation EER of {}\% (clean/stats/L4) and {}\% (trimmed/mean/L4), against {}\% "
      r"and {}\% without padding, and an In-the-Wild EER of {}\% and {}\%, against {}\% and {}\%",
      lambda t: [_pvu(t, config, extraction, f"{test_set}_eer_mean")
                 for test_set in ("eval", "itw") for extraction in ("padded", "unpadded")
                 for config in ("clean/stats/L4", "trim40/mean/L4")]),
    E("padding_median", "4.2", r"(median {}~s against {}~s on clean audio)",
      lambda t: [(np.median(t.col(PADDING_PER_FILE, "pad_s", sample="random", condition=c)), 2)
                 for c in ("trim40", "clean")],
      note="median padding of the random sample of the padding check, both classes"),
    E("padding_by_amount", "4.2",
      r"it is about {} for less than 1~s of padding and about {} for more than 3~s",
      lambda t: [(t.val(PADDING_BY_AMOUNT, "mean_rel_L4", condition="trim40", sample="random",
                        comparison="no_mask", pad_bin=b), 2) for b in ("<1 s", ">3 s")]),

    # 4.3 Layer-wise results and the 2x2 design -------------------------------------------------------------
    E("layer1_matched", "4.3", r"layer~1: {}\% clean against {}\% trimmed",
      lambda t: [(_matched(t, "eval_eer_mean", condition=c, pooling="stats", layer=1)[0], 2)
                 for c in ("clean", "trim40")]),
    E("total_eval_six", "4.3", r"configurations, by {}--{} percentage points, and the increase",
      lambda t: span(_effect(t, "delta_total", "eval", **SIX), 1)),
    E("tdcf_ranges", "4.3",
      r"from {}--{} in the clean condition to {}--{} in the trimmed condition",
      lambda t: span(_selected(t, "eval_tdcf_mean", "clean"), 4)
      + span(_selected(t, "eval_tdcf_mean", "trim40"), 4)),
    E("speakers_itw", "4.3", r"With {} speakers per class, the speaker-level intervals",
      lambda t: [(t.col(T.SELECTED_CI, "n_speakers_bonafide", test_set="itw").max(), 0)]),
    E("total_itw_six", "4.3", r"trimming lowers the EER by {}--{} points; the decrease",
      lambda t: magnitudes(_effect(t, "delta_total", "itw", **SIX), 1)),
    E("delta_test_eval", "4.3",
      r"rises by {}--{} percentage points when silence is trimmed from the test audio only",
      lambda t: span(_effect(t, "delta_test", "eval"), 2)),
    E("delta_test_t_eval", "4.3",
      r"({} to {} points; the interval excludes zero in {} of the 24 configurations",
      lambda t: span(_effect(t, "delta_test_t", "eval"), 2)
      + [(_excluding_zero(t, "delta_test_t", "eval"), 0)]),
    E("delta_train_c_eval", "4.3", r"raises the EER by {}--{} points, with an interval excluding",
      lambda t: span(_effect(t, "delta_train_c", "eval"), 2)),
    E("cells_eval", "4.3",
      r"cells are $E(\mathrm{C},\mathrm{C})={}\%$, $E(\mathrm{C},\mathrm{T})={}\%$, "
      r"$E(\mathrm{T},\mathrm{C})={}\%$, and $E(\mathrm{T},\mathrm{T})={}\%$. The interaction lies "
      r"between {} and {} points",
      lambda t: _cells(t, "eval", "stats", 4) + by_magnitude(_effect(t, "interaction", "eval"), 2)),
    E("delta_test_itw_max", "4.3", r"changes the EER by at most {} points in any of the 24",
      lambda t: [(np.abs(_effect(t, "delta_test", "itw")).max(), 2)]),
    E("total_itw_count", "4.3", r"excluding zero in {} configurations (mean pooling at layers 5--9",
      lambda t: [(_excluding_zero(t, "delta_total", "itw", sign=-1), 0)]),
    E("interaction_itw_count", "4.3", r"set of {} configurations (mean pooling at layers 1--5",
      lambda t: [(_excluding_zero(t, "interaction", "itw", sign=-1), 0)]),
    E("delta_test_t_itw_six", "4.3", r"($\Delta_{\mathrm{test}}^{\mathrm{T}}$, by {}--{} points)",
      lambda t: magnitudes(_effect(t, "delta_test_t", "itw", **SIX), 2)),
    E("delta_train_c_itw", "4.3", r"by {}--{} points with both pooling methods, with intervals",
      lambda t: magnitudes(_effect(t, "delta_train_c", "itw", layer=[6, 8, 9]), 2)),
    E("interaction_itw_layer12", "4.3", r"at layer~12 ({} [{}, {}])",
      lambda t: [(_effect(t, "interaction", "itw", c, pooling="stats", layer=12)[0], 2)
                 for c in ("estimate", "spk_ci_low", "spk_ci_high")]),
    E("interaction_itw_rest", "4.3", r"its interval includes zero in the remaining {}",
      lambda t: [(len(_effect(t, "interaction", "itw")) - _excluding_zero(t, "interaction", "itw"), 0)]),
    E("delta_test_t_itw_late", "4.3",
      r"by {} (mean) and {} (statistics) points, as it does for statistics pooling at layer~8 "
      r"({} points)",
      lambda t: [(_effect(t, "delta_test_t", "itw", pooling=p, layer=k)[0], 2)
                 for p, k in (("mean", 12), ("stats", 12), ("stats", 8))]),
    E("cells_itw", "4.3",
      r"the cells are $E(\mathrm{C},\mathrm{C})={}\%$, $E(\mathrm{C},\mathrm{T})={}\%$, "
      r"$E(\mathrm{T},\mathrm{C})={}\%$, and $E(\mathrm{T},\mathrm{T})={}\%$",
      lambda t: _cells(t, "itw", "stats", 4)),

    # 4.4 Preservation of the layer ranking ------------------------------------------------------------------
    E("tau_text", "4.4", r"($\tau = {}$). On In-the-Wild, $\tau$ is {}--{}, but its intervals are wide",
      lambda t: [(t.col(T.RANK, "kendall_tau", condition="trim40", test_set="eval").max(), 2)]
      + span(t.col(T.RANK, "kendall_tau", condition="trim40", test_set="itw"), 2)),
    E("best_layer_shares", "4.4",
      r"ranges from {}\% (trimmed, mean pooling, evaluation set) to {}\%", _best_shares),
    E("first_layers", "4.4",
      r"({} and {} points) and fall from ranks {} and {} to ranks {} and {}", _rank_detail),
    E("late_layers_ratio", "4.4", r"($\times${}--{}) and keep their relative order", _ratio_span),

    # 4.5 In-the-Wild by original duration -----------------------------------------------------------------------
    E("duration_counts", "4.5", r"({}) and files longer than 4~s ({})",
      lambda t: [(cfg.ITW_DURATION_COUNTS[0], 0), (cfg.ITW_DURATION_COUNTS[1], 0)], kind="config"),
    E("duration_long_test", "4.5", r"($\Delta_{\mathrm{test}} = {} \pm {}$)",
      lambda t: [(t.val(T.BY_DURATION, c, condition="trim40", pooling="stats", layer=4,
                        subset=">4s", quantity="delta_test"), 2) for c in ("mean", "sd")]),
    E("duration_short_test", "4.5", r"it lowers the EER by ${} \pm {}$ points",
      lambda t: [(abs(t.val(T.BY_DURATION, "mean", condition="trim40", pooling="stats", layer=4,
                            subset="<=4s", quantity="delta_test")), 2),
                 (t.val(T.BY_DURATION, "sd", condition="trim40", pooling="stats", layer=4,
                        subset="<=4s", quantity="delta_test"), 2)]),
    # Two statements without a number: on the files of at most 4 s, the speaker-level interval of
    # the test-time effect includes zero in all six configurations of layers 3 to 5 for the models
    # trained on untrimmed audio (none excludes zero), and excludes zero in all six for the models
    # trained on trimmed audio.
    E("duration_short_test_interval", "4.5",
      r"but the speaker-level interval of this effect includes zero, here and at layers 3 and 5 "
      r"with both pooling methods",
      lambda t: _subset_excluding_zero(t, "delta_test", "<=4s"), kind="statement", expected=(0, 6)),
    E("duration_short_test_t_interval", "4.5",
      r"lowers the EER of the short files with a speaker-level interval that excludes zero in all "
      r"of these configurations",
      lambda t: _subset_excluding_zero(t, "delta_test_t", "<=4s"), kind="statement",
      expected=(6, 6)),
    E("duration_long_cells", "4.5",
      r"$E(\mathrm{T},\mathrm{C}) = {}\%$ against $E(\mathrm{C},\mathrm{C}) = {}\%$",
      lambda t: [(t.val(T.BY_DURATION, "mean", condition="trim40", pooling="stats", layer=4,
                        subset=">4s", quantity=q), 2) for q in ("eer_tc", "eer_cc")]),

    # 4.6 Per-attack analysis ---------------------------------------------------------------------------------------
    E("attack_files", "4.6", r"contributes {} spoofed files from {} speakers",
      lambda t: [(t.col(T.PER_ATTACK, "n_spoof", condition="trim40").max(), 0),
                 (t.col(T.PER_ATTACK, "n_speakers_spoof", condition="trim40").max(), 0)]),
    E("attack_a10", "4.6", r"with an increase of {}--{} points across the six",
      lambda t: span(_attack(t, ["A10"]), 1)),
    E("attack_a10_eer", "4.6", r"from about {}--{}\% to {}--{}\% EER",
      lambda t: [(t.col(T.PER_ATTACK, "eer_clean", condition="trim40", attack="A10").min(), 0),
                 (t.col(T.PER_ATTACK, "eer_clean", condition="trim40", attack="A10").max(), 1),
                 (t.col(T.PER_ATTACK, "eer_trimmed", condition="trim40", attack="A10").min(), 0),
                 (t.col(T.PER_ATTACK, "eer_trimmed", condition="trim40", attack="A10").max(), 0)],
      note="the manuscript says 'about'"),
    E("attack_a11_a15", "4.6", r"with increases of {}--{} and {}--{} points",
      lambda t: span(_attack(t, ["A11"]), 1) + span(_attack(t, ["A15"]), 1)),
    E("attack_moderate", "4.6", r"increase by {}--{} points, significantly in all six",
      lambda t: span(_attack(t, ["A07", "A08", "A09", "A12", "A14", "A16"]), 1)),
    E("attack_small", "4.6", r"A13 by {} to {} points",
      lambda t: span(_attack(t, ["A13"]), 2)),
    E("attack_a18", "4.6", r"A18 by {} to {} (not significant",
      lambda t: span(_attack(t, ["A18"]), 1)),
    E("attack_a19", "4.6", r"A19 by {} to {} (significant only",
      lambda t: span(_attack(t, ["A19"]), 1)),
    E("attack_a17", "4.6", r"decreases after trimming, by {}--{} points",
      lambda t: magnitudes(_attack(t, ["A17"]), 2)),
    E("attack_by_depth", "4.6",
      r"it is {}, {}, and {} points for A11 and {}, {}, and {} points for A15",
      lambda t: [(_attack(t, [a], pooling="stats", layer=k)[0], 1)
                 for a in ("A11", "A15") for k in (3, 4, 5)]),

    # 4.7 Sensitivity to the trimming threshold -------------------------------------------------------------------------
    E("residual_medians", "4.7",
      r"set is {}~s for bonafide and {}~s for spoofed files when silence is defined by a 15~dB "
      r"threshold, and {}~s and {}~s at 20~dB",
      lambda t: [_median(t, RESIDUAL_SUMMARY, "eval", f"lead_{d}", label, top_db=d)
                 for d in (15, 20) for label in (0, 1)]),
    E("residual_classifier", "4.7",
      r"reaches {}\%, {}\%, and {}\% EER on the evaluation set with silence measured at 15, 20, "
      r"and 30~dB (sil), and {}\%--{}\% with sil+dur, compared with {}\% and {}\% on the",
      lambda t: [(t.val(RESIDUAL_BASELINE, "eval", features="sil", top_db=d), 2) for d in (15, 20, 30)]
      + span(t.col(RESIDUAL_BASELINE, "eval", features="sil+dur", top_db=[15, 20, 30]), 2)
      + [(t.val(T.SILENCE_BASELINE, "eval", condition="clean", features="sil", top_db=d), 2)
         for d in (30, 40)]),
    E("residual_classifier_itw", "4.7", r"classifiers give EERs between {}\% and {}\%.",
      lambda t: span(t.col(RESIDUAL_BASELINE, "itw", top_db=[15, 20, 30]), 2)),
    # Every file is counted, also a file that the fallback rule kept untrimmed
    # (``silence.remaining_silence``); the medians are those of the whole file.
    E("remaining_silence_share", "4.7",
      r"leading silence remains in {}--{}\% of the files of each set ({}--{}\% within the "
      r"4-second window)",
      lambda t: span(t.col(RESIDUAL_SHARE, "pct_lead_over", label="all"), 1)
      + span(t.col(RESIDUAL_SHARE, "pct_win_lead_over", label="all"), 1)),
    E("remaining_silence_by_class", "4.7",
      r"({}\% and {}\% on the evaluation set), with a median of {}--{}~s",
      lambda t: [(t.val(RESIDUAL_SHARE, "pct_lead_over", table="eval", label=label), 1)
                 for label in ("0", "1")]
      + span(t.col(RESIDUAL_SHARE, "lead_median_s", label="all"), 2)),
    E("removed_eval", "4.7",
      r"grows from {}~s at 40~dB to {}~s at 20~dB and {}~s at 15~dB for bonafide files, and from "
      r"{}~s to {}~s and {}~s for spoofed files; at 15~dB, this is a median of {}\% of the",
      lambda t: [(_removed(t, "eval", label, d), 2) for label in (0, 1) for d in (40, 20, 15)]
      + [(100 * _removed(t, "eval", 0, 15, "removed_frac_median"), 0)]),
    E("removed_itw", "4.7",
      r"median of {}~s from bonafide and {}~s from spoofed files, the stricter thresholds remove "
      r"{}~s and {}~s at 20~dB and {}~s and {}~s at 15~dB",
      lambda t: [(_removed(t, "itw", label, d), 2) for d in (40, 20, 15) for label in (0, 1)]),
    E("dev_selected_eer", "4.7", r"{}\% at 40~dB to {}\% at 20~dB and {}\% at 15~dB. At the",
      lambda t: [(_matched(t, "eval_eer_mean", condition=c, pooling="stats", layer=5)[0], 2)
                 for c in T.THRESHOLD_CONDITIONS]),
    E("delta_test_thresholds", "4.7",
      r"grows from {}--{} points at 40~dB to {}--{} points at 20~dB and {}--{} points at 15~dB",
      lambda t: [v for c in T.THRESHOLD_CONDITIONS
                 for v in span(_effect(t, "delta_test", "eval", condition=c, **SIX), 1)]),
    E("interaction_thresholds", "4.7", r"at layer~4 from {} to {} and {} points",
      lambda t: [(_effect(t, "interaction", "eval", condition=c, pooling="stats", layer=4)[0], 2)
                 for c in T.THRESHOLD_CONDITIONS]),
    E("best_layer_strict", "4.7",
      r"(in {}\% and {}\% of the resamples) and at 20~dB with mean pooling ({}\%)",
      lambda t: [(100 * t.val(T.THRESHOLDS, "p_best_eval_trimmed", condition=c, pooling=p), 0)
                 for c, p in (("trim15", "stats"), ("trim15", "mean"), ("trim20", "mean"))]),
    E("tie_at_20", "4.7", r"({}\% and {}\%), and layer~6 is best in {}\% of the resamples",
      lambda t: [(_matched(t, "eval_eer_mean", condition="trim20", pooling="stats", layer=k)[0], 2)
                 for k in (5, 6)]
      + [(100 * t.val(T.RANK_LAYERS, "p_best_trimmed", condition="trim20", test_set="eval",
                      pooling="stats", layer=6), 0)]),
    E("best_layer_itw_strict", "4.7", r"remains layer~4, in {}\%--{}\% of the resamples",
      lambda t: span(100 * t.col(T.THRESHOLDS, "p_best_itw_trimmed", condition=STRICT,
                                 pooling="stats"), 0)),
    E("delta_test_itw_thresholds", "4.7",
      r"by {}--{} points at 20~dB and {}--{} points at 15~dB, whereas at 40~dB it lowered it by "
      r"{}--{} points",
      lambda t: [v for c in ("trim20", "trim15", "trim40")
                 for v in magnitudes(_effect(t, "delta_test", "itw", condition=c, **SIX), 1)]),
    E("attack_a10_strict", "4.7",
      r"with an increase of {}--{} points at 20~dB and {}--{} points at 15~dB",
      lambda t: [v for c in STRICT for v in span(_attack(t, ["A10"], condition=c), 1)]),

    # 4.8 What the layers encode -----------------------------------------------------------------------------------------
    E("probe_sample", "4.8",
      r"features ({} dimensions) of each layer, using {} training files from {} speakers",
      lambda t: [(cfg.encoder(ENCODER)["hidden_size"], 0),
                 (t.col(T.PROBE_R2, "n_files", condition="clean", target="pad").max(), 0),
                 (t.col(T.PROBE_R2, "n_groups", condition="clean", target="pad").max(), 0)]),
    E("probe_lead", "4.8", r"($R^2 \approx {}$)",
      lambda t: [(np.median(t.col(T.PROBE_R2, "r2_mean", condition="clean", cv="speaker",
                                  control="real", target="win_lead_40")), 2)],
      note="the median over the 12 layers"),
    E("probe_length", "4.8",
      r"with $R^2$ between {} and {} on clean audio and at least {} on trimmed audio",
      lambda t: span(t.col(T.PROBE_R2, "r2_mean", condition="clean", cv="speaker", control="real",
                           target="length_in_window"), 2)
      # "at least": a lower bound, cut off after two decimals, not rounded
      # (decided by the first author, 2026-11-22).
      + [lower_bound(t.col(T.PROBE_R2, "r2_mean", condition="trim40", cv="speaker",
                           control="real", target="length_in_window").min(), 2)]),

    # 5 Discussion ---------------------------------------------------------------------------------------------------------
    E("discussion_silence_only", "5.1", r"silence features alone give {}\% EER at 30~dB",
      lambda t: [(t.val(T.SILENCE_BASELINE, "eval", condition="clean", features="sil", top_db=30), 2)]),
    E("discussion_itw_medians", "5.1",
      r"silence is {}~s for bonafide and {}~s for spoofed files (Section",
      lambda t: [_median(t, SILENCE_SUMMARY, "itw", "lead_40", label) for label in (0, 1)]),
    E("discussion_itw_classifier", "5.1", r"gives EERs between {}\% and {}\%",
      lambda t: span(t.col(T.SILENCE_BASELINE, "itw", condition="clean", top_db=[30, 40]), 2)),
    E("discussion_delta_test_t", "5.1",
      r"($\Delta_{\mathrm{test}}^{\mathrm{T}}$ is {} to {} points)",
      lambda t: span(_effect(t, "delta_test_t", "eval"), 2)),
    # E(T,T) / E(C,C) and E(T,C) / E(C,C), layers 3--5, statistics pooling, evaluation set.
    E("discussion_factor", "5.2", r"by a factor of {}--{} when the test audio is also trimmed",
      lambda t: _ratio_span(t, layers=(3, 4, 5))),
    E("discussion_factor_clean_test", "5.2", r"by a factor of {}--{} when it is clean",
      lambda t: _ratio_span(t, layers=(3, 4, 5), cell="tc")),
    E("limitations_speakers", "5.5", r"In-the-Wild contains {} speakers per class",
      lambda t: [(t.col(T.SELECTED_CI, "n_speakers_bonafide", test_set="itw").max(), 0)]),
    E("discussion_itw_eer", "5.3", r"between {}\% and {}\%. For frozen encoders",
      lambda t: span(np.concatenate([_selected(t, "itw_eer_mean", c, ["dev-selected"])
                                     for c in ("clean", "trim40")]), 2)),

    # Rounded statements ("about ...") ---------------------------------------------------------------------------------------
    E("approximate_interval_width", "4.3", r"span about $\pm${} points", kind="approximate"),
    E("approximate_interval_width_limitations", "5.5", r"wide, about $\pm${} percentage points",
      kind="approximate"),

    # Prior work ------------------------------------------------------------------------------------------------------------
    E("cited_rawnet2", "5.1", r"RawNet2 rose from {}\% to {}\%", kind="cited"),
    E("cited_frozen_encoders", "5.3",
      r"{}\% for WavLM Large and {}\% for HuBERT Base, both trained on ASVspoof", kind="cited"),
    E("cited_wavlm_large", "5.3", r"({}\% with WavLM Large in the same study", kind="cited"),
    E("cited_muller", "5.1", r"obtained {}\% EER on the ASVspoof 2019 ev", kind="cited"),
    E("cited_layer_ranges", "5.2", r"({}--{} for base and {}--{} for large models)", kind="cited"),
    E("cited_serrano", "5.3", r"reported {}\% for layer~4 of WavLM Base", kind="cited"),
    E("cited_beheshti", "5.3", r"reached {}\% with four selected layers", kind="cited"),
    E("cited_pascu", "5.3", r"reached {}\% with frozen wav2vec2", kind="cited"),
)


# The padding numbers --------------------------------------------------------------------------------

# Entries of the running text whose numbers come from the padding check.
PADDING_ENTRIES = ("padding_deviation_layer4", "padding_median", "padding_by_amount")
# A statement of the analysis plan (not an entry of the running text): on the
# most padded files, trimmed audio, layer 4, the deviation with an attention
# mask and without one.
PLAN_PADDING_ENTRIES = (
    Entry("padding_most_padded_layer4", "analysis plan", "",
          lambda t: [(t.val(PADDING_DEVIATION, "rel_mean_diff", condition="trim40",
                            sample="most_padded", comparison=m, statistic="mean", layer=4), 2)
                     for m in ("mask", "no_mask")]),
)


def printed_values(tables: Tables, entries: Sequence[Entry]) -> pd.DataFrame:
    """The numbers of ``entries`` from ``tables``: key, position (in the
    phrase), digits, value and printed (rounded once to the digits printed)."""
    rows = []
    for entry in entries:
        for position, (value, digits) in enumerate(entry.values(tables)):
            rows.append({"key": entry.key, "position": position, "digits": int(digits),
                         "value": float(value), "printed": printed(value, digits)})
    return pd.DataFrame(rows)


def padding_entries() -> tuple[Entry, ...]:
    """The entries of PADDING_ENTRIES, then PLAN_PADDING_ENTRIES."""
    by_key = {entry.key: entry for entry in ENTRIES}
    return tuple(by_key[key] for key in PADDING_ENTRIES) + PLAN_PADDING_ENTRIES


def compare_printed(new: Tables, reference: Tables, entries: Sequence[Entry]) -> pd.DataFrame:
    """Every number of ``entries`` from two sets of tables, at the printed precision.

    One row per number: key, position, digits, value, reference_value,
    printed, reference_printed and ok (the two printed numbers are equal).
    """
    mine, other = printed_values(new, entries), printed_values(reference, entries)
    if mine[["key", "position", "digits"]].to_dict("list") != \
            other[["key", "position", "digits"]].to_dict("list"):
        raise ValueError("the two sets of tables give different numbers of values")
    out = mine.rename(columns={"printed": "printed_new"})
    out["reference_value"], out["reference_printed"] = other["value"], other["printed"]
    out["ok"] = out["printed_new"] == out["reference_printed"]
    return out.rename(columns={"printed_new": "printed"})[
        ["key", "position", "digits", "value", "reference_value", "printed", "reference_printed", "ok"]]


# The silence that remains after trimming (Section 4.7) -----------------------------------------------

# Entries of the running text whose numbers come from the remaining leading silence.
REMAINING_SILENCE_ENTRIES = ("remaining_silence_share", "remaining_silence_by_class")
# The numbers of these entries as the manuscript prints them. The source of the
# manuscript is not in this repository; the sentence was given by the first
# author on 2026-11-21: "After trimming at 40~dB, more than 50~ms of leading
# silence remains in 3.1--4.2\% of the files of each set (2.2--4.0\% within the
# 4-second window), more often in bonafide than in spoofed files (6.7\% and
# 3.0\% on the evaluation set), with a median of 0.16--0.23~s."
# The smallest share within the window is stated as 2.1, not 2.2: the first
# author corrected the manuscript to 2.1 on 2026-11-21 (683 of 31,779
# In-the-Wild files, 2.149218%, rounded once).
STATED = {"remaining_silence_share": (3.1, 4.2, 2.1, 4.0),
          "remaining_silence_by_class": (6.7, 3.0, 0.16, 0.23)}
# "Five bonafide In-the-Wild files are too quiet to be trimmed at 40~dB
# (loudest-frame RMS below $10^{-3}$ ...)": the table, the threshold, the
# number of files and their label. The sentence prints the number as a word,
# so no entry of the registry reads it.
STATED_QUIET = {"table": "itw", "top_db": 40, "n_files": 5, "label": 0}


def remaining_silence_entries() -> tuple[Entry, ...]:
    """The entries of REMAINING_SILENCE_ENTRIES."""
    by_key = {entry.key: entry for entry in ENTRIES}
    return tuple(by_key[key] for key in REMAINING_SILENCE_ENTRIES)


def compare_stated(tables: Tables, entries: Sequence[Entry],
                   stated: Mapping[str, Sequence[float]] = STATED) -> pd.DataFrame:
    """Every number of ``entries`` from ``tables`` against the number that the manuscript prints.

    ``stated`` maps the key of an entry to its numbers as printed, in the
    order of the phrase. One row per number: key, position, digits, value,
    printed (rounded once to the digits printed), stated and ok (printed
    equals stated). Raises if an entry has another number of values than
    ``stated`` gives.
    """
    out = printed_values(tables, entries)
    expected = []
    for entry in entries:
        n = int((out["key"] == entry.key).sum())
        if n != len(stated[entry.key]):
            raise ValueError(f"{entry.key}: {n} values computed, {len(stated[entry.key])} stated")
        expected += [float(value) for value in stated[entry.key]]
    out["stated"] = expected
    out["ok"] = out["printed"] == out["stated"]
    return out


def compare_quiet(quiet: pd.DataFrame, stated: Mapping[str, object] = STATED_QUIET) -> pd.DataFrame:
    """The files too quiet to be trimmed against the number that the manuscript states.

    ``quiet``: the rows of ``silence.quiet_files`` at the threshold of
    ``stated``. One row: table, top_db, n_files, n_bonafide, n_spoof,
    stated_n_files, stated_label and ok (the stated number of files, all
    with the stated label).
    """
    if not quiet.empty and set(quiet["top_db"].astype(float)) != {float(stated["top_db"])}:
        raise ValueError(f"the quiet files were not listed at {stated['top_db']} dB")
    labels = quiet.loc[quiet["table"] == stated["table"], "label"].astype(int)
    n = int(len(labels))
    return pd.DataFrame([{
        "table": stated["table"], "top_db": float(stated["top_db"]), "n_files": n,
        "n_bonafide": int((labels == 0).sum()), "n_spoof": int((labels == 1).sum()),
        "stated_n_files": int(stated["n_files"]), "stated_label": int(stated["label"]),
        "ok": bool(n == stated["n_files"] and (labels == stated["label"]).all())}])


# The probes ------------------------------------------------------------------------------------------

# Entries of Section 4.8, compared by notebooks/07_probes.ipynb (gate A6) with
# the numbers the manuscript prints (source of 2026-11-21): "({} dimensions)
# of each layer, using {} training files from {} speakers", "$R^2 \approx
# 0.74$" and "between 0.90 and 0.96 on clean audio and at least 0.88 on
# trimmed audio".
PROBE_ENTRIES = ("probe_sample", "probe_lead", "probe_length")
STATED_PROBES = {"probe_sample": (768, 5000, 20), "probe_lead": (0.74,),
                 "probe_length": (0.90, 0.96, 0.88)}


def probe_entries() -> tuple[Entry, ...]:
    """The entries of PROBE_ENTRIES, in that order."""
    by_key = {entry.key: entry for entry in ENTRIES}
    return tuple(by_key[key] for key in PROBE_ENTRIES)


# Reading the manuscript ---------------------------------------------------------------------------

def normalise(tex: str) -> str:
    """The source without comments, every run of white space as one space."""
    return re.sub(r"\s+", " ", re.sub(r"(?<!\\)%.*", "", tex))


def template_regex(template: str) -> re.Pattern:
    """The regular expression of a template: literal text with ``{}`` for a number."""
    parts = [re.escape(part).replace(r"\ ", r"\s+") for part in template.split("{}")]
    return re.compile(NUMBER.join(parts))


def read_number(text: str) -> float:
    """A number as the manuscript writes it (``19{,}387``, ``+0.11``, ``-5.65``)."""
    return float(text.replace("{,}", ""))


def printed(value: float, digits: int) -> float:
    """``value`` as it is printed with ``digits`` decimals (rounded once)."""
    return float(f"{round(float(value), digits):.{digits}f}")


def check(tables_dir: str, tex: str, entries: Sequence[Entry] = ENTRIES) -> pd.DataFrame:
    """Compare every entry with the manuscript source ``tex``.

    Returns one row per entry: key, section, kind, status, n_numbers,
    manuscript (the numbers read), computed (the numbers of the tables) and
    detail. status is "ok", "cited" or "approximate" (found, nothing to
    compare), "different" (for a statement: the tables do not give what the
    phrase states; ``detail`` says what it needs), "pending correction" (different, and the entry
    says that the manuscript is to be corrected), "not found" (the phrase
    does not occur), "ambiguous" (it
    occurs more than once) or "source missing" (a table or its rows are not
    saved yet).
    """
    text = normalise(tex)
    tables = Tables(tables_dir)
    rows = []
    for entry in entries:
        row = {"key": entry.key, "section": entry.section, "kind": entry.kind, "status": "",
               "n_numbers": 0, "manuscript": "", "computed": "", "detail": entry.note}
        found = list(template_regex(entry.template).finditer(text))
        if len(found) != 1:
            row["status"] = "not found" if not found else "ambiguous"
            row["detail"] = f"{len(found)} occurrences of the phrase"
            rows.append(row)
            continue
        read = [read_number(g) for g in found[0].groups()]
        row["n_numbers"], row["manuscript"] = len(read), ", ".join(f"{v:g}" for v in read)
        if entry.kind in UNCOMPARED:
            row["status"] = entry.kind
            rows.append(row)
            continue
        try:
            computed = [printed(value, digits) for value, digits in entry.values(tables)]
        except SourceMissing as error:
            row["status"], row["detail"] = "source missing", str(error)
            rows.append(row)
            continue
        row["computed"] = ", ".join(f"{v:g}" for v in computed)
        if entry.kind == "statement":
            needed = [float(value) for value in entry.expected]
            row["status"] = "ok" if computed == needed else "different"
            if row["status"] == "different":
                row["detail"] = "the statement needs " + ", ".join(f"{v:g}" for v in needed)
        elif len(computed) != len(read):
            row["status"] = "different"
            row["detail"] = f"{len(read)} numbers in the phrase, {len(computed)} computed"
        else:
            row["status"] = "ok" if computed == read else "different"
        if row["status"] == "different" and entry.pending:
            row["status"], row["detail"] = "pending correction", entry.pending
        rows.append(row)
    return pd.DataFrame(rows)


# Coverage -----------------------------------------------------------------------------------------

SCOPE_START = (r"\abstract{", r"\subsection{Preprocessing and silence trimming}", r"\section{Results}")
SCOPE_END = (r"\keywords{", r"\subsection{Frozen feature extraction}", r"\bibliography")
ENVIRONMENTS = ("table", "figure", "equation", "align", "tabular")
STRUCTURAL = (
    r"\\(?:ref|eqref|cite[a-z]*|label|includegraphics|section|subsection)\*?(?:\[[^\]]*\])?\{[^}]*\}",
    r"[Ll]ayers?[~ ]\d+(?:(?:--|, and | and |, )\d+)*",          # layer 4, layers 3--5, layers 6, 8, and 9
    r"\bL\d+\b", r"\bA\d\d\b",                                    # L4, A10
    r"\d+(?:\.\d+)?~dB", r"\b(?:20|15|40|30|50) and \d+~dB", r"\b15, 20, and 30~dB",
    r"\b4~s\b", r"\b0\.5~s\b", r"\b1~s\b", r"\b3~s\b", r"\b50~ms\b", r"\b[34]-second\b",
    r"ASVspoof~?\s?(?:2019|2021|5)", r"\b2\$\\times\$2\b", r"\b95\\%", r"\b1\{,\}000\b",
    r"Tacotron~2", r"wav2vec2", r"xls-r-2b", r"XLS-R-300M", r"\$n = 12\$",
    r"\b24 (?:configurations|comparisons)", r"\bof the 24\b", r"\b12 layers\b",
    r"exceeds 0\.05",
    r"librosa~[\d.]+", r"10\^\{-8\}", r"10\^\{-3\}", r"10\^\{-5\}", r"\\log\(1\+x\)", r"R\^2",
    r"\\url\{[^}]*\}", r"\\setcounter\{[^}]*\}\{[^}]*\}",
    r"dev-selected \(3 and 5\) and reference \(4\)",
)


def scoped_text(tex: str) -> str:
    """The checked parts of the source: the abstract, the preprocessing
    subsection and everything from the results to the references, without
    tables, figures and displayed equations."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    parts = []
    for start, end in zip(SCOPE_START, SCOPE_END):
        a = tex.find(start)
        b = tex.find(end, a) if a >= 0 else -1
        if a >= 0:
            parts.append(tex[a:b if b >= 0 else len(tex)])
    text = "\n".join(parts)
    for name in ENVIRONMENTS:
        text = re.sub(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (name, name), " ", text, flags=re.S)
    return re.sub(r"\s+", " ", text)


def coverage(tex: str, entries: Sequence[Entry] = ENTRIES) -> pd.DataFrame:
    """Every number of the checked text with its state.

    Returns one row per number: number, state (one of STATES), entry (the
    key of the entry that names it) and context (the text around the number).
    """
    text = scoped_text(tex)
    owner: dict[int, tuple[str, str]] = {}
    for entry in entries:
        matches = list(template_regex(entry.template).finditer(text))
        if len(matches) != 1:                      # a phrase that is absent or ambiguous covers nothing
            continue
        for found in matches:
            for k in range(1, (found.lastindex or 0) + 1):
                for position in range(found.start(k), found.end(k)):
                    owner[position] = (entry.kind if entry.kind in UNCOMPARED else "checked",
                                       entry.key)
    structural = set()
    for pattern in STRUCTURAL:
        for found in re.finditer(pattern, text):
            structural.update(range(found.start(), found.end()))
    rows = []
    for found in re.finditer(r"(?<![\w.])[-+]?\d+(?:\{,\}\d{3})*(?:\.\d+)?", text):
        position = found.end() - 1
        if position in owner:
            state, key = owner[position]
        elif position in structural:
            state, key = "structural", ""
        else:
            state, key = "unchecked", ""
        number = found.group(0)
        if number[0] == "-" and text[max(0, found.start() - 1):found.start()] == "-":
            number = number[1:]                                # the second number of a range a--b
        context = text[max(0, found.start() - 45):found.end() + 30]
        rows.append({"number": number, "state": state, "entry": key, "context": context})
    return pd.DataFrame(rows, columns=["number", "state", "entry", "context"])


def coverage_summary(report: pd.DataFrame) -> pd.DataFrame:
    """Number of numbers per state."""
    counts = report["state"].value_counts()
    return pd.DataFrame({"state": list(STATES), "n": [int(counts.get(s, 0)) for s in STATES]})
