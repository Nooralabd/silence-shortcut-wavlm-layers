"""LaTeX bodies of the tables of the manuscript, written from saved tables.

Every table is identified by its LaTeX label (``tab:effects``). A builder
takes the saved tables and returns (header, rows); a row is a list of cells
and the first ``label_cells`` of them are labels. Values are read from the
tables, aggregated only where the manuscript does (ranges, ratios and ranks)
and rounded once with ``round`` on the full-precision value. Negative
numbers are written ``$-$x``; an interval that excludes zero is bold where
the manuscript makes it bold; a group label is written on the first row of
its group only.

Sources are CSV files under one folder (``results/`` of the repository),
named as in REPRODUCIBILITY.md.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Mapping, Sequence

import pandas as pd

from .. import config as cfg

SILENCE_BASELINE = "measurements/silence_only_baseline.csv"
SEED_SUMMARY = "seed_summary.csv"
SELECTED_CI = "selected_configs_eer_ci.csv"
EFFECTS = "effects_2x2.csv"
RANK = "rank_agreement.csv"
RANK_LAYERS = "rank_agreement_layers.csv"
CELLS = "cells_2x2.csv"
BY_DURATION = "itw_by_duration.csv"
PER_ATTACK = "per_attack_effect.csv"
THRESHOLDS = "threshold_summary.csv"
FALLBACK = "fallback_counts.csv"
TRIM_AMOUNT = "measurements/trim_amount.csv"
PROBE_R2 = "probe_r2.csv"

BASE = "clean"
MAIN = "trim40"
CONDITION_NAMES = {"clean": "clean", "trim40": "trimmed"}     # as printed in a configuration
THRESHOLD_CONDITIONS = ("trim40", "trim20", "trim15")
TEST_SETS = (("eval", "Eval"), ("itw", "ITW"))
TEST_SET_LONG = {"eval": "Evaluation", "itw": "In-the-Wild"}
POOLINGS = ("stats", "mean")
LAYERS = tuple(range(1, 13))
# Presentation choices of the manuscript (which rows a table shows).
RANK_DETAIL_LAYERS = (1, 2, 3, 4, 5, 6)
RANK_DETAIL_RANGE = (7, 8, 9, 10, 11, 12)
DURATION_LAYER = 4
ATTACK_GROUPS = (            # group, attack sets (one range each), decimals per range
    ("Most affected", (("A10",),), (1,)),
    ("Large", (("A11",), ("A15",)), (1, 1)),
    ("Moderate", (("A07", "A08", "A09", "A12", "A14", "A16"),), (1,)),
    ("Small", (("A13",), ("A18",), ("A19",)), (2, 1, 1)),
    ("Decrease", (("A17",),), (2,)),
)
THRESHOLD_LAYER = 5
THRESHOLD_EFFECT_LAYERS = (3, 4, 5)
THRESHOLD_EFFECT_LAYER = 4
PROBE_TARGETS = (("f0_mean_hz", "Mean F0"), ("hnr_mean_db", "Mean HNR"),
                 ("tilt_db_per_khz", "Spectral tilt"), ("win_lead_40", "Leading silence"),
                 ("length_in_window", "Length in the 4~s window"))
PROBE_LAYERS = (1, 4, 8, 12)

Tables = Mapping[str, pd.DataFrame]


# Formatting -------------------------------------------------------------------------------

def num(value: object, digits: int, signed: bool = False, scale: float = 1.0,
        fmt: str | None = None) -> str:
    """``value`` times ``scale``, rounded once; "--" when missing.

    A negative value is written ``$-$x``; ``signed`` adds ``+``. With
    ``digits`` = 0 an integer with ``{,}`` as thousands separator.
    ``fmt="pvalue"``: three decimals, one significant digit below 0.001.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "--"
    if math.isnan(v):
        return "--"
    v = v * scale
    if fmt == "pvalue":
        text = f"{v:.1g}" if 0 < v < 0.001 else f"{round(v, 3):.3f}"
    elif digits == 0:
        text = f"{int(round(v)):,d}".replace(",", "{,}")
    else:
        text = f"{round(v, digits):.{digits}f}"
        if signed and not text.startswith("-"):
            text = "+" + text
    return "$-$" + text[1:] if text.startswith("-") else text


def limit(value: object, digits: int, signed: bool = False) -> str:
    """A limit of an interval, written so that it does not read as zero.

    Rule of the manuscript: a limit that is not zero but rounds to zero at
    ``digits`` decimals is printed to its first significant digit (0.002211
    is printed 0.002, not 0.00), because a limit printed as zero hides on
    which side of zero the interval ends. Every other limit, and a limit that
    is exactly zero, is printed as ``num`` prints it. The rule applies to
    the limits of an interval only, not to the estimate before it.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return num(value, digits, signed)
    if digits <= 0 or v == 0 or math.isnan(v) or math.isinf(v) or round(v, digits) != 0:
        return num(value, digits, signed)
    v = float(f"{v:.1g}")                                   # one significant digit, rounded once
    return num(v, max(digits, -math.floor(math.log10(abs(v)))), signed)


def interval(low: object, high: object, digits: int, signed: bool = False) -> str:
    """``[low, high]``, each limit written by ``limit``."""
    return f"[{limit(low, digits, signed)}, {limit(high, digits, signed)}]"


def value_ci(value: object, low: object, high: object, digits: int, signed: bool = False,
             bold_if_excludes_zero: bool = False) -> str:
    """``value [low, high]``; bold when asked and the interval excludes zero."""
    if num(value, digits) == "--":
        return "--"
    text = f"{num(value, digits, signed)} {interval(low, high, digits, signed)}"
    try:
        excludes = float(low) > 0 or float(high) < 0
    except (TypeError, ValueError):
        excludes = False
    return rf"\textbf{{{text}}}" if bold_if_excludes_zero and excludes else text


def pm(value: object, sd: object, digits: int) -> str:
    """``value $\\pm$ sd``."""
    return f"{num(value, digits)} $\\pm$ {num(sd, digits)}"


def magnitude_range(values, digits: int, signed: bool = True) -> str:
    """``a to b`` with the smaller magnitude first."""
    present = [float(x) for x in values if x == x]
    if not present:
        return "--"
    low, high = min(present, key=abs), max(present, key=abs)
    return f"{num(low, digits, signed)} to {num(high, digits, signed)}"


def best_layer(layer, proportion) -> str:
    """``L5 (82\\%)``."""
    try:
        return f"L{int(layer)} ({num(proportion, 0, scale=100)}\\%)"
    except (TypeError, ValueError):
        return "--"


def group_labels(rows: list[list[str]], column: int = 0) -> list[list[str]]:
    """Blank a group label on every row but the first of its group."""
    out, last = [], object()
    for row in rows:
        row = list(row)
        if row[column] == last:
            row[column] = ""
        else:
            last = row[column]
        out.append(row)
    return out


def one(table: pd.DataFrame | None, **where) -> pd.Series | None:
    """The first row with the given values, or None."""
    if table is None:
        return None
    mask = pd.Series(True, index=table.index)
    for column, value in where.items():
        mask &= table[column] == value
    hit = table[mask]
    return hit.iloc[0] if len(hit) else None


def get(row: pd.Series | None, column: str):
    """A value of a row, NaN when the row or the column is missing."""
    return float("nan") if row is None or column not in row.index else row[column]


def of_encoder(table: pd.DataFrame, encoder: str) -> pd.DataFrame:
    """The rows of one encoder (all rows of a table without an encoder column)."""
    return table[table["encoder"] == encoder] if "encoder" in table.columns else table


# Builders -----------------------------------------------------------------------------------

def silence(tables: Tables, encoder: str = "wavlm-base"):
    """tab:silence: the silence-only classifier on the untrimmed audio."""
    table = tables[SILENCE_BASELINE]
    table = table[table["condition"] == BASE]
    rows = []
    for features in ("sil", "sil+dur"):
        for top_db in (30, 40):
            r = one(table, features=features, top_db=top_db)
            rows.append([features, f"{top_db}~dB", num(get(r, "dev"), 2), num(get(r, "eval"), 2),
                         num(get(r, "itw"), 2)])
    return ["Features", "Threshold", "Development", "Evaluation", "In-the-Wild"], rows


def matched_summary(tables: Tables, encoder: str) -> pd.DataFrame:
    """Seed summary of the unpadded matched cells (trained and tested in one condition)."""
    summary = of_encoder(tables[SEED_SUMMARY], encoder)
    if "extraction" in summary.columns:
        summary = summary[summary["extraction"] == "unpadded"]
    return summary[summary["condition"] == summary["test_condition"]]


def main(tables: Tables, encoder: str = "wavlm-base"):
    """tab:main: the dev-selected and reference configurations."""
    means = matched_summary(tables, encoder)
    ci = of_encoder(tables[SELECTED_CI], encoder)
    configs = ci.drop_duplicates(["condition", "pooling", "layer"])
    order = {BASE: 0, MAIN: 1}
    configs = configs.assign(_order=configs["condition"].map(order).fillna(9)).sort_values(
        ["_order", "pooling", "layer"])
    rows = []
    for _, c in configs.iterrows():
        key = dict(condition=c["condition"], pooling=c["pooling"], layer=int(c["layer"]))
        m = one(means, **key)
        e, w = one(ci, **key, test_set="eval"), one(ci, **key, test_set="itw")
        name = CONDITION_NAMES.get(c["condition"], c["condition"])
        rows.append([f"{name}/{c['pooling']}/L{int(c['layer'])}",
                     "dev" if c["selection"] == "dev-selected" else "ref",
                     pm(get(m, "dev_eer_mean"), get(m, "dev_eer_sd"), 2),
                     pm(get(m, "eval_eer_mean"), get(m, "eval_eer_sd"), 2),
                     interval(get(e, "spk_ci_low"), get(e, "spk_ci_high"), 2),
                     pm(get(m, "eval_tdcf_mean"), get(m, "eval_tdcf_sd"), 4),
                     pm(get(m, "itw_eer_mean"), get(m, "itw_eer_sd"), 2),
                     interval(get(w, "spk_ci_low"), get(w, "spk_ci_high"), 2)])
    return (["Configuration", "Sel.", "Dev EER", "Eval EER", "Eval 95% CI", "Eval min t-DCF",
             "ITW EER", "ITW 95% CI"], rows)


def effects_of(tables: Tables, encoder: str, condition: str = MAIN) -> pd.DataFrame:
    table = of_encoder(tables[EFFECTS], encoder)
    return table[table["condition"] == condition]


def effect_cell(table: pd.DataFrame, **where) -> str:
    e = one(table, **where)
    return value_ci(get(e, "estimate"), get(e, "spk_ci_low"), get(e, "spk_ci_high"), 2,
                    signed=True, bold_if_excludes_zero=True)


def effects_for_pooling(tables: Tables, encoder: str, pooling: str):
    table = effects_of(tables, encoder)
    rows = []
    for test_set, label in TEST_SETS:
        for layer in LAYERS:
            rows.append([label, str(layer)] + [
                effect_cell(table, test_set=test_set, pooling=pooling, layer=layer, effect=effect)
                for effect in ("delta_test", "delta_train", "delta_total", "interaction")])
    return (["Set", "L", r"$\Delta_{\mathrm{test}}$", r"$\Delta_{\mathrm{train}}$",
             r"$\Delta_{\mathrm{total}}$", "$I$"], group_labels(rows))


def effects(tables: Tables, encoder: str = "wavlm-base"):
    """tab:effects: four effects by layer, statistics pooling."""
    return effects_for_pooling(tables, encoder, "stats")


def effects_mean(tables: Tables, encoder: str = "wavlm-base"):
    """tab:effects-mean: the same with mean pooling."""
    return effects_for_pooling(tables, encoder, "mean")


def effects_other(tables: Tables, encoder: str = "wavlm-base"):
    """tab:effects-other: the two remaining effects, both poolings."""
    table = effects_of(tables, encoder)
    rows = []
    for test_set, label in TEST_SETS:
        for layer in LAYERS:
            rows.append([label, str(layer)] + [
                effect_cell(table, test_set=test_set, pooling=pooling, layer=layer, effect=effect)
                for effect in ("delta_train_c", "delta_test_t") for pooling in POOLINGS])
    return (["Set", "L", r"$\Delta_{\mathrm{train}}^{\mathrm{C}}$ stats", "mean",
             r"$\Delta_{\mathrm{test}}^{\mathrm{T}}$ stats", "mean"], group_labels(rows))


def rank(tables: Tables, encoder: str = "wavlm-base"):
    """tab:rank: agreement of the layer rankings at 40 dB."""
    table = of_encoder(tables[RANK], encoder)
    table = table[table["condition"] == MAIN]
    detail = of_encoder(tables[RANK_LAYERS], encoder)
    detail = detail[detail["condition"] == MAIN]
    rows = []
    for test_set in ("eval", "itw"):
        for pooling in POOLINGS:
            r = one(table, test_set=test_set, pooling=pooling)
            best = []
            for name in ("clean", "trimmed"):
                layer = get(r, f"best_layer_{name}")
                d = (one(detail, test_set=test_set, pooling=pooling, layer=int(layer))
                     if layer == layer else None)
                best.append(best_layer(layer, get(d, f"p_best_{name}")))
            rows.append([TEST_SET_LONG[test_set], pooling,
                         value_ci(get(r, "kendall_tau"), get(r, "tau_ci_low"), get(r, "tau_ci_high"), 2),
                         num(get(r, "kendall_p_exact"), 3, fmt="pvalue"),
                         value_ci(get(r, "spearman_rho"), get(r, "rho_ci_low"), get(r, "rho_ci_high"), 2),
                         " / ".join(best)])
    return (["Set", "Pooling", r"Kendall's $\tau$", "$p$ (exact)", r"Spearman's $\rho$",
             "Best layer (clean / trimmed)"], rows)


def rankdetail(tables: Tables, encoder: str = "wavlm-base"):
    """tab:rankdetail: EER per layer in the two matched cells, statistics pooling."""
    cells = of_encoder(tables[CELLS], encoder)
    cells = cells[(cells["condition"] == MAIN) & (cells["pooling"] == "stats")]
    cc = cells[cells["cell"] == "cc"].set_index("layer")["eval_eer_mean"]
    tt = cells[cells["cell"] == "tt"].set_index("layer")["eval_eer_mean"]
    ratio = (tt / cc).dropna()
    rank_cc, rank_tt = cc.rank(method="min"), tt.rank(method="min")
    rows = []
    for layer in RANK_DETAIL_LAYERS:
        if layer in cc.index and layer in tt.index:
            rows.append([str(layer), num(cc[layer], 2), num(tt[layer], 2),
                         f"$\\times${num(ratio[layer], 1)}",
                         f"{int(rank_cc[layer])} $\\rightarrow$ {int(rank_tt[layer])}"])
        else:
            rows.append([str(layer), "--", "--", "--", "--"])
    span = [layer for layer in RANK_DETAIL_RANGE if layer in cc.index and layer in tt.index]
    label = f"{RANK_DETAIL_RANGE[0]}--{RANK_DETAIL_RANGE[-1]}"
    if span:
        rows.append([label, f"{num(cc[span].min(), 2)}--{num(cc[span].max(), 2)}",
                     f"{num(tt[span].min(), 2)}--{num(tt[span].max(), 2)}",
                     f"$\\times${num(ratio[span].min(), 1)}--{num(ratio[span].max(), 1)}",
                     "same relative order"])
    else:
        rows.append([label, "--", "--", "--", "same relative order"])
    return ["Layer", "Clean", "Trimmed", "Ratio", r"Rank (clean $\rightarrow$ trimmed)"], rows


def duration(tables: Tables, encoder: str = "wavlm-base"):
    """tab:duration: In-the-Wild by original duration, statistics pooling, one layer."""
    table = of_encoder(tables[BY_DURATION], encoder)
    table = table[(table["condition"] == MAIN) & (table["pooling"] == "stats")
                  & (table["layer"].astype(int) == DURATION_LAYER)]
    subsets = [s for s in table["subset"].unique() if s != "all"]
    short = next((s for s in subsets if s.startswith("<=")), "<=4s")
    long = next((s for s in subsets if s.startswith(">")), ">4s")
    rows = []
    for quantity, label in (("eer_cc", r"$E(\mathrm{C},\mathrm{C})$"),
                            ("eer_ct", r"$E(\mathrm{C},\mathrm{T})$"),
                            ("eer_tc", r"$E(\mathrm{T},\mathrm{C})$"),
                            ("eer_tt", r"$E(\mathrm{T},\mathrm{T})$")):
        a, b = one(table, quantity=quantity, subset=short), one(table, quantity=quantity, subset=long)
        rows.append([label, num(get(a, "mean"), 2), num(get(b, "mean"), 2)])
    for quantity, label in (("delta_test", r"$\Delta_{\mathrm{test}}$"),
                            ("delta_train", r"$\Delta_{\mathrm{train}}$"),
                            ("delta_total", r"$\Delta_{\mathrm{total}}$")):
        a, b = one(table, quantity=quantity, subset=short), one(table, quantity=quantity, subset=long)
        rows.append([label, pm(get(a, "mean"), get(a, "sd"), 2), pm(get(b, "mean"), get(b, "sd"), 2)])
    return ["", r"$\le$4~s", "$>$4~s"], rows


def attack(tables: Tables, encoder: str = "wavlm-base"):
    """tab:attack: the change per attack as ranges over the six configurations."""
    table = of_encoder(tables[PER_ATTACK], encoder)
    table = table[table["condition"] == MAIN]
    rows = []
    for group, attack_sets, decimals in ATTACK_GROUPS:
        ranges = [magnitude_range(table[table["attack"].isin(attacks)]["estimate"], d)
                  for attacks, d in zip(attack_sets, decimals)]
        rows.append([group, "; ".join(", ".join(a) for a in attack_sets), "; ".join(ranges)])
    return ["Group", "Attacks", "Change"], rows


def threshold(tables: Tables, encoder: str = "wavlm-base"):
    """tab:threshold: the main results at the three trimming thresholds."""
    summary = of_encoder(tables[THRESHOLDS], encoder)
    rank_table = of_encoder(tables[RANK], encoder)
    removed = tables[TRIM_AMOUNT]
    fallback = tables.get(FALLBACK)
    means = matched_summary(tables, encoder)
    effect_table = of_encoder(tables[EFFECTS], encoder)
    columns: dict[str, dict[str, str]] = {}
    for condition in THRESHOLD_CONDITIONS:
        top_db = cfg.trim_top_db(condition)
        c: dict[str, str] = {}
        counts = None
        if fallback is not None:
            counts = fallback[(fallback["condition"] == condition)
                              & (fallback["label"].astype(str) == "all")]
        c["fallback"] = " / ".join(num(get(one(counts, split=s), "n_fallback"), 0)
                                   for s in ("eval", "itw"))
        for s in ("eval", "itw"):
            part = removed[(removed["top_db"] == top_db) & (removed["split"] == s)]
            c[f"removed_{s}"] = " / ".join(
                num(get(one(part, label=label), "removed_total_median_s"), 2) for label in (0, 1))
        stats_row = one(summary, condition=condition, pooling="stats")
        dev = get(stats_row, "dev_selected_layer")
        c["dev_selected"] = f"L{int(dev)}" if dev == dev else "--"
        m = one(means, condition=condition, pooling="stats", layer=THRESHOLD_LAYER)
        c["eer_eval"] = num(get(m, "eval_eer_mean"), 2)
        c["eer_itw"] = num(get(m, "itw_eer_mean"), 2)
        for s in ("eval", "itw"):
            taus = [get(one(rank_table, condition=condition, pooling=p, test_set=s), "kendall_tau")
                    for p in POOLINGS]
            c[f"tau_{s}"] = " / ".join(num(t, 2) for t in taus)
            c[f"best_{s}"] = best_layer(get(stats_row, f"best_layer_{s}_trimmed"),
                                        get(stats_row, f"p_best_{s}_trimmed"))
        part = effect_table[effect_table["condition"] == condition]
        values = [get(one(part, test_set="eval", pooling=p, layer=layer, effect="delta_test"), "estimate")
                  for layer in THRESHOLD_EFFECT_LAYERS for p in POOLINGS]
        c["delta_test_range"] = magnitude_range(values, 1, signed=False)
        for s in ("eval", "itw"):
            for key, name in (("total", "delta_total"), ("inter", "interaction")):
                base = f"{name}_{s}_L{THRESHOLD_EFFECT_LAYER}"
                c[f"{key}_{s}"] = value_ci(get(stats_row, base), get(stats_row, f"{base}_spk_ci_low"),
                                           get(stats_row, f"{base}_spk_ci_high"), 2)
        columns[condition] = c
    labels = [
        ("fallback", "Fallback files, eval / ITW"),
        ("removed_eval", "Median removed (s), eval, bona / spoof"),
        ("removed_itw", "Median removed (s), ITW, bona / spoof"),
        ("dev_selected", "Dev-selected layer, trimmed"),
        ("eer_eval", f"Eval EER, stats, L{THRESHOLD_LAYER} (\\%)"),
        ("eer_itw", f"ITW EER, stats, L{THRESHOLD_LAYER} (\\%)"),
        ("tau_eval", r"Kendall's $\tau$, eval, stats / mean"),
        ("tau_itw", r"Kendall's $\tau$, ITW, stats / mean"),
        ("best_eval", "Best layer, eval"),
        ("best_itw", "Best layer, ITW"),
        ("delta_test_range", r"$\Delta_{\mathrm{test}}$, eval, L3--L5, both poolings"),
        ("total_eval", r"$\Delta_{\mathrm{total}}$, eval"),
        ("inter_eval", "Interaction, eval"),
        ("total_itw", r"$\Delta_{\mathrm{total}}$, ITW"),
        ("inter_itw", "Interaction, ITW"),
    ]
    rows = [[label] + [columns[condition][key] for condition in THRESHOLD_CONDITIONS]
            for key, label in labels]
    header = [""] + [f"{cfg.trim_top_db(condition):g}~dB" for condition in THRESHOLD_CONDITIONS]
    return header, rows


def probes(tables: Tables, encoder: str = "wavlm-base"):
    """tab:probes: probe R2 at selected layers, clean audio, speaker folds."""
    r2 = of_encoder(tables[PROBE_R2], encoder)
    if "condition" in r2.columns:
        r2 = r2[r2["condition"] == BASE]
    r2 = r2[(r2["cv"] == "speaker") & (r2["control"] == "real")]
    rows = [[label] + [num(get(one(r2, target=target, layer=layer), "r2_mean"), 2)
                       for layer in PROBE_LAYERS] for target, label in PROBE_TARGETS]
    return ["Target"] + [f"L{layer}" for layer in PROBE_LAYERS], rows


def bca(tables: Tables, encoder: str = "wavlm-base"):
    """tab:bca: percentile and BCa intervals of tau at the three thresholds."""
    table = of_encoder(tables[RANK], encoder)
    rows = []
    for condition in THRESHOLD_CONDITIONS:
        for test_set, label in TEST_SETS:
            for pooling in POOLINGS:
                r = one(table, condition=condition, test_set=test_set, pooling=pooling)
                if r is None:
                    continue
                rows.append([f"{cfg.trim_top_db(condition):g}~dB", label, pooling,
                             num(r["kendall_tau"], 2),
                             interval(r["tau_ci_low"], r["tau_ci_high"], 2),
                             interval(r["tau_bca_low"], r["tau_bca_high"], 2),
                             num(r["z0_tau"], 2), num(r["a_tau"], 3),
                             f"{num(r['tie_share_tau'], 0, scale=100)}\\%",
                             num(r["kendall_p_exact"], 3, fmt="pvalue")])
    return (["Threshold", "Set", "Pooling", r"$\tau$", "Percentile", "BCa", "$z_0$", "$a$", "Ties",
             "$p$"], group_labels(rows))


TABLES: dict[str, dict] = {
    "tab:silence": {"build": silence, "label_cells": 2, "sources": [SILENCE_BASELINE]},
    "tab:main": {"build": main, "label_cells": 2, "sources": [SEED_SUMMARY, SELECTED_CI]},
    "tab:effects": {"build": effects, "label_cells": 2, "sources": [EFFECTS]},
    "tab:rank": {"build": rank, "label_cells": 2, "sources": [RANK, RANK_LAYERS]},
    "tab:rankdetail": {"build": rankdetail, "label_cells": 1, "sources": [CELLS]},
    "tab:duration": {"build": duration, "label_cells": 1, "sources": [BY_DURATION]},
    "tab:attack": {"build": attack, "label_cells": 2, "sources": [PER_ATTACK]},
    "tab:threshold": {"build": threshold, "label_cells": 1,
                      "sources": [THRESHOLDS, RANK, TRIM_AMOUNT, FALLBACK, SEED_SUMMARY, EFFECTS]},
    "tab:probes": {"build": probes, "label_cells": 1, "sources": [PROBE_R2]},
    "tab:effects-mean": {"build": effects_mean, "label_cells": 2, "sources": [EFFECTS]},
    "tab:effects-other": {"build": effects_other, "label_cells": 2, "sources": [EFFECTS]},
    "tab:bca": {"build": bca, "label_cells": 3, "sources": [RANK]},
}


# Reading and writing ---------------------------------------------------------------------------

def read_tables(tables_dir: str, names: Sequence[str]) -> dict[str, pd.DataFrame]:
    """The named CSV files of a folder, numbers read exactly as written; a missing file raises."""
    out = {}
    for name in dict.fromkeys(names):
        path = os.path.join(tables_dir, *name.split("/"))
        if not os.path.exists(path):
            raise FileNotFoundError(f"table {name} is not in {tables_dir}")
        out[name] = pd.read_csv(path, float_precision="round_trip")
    return out


def build(label: str, tables_dir: str, encoder: str = "wavlm-base") -> tuple[list[str], list[list[str]]]:
    """(header, rows) of the table with a LaTeX label, from the CSV files of ``tables_dir``."""
    spec = TABLES[label]
    return spec["build"](read_tables(tables_dir, spec["sources"]), encoder)


def file_name(label: str) -> str:
    """File of a table body: ``tab:effects-mean`` -> ``effects-mean.tex``."""
    return label.partition(":")[2] + ".tex"


def latex_body(label: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """The body of a tabular: comment lines with the label, sources and columns, then the rows."""
    lines = [f"% {label}", "% sources: " + ", ".join(TABLES[label]["sources"]),
             "% columns: " + " | ".join(header)]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    return "\n".join(lines) + "\n"


def write_tables(tables_dir: str, out_dir: str, labels: Sequence[str] | None = None,
                 encoder: str = "wavlm-base", log: Callable[[str], None] = print) -> dict[str, str]:
    """Write ``<out_dir>/<name>.tex`` for every table (or ``labels``). Returns label -> path."""
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for label in (labels or TABLES):
        header, rows = build(label, tables_dir, encoder)
        path = os.path.join(out_dir, file_name(label))
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(latex_body(label, header, rows))
        log(f"{label}: {len(rows)} rows x {len(header)} columns -> {path}")
        written[label] = path
    return written
