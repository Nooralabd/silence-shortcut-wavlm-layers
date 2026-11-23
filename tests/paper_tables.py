"""Invented result tables in the layout that the paper modules read.

Every value is a simple function of its keys, so a test can state the
expected cell by hand.
"""

from __future__ import annotations

import os

import pandas as pd

from ssl_silence import config as cfg
from ssl_silence.paper import tables as T

ENCODER = "wavlm-base"
CONDITIONS = ("clean", "trim40", "trim20", "trim15")
TRIMMED = CONDITIONS[1:]
POOLINGS = ("stats", "mean")
LAYERS = tuple(range(1, 13))
ATTACKS = cfg.SPOOF_SYSTEM_IDS["eval"]


def eval_eer(condition: str, pooling: str, layer: int) -> float:
    """EER of a matched cell on the evaluation set."""
    return 1.0 + 0.25 * layer + 3.0 * CONDITIONS.index(condition) + (0.125 if pooling == "mean" else 0)


def effect_value(condition: str, test_set: str, pooling: str, layer: int, effect: str) -> float:
    sign = -1.0 if effect == "interaction" else 1.0
    return sign * (cfg.EFFECTS.index(effect) + 1 + layer / 8 + (0.5 if test_set == "itw" else 0)
                   + (0.25 if pooling == "mean" else 0) + TRIMMED.index(condition))


def seed_summary() -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        for pooling in POOLINGS:
            for layer in LAYERS:
                eer = eval_eer(condition, pooling, layer)
                rows.append({"encoder": ENCODER, "extraction": "unpadded", "condition": condition,
                             "test_condition": condition, "pooling": pooling, "layer": layer,
                             "n_seeds": 5, "dev_eer_mean": eer / 4, "dev_eer_sd": 0.0125,
                             "eval_eer_mean": eer, "eval_eer_sd": 0.0375,
                             "eval_tdcf_mean": eer / 40, "eval_tdcf_sd": 0.00025,
                             "itw_eer_mean": 30 - eer, "itw_eer_sd": 0.25})
    # A cross-condition cell and a padded run must not be read as matched cells.
    rows.append({**rows[0], "test_condition": "trim40", "eval_eer_mean": 99.0})
    rows.append({**rows[0], "extraction": "padded", "eval_eer_mean": 98.0})
    return pd.DataFrame(rows)


def selected_ci() -> pd.DataFrame:
    configs = (("clean", "stats", 3, "dev-selected"), ("clean", "stats", 4, "reference"),
               ("trim40", "mean", 4, "reference + dev-selected"), ("trim40", "stats", 5, "dev-selected"))
    rows = []
    for condition, pooling, layer, selection in configs:
        for test_set in ("eval", "itw"):
            centre = eval_eer(condition, pooling, layer) if test_set == "eval" else 20.0
            rows.append({"encoder": ENCODER, "config": f"{condition}/{pooling}/L{layer}",
                         "selection": selection, "condition": condition, "pooling": pooling,
                         "layer": layer, "test_set": test_set, "eer": centre,
                         "spk_ci_low": centre - 0.5, "spk_ci_high": centre + 0.5})
    return pd.DataFrame(rows)


def effects() -> pd.DataFrame:
    rows = []
    for condition in TRIMMED:
        for test_set in ("eval", "itw"):
            for pooling in POOLINGS:
                for layer in LAYERS:
                    for effect in cfg.EFFECTS:
                        value = effect_value(condition, test_set, pooling, layer, effect)
                        # On In-the-Wild the intervals of delta_train include zero.
                        wide = test_set == "itw" and effect == "delta_train"
                        half = abs(value) + 1.0 if wide else 0.5
                        rows.append({"encoder": ENCODER, "condition": condition, "test_set": test_set,
                                     "pooling": pooling, "layer": layer, "effect": effect,
                                     "estimate": value, "spk_ci_low": value - half,
                                     "spk_ci_high": value + half, "file_ci_low": value - 0.1,
                                     "file_ci_high": value + 0.1})
    return pd.DataFrame(rows)


def rank() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, detail = [], []
    for k, condition in enumerate(TRIMMED):
        for test_set in ("itw", "eval"):
            for pooling in POOLINGS:
                tau = 0.75 - 0.125 * k - (0.0625 if pooling == "mean" else 0)
                best = 4 if test_set == "itw" else 5
                summary.append({"encoder": ENCODER, "condition": condition, "test_set": test_set,
                                "pooling": pooling, "kendall_tau": tau, "tau_ci_low": tau - 0.25,
                                "tau_ci_high": tau + 0.125, "tau_bca_low": tau - 0.125,
                                "tau_bca_high": tau + 0.1875, "z0_tau": 0.5, "a_tau": 0.0125,
                                "tie_share_tau": 0.125, "kendall_p_exact": 0.0004 if k == 0 else 0.0625,
                                "spearman_rho": tau + 0.125, "rho_ci_low": tau, "rho_ci_high": tau + 0.1875,
                                "best_layer_clean": best, "best_layer_trimmed": best + (k > 0)})
                for layer in LAYERS:
                    detail.append({"encoder": ENCODER, "condition": condition, "test_set": test_set,
                                   "pooling": pooling, "layer": layer,
                                   "p_best_clean": 0.875 if layer == best else 0.0,
                                   "p_best_trimmed": 0.625 if layer == best + (k > 0) else 0.0})
    return pd.DataFrame(summary), pd.DataFrame(detail)


def cells() -> pd.DataFrame:
    rows = []
    for pooling in POOLINGS:
        for layer in LAYERS:
            for cell, condition in (("cc", "clean"), ("tt", "trim40")):
                # Trimmed: layers 1 and 2 become the worst, the others keep their order.
                value = eval_eer(condition, pooling, layer) + (20.0 if cell == "tt" and layer <= 2 else 0)
                rows.append({"encoder": ENCODER, "condition": "trim40", "pooling": pooling,
                             "layer": layer, "cell": cell, "eval_eer_mean": value,
                             "itw_eer_mean": 30 - value})
    return pd.DataFrame(rows)


def by_duration() -> pd.DataFrame:
    rows = []
    quantities = ("eer_cc", "eer_ct", "eer_tc", "eer_tt", "delta_test", "delta_train", "delta_total")
    for pooling in POOLINGS:
        for layer in (3, 4, 5):
            for s, subset in enumerate(("all", "<=4s", ">4s")):
                for q, quantity in enumerate(quantities):
                    mean = (20.0 - q - s) if quantity.startswith("eer") else -(q - 3) / 4 - s / 8
                    rows.append({"encoder": ENCODER, "condition": "trim40", "pooling": pooling,
                                 "layer": layer, "subset": subset, "quantity": quantity,
                                 "mean": mean + layer / 100, "sd": 0.125})
    return pd.DataFrame(rows)


def per_attack() -> pd.DataFrame:
    rows = []
    for a, attack in enumerate(ATTACKS):
        for p, pooling in enumerate(POOLINGS):
            for layer in (3, 4, 5):
                value = -1.5 + layer / 8 if attack == "A17" else 0.5 * (a + 1) + layer + p / 4
                rows.append({"encoder": ENCODER, "condition": "trim40", "attack": attack,
                             "pooling": pooling, "layer": layer, "estimate": value,
                             "spk_ci_low": value - 0.5, "spk_ci_high": value + 0.5})
    return pd.DataFrame(rows)


def thresholds() -> pd.DataFrame:
    rows = []
    for k, condition in enumerate(TRIMMED):
        for pooling in POOLINGS:
            row = {"encoder": ENCODER, "condition": condition, "pooling": pooling,
                   "dev_selected_layer": 5, "best_layer_eval_trimmed": 5 + (k > 0),
                   "p_best_eval_trimmed": 0.625, "best_layer_itw_trimmed": 4,
                   "p_best_itw_trimmed": 0.875}
            for test_set in ("eval", "itw"):
                for name, sign in (("delta_total", 1.0), ("interaction", -1.0)):
                    value = sign * (4.0 + k + (0.5 if test_set == "itw" else 0))
                    base = f"{name}_{test_set}_L4"
                    row.update({base: value, f"{base}_spk_ci_low": value - 0.75,
                                f"{base}_spk_ci_high": value + 0.75})
            rows.append(row)
    return pd.DataFrame(rows)


def fallback() -> pd.DataFrame:
    rows = []
    for k, condition in enumerate(TRIMMED):
        for split, n in (("train", 8), ("dev", 5), ("eval", 21), ("itw", 49)):
            for label, share in (("0", 0.25), ("1", 0.75), ("all", 1.0)):
                rows.append({"condition": condition, "split": split, "label": label,
                             "n_fallback": int(n * share) * (1 + 25 * k)})
    return pd.DataFrame(rows)


def trim_amount() -> pd.DataFrame:
    rows = []
    for k, top_db in enumerate((40, 20, 15)):
        for split in ("train", "dev", "eval", "itw"):
            for label in (0, 1):
                rows.append({"top_db": top_db, "dataset": "x", "split": split, "label": label,
                             "removed_total_median_s": 1.0 + k / 4 - label / 2 + (split == "itw") / 8,
                             "removed_frac_median": 0.25 + k / 8})
    return pd.DataFrame(rows)


def silence_baseline() -> pd.DataFrame:
    rows = [{"condition": condition, "features": features, "top_db": top_db,
             "dev": 30.0 + top_db / 8 + f, "eval": 20.0 + top_db / 16 + f, "itw": 50.0 + f + c}
            for c, condition in enumerate(("clean", "trim40")) for f, features in enumerate(("sil", "sil+dur"))
            for top_db in (15, 20, 30, 40, 50)]
    return pd.DataFrame(rows)


def probe_r2() -> pd.DataFrame:
    targets = ("f0_mean_hz", "hnr_mean_db", "tilt_db_per_khz", "win_lead_40", "pad", "length_in_window")
    rows = [{"encoder": ENCODER, "condition": condition, "cv": cv, "layer": layer, "target": target,
             "control": control,
             "r2_mean": (0.9 - t / 16 - layer / 64 + (cv == "random") / 32 - (condition == "trim40") / 8)
             if control == "real" else -0.01, "r2_sd": 0.02}
            for condition in ("clean", "trim40") for cv in ("random", "speaker") for layer in LAYERS
            for t, target in enumerate(targets) for control in ("real", "shuffled")]
    return pd.DataFrame(rows)


def all_tables() -> dict[str, pd.DataFrame]:
    summary, detail = rank()
    return {T.SILENCE_BASELINE: silence_baseline(), T.SEED_SUMMARY: seed_summary(),
            T.SELECTED_CI: selected_ci(), T.EFFECTS: effects(), T.RANK: summary,
            T.RANK_LAYERS: detail, T.CELLS: cells(), T.BY_DURATION: by_duration(),
            T.PER_ATTACK: per_attack(), T.THRESHOLDS: thresholds(), T.FALLBACK: fallback(),
            T.TRIM_AMOUNT: trim_amount(), T.PROBE_R2: probe_r2()}


def write_tables(folder: str) -> str:
    """Write every table as a CSV file under ``folder`` and return the folder."""
    for name, table in all_tables().items():
        path = os.path.join(folder, *name.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        table.to_csv(path, index=False)
    return folder


def fake_manuscript(folder: str) -> str:
    """A LaTeX source that holds every table as the builders write it, with
    header rows, rules and a comment, as in a manuscript."""
    parts = [r"\documentclass{article}\begin{document}"]
    for label in T.TABLES:
        header, rows = T.build(label, folder)
        body = "\n".join(" & ".join(row) + r" \\" for row in rows)
        parts.append("\n".join([
            r"\begin{table}[t]", r"\caption{A caption with the number 12.5 in it}",
            rf"\label{{{label}}}", r"\centering", r"\begin{tabular}{@{}" + "l" * len(header) + "@{}}",
            r"\toprule", " & ".join(header) + r" \\  % header with 3 numbers 1 2 3", r"\midrule",
            body, r"\bottomrule", r"\end{tabular}", r"\end{table}"]))
    parts.append(r"\end{document}")
    return "\n\n".join(parts) + "\n"

