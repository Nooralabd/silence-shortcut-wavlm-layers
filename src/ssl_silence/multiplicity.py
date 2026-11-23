"""Holm and Benjamini-Hochberg adjustments of the effects of the 2x2 design.

A descriptive note, not a rule: the intervals of the effects are not
adjusted, and the replication rule of ``docs/analysis_plan_hubert.md`` uses
the unadjusted intervals.

A family is the configurations (layer and pooling) of one effect on one test
set at one condition: 24 at 40 dB. The p-value of a configuration is a
normal approximation, two-sided, from the estimate divided by its
speaker-level bootstrap standard error (``spk_boot_se`` of
``effects_2x2.csv``). It is an approximation: the intervals of the effects
are percentile intervals, and the 1,000 resamples are too few for resampled
p-values at this number of comparisons. A configuration whose percentile
interval excludes zero can therefore have p >= 0.05 here, and the reverse.

Run as a module, it reads the saved effects table and writes two tables with
their meta files:

    python -m ssl_silence.multiplicity <effects_2x2.csv> <output folder>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import provenance

ALPHA = 0.05
METHOD = "two-sided normal approximation: estimate / speaker-level bootstrap standard error"
FAMILY = ["encoder", "condition", "test_set", "effect"]
TABLE = "multiple_comparisons"
SUMMARY = "multiple_comparisons_summary"


def normal_p(estimate, standard_error) -> np.ndarray:
    """Two-sided normal p-value of estimate / standard_error."""
    from scipy.stats import norm

    z = np.asarray(estimate, dtype=np.float64) / np.asarray(standard_error, dtype=np.float64)
    return 2.0 * norm.sf(np.abs(z))


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm's step-down adjusted p-values (Holm 1979), in the order given:
    the k-th smallest p times (m - k + 1), made monotone and capped at 1."""
    p = np.asarray(p, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    stepped = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(stepped, 1.0)
    return adjusted


def benjamini_hochberg(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (1995), in the order given: the
    k-th smallest p times m / k, made monotone from the largest and capped at 1."""
    p = np.asarray(p, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    stepped = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(stepped, 1.0)
    return adjusted


def adjusted_effects(effects: pd.DataFrame, condition: str = "trim40",
                     alpha: float = ALPHA) -> pd.DataFrame:
    """The effects of one condition with Holm and Benjamini-Hochberg adjusted
    p-values inside each family (one effect on one test set).

    ``effects``: the table ``effects_2x2`` (speaker-level columns).

    Columns: encoder, condition, test_set, effect, pooling, layer, n_family,
    estimate, spk_ci_low, spk_ci_high, spk_boot_se, spk_excludes_zero, z,
    p_normal, p_holm, p_bh, significant_unadjusted, significant_holm,
    significant_bh (each p below ``alpha``).
    """
    chosen = effects[effects["condition"] == condition]
    if chosen.empty:
        raise ValueError(f"the effects table has no row of {condition}")
    if chosen["spk_boot_se"].isna().any() or (chosen["spk_boot_se"] <= 0).any():
        raise ValueError("a speaker-level bootstrap standard error is missing or not positive")
    parts = []
    for _, family in chosen.groupby(FAMILY, sort=False):
        if family.duplicated(["pooling", "layer"]).any():
            raise ValueError("a configuration appears twice in a family")
        p = normal_p(family["estimate"], family["spk_boot_se"])
        part = family[FAMILY + ["pooling", "layer"]].copy()
        part["n_family"] = len(family)
        for column in ("estimate", "spk_ci_low", "spk_ci_high", "spk_boot_se", "spk_excludes_zero"):
            part[column] = family[column]
        part["z"] = family["estimate"] / family["spk_boot_se"]
        part["p_normal"], part["p_holm"], part["p_bh"] = p, holm(p), benjamini_hochberg(p)
        part["significant_unadjusted"] = part["p_normal"] < alpha
        part["significant_holm"] = part["p_holm"] < alpha
        part["significant_bh"] = part["p_bh"] < alpha
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _names(rows: pd.DataFrame) -> str:
    return " ".join(f"{pooling}/L{layer}" for pooling, layer in zip(rows["pooling"], rows["layer"]))


def summarise(adjusted: pd.DataFrame) -> pd.DataFrame:
    """One row per family: how many configurations are significant, and which
    ones lose it.

    Columns: encoder, condition, test_set, effect, n_family,
    n_interval_excludes_zero (the unadjusted percentile interval),
    n_unadjusted, n_holm, n_bh (p below alpha), interval_but_not_unadjusted
    (the interval excludes zero and the normal p does not reach alpha),
    interval_but_not_holm, interval_but_not_bh (configurations as
    "pooling/Llayer", separated by spaces).
    """
    rows = []
    for key, family in adjusted.groupby(FAMILY, sort=False):
        excluded = family[family["spk_excludes_zero"].astype(bool)]
        rows.append({**dict(zip(FAMILY, key)), "n_family": len(family),
                     "n_interval_excludes_zero": len(excluded),
                     "n_unadjusted": int(family["significant_unadjusted"].sum()),
                     "n_holm": int(family["significant_holm"].sum()),
                     "n_bh": int(family["significant_bh"].sum()),
                     "interval_but_not_unadjusted": _names(excluded[~excluded["significant_unadjusted"]]),
                     "interval_but_not_holm": _names(excluded[~excluded["significant_holm"]]),
                     "interval_but_not_bh": _names(excluded[~excluded["significant_bh"]])})
    return pd.DataFrame(rows)


def write_tables(effects_path: str | Path, out_dir: str | Path, condition: str = "trim40",
                 alpha: float = ALPHA) -> tuple[Path, Path]:
    """Read ``effects_2x2.csv`` and write ``multiple_comparisons_<encoder>.csv``
    and ``multiple_comparisons_summary_<encoder>.csv`` with their meta files,
    which record the sha256 of the effects table, the method and alpha."""
    effects = pd.read_csv(effects_path, float_precision="round_trip")
    encoders = sorted(set(effects["encoder"]))
    if len(encoders) != 1:
        raise ValueError(f"the effects table holds {len(encoders)} encoders")
    adjusted = adjusted_effects(effects, condition, alpha)
    extra = {"effects_table": Path(effects_path).name,
             "effects_sha256": provenance.sha256_file(effects_path), "condition": condition,
             "alpha": alpha, "method": METHOD, "approximation": True,
             "family": "the configurations (layer, pooling) of one effect on one test set"}
    out = Path(out_dir)
    paths = (out / f"{TABLE}_{encoders[0]}.csv", out / f"{SUMMARY}_{encoders[0]}.csv")
    provenance.write_table(adjusted, paths[0], **extra)
    provenance.write_table(summarise(adjusted), paths[1], **extra)
    return paths


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m ssl_silence.multiplicity <effects_2x2.csv> <output folder>")
    for written in write_tables(sys.argv[1], sys.argv[2]):
        print(written, provenance.sha256_file(written))
