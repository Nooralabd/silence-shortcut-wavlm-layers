"""Units of the replication rule and the comparison of two encoders on them.

A unit is one estimate with its speaker-level percentile interval. It is in
the rule when the interval of the reference encoder excludes zero.

The rule (``docs/analysis_plan_hubert.md``, Section 6): a unit of the rule
is replicated if the estimate of the second encoder has the same sign and
its speaker-level percentile interval excludes zero. Kendall's tau follows
the same rule (``docs/deviations.md``). The findings of Section 6a are
FINDINGS.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Mapping

import numpy as np
import pandas as pd

from . import config as cfg

KEYS = ["finding", "quantity", "condition", "test_set", "subset", "attack", "pooling", "layer"]
UNIT_COLUMNS = KEYS + ["estimate", "spk_ci_low", "spk_ci_high", "sign", "in_rule"]
GROUP_KEYS = ["finding", "quantity", "condition", "test_set", "subset"]
SUMMARY_COLUMNS = GROUP_KEYS + ["n_units", "n_in_rule", "n_positive_in_rule",
                                "n_negative_in_rule"]
OUTCOME_COUNTS = ["both_exclude_same_sign", "both_exclude_opposite_sign", "reference_only",
                  "replica_only", "neither"]
OUTCOME_COLUMNS = GROUP_KEYS + ["n_units", "n_in_rule", "n_replicated"] + OUTCOME_COUNTS
VERDICTS = ("replicated", "not replicated", "descriptive")

MAIN_CONDITION = "trim40"
THRESHOLD_CONDITIONS = ("trim20", "trim15")
ALL_LAYERS = tuple(range(1, 13))
# At the stricter thresholds the cross-condition cells of the reference encoder
# exist for these layers only; delta_total needs the matched cells alone.
THRESHOLD_LAYERS = (3, 4, 5)
MATCHED_CELL_EFFECTS = ("delta_total",)
# The duration split: the label of the whole test set, and the number of subsets.
WHOLE_SET = "all"
N_DURATION_SUBSETS = 2
SHORT_SUBSET, LONG_SUBSET = "<=4s", ">4s"

# The findings of Section 6a of the analysis plan: (id, type, unit family, quantities,
# conditions, test sets, subsets); None: any. Findings without units (the identity of the
# best layer, the padding deviation) are not listed. A unit belongs to the first finding it
# fits; the last entry holds the units that the plan lists after the table ("the remaining
# effect units at 20 and 15 dB").
EFFECT, ABSENCE = "effect", "absence finding"
REMAINING = "remaining units at 20 and 15 dB"
_MAIN, _STRICT = (MAIN_CONDITION,), THRESHOLD_CONDITIONS
FINDINGS: tuple[tuple, ...] = (
    ("F1", EFFECT, "effect", ("delta_total",), _MAIN, ("eval",), None),
    ("F2", EFFECT, "effect", ("delta_test",), _MAIN, ("eval",), None),
    ("F3", EFFECT, "effect", ("delta_test_t",), _MAIN, ("eval",), None),
    ("F4", EFFECT, "effect", ("delta_train_c",), _MAIN, ("eval",), None),
    ("F5", EFFECT, "effect", ("interaction",), _MAIN, ("eval",), None),
    ("F6", EFFECT, "effect", ("delta_train",), _MAIN, ("eval",), None),
    ("F7", ABSENCE, "effect", ("delta_test",), _MAIN, ("itw",), None),
    ("F8", EFFECT, "effect", ("delta_total",), _MAIN, ("itw",), None),
    ("F9", EFFECT, "effect", ("interaction",), _MAIN, ("itw",), None),
    ("F10", EFFECT, "effect", ("delta_test_t",), _MAIN, ("itw",), None),
    ("F11", EFFECT, "effect", ("delta_train_c",), _MAIN, ("itw",), None),
    ("F12", EFFECT, "effect", ("delta_train",), _MAIN, ("itw",), None),
    ("F13", EFFECT, "rank_agreement", ("kendall_tau",), _MAIN, None, None),
    ("F15", EFFECT, "duration", None, _MAIN, ("itw",), (SHORT_SUBSET,)),
    ("F16", ABSENCE, "duration", ("delta_test",), _MAIN, ("itw",), (LONG_SUBSET,)),
    ("F17", EFFECT, "duration", None, _MAIN, ("itw",), (LONG_SUBSET,)),
    ("F18", EFFECT, "per_attack", None, _MAIN, ("eval",), None),
    ("F19", EFFECT, "effect", ("delta_test",), _STRICT, ("eval",), None),
    ("F20", EFFECT, "effect", ("delta_total",), _STRICT, ("eval",), None),
    ("F21", EFFECT, "effect", ("interaction",), _STRICT, ("eval",), None),
    ("F22", EFFECT, "effect", ("interaction",), _STRICT, ("itw",), None),
    ("F23", EFFECT, "effect", ("delta_test",), _STRICT, ("itw",), None),
    ("F24", EFFECT, "rank_agreement", ("kendall_tau", "tau_difference"), _STRICT, None, None),
    ("F26", EFFECT, "per_attack", None, _STRICT, ("eval",), None),
    (REMAINING, EFFECT, "effect", None, _STRICT, None, None),
)
ALL_UNITS = "all units"
FINDING_COLUMNS = ["finding_id", "type", "n_units", "n_in_rule", "n_replicated"] + OUTCOME_COUNTS

PREREGISTRATION = "docs/preregistration.md"
PREREGISTERED_PLAN = "docs/analysis_plan_hubert.md"
PREREGISTERED_UNITS = "docs/replication_units.csv"
PREREGISTERED_SUMMARY = "docs/replication_units_summary.csv"


def _require(table: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise KeyError(f"{name}: missing columns {missing}")


def _units(table: pd.DataFrame, finding: str, estimate: str = "estimate",
           low: str = "spk_ci_low", high: str = "spk_ci_high", **fixed: object) -> pd.DataFrame:
    """Rows of ``table`` in the unit layout; ``fixed`` sets columns the table lacks."""
    out = pd.DataFrame(index=table.index)
    out["finding"] = finding
    for column in KEYS[1:]:
        if column in fixed:
            out[column] = fixed[column]
        elif column in table.columns:
            out[column] = table[column]
        else:
            out[column] = "" if column != "layer" else pd.NA
    value = table[estimate].to_numpy(dtype=float)
    lo = table[low].to_numpy(dtype=float)
    hi = table[high].to_numpy(dtype=float)
    if not (np.isfinite(value).all() and np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError(f"{finding}: non-finite estimate or interval")
    if (lo > hi).any():
        raise ValueError(f"{finding}: interval with low > high")
    out["estimate"], out["spk_ci_low"], out["spk_ci_high"] = value, lo, hi
    out["sign"] = np.select([value > 0, value < 0], ["positive", "negative"], "zero")
    out["in_rule"] = (lo > 0) | (hi < 0)
    out["layer"] = out["layer"].astype("Int64")
    if out.duplicated(KEYS).any():
        raise ValueError(f"{finding}: duplicated units")
    return out[UNIT_COLUMNS].reset_index(drop=True)


def effect_units(effects: pd.DataFrame, condition: str = MAIN_CONDITION,
                 layers: tuple[int, ...] = ALL_LAYERS,
                 test_sets: tuple[str, ...] = cfg.TEST_SETS,
                 poolings: tuple[str, ...] = cfg.POOLINGS,
                 names: tuple[str, ...] = cfg.EFFECTS) -> pd.DataFrame:
    """One unit per effect x test set x layer x pooling of the 2x2 design of one condition.

    ``effects`` has columns condition, test_set, pooling, layer, effect,
    estimate, spk_ci_low, spk_ci_high. ``names`` are the effects to list.
    Every requested unit must be present.
    """
    _require(effects, ["condition", "test_set", "pooling", "layer", "effect", "estimate",
                       "spk_ci_low", "spk_ci_high"], "effects")
    part = effects[(effects["condition"] == condition) & effects["effect"].isin(names)
                   & effects["test_set"].isin(test_sets) & effects["pooling"].isin(poolings)
                   & effects["layer"].isin(layers)]
    units = _units(part.rename(columns={"effect": "quantity"}), "effect")
    expected = len(names) * len(test_sets) * len(layers) * len(poolings)
    if len(units) != expected:
        raise ValueError(f"effects ({condition}): {len(units)} units, expected {expected}")
    order = {name: i for i, name in enumerate(cfg.EFFECTS)}
    units["_order"] = units["quantity"].map(order)
    units = units.sort_values(["_order", "test_set", "pooling", "layer"], kind="mergesort")
    return units.drop(columns="_order").reset_index(drop=True)


def rank_units(rank: pd.DataFrame, condition: str = MAIN_CONDITION,
               test_sets: tuple[str, ...] = cfg.TEST_SETS,
               poolings: tuple[str, ...] = cfg.POOLINGS) -> pd.DataFrame:
    """One unit per test set x pooling: Kendall's tau between the clean and the
    ``condition`` layer curves, with its percentile interval.

    ``rank`` has columns condition, test_set, pooling, kendall_tau,
    tau_ci_low, tau_ci_high.
    """
    _require(rank, ["condition", "test_set", "pooling", "kendall_tau", "tau_ci_low",
                    "tau_ci_high"], "rank agreement")
    part = rank[(rank["condition"] == condition) & rank["test_set"].isin(test_sets)
                & rank["pooling"].isin(poolings)]
    units = _units(part, "rank_agreement", "kendall_tau", "tau_ci_low", "tau_ci_high",
                   quantity="kendall_tau")
    if len(units) != len(test_sets) * len(poolings):
        raise ValueError(f"rank agreement ({condition}): {len(units)} units")
    return units.sort_values(["test_set", "pooling"], kind="mergesort").reset_index(drop=True)


def tau_difference_units(differences: pd.DataFrame, condition: str,
                         test_sets: tuple[str, ...] = cfg.TEST_SETS,
                         poolings: tuple[str, ...] = cfg.POOLINGS) -> pd.DataFrame:
    """One unit per test set x pooling: tau at 40 dB minus tau at ``condition``.

    ``differences`` has columns condition, test_set, pooling, estimate,
    spk_ci_low, spk_ci_high.
    """
    _require(differences, ["condition", "test_set", "pooling", "estimate", "spk_ci_low",
                           "spk_ci_high"], "tau difference")
    part = differences[(differences["condition"] == condition)
                       & differences["test_set"].isin(test_sets)
                       & differences["pooling"].isin(poolings)]
    units = _units(part, "rank_agreement", quantity="tau_difference")
    if len(units) != len(test_sets) * len(poolings):
        raise ValueError(f"tau difference ({condition}): {len(units)} units")
    return units.sort_values(["test_set", "pooling"], kind="mergesort").reset_index(drop=True)


def per_attack_units(per_attack: pd.DataFrame, condition: str = MAIN_CONDITION) -> pd.DataFrame:
    """One unit per attack x layer x pooling: trimmed minus clean on the evaluation set.

    ``per_attack`` has columns condition, attack, pooling, layer, estimate,
    spk_ci_low, spk_ci_high.
    """
    _require(per_attack, ["condition", "attack", "pooling", "layer", "estimate", "spk_ci_low",
                          "spk_ci_high"], "per-attack effect")
    part = per_attack[per_attack["condition"] == condition]
    units = _units(part, "per_attack", quantity="delta_total", test_set="eval")
    return units.sort_values(["attack", "layer", "pooling"], kind="mergesort").reset_index(drop=True)


def duration_units(by_duration: pd.DataFrame, condition: str = MAIN_CONDITION,
                   layers: tuple[int, ...] = THRESHOLD_LAYERS,
                   poolings: tuple[str, ...] = cfg.POOLINGS) -> pd.DataFrame:
    """One unit per effect x duration subset x layer x pooling on In-the-Wild.

    ``by_duration`` has columns condition, subset, pooling, layer, effect,
    estimate, spk_ci_low, spk_ci_high. The subsets are the two duration
    subsets; the rows of the whole test set (subset WHOLE_SET) repeat the
    In-the-Wild effect units and are not duration units. With the defaults:
    6 effects x 2 subsets x 3 layers x 2 poolings = 72 units.
    """
    _require(by_duration, ["condition", "subset", "pooling", "layer", "effect", "estimate",
                           "spk_ci_low", "spk_ci_high"], "duration split")
    part = by_duration[(by_duration["condition"] == condition)
                       & by_duration["effect"].isin(cfg.EFFECTS)
                       & (by_duration["subset"] != WHOLE_SET)
                       & by_duration["pooling"].isin(poolings)
                       & by_duration["layer"].isin(layers)]
    units = _units(part.rename(columns={"effect": "quantity"}), "duration", test_set="itw")
    subsets = sorted(set(units["subset"]))
    expected = len(cfg.EFFECTS) * N_DURATION_SUBSETS * len(layers) * len(poolings)
    if len(subsets) != N_DURATION_SUBSETS or len(units) != expected:
        raise ValueError(f"duration split ({condition}): {len(units)} units in the subsets "
                         f"{subsets}, expected {expected} in {N_DURATION_SUBSETS} subsets")
    return units.sort_values(["quantity", "subset", "pooling", "layer"],
                             kind="mergesort").reset_index(drop=True)


def add_matched_totals(effects: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """The effects table with the ``delta_total`` rows of ``matched``
    (``effects.total_effect_ci``) for the layers it lacks.

    Where both tables have a row, its estimate and speaker-level interval
    must be identical; a difference raises. Returns the columns that both
    tables have.
    """
    keys = ["condition", "test_set", "pooling", "layer", "effect"]
    values = ["estimate", "spk_ci_low", "spk_ci_high"]
    _require(effects, keys + values, "effects")
    _require(matched, keys + values, "matched totals")
    both = effects[keys + values].merge(matched[keys + values], on=keys, suffixes=("", "_matched"))
    for column in values:
        differs = both[column].to_numpy(dtype=float) != both[f"{column}_matched"].to_numpy(dtype=float)
        if differs.any():
            first = both[differs].iloc[0]
            raise ValueError(f"{int(differs.sum())} rows of the matched totals differ from the "
                             f"effects table in {column} (first: {dict(first[keys])})")
    index = pd.MultiIndex.from_frame(effects[keys])
    new = matched[~pd.MultiIndex.from_frame(matched[keys]).isin(index)]
    columns = [c for c in effects.columns if c in matched.columns]
    return pd.concat([effects[columns], new[columns]], ignore_index=True)


def replication_units(effects: pd.DataFrame, rank: pd.DataFrame, per_attack: pd.DataFrame,
                      by_duration: pd.DataFrame | None = None,
                      tau_differences: pd.DataFrame | None = None,
                      threshold_conditions: tuple[str, ...] = THRESHOLD_CONDITIONS,
                      threshold_layers: tuple[int, ...] = THRESHOLD_LAYERS,
                      layers: tuple[int, ...] = ALL_LAYERS) -> pd.DataFrame:
    """All units, in the UNIT_COLUMNS layout.

    40 dB: the effects of every layer of ``layers``, tau, the per-attack
    changes and, when given, the duration split (``threshold_layers``). Each
    stricter threshold: delta_total of every layer, the other effects of
    ``threshold_layers``, tau, the per-attack changes and, when given, the
    tau difference to 40 dB.
    """
    parts = [effect_units(effects, layers=layers), rank_units(rank), per_attack_units(per_attack)]
    if by_duration is not None:
        parts.append(duration_units(by_duration, layers=threshold_layers))
    cross_cell_effects = tuple(e for e in cfg.EFFECTS if e not in MATCHED_CELL_EFFECTS)
    for condition in threshold_conditions:
        parts.append(effect_units(effects, condition, layers=layers, names=MATCHED_CELL_EFFECTS))
        parts.append(effect_units(effects, condition, layers=threshold_layers,
                                  names=cross_cell_effects))
        parts.append(rank_units(rank, condition))
        parts.append(per_attack_units(per_attack, condition))
        if tau_differences is not None:
            parts.append(tau_difference_units(tau_differences, condition))
    return pd.concat(parts, ignore_index=True)


def summarise_units(units: pd.DataFrame) -> pd.DataFrame:
    """Count of units and of units in the rule per finding x quantity x condition x test set x subset."""
    rows = []
    for key, group in units.groupby(GROUP_KEYS, sort=False, dropna=False):
        in_rule = group[group["in_rule"]]
        rows.append({**dict(zip(GROUP_KEYS, key)), "n_units": len(group),
                     "n_in_rule": len(in_rule),
                     "n_positive_in_rule": int((in_rule["sign"] == "positive").sum()),
                     "n_negative_in_rule": int((in_rule["sign"] == "negative").sum())})
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def read_units(path: str) -> pd.DataFrame:
    """A units file (``replication_units.csv``) in the UNIT_COLUMNS layout:
    empty text keys are "", ``layer`` is a nullable integer, ``in_rule`` a boolean."""
    table = pd.read_csv(path, float_precision="round_trip", keep_default_na=False, na_values=[""],
                        dtype={key: str for key in KEYS if key != "layer"})
    _require(table, UNIT_COLUMNS, os.path.basename(path))
    return normalise_units(table[UNIT_COLUMNS])


def normalise_units(units: pd.DataFrame) -> pd.DataFrame:
    """The keys of a units table in one form, whether it was computed or read from a file."""
    out = units.copy()
    for key in KEYS:
        if key == "layer":
            out[key] = pd.to_numeric(out[key], errors="coerce").astype("Int64")
        else:
            out[key] = out[key].fillna("").astype(str)
    in_rule = out["in_rule"]
    if in_rule.dtype != bool:
        text = in_rule.astype(str)
        if not text.isin(["True", "False"]).all():
            raise ValueError("in_rule must be True or False in every unit")
        out["in_rule"] = text == "True"
    return out.reset_index(drop=True)


def compare_units(reference: pd.DataFrame, replica: pd.DataFrame) -> pd.DataFrame:
    """Unit-level comparison of two encoders on the units of ``reference``.

    Adds the replica's estimate, interval, sign and exclusion flag, the sign
    agreement, ``replicated`` and ``verdict``. A unit is replicated if the
    reference is in the rule, the replica interval excludes zero and the
    signs are equal, for every quantity (Kendall's tau included; ``sign``
    and ``replica_sign`` are the signs of the two estimates). ``verdict`` is
    "replicated" or "not replicated" for a unit in the rule and
    "descriptive" for every other unit. Every reference unit must exist in
    ``replica``.
    """
    ref = normalise_units(reference[UNIT_COLUMNS])
    rep = normalise_units(replica[UNIT_COLUMNS]).rename(columns={
        "estimate": "replica_estimate", "spk_ci_low": "replica_spk_ci_low",
        "spk_ci_high": "replica_spk_ci_high", "sign": "replica_sign",
        "in_rule": "replica_excludes_zero"})
    merged = ref.merge(rep, on=KEYS, how="left", validate="one_to_one")
    if merged["replica_estimate"].isna().any():
        raise KeyError("some reference units are missing from the replica")
    merged["replica_excludes_zero"] = merged["replica_excludes_zero"].astype(bool)
    merged["same_sign"] = merged["sign"] == merged["replica_sign"]
    merged["replicated"] = merged["in_rule"] & merged["replica_excludes_zero"] & merged["same_sign"]
    merged["verdict"] = np.select([merged["replicated"], merged["in_rule"]], VERDICTS[:2],
                                  VERDICTS[2])
    return merged


def _counts(group: pd.DataFrame) -> dict[str, int]:
    """The 2x2 classification of a group of compared units."""
    ref, rep, same = group["in_rule"], group["replica_excludes_zero"], group["same_sign"]
    return {"both_exclude_same_sign": int((ref & rep & same).sum()),
            "both_exclude_opposite_sign": int((ref & rep & ~same).sum()),
            "reference_only": int((ref & ~rep).sum()),
            "replica_only": int((~ref & rep).sum()),
            "neither": int((~ref & ~rep).sum())}


def outcome_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Counts per finding x quantity x condition x test set x subset (the rows
    of ``summarise_units``): the units, the units in the rule, the replicated
    units, and the 2x2 classification (reference interval excludes zero or
    not x replica interval excludes zero or not), with sign agreement where
    both exclude zero. ``replica_only`` is the reverse count."""
    rows = []
    for key, group in comparison.groupby(GROUP_KEYS, sort=False, dropna=False):
        rows.append({**dict(zip(GROUP_KEYS, key)), "n_units": len(group),
                     "n_in_rule": int(group["in_rule"].sum()),
                     "n_replicated": int(group["replicated"].sum()), **_counts(group)})
    return pd.DataFrame(rows, columns=OUTCOME_COLUMNS)


# The findings of the analysis plan -----------------------------------------------------------------

def finding_ids(units: pd.DataFrame) -> pd.Series:
    """The finding of FINDINGS of every unit. A unit that fits none raises."""
    found = pd.Series("", index=units.index, dtype=object)
    for name, _, family, quantities, conditions, test_sets, subsets in FINDINGS:
        fits = (units["finding"] == family) & (found == "")
        for column, allowed in (("quantity", quantities), ("condition", conditions),
                                ("test_set", test_sets), ("subset", subsets)):
            if allowed is not None:
                fits &= units[column].isin(allowed)
        found[fits] = name
    if (found == "").any():
        first = units[found == ""].iloc[0]
        raise ValueError(f"{int((found == '').sum())} units fit no finding of the plan "
                         f"(first: {dict(first[KEYS])})")
    return found


def findings_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """The outcome per finding of the plan, in the order of FINDINGS, and a
    last row for all units.

    Columns: finding_id, type, n_units, n_in_rule, n_replicated and the 2x2
    classification. An absence finding is reported with the 2x2
    classification only: its ``n_replicated`` is empty, because the plan
    counts it neither as replicated nor as failed. The last row counts the
    units, whatever their finding.
    """
    ids = finding_ids(comparison)
    rows = []
    for name, kind, *_ in FINDINGS:
        group = comparison[ids == name]
        rows.append({"finding_id": name, "type": kind, "n_units": len(group),
                     "n_in_rule": int(group["in_rule"].sum()),
                     "n_replicated": pd.NA if kind == ABSENCE else int(group["replicated"].sum()),
                     **_counts(group)})
    rows.append({"finding_id": ALL_UNITS, "type": "", "n_units": len(comparison),
                 "n_in_rule": int(comparison["in_rule"].sum()),
                 "n_replicated": int(comparison["replicated"].sum()), **_counts(comparison)})
    table = pd.DataFrame(rows, columns=FINDING_COLUMNS)
    table["n_replicated"] = table["n_replicated"].astype("Int64")
    return table


def absence_units(comparison: pd.DataFrame) -> pd.DataFrame:
    """The compared units of the absence findings, with a leading column ``finding_id``."""
    ids = finding_ids(comparison)
    absent = [name for name, kind, *_ in FINDINGS if kind == ABSENCE]
    out = comparison[ids.isin(absent)].copy()
    out.insert(0, "finding_id", ids[ids.isin(absent)])
    return out.reset_index(drop=True)


# The pre-registered files ---------------------------------------------------------------------------

def sha256_lf(path: str) -> str:
    """sha256 of a file with LF line endings, whatever the checkout wrote."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def preregistered_files(repo: str) -> dict[str, str]:
    """{file: sha256} of the table in PREREGISTRATION of the repository ``repo``."""
    with open(os.path.join(repo, PREREGISTRATION), encoding="utf-8") as fh:
        rows = re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$", fh.read(), flags=re.M)
    if not rows:
        raise ValueError(f"{PREREGISTRATION} names no file with a sha256")
    return dict(rows)


def preregistration_check(repo: str, copies: Mapping[str, str] | None = None) -> pd.DataFrame:
    """The pre-registered files of ``repo`` against the sha256 recorded in PREREGISTRATION.

    ``copies`` maps the path of another copy of a pre-registered file to the
    name of that file; it must have the same sha256. Columns: file,
    registered_as, recorded_sha256, sha256 ("missing" without the file), ok.
    """
    recorded = preregistered_files(repo)
    files = [(os.path.join(repo, name), name, name) for name in recorded]
    for path, name in (copies or {}).items():
        if name not in recorded:
            raise KeyError(f"{name} is not a pre-registered file")
        files.append((path, path, name))
    rows = []
    for path, shown, name in files:
        digest = sha256_lf(path) if os.path.exists(path) else "missing"
        rows.append({"file": shown, "registered_as": name, "recorded_sha256": recorded[name],
                     "sha256": digest, "ok": digest == recorded[name]})
    return pd.DataFrame(rows)
