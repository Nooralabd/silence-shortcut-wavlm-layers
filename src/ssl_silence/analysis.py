"""The analysis tables of one encoder from its score files, and their
comparison with the original tables.

``ScoreTables`` computes the tables that follow from the scores: metrics per
run, seed summary, the four train/test cells, the effects, the rank
agreement, the selected configurations, the per-attack effect, In-the-Wild
by duration, the threshold summary and the padded comparison. The functions
that compute them are those of ``selection``, ``effects`` and ``ranking``;
this module only calls them in order and writes the tables.

Every bootstrap table is computed per trimmed condition as a *part*
(``<folder>/parts/<table>_<condition>.csv``) and the parts are stacked into
the table ``<folder>/<table>.csv``. A part is reused after a restart when it
was written at the same commit of this repository from the same inputs (the
sha256 of ``runs.csv``, the seeds, the layers and the number of resamples)
and its file still has the sha256 of its meta file; otherwise it is computed
again. A part is never reused from a working tree with uncommitted changes.
A part written at another commit is reused, under the same conditions, only
if that commit is named explicitly (``reuse_commits``).

Every table that a method returns was read back from its file, so that what
is compared is what was written.

For a second encoder the same tables are computed, and with the tables of
the reference encoder: the paired differences between the encoders and the
outcome of the replication rule (``replication``). PLAN_TABLES names every
table of the analysis plan; a table that the plan does not name is written
to the folder EXPLORATORY with EXPLORATORY_PREFIX in its file name.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as cfg
from . import effects as E
from . import provenance
from . import ranking as R
from . import replication
from . import selection as S
from . import shards
from . import training
from . import verify
from .bootstrap import Resamples, ScoreStore

PARTS = "parts"
EXPLORATORY = "exploratory"
EXPLORATORY_PREFIX = "exploratory_"
# Tables of a second encoder -> where docs/analysis_plan_hubert.md names them.
PLAN_TABLES: dict[str, str] = {
    "metrics_per_run": "Section 5: the official EER per seed; tandem t-DCF (item 4)",
    "seed_summary": "Section 5: the EER averaged over the five seeds",
    "cells_2x2": "Section 5, item 1: the four cells, at each threshold",
    "fallback_counts": "Section 3: the fallback counts per split and class",
    "dev_selected": "Section 4 and Section 5, item 4: the dev-selected layers, EER and t-DCF",
    "selected_configs_eer_ci": "Section 5, item 4: the dev-selected layers with intervals",
    "itw_by_duration": "Section 5, item 5: In-the-Wild by duration at 40 dB, layers 3-5",
    "effects_2x2": "Section 5, item 2: the six effects with intervals, at each threshold",
    "rank_agreement": "Section 5, item 3: tau and rho, percentile and BCa intervals, exact p",
    "rank_agreement_layers": "Section 5, item 8: the best layer with its bootstrap proportion",
    "threshold_summary": "Section 9: the thresholds side by side, from the tables above",
    "per_attack_effect": "Section 5, item 6: trimmed minus clean per attack, layers 3-5",
    "effects_matched_cells": "Section 6, Units: the total effect at 20 and 15 dB",
    "rank_agreement_threshold_difference": "Section 5, intervals added: tau(40 dB) - tau(threshold)",
    "itw_by_duration_ci": "Section 5, intervals added: the effects per duration subset",
    "itw_by_duration_difference": "Section 9: longer than 4 s minus at most 4 s",
    "replication_units": "Section 6, Units: the units of this encoder",
    "replication_units_summary": "Section 6, Units: their count per row",
    "encoder_difference_cells_effects": "Section 8: reference minus this encoder, cells and effects",
    "encoder_difference_rank_agreement": "Section 8: reference minus this encoder, Kendall's tau",
    "replication_comparison": "Section 6, Reporting: the unit-level table with the verdict",
    "replication_outcome": "Section 6, Reporting: the 2x2 outcome per row of the summary",
    "replication_findings": "Section 6a: the outcome per finding",
    "replication_absence_findings": "Section 6a: the units of the absence findings",
}
# Written only if a dev-selected layer lies outside the layers of the duration
# and per-attack analyses (Section 4 of the plan).
SELECTED_LAYERS_SUFFIX = "_dev_selected_layers"
SELECTED_LAYER_TABLES: dict[str, str] = {
    f"{name}{SELECTED_LAYERS_SUFFIX}": "Section 4: the same analysis for a dev-selected layer outside 3-5"
    for name in ("per_attack_effect", "itw_by_duration", "itw_by_duration_ci",
                 "itw_by_duration_difference")}
# Not named by the plan: file name (in the folder EXPLORATORY) -> what it is.
EXPLORATORY_TABLES: dict[str, str] = {
    "exploratory_itw_by_duration_thresholds":
        "In-the-Wild by duration at 20 and 15 dB (the plan names 40 dB only): cells and effects, "
        "mean and SD over seeds, no intervals",
}
# Table of this package -> conditions of the original tables it is compared
# with (None: one original table for every condition).
ORIGINAL_TABLES: dict[str, tuple[str | None, ...]] = {
    "cells_2x2": ("trim40",),
    "padded_vs_unpadded": (None,),
    "fallback_counts": ("trim40", "trim20", "trim15"),
    "selected_configs_eer_ci": (None,),
    "itw_by_duration": ("trim40",),
    "effects_2x2": ("trim40", "trim20", "trim15"),
    "rank_agreement": ("trim40", "trim20", "trim15"),
    "rank_agreement_layers": ("trim40",),
    "threshold_summary": ("trim40", "trim20", "trim15"),
    "per_attack_effect": ("trim40", "trim20", "trim15"),
}
# The original duration table has the four cells and three of the six effects.
ORIGINAL_DURATION_QUANTITIES = ("eer_cc", "eer_ct", "eer_tc", "eer_tt",
                                "delta_test", "delta_train", "delta_total")


# Layers and conditions of the experiment configuration ----------------------------------------

def trained_conditions(encoder: str, extraction: str = "unpadded", base: str = E.BASE
                       ) -> tuple[str, ...]:
    """The training conditions of the configuration other than ``base``, in the order of the
    conditions of the configuration."""
    trained = {p.spec.condition for p in training.plan_runs(encoder, extraction, cfg.SEEDS[:1])}
    return tuple(c for c in cfg.conditions() if c in trained and c != base)


def cross_layers_of(encoder: str, condition: str, extraction: str = "unpadded",
                    base: str = E.BASE) -> tuple[int, ...]:
    """Layers with all four cells of ``condition``: for every pooling, the
    ``base`` model is scored in ``condition`` and the ``condition`` model in ``base``."""
    scored: dict[tuple[str, int, str], tuple[str, ...]] = {
        (p.spec.condition, p.spec.layer, p.spec.pooling): p.test_conditions
        for p in training.plan_runs(encoder, extraction, cfg.SEEDS[:1])}
    layers = sorted({layer for _, layer, _ in scored})
    return tuple(layer for layer in layers if all(
        condition in scored.get((base, layer, pooling), ())
        and base in scored.get((condition, layer, pooling), ()) for pooling in cfg.POOLINGS))


# Files -----------------------------------------------------------------------------------------

def read_table(path: str) -> pd.DataFrame:
    """Read a table written by ``provenance.write_table``: every number exactly
    as written, only an empty field is missing, and ``frames`` is "" where empty."""
    table = pd.read_csv(path, float_precision="round_trip", keep_default_na=False, na_values=[""])
    if "frames" in table.columns:
        table["frames"] = table["frames"].fillna("").astype(str)
    return table


def _meta(path: str) -> dict[str, Any] | None:
    meta_path = os.path.splitext(path)[0] + ".meta.json"
    if not (os.path.exists(path) and os.path.exists(meta_path)):
        return None
    with open(meta_path, encoding="utf-8") as fh:
        return json.load(fh)


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.0f} s"


# Tables ----------------------------------------------------------------------------------------

class ScoreTables:
    """The analysis tables of one encoder, written under ``out_dir``.

    ``speaker_maps`` maps a test set to {audio id: speaker}; ``attack_map``
    maps the evaluation files to their system id; ``duration_map`` maps the
    In-the-Wild files to their original duration in seconds; ``asv`` are the
    ASV error rates of ``selection.load_asv``; ``padded_runs_dir`` holds the
    runs of the padded comparison (None: no padded comparison).

    ``conditions`` are the trimmed conditions, ``layers`` the layers of the
    rank agreement, ``cross_layers`` the layers with all four cells per
    condition and ``detail_layers`` the layers of the per-attack and the
    duration tables; the defaults are those of the experiment configuration.
    ``reuse_commits``: full hashes of earlier commits whose parts are reused
    too. ``base_runs_dir``: a second runs folder from which the cell trained
    and tested on untrimmed audio is read (``bootstrap.ScoreStore``).
    ``steps`` lists what was computed and what was reused, with the
    commit that computed it.
    """

    def __init__(self, encoder: str, runs_dir: str, out_dir: str, *,
                 speaker_maps: Mapping[str, Mapping[str, str]],
                 attack_map: Mapping[str, str] | None = None,
                 duration_map: Mapping[str, float] | None = None,
                 asv: Mapping[str, Any] | None = None, padded_runs_dir: str | None = None,
                 conditions: Sequence[str] | None = None, layers: Sequence[int] | None = None,
                 cross_layers: Mapping[str, Sequence[int]] | None = None,
                 detail_layers: Sequence[int] = E.ATTACK_LAYERS,
                 seeds: Sequence[int] = cfg.SEEDS, n_boot: int = cfg.N_BOOT,
                 reuse_commits: Sequence[str] = (), base_runs_dir: str | None = None,
                 log: Callable[[str], None] = print):
        self.encoder, self.runs_dir, self.out_dir = encoder, runs_dir, out_dir
        self.reuse_commits = tuple(reuse_commits)
        if any(len(c) != 40 or set(c) - set("0123456789abcdef") for c in self.reuse_commits):
            raise ValueError("reuse_commits takes full commit hashes (40 hexadecimal characters)")
        self._rank_statistics: dict = {}
        self.padded_runs_dir = padded_runs_dir
        self.attack_map, self.duration_map, self.asv = attack_map, duration_map, asv
        self.conditions = tuple(conditions if conditions is not None
                                else trained_conditions(encoder))
        self.layers = tuple(int(x) for x in (
            layers if layers is not None else range(1, cfg.encoder(encoder)["n_layers"] + 1)))
        found = cross_layers if cross_layers is not None else {
            c: cross_layers_of(encoder, c) for c in self.conditions}
        self.cross_layers = {c: tuple(int(x) for x in found[c]) for c in self.conditions}
        self.detail_layers = tuple(int(x) for x in detail_layers)
        self.seeds, self.n_boot, self.log = tuple(int(s) for s in seeds), int(n_boot), log
        self.resamples = Resamples(ScoreStore(runs_dir, encoder, "unpadded", self.seeds,
                                              base_runs_dir=base_runs_dir),
                                   speaker_maps, n_boot=self.n_boot)
        runs = {"unpadded": runs_dir, **({"padded": padded_runs_dir} if padded_runs_dir else {}),
                **({"base": base_runs_dir} if base_runs_dir else {})}
        self.inputs = json.loads(json.dumps({
            "encoder": encoder, "seeds": self.seeds, "n_boot": self.n_boot, "layers": self.layers,
            "cross_layers": self.cross_layers, "detail_layers": self.detail_layers,
            "runs_csv_sha256": {kind: provenance.sha256_file(os.path.join(folder, "runs.csv"))
                                for kind, folder in runs.items()}}))
        self.steps: list[dict[str, object]] = []

    # Parts and stacked tables

    def path(self, name: str, part: bool = False) -> str:
        return os.path.join(self.out_dir, *((PARTS,) if part else ()), f"{name}.csv")

    def part(self, name: str, compute: Callable[[], pd.DataFrame],
             inputs: Mapping[str, Any] | None = None) -> pd.DataFrame:
        """The part ``name``: reused if it was written at this commit from the
        same inputs and its file is unchanged, otherwise computed and written.
        ``inputs``: the inputs of this part where they are not ``self.inputs``."""
        inputs = self.inputs if inputs is None else json.loads(json.dumps(inputs))
        path, commit = self.path(name, part=True), provenance.git_commit()
        meta = _meta(path)
        accepted = set(self.reuse_commits)
        if commit != "unknown" and not commit.endswith("-dirty"):
            accepted.add(commit)
        reusable = (meta is not None and meta.get("git_commit") in accepted
                    and meta.get("inputs") == inputs
                    and meta.get("sha256") == provenance.sha256_file(path))
        if reusable:
            status, seconds, computed_at = "reused", float(meta.get("seconds", float("nan"))), \
                meta["git_commit"]
            where = "this commit" if computed_at == commit else f"commit {computed_at[:7]}"
            self.log(f"{name}: reused (computed in {_minutes(seconds)} at {where})")
        else:
            start = time.time()
            table = compute()
            status, seconds, computed_at = "computed", time.time() - start, commit
            provenance.write_table(table, path, inputs=inputs, seconds=seconds)
            self.log(f"{name}: computed in {_minutes(seconds)}")
        table = read_table(path)
        self.steps.append({"step": name, "status": status, "seconds": seconds, "rows": len(table),
                           "sha256": provenance.sha256_file(path), "git_commit": computed_at})
        return table

    def write(self, name: str, table: pd.DataFrame,
              inputs: Mapping[str, Any] | None = None) -> pd.DataFrame:
        """Write the table ``name`` and return it as read back from its file."""
        provenance.write_table(table, self.path(name),
                               inputs=self.inputs if inputs is None else inputs)
        return read_table(self.path(name))

    def exploratory_path(self, name: str) -> str:
        return os.path.join(self.out_dir, EXPLORATORY, f"{name}.csv")

    def write_exploratory(self, name: str, table: pd.DataFrame) -> pd.DataFrame:
        """Write a table that the analysis plan does not name: to the folder
        EXPLORATORY, under a name of EXPLORATORY_TABLES."""
        if name not in EXPLORATORY_TABLES or not name.startswith(EXPLORATORY_PREFIX):
            raise ValueError(f"{name} is not a table of EXPLORATORY_TABLES")
        path = self.exploratory_path(name)
        provenance.write_table(table, path, inputs=self.inputs, exploratory=True)
        return read_table(path)

    def stacked(self, name: str, compute: Callable[[str], pd.DataFrame],
                conditions: Sequence[str] | None = None,
                inputs: Mapping[str, Any] | None = None) -> pd.DataFrame:
        """One part per trimmed condition (or per condition of ``conditions``),
        stacked into the table ``name``."""
        conditions = self.conditions if conditions is None else conditions
        parts = [self.part(f"{name}_{c}", lambda c=c: compute(c), inputs) for c in conditions]
        return self.write(name, pd.concat(parts, ignore_index=True), inputs)

    def steps_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps, columns=["step", "status", "seconds", "rows", "sha256",
                                                 "git_commit"])

    # Tables without a bootstrap

    def metrics(self) -> pd.DataFrame:
        """``metrics_per_run`` of the unpadded and the padded runs (a part), and
        ``seed_summary``. Returns the seed summary."""
        def compute() -> pd.DataFrame:
            folders = [self.runs_dir] + ([self.padded_runs_dir] if self.padded_runs_dir else [])
            return pd.concat([S.metrics_per_run(folder, self.asv) for folder in folders],
                             ignore_index=True)
        per_run = self.write("metrics_per_run", self.part("metrics_per_run", compute))
        return self.write("seed_summary", S.seed_summary(per_run))

    def summary(self, extraction: str = "unpadded") -> pd.DataFrame:
        """The seed summary of one extraction of this encoder, from its file."""
        table = read_table(self.path("seed_summary"))
        return table[(table["encoder"] == self.encoder)
                     & (table["extraction"] == extraction)].reset_index(drop=True)

    def cells(self) -> pd.DataFrame:
        """``cells_2x2``: the four cells per trimmed condition, pooling and layer."""
        summary = self.summary()
        return self.write("cells_2x2", pd.concat(
            [E.cells_table(summary, c) for c in self.conditions], ignore_index=True))

    def padded_vs_unpadded(self) -> pd.DataFrame:
        """``padded_vs_unpadded``: the configurations of the padded comparison."""
        if not self.padded_runs_dir:
            raise ValueError("the padded comparison needs the folder of the padded runs")
        table = read_table(self.path("seed_summary"))
        return self.write("padded_vs_unpadded", S.padded_vs_unpadded(table, self.encoder))

    def fallback_counts(self, feature_dirs: Mapping[str, str]) -> pd.DataFrame:
        """``fallback_counts``: ``shards.fallback_counts`` of the feature folder
        of every trimmed condition, with a leading column ``condition``."""
        def compute(condition: str) -> pd.DataFrame:
            table = shards.fallback_counts(feature_dirs[condition])
            table.insert(0, "condition", condition)
            return table
        return self.stacked("fallback_counts", compute)

    def _duration_table(self, conditions: Sequence[str],
                        layers: Sequence[int] | None = None) -> pd.DataFrame:
        if self.duration_map is None:
            raise ValueError("the duration table needs the durations of the In-the-Wild files")
        return pd.concat([
            E.itw_by_duration(self.resamples, c, self.duration_map,
                              layers=self._detail_cross_layers(c, layers))
            for c in conditions], ignore_index=True)

    def itw_by_duration(self, conditions: Sequence[str] | None = None,
                        layers: Sequence[int] | None = None, suffix: str = "") -> pd.DataFrame:
        """``itw_by_duration``: cells and effects on In-the-Wild by original
        duration, for every trimmed condition (or those of ``conditions``).
        ``layers`` and ``suffix``: the same table for other layers, under the
        name ``itw_by_duration<suffix>``."""
        return self.write(f"itw_by_duration{suffix}", self._duration_table(
            self.conditions if conditions is None else conditions, layers),
            self._inputs_for(layers))

    def itw_by_duration_thresholds(self) -> pd.DataFrame:
        """``exploratory_itw_by_duration_thresholds``: the table of
        ``itw_by_duration`` at the stricter thresholds. The analysis plan
        names the duration split at the main condition only, so this table
        is exploratory."""
        return self.write_exploratory("exploratory_itw_by_duration_thresholds",
                                      self._duration_table(self.threshold_conditions))

    def _detail_cross_layers(self, condition: str, layers: Sequence[int] | None = None
                             ) -> list[int]:
        chosen = self.detail_layers if layers is None else layers
        return [int(x) for x in chosen if x in self.cross_layers[condition]]

    def _inputs_for(self, layers: Sequence[int] | None) -> dict[str, Any] | None:
        """The inputs of a table of other layers than ``detail_layers`` (None: ``self.inputs``)."""
        if layers is None:
            return None
        return {**self.inputs, "detail_layers": [int(x) for x in layers]}

    # The dev-selected layers

    def dev_selected(self) -> pd.DataFrame:
        """``dev_selected``: the layer selected on the development set per
        condition and pooling (``selection.select_on_dev``), with the seed
        means and standard deviations of that configuration."""
        return self.write("dev_selected", S.select_on_dev(self.summary()))

    def selected_layers_outside_detail(self) -> tuple[int, ...]:
        """The dev-selected layers (any condition, any pooling) that are not in ``detail_layers``."""
        selected = {int(x) for x in S.select_on_dev(self.summary())["layer"]}
        return tuple(sorted(selected - set(self.detail_layers)))

    def selected_layer_tables(self) -> dict[str, pd.DataFrame]:
        """The per-attack and the duration tables for the dev-selected layers
        outside ``detail_layers``, under the names of SELECTED_LAYER_TABLES;
        nothing if there is no such layer."""
        layers = self.selected_layers_outside_detail()
        if not layers:
            return {}
        suffix, main = SELECTED_LAYERS_SUFFIX, (replication.MAIN_CONDITION,)
        by_subset, difference = self.itw_by_duration_ci(layers, suffix)
        return {f"per_attack_effect{suffix}": self.per_attack_effect(layers, suffix),
                f"itw_by_duration{suffix}": self.itw_by_duration(main, layers, suffix),
                f"itw_by_duration_ci{suffix}": by_subset,
                f"itw_by_duration_difference{suffix}": difference}

    # Bootstrap tables

    def selected_configs_eer_ci(self, all_conditions: bool = False) -> pd.DataFrame:
        """``selected_configs_eer_ci``: the reference and the dev-selected
        configurations of the untrimmed and the main condition, or of every
        condition (``all_conditions``)."""
        configs = (S.selected_configs(self.summary(), self.encoder, (E.BASE, *self.conditions))
                   if all_conditions else S.selected_configs(self.summary(), self.encoder))
        return self.write("selected_configs_eer_ci", self.part(
            "selected_configs_eer_ci", lambda: E.selected_configs_eer_ci(self.resamples, configs)))

    def effects(self) -> pd.DataFrame:
        """``effects_2x2``: the six effects for the cross layers of every condition."""
        return self.stacked("effects_2x2", lambda c: E.effects_ci(self.resamples, c,
                                                                 self.cross_layers[c]))

    def rank_agreement(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """``rank_agreement`` and ``rank_agreement_layers`` of every condition."""
        summaries, details = [], []
        for c in self.conditions:
            computed: dict[str, pd.DataFrame] = {}

            def compute(which: str, c: str = c, computed: dict = computed) -> pd.DataFrame:
                if not computed:
                    computed["rank_agreement"], computed["rank_agreement_layers"] = \
                        R.rank_agreement_ci(self.resamples, c, layers=self.layers,
                                            statistics=self._rank_statistics)
                return computed[which]
            # Both parts come from one computation; if only one of them can be
            # reused, the other is computed again (and the first with it).
            summaries.append(self.part(f"rank_agreement_{c}", lambda: compute("rank_agreement")))
            details.append(self.part(f"rank_agreement_layers_{c}",
                                     lambda: compute("rank_agreement_layers")))
        return (self.write("rank_agreement", pd.concat(summaries, ignore_index=True)),
                self.write("rank_agreement_layers", pd.concat(details, ignore_index=True)))

    def per_attack_effect(self, layers: Sequence[int] | None = None, suffix: str = ""
                          ) -> pd.DataFrame:
        """``per_attack_effect``: trimmed minus clean per attack, matched cells.
        ``layers`` and ``suffix``: the same table for other layers, under the
        name ``per_attack_effect<suffix>``."""
        if self.attack_map is None:
            raise ValueError("the per-attack table needs the attack of every evaluation file")
        chosen = self.detail_layers if layers is None else tuple(int(x) for x in layers)
        return self.stacked(f"per_attack_effect{suffix}", lambda c: E.per_attack_effect_ci(
            self.resamples, c, self.attack_map, layers=chosen), inputs=self._inputs_for(layers))

    def threshold_summary(self) -> pd.DataFrame:
        """``threshold_summary`` from the written tables of the rank agreement,
        the effects and the fallback counts."""
        tables = {name: read_table(self.path(name)) for name in (
            "rank_agreement", "rank_agreement_layers", "effects_2x2", "fallback_counts")}
        counts = tables["fallback_counts"]
        fallback = {c: counts[counts["condition"] == c] for c in self.conditions}
        return self.write("threshold_summary", R.threshold_summary(
            tables["rank_agreement"], tables["rank_agreement_layers"], tables["effects_2x2"],
            fallback, S.select_on_dev(self.summary()), conditions=self.conditions,
            layers=self.detail_layers))

    # Units of the replication rule

    @property
    def threshold_conditions(self) -> tuple[str, ...]:
        """The trimmed conditions other than the main one (``replication.MAIN_CONDITION``)."""
        return tuple(c for c in self.conditions if c != replication.MAIN_CONDITION)

    def matched_total_effects(self) -> pd.DataFrame:
        """``effects_matched_cells``: delta_total of every layer at the stricter
        thresholds, from the matched cells (the cross-condition cells of some
        layers were not scored there)."""
        return self.stacked("effects_matched_cells", lambda c: E.total_effect_ci(
            self.resamples, c, self.layers), self.threshold_conditions)

    def tau_differences(self) -> pd.DataFrame:
        """``rank_agreement_threshold_difference``: tau at the main condition
        minus tau at each stricter threshold, in the same resamples."""
        return self.stacked("rank_agreement_threshold_difference", lambda c: R.tau_difference_ci(
            self.resamples, c, reference=replication.MAIN_CONDITION, layers=self.layers,
            statistics=self._rank_statistics), self.threshold_conditions)

    def rank_noise_ceiling(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """``rank_noise_ceiling`` and ``rank_noise_ceiling_splits`` at the main
        condition: Kendall's tau between the layer rankings of two
        speaker-disjoint halves (``ranking.noise_ceiling``). Not a table of
        the analysis plan: computed for the reference encoder only."""
        c = replication.MAIN_CONDITION
        computed: dict[str, pd.DataFrame] = {}

        def compute(which: str) -> pd.DataFrame:
            if not computed:
                computed["rank_noise_ceiling"], computed["rank_noise_ceiling_splits"] = \
                    R.noise_ceiling(self.resamples, c, layers=self.layers)
            return computed[which]
        summary = self.part(f"rank_noise_ceiling_{c}", lambda: compute("rank_noise_ceiling"))
        splits = self.part(f"rank_noise_ceiling_splits_{c}",
                           lambda: compute("rank_noise_ceiling_splits"))
        return (self.write("rank_noise_ceiling", summary),
                self.write("rank_noise_ceiling_splits", splits))

    def itw_by_duration_ci(self, layers: Sequence[int] | None = None, suffix: str = ""
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """``itw_by_duration_ci`` and ``itw_by_duration_difference`` of the main
        condition: the effects per duration subset with paired intervals, and
        their difference between the two subsets. ``layers`` and ``suffix``:
        the same tables for other layers, under names that end in ``suffix``."""
        if self.duration_map is None:
            raise ValueError("the duration table needs the durations of the In-the-Wild files")
        condition = replication.MAIN_CONDITION
        computed: dict[str, pd.DataFrame] = {}
        inputs = self._inputs_for(layers)

        def compute(which: str) -> pd.DataFrame:
            if not computed:
                computed["itw_by_duration_ci"], computed["itw_by_duration_difference"] = \
                    E.itw_by_duration_ci(self.resamples, condition, self.duration_map,
                                         layers=self._detail_cross_layers(condition, layers))
            return computed[which]
        names = ("itw_by_duration_ci", "itw_by_duration_difference")
        parts = [self.part(f"{name}{suffix}_{condition}", lambda name=name: compute(name), inputs)
                 for name in names]
        return (self.write(f"{names[0]}{suffix}", parts[0], inputs),
                self.write(f"{names[1]}{suffix}", parts[1], inputs))

    def replication_units(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """``replication_units`` and ``replication_units_summary`` from the
        written tables: the effects (with the matched totals of the layers the
        effects table lacks at the stricter thresholds), the rank agreement,
        its threshold differences, the per-attack effect and the duration
        split with intervals. Returns (units, summary) as computed."""
        tables = {name: read_table(self.path(name)) for name in (
            "effects_2x2", "effects_matched_cells", "rank_agreement",
            "rank_agreement_threshold_difference", "per_attack_effect", "itw_by_duration_ci")}
        effects = replication.add_matched_totals(tables["effects_2x2"],
                                                 tables["effects_matched_cells"])
        units = replication.replication_units(
            effects, tables["rank_agreement"], tables["per_attack_effect"],
            tables["itw_by_duration_ci"], tables["rank_agreement_threshold_difference"],
            threshold_conditions=self.threshold_conditions, threshold_layers=self.detail_layers,
            layers=self.layers)
        summary = replication.summarise_units(units)
        self.write("replication_units", units)
        self.write("replication_units_summary", summary)
        return units, summary


    # Two encoders: paired differences and the replication rule

    def _paired_inputs(self, reference: "ScoreTables") -> dict[str, Any]:
        return {**self.inputs, "reference": reference.inputs}

    def encoder_differences(self, reference: "ScoreTables") -> pd.DataFrame:
        """``encoder_difference_cells_effects``: the reference encoder minus this
        encoder, paired, per trimmed condition that both have. For the layers
        with all four cells in both encoders: the four cells and the six
        effects; for the other layers: the two matched cells and the total
        effect."""
        def compute(condition: str) -> pd.DataFrame:
            shared = [x for x in self.cross_layers[condition]
                      if x in reference.cross_layers.get(condition, ())]
            others = [x for x in self.layers if x not in shared and x in reference.layers]
            parts = []
            if shared:
                parts.append(E.encoder_differences(reference.resamples, self.resamples, condition,
                                                   shared))
            if others:
                parts.append(E.encoder_differences(reference.resamples, self.resamples, condition,
                                                   others, cells=E.MATCHED_CELLS))
            return pd.concat(parts, ignore_index=True)
        conditions = [c for c in self.conditions if c in reference.conditions]
        return self.stacked("encoder_difference_cells_effects", compute, conditions,
                            self._paired_inputs(reference))

    def encoder_tau_differences(self, reference: "ScoreTables") -> pd.DataFrame:
        """``encoder_difference_rank_agreement``: Kendall's tau (and Spearman's
        rho) of the reference encoder minus that of this encoder, paired, per
        trimmed condition that both have."""
        conditions = [c for c in self.conditions if c in reference.conditions]
        return self.stacked("encoder_difference_rank_agreement", lambda c: R.encoder_tau_differences(
            reference.resamples, self.resamples, c, layers=self.layers,
            statistics=self._rank_statistics), conditions, self._paired_inputs(reference))

    def replication_outcome(self, reference_units_path: str) -> dict[str, pd.DataFrame]:
        """The replication rule on the units of the reference encoder
        (``reference_units_path``, a file of ``replication.replication_units``)
        and the written units of this encoder: ``replication_comparison``
        (every unit with both estimates, both intervals and the verdict),
        ``replication_outcome`` (the 2x2 outcome per row of the summary),
        ``replication_findings`` (per finding of the plan) and
        ``replication_absence_findings`` (the units of the absence findings).
        The two encoders must have the same units."""
        reference = replication.read_units(reference_units_path)
        units = replication.read_units(self.path("replication_units"))
        keys = [set(map(tuple, t[replication.KEYS].astype(str).to_numpy())) for t in (reference, units)]
        if keys[0] != keys[1]:
            raise ValueError(f"the units of the two encoders differ: {len(keys[0] - keys[1])} only "
                             f"in the reference, {len(keys[1] - keys[0])} only in {self.encoder}")
        comparison = replication.compare_units(reference, units)
        inputs = {**self.inputs,
                  "reference_units_sha256": replication.sha256_lf(reference_units_path)}
        tables = {"replication_comparison": comparison,
                  "replication_outcome": replication.outcome_table(comparison),
                  "replication_findings": replication.findings_table(comparison),
                  "replication_absence_findings": replication.absence_units(comparison)}
        return {name: self.write(name, table, inputs) for name, table in tables.items()}

    def written_tables_check(self) -> pd.DataFrame:
        """Every table of the analysis plan is written, the tables for
        dev-selected layers exist exactly when there is such a layer, every
        exploratory table is in the folder EXPLORATORY, and no other table is
        in either folder. Columns: table, kind, plan, written, ok."""
        extra_layers = bool(self.selected_layers_outside_detail())
        rows = []
        for name, where in PLAN_TABLES.items():
            written = os.path.exists(self.path(name))
            rows.append({"table": name, "kind": "analysis plan", "plan": where,
                         "written": written, "ok": written})
        for name, where in SELECTED_LAYER_TABLES.items():
            written = os.path.exists(self.path(name))
            rows.append({"table": name, "kind": "analysis plan, dev-selected layers",
                         "plan": where, "written": written, "ok": written == extra_layers})
        for name, what in EXPLORATORY_TABLES.items():
            written = os.path.exists(self.exploratory_path(name))
            rows.append({"table": f"{EXPLORATORY}/{name}", "kind": "exploratory", "plan": what,
                         "written": written, "ok": written})
        listed = {"": set(PLAN_TABLES) | set(SELECTED_LAYER_TABLES), EXPLORATORY: set(EXPLORATORY_TABLES)}
        for folder, names in listed.items():
            path = os.path.join(self.out_dir, folder)
            found = sorted(f[:-4] for f in (os.listdir(path) if os.path.isdir(path) else [])
                           if f.endswith(".csv"))
            for name in found:
                if name not in names:
                    rows.append({"table": f"{folder}/{name}" if folder else name,
                                 "kind": "not listed", "plan": "", "written": True, "ok": False})
        return pd.DataFrame(rows, columns=["table", "kind", "plan", "written", "ok"])


def planned_tables() -> pd.DataFrame:
    """The tables of a second encoder before anything is computed: those of
    the analysis plan (PLAN_TABLES), those written only for a dev-selected
    layer outside the detail layers (SELECTED_LAYER_TABLES), and the
    exploratory ones (EXPLORATORY_TABLES). Columns: table, kind, plan."""
    rows = [{"table": name, "kind": "analysis plan", "plan": where}
            for name, where in PLAN_TABLES.items()]
    rows += [{"table": name, "kind": "analysis plan, dev-selected layers", "plan": where}
             for name, where in SELECTED_LAYER_TABLES.items()]
    rows += [{"table": f"{EXPLORATORY}/{name}", "kind": "exploratory", "plan": what}
             for name, what in EXPLORATORY_TABLES.items()]
    return pd.DataFrame(rows, columns=["table", "kind", "plan"])


def paired_resamples_check(reference: ScoreTables, replica: ScoreTables) -> pd.DataFrame:
    """The resamples of two encoders are the same resamples of the same files.

    One row per test set and level (file, spk): the number of files and of
    resamples, the seed, whether the files and their order, the labels and
    the speakers are equal, and the sha256 of the multiplicities of each
    encoder. ``ok``: everything is equal.
    """
    import hashlib

    a, b = reference.resamples, replica.resamples
    rows = []
    for test_set in cfg.TEST_SETS:
        ea, eb = a.get(test_set), b.get(test_set)
        same_files = bool(np.array_equal(ea["ids"], eb["ids"]))
        same_labels = bool(np.array_equal(ea["labels"], eb["labels"]))
        same_speakers = bool(ea["groups"] is not None and eb["groups"] is not None
                             and np.array_equal(ea["groups"], eb["groups"]))
        for level in ("file", "spk"):
            digests = [hashlib.sha256(np.ascontiguousarray(e[level]).tobytes()).hexdigest()
                       if e[level] is not None else "none" for e in (ea, eb)]
            ok = (same_files and same_labels and same_speakers and digests[0] == digests[1]
                  and digests[0] != "none" and (a.n_boot, a.seed) == (b.n_boot, b.seed))
            rows.append({"test_set": test_set, "level": level, "n_files": int(len(ea["ids"])),
                         "n_boot": a.n_boot, "n_boot_replica": b.n_boot, "seed": a.seed,
                         "seed_replica": b.seed, "same_files_in_order": same_files,
                         "same_labels": same_labels, "same_speakers": same_speakers,
                         f"sha256_{reference.encoder}": digests[0],
                         f"sha256_{replica.encoder}": digests[1], "ok": bool(ok)})
    return pd.DataFrame(rows)


# Comparison with the original tables ------------------------------------------------------------

def original_table_names(tables: Mapping[str, Sequence[str | None]] = ORIGINAL_TABLES
                         ) -> dict[str, list[str]]:
    """Table of this package -> names of the original tables in the mapping file."""
    return {name: [name if c is None else f"{name}_{c}" for c in conditions]
            for name, conditions in tables.items()}


def compare_with_original(name: str, table: pd.DataFrame, legacy: Mapping[str, Any],
                          copies: str | None = None,
                          conditions: Sequence[str | None] | None = None
                          ) -> dict[str, pd.DataFrame]:
    """The table ``name`` of this package against its original tables.

    One check table per original table (``verify.compare_with_legacy``),
    under the name ``table_<original table>``. The rows of one condition are
    compared with the original table of that condition; of the duration
    table, the quantities that the original table has. A row that only one
    of the two tables has fails. ``copies``: the folder of the migrated
    copies of the original tables (default: the original files).
    """
    conditions = ORIGINAL_TABLES[name] if conditions is None else tuple(conditions)
    if name == "itw_by_duration":
        table = table[table["quantity"].isin(ORIGINAL_DURATION_QUANTITIES)]
    checks = {}
    for condition in conditions:
        original = name if condition is None else f"{name}_{condition}"
        part = table if condition is None else table[table["condition"] == condition]
        checks[f"table_{original}"] = verify.compare_with_legacy(
            part.reset_index(drop=True), original, legacy, copies)
    return checks
