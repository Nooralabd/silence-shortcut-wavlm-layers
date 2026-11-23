"""Figures of the manuscript, drawn from saved tables only.

No metric is computed here: every plotted value is read from a table written
by the analysis modules. Each figure keeps the style of the published figure
(colours, line styles, markers, layout, legend position). The palette is
defined once; ``save_figure`` checks every figure against it.

Two conventions hold in every figure that shows them: the clean condition is
teal and the trimmed condition red; statistics pooling has a solid line and
a filled circle, mean pooling a dashed line and an open square.

Every figure is saved as a 300 dpi PNG and a vector PDF.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import config as cfg

# Palette (fixed; no other colour is used) ----------------------------------------------
FSM_TEAL = "#00687F"     # clean condition, bonafide, first series
FSM_RED = "#EE3342"      # trimmed condition, spoof, zero line of the forest plot
DEEP_TEAL = "#0B3C49"    # text, axes, reference lines
SKY_TEAL = "#6FB3C6"     # second series, file-level intervals
SLATE_GRAY = "#6C757D"   # reference lines, controls
MIST = "#E6F0F3"         # grid lines and bands
BLUSH = "#FDE8EA"        # warm background (not used by a figure so far)
PALETTE = {"primary": FSM_TEAL, "accent": FSM_RED, "dark": DEEP_TEAL, "light": SKY_TEAL,
           "neutral": SLATE_GRAY, "bg": MIST, "bg-warm": BLUSH}
WHITE = "#FFFFFF"

BASE = "clean"
CONDITION_LABELS = {"clean": "clean", "trim40": "trimmed", "trim20": "trimmed at 20 dB",
                    "trim15": "trimmed at 15 dB"}
ENCODER_LABELS = {"wavlm-base": "WavLM", "hubert-base": "HuBERT", "wavlm-large": "WavLM Large"}
POOLING_LINE = {"stats": "-", "mean": "--"}
POOLING_MARKER = {"stats": "o", "mean": "s"}
CLASS_STYLE = {
    0: {"color": FSM_TEAL, "linestyle": "-", "marker": "o", "label": "bonafide"},
    1: {"color": FSM_RED, "linestyle": "--", "marker": "s", "label": "spoof"},
}
LAYER_COLOR = {3: SKY_TEAL, 4: DEEP_TEAL, 5: FSM_TEAL}
TEST_SET_PANELS = (("eval", "(a) ASVspoof 2019 LA eval"), ("itw", "(b) In-the-Wild"))
TEST_SET_NAMES = {"eval": "ASVspoof 2019 eval", "itw": "In-the-Wild"}
SPLIT_PANELS = {"train": "ASVspoof train", "dev": "ASVspoof dev", "eval": "ASVspoof eval",
                "itw": "In-the-Wild"}

# The manuscript prints figures at its text width; a print-width figure is
# drawn exactly this wide with no font below MIN_FONT_PT, so it enters at scale 1.
PRINT_WIDTH_CM = 13.0
PRINT_WIDTH_IN = PRINT_WIDTH_CM / 2.54
PRINT_PAD_IN = 0.02
MIN_FONT_PT = 7.0
MAX_FILE_BYTES = 5 * 1024 * 1024


class MissingRows(ValueError):
    """A figure lacks rows or values it needs; nothing is drawn from an incomplete table."""


# Style and checks -----------------------------------------------------------------------

def pyplot(font_size: float = 9, axes_linewidth: float | None = None):
    """matplotlib.pyplot with the shared settings (Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcdefaults()
    settings = {
        "font.size": font_size, "text.color": DEEP_TEAL, "axes.titlecolor": DEEP_TEAL,
        "axes.edgecolor": DEEP_TEAL, "axes.labelcolor": DEEP_TEAL, "xtick.color": DEEP_TEAL,
        "ytick.color": DEEP_TEAL, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": MIST, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    }
    if axes_linewidth is not None:
        settings["axes.linewidth"] = axes_linewidth
    plt.rcParams.update(settings)
    return plt


def _hex(colour) -> str:
    from matplotlib.colors import to_hex

    return to_hex(colour, keep_alpha=False).upper()


def figure_colours(fig) -> set[str]:
    """Hex colours of every visible line, marker, patch, collection and text of a figure."""
    from matplotlib.collections import Collection
    from matplotlib.colors import to_rgba_array
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    found: set[str] = set()

    def add(value) -> None:
        if value is None or (isinstance(value, str) and value in ("none", "None")):
            return
        for rgba in to_rgba_array(value):
            if rgba[3] > 0:
                found.add(_hex(rgba))

    for artist in fig.findobj():
        if not artist.get_visible():
            continue
        if isinstance(artist, Line2D):
            if artist.get_linestyle() not in ("None", "none", " ", ""):
                add(artist.get_color())
            if artist.get_marker() not in ("None", "none", " ", "", None):
                add(artist.get_markerfacecolor())
                add(artist.get_markeredgecolor())
        elif isinstance(artist, Collection):
            add(artist.get_facecolor())
            add(artist.get_edgecolor())
        elif isinstance(artist, Patch):
            if artist.get_fill():
                add(artist.get_facecolor())
            if artist.get_linewidth():
                add(artist.get_edgecolor())
        elif isinstance(artist, Text) and artist.get_text():
            add(artist.get_color())
    return found


def series_styles(fig) -> list[tuple[str, str, str, str]]:
    """(label, colour, line style, marker) of every labelled line of a figure."""
    from matplotlib.lines import Line2D

    out = []
    for artist in fig.findobj(Line2D):
        label = str(artist.get_label())
        if label and not label.startswith("_"):
            out.append((label, _hex(artist.get_color()), str(artist.get_linestyle()),
                        str(artist.get_marker())))
    return out


def font_sizes(fig) -> list[float]:
    """Font size of every visible, non-empty text of a figure."""
    from matplotlib.text import Text

    return [float(t.get_fontsize()) for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip()]


def check_figure(fig, print_width: bool = True) -> None:
    """Raise unless the figure follows the rules.

    Always: palette colours (and white) only; no figure title (the caption
    carries that text); a red and a grey labelled series never share both
    line style and marker. With ``print_width``: exactly PRINT_WIDTH_CM wide
    and no font below MIN_FONT_PT.
    """
    allowed = {c.upper() for c in PALETTE.values()} | {WHITE}
    foreign = sorted(figure_colours(fig) - allowed)
    if foreign:
        raise ValueError(f"colours outside the palette: {foreign}")
    if fig.get_suptitle():
        raise ValueError("a figure carries no figure-level title")
    styles = series_styles(fig)
    red = {(line, marker) for _, colour, line, marker in styles if colour == FSM_RED.upper()}
    grey = {(line, marker) for _, colour, line, marker in styles if colour == SLATE_GRAY.upper()}
    if red & grey:
        raise ValueError(f"red and grey series differ by colour alone: {sorted(red & grey)}")
    if print_width:
        width = fig.get_size_inches()[0] * 2.54
        if abs(width - PRINT_WIDTH_CM) > 1e-6:
            raise ValueError(f"the figure is {width:.3f} cm wide, not {PRINT_WIDTH_CM} cm")
        small = sorted({s for s in font_sizes(fig) if s < MIN_FONT_PT})
        if small:
            raise ValueError(f"fonts below {MIN_FONT_PT} pt: {small}")


def save_figure(fig, out_path_no_ext: str, print_width: bool = True) -> list[str]:
    """Check the figure and write ``<path>.png`` (300 dpi) and ``<path>.pdf``.

    A print-width figure is saved with a small fixed padding so that it comes
    out at its nominal width. Returns both paths.
    """
    check_figure(fig, print_width)
    cfg.makedirs(os.path.dirname(out_path_no_ext) or ".")
    paths = [out_path_no_ext + ".png", out_path_no_ext + ".pdf"]
    extra = {"pad_inches": PRINT_PAD_IN} if print_width else {}
    fig.savefig(paths[0], dpi=300, **extra)
    fig.savefig(paths[1], **extra)
    for path in paths:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            raise ValueError(f"{path} is larger than 5 MB")
    return paths


def _require(table: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise MissingRows(f"{name}: missing columns {missing}")
    if table.empty:
        raise MissingRows(f"{name}: the table is empty")


def _one(table: pd.DataFrame, name: str, **keys):
    """The single row of ``table`` with the given key values."""
    part = table
    for column, value in keys.items():
        part = part[part[column] == value]
    if len(part) != 1:
        raise MissingRows(f"{name}: {len(part)} rows for {keys}, expected 1")
    return part.iloc[0]


def _finite(values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise MissingRows(f"{name}: a value to be drawn is missing")
    return values


def six_configs(table: pd.DataFrame, name: str) -> list[tuple[str, int]]:
    """The six (pooling, layer) configurations of a table, by layer, statistics first."""
    configs = sorted({(p, int(k)) for p, k in zip(table["pooling"], table["layer"])},
                     key=lambda key: (key[1], key[0] != "stats"))
    if len(configs) != 6:
        raise MissingRows(f"{name}: expected 6 (pooling, layer) configurations, found {configs}")
    return configs


def encoder_label(encoder: str) -> str:
    """Name of an encoder on an axis."""
    return ENCODER_LABELS.get(encoder, encoder)


# Silence in the data ------------------------------------------------------------------------

def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical distribution function: values (N,) -> (sorted x, F(x) in (0, 1])."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    return x, np.arange(1, x.size + 1) / x.size


def ecdf_x_limit(tables: Mapping[str, pd.DataFrame], column: str, q: float = 0.99) -> float | None:
    """Upper x limit in seconds: the largest q-quantile over every panel and
    class, so that each curve shows at least a share q of its files.
    ``column`` is in samples. None when every quantile is 0."""
    limit = 0.0
    for table in tables.values():
        for label in CLASS_STYLE:
            values = table.loc[table["label"] == label, column].to_numpy(dtype=np.float64)
            if values.size:
                limit = max(limit, float(np.quantile(values, q)))
    return limit / cfg.SAMPLE_RATE if limit > 0 else None


def silence_ecdf_figure(tables: Mapping[str, pd.DataFrame], column: str = "lead_40",
                        xlabel: str = "leading silence in file (s)",
                        x_max_s: float | None = None):
    """Empirical distribution of one silence measurement, bonafide against
    spoof, one panel per table (panel title -> ``silence.silence_table``).

    ``column`` is in samples and drawn in seconds. The x limit is shared
    (default ``ecdf_x_limit``); the curves are not truncated, only the view.
    The classes differ by colour, line style and marker. Not a print-width
    figure. Returns the figure.
    """
    plt = pyplot(font_size=10, axes_linewidth=0.8)
    for title, table in tables.items():
        _require(table, ["label", column], f"silence distribution, {title}")
    if x_max_s is None:
        x_max_s = ecdf_x_limit(tables, column)
    n = len(tables)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n + 0.6, 2.8), sharey=True, squeeze=False)
    for ax, (title, table) in zip(axes[0], tables.items()):
        for label, style in CLASS_STYLE.items():
            values = table.loc[table["label"] == label, column].to_numpy() / cfg.SAMPLE_RATE
            if values.size:
                x, f = ecdf(values)
                marks = np.unique(np.clip((np.array([0.1, 0.3, 0.5, 0.7, 0.9]) * x.size).astype(int),
                                          0, x.size - 1))
                ax.step(x, f, where="post", linewidth=1.6, markersize=4, markevery=list(marks),
                        **style)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.0)
        if x_max_s is not None:
            ax.set_xlim(0, x_max_s)
    axes[0][0].set_ylabel("cumulative fraction of files")
    axes[0][-1].legend(loc="lower right")
    fig.tight_layout()
    return fig


# EER per layer ------------------------------------------------------------------------------

def _condition_colours(condition: str) -> dict[str, str]:
    return {BASE: FSM_TEAL, condition: FSM_RED}


def layer_eer_figure(summary: pd.DataFrame, dev_selected: pd.DataFrame,
                     condition: str = "trim40"):
    """EER of each layer on the two test sets, clean and trimmed, both poolings.

    ``summary``: the matched cells of ``selection.seed_summary`` (encoder,
    condition, pooling, layer, eval_eer_mean, itw_eer_mean). ``dev_selected``:
    ``selection.select_on_dev`` (encoder, condition, pooling, layer). One row
    of panels per encoder, columns = test sets. The band marks layers 3-5,
    the vertical line layer 4; a marker shows the layer selected on the
    development set (filled: statistics; open: mean). One legend below the
    axes. Returns the figure.
    """
    _require(summary, ["encoder", "condition", "pooling", "layer", "eval_eer_mean", "itw_eer_mean"],
             "EER per layer")
    _require(dev_selected, ["encoder", "condition", "pooling", "layer"], "dev-selected layers")
    encoders = list(pd.unique(summary["encoder"]))
    colours = _condition_colours(condition)
    plt = pyplot()
    fig, axes = plt.subplots(len(encoders), 2, squeeze=False, layout="constrained",
                             figsize=(PRINT_WIDTH_IN, 7.4 / 2.54 * len(encoders)))
    for row, encoder in enumerate(encoders):
        for ax, (test_set, panel) in zip(axes[row], TEST_SET_PANELS):
            value = f"{test_set}_eer_mean"
            ax.axvspan(2.5, 5.5, color=MIST, zorder=0)
            ax.axvline(4, color=SLATE_GRAY, linewidth=0.6, zorder=1)
            for cond in (BASE, condition):
                for pooling in ("stats", "mean"):
                    part = summary[(summary["encoder"] == encoder) & (summary["condition"] == cond)
                                   & (summary["pooling"] == pooling)].sort_values("layer")
                    if part.empty:
                        raise MissingRows(f"EER per layer: no rows for {encoder}, {cond}, {pooling}")
                    ax.plot(part["layer"], _finite(part[value], "EER per layer"),
                            color=colours[cond], linestyle=POOLING_LINE[pooling], linewidth=1.2,
                            label=f"{CONDITION_LABELS[cond]}, {pooling} pooling", zorder=2)
                    selected = int(_one(dev_selected, "dev-selected layers", encoder=encoder,
                                        condition=cond, pooling=pooling)["layer"])
                    y = part.loc[part["layer"] == selected, value]
                    ax.plot([selected], y, marker=POOLING_MARKER[pooling], color=colours[cond],
                            markersize=5, linestyle="none", zorder=3,
                            markerfacecolor="white" if pooling == "mean" else colours[cond])
            title = panel if len(encoders) == 1 else f"{panel}, {encoder_label(encoder)}"
            ax.set_title(title, fontsize=8)
            ax.set_xlabel(f"{encoder_label(encoder)} layer", fontsize=8)
            ax.set_xticks(sorted(int(k) for k in pd.unique(summary["layer"])))
            ax.tick_params(labelsize=7)
        axes[row][0].set_ylabel("EER (%)", fontsize=8)
    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7, frameon=False,
               title="marker = dev-selected layer", title_fontsize=7, columnspacing=2.0,
               handlelength=2.2)
    return fig


# The four train/test cells ----------------------------------------------------------------------

def cells_figure(cells: pd.DataFrame):
    """EER in the four train/test cells of six configurations.

    ``cells``: ``effects.cells_table`` of the layers to show (condition,
    pooling, layer, train_condition, test_condition, eval_eer_mean,
    itw_eer_mean). Rows = test sets, columns = configurations. The x axis is
    the test condition (C = clean, T = trimmed); each line is one training
    condition. Legend above the panels. Returns the figure.
    """
    _require(cells, ["condition", "pooling", "layer", "train_condition", "test_condition",
                     "eval_eer_mean", "itw_eer_mean"], "train/test cells")
    conditions = list(pd.unique(cells["condition"]))
    if len(conditions) != 1:
        raise MissingRows(f"train/test cells: one trimmed condition per figure, found {conditions}")
    condition = conditions[0]
    colours = _condition_colours(condition)
    configs = six_configs(cells, "train/test cells")
    plt = pyplot()
    fig, axes = plt.subplots(2, 6, figsize=(PRINT_WIDTH_IN, 6.6 / 2.54), sharey="row",
                             layout="constrained")
    for col, (pooling, layer) in enumerate(configs):
        part = cells[(cells["pooling"] == pooling) & (cells["layer"] == layer)]
        for row, test_set in enumerate(("eval", "itw")):
            value = f"{test_set}_eer_mean"
            ax = axes[row][col]
            for train in (BASE, condition):
                ends = [float(_one(part, "train/test cells", train_condition=train,
                                   test_condition=test)[value]) for test in (BASE, condition)]
                ax.plot([0, 1], _finite(ends, "train/test cells"), color=colours[train],
                        linestyle=POOLING_LINE[pooling], marker=POOLING_MARKER[pooling],
                        markersize=3, linewidth=1.1, label=f"trained {CONDITION_LABELS[train]}")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["C", "T"])
            ax.set_xlim(-0.35, 1.35)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(f"L{layer}, {pooling}", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{TEST_SET_NAMES[test_set]}\nEER (%)", fontsize=7)
    fig.supxlabel("test condition (C = clean, T = trimmed)", fontsize=7)
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=7, frameon=False,
               columnspacing=2.0)
    return fig


# Effects ------------------------------------------------------------------------------------------

EFFECT_PANELS = (("delta_test", "Test effect, $\\Delta_{\\mathrm{test}}$", FSM_TEAL),
                 ("interaction", "Interaction, $I$", FSM_RED))
POOLING_SERIES = {
    "stats": {"linestyle": "-", "marker": "o", "offset": -0.12, "label": "statistics pooling"},
    "mean": {"linestyle": "--", "marker": "s", "offset": 0.12, "label": "mean pooling"},
}
INTERVAL_COLUMNS = {"spk": ("spk_ci_low", "spk_ci_high"), "file": ("file_ci_low", "file_ci_high")}
FOREST_PANELS = (("delta_total", "(a) trimmed - clean"),
                 ("delta_test", "(b) test trimming, clean-trained model"),
                 ("delta_test_t", "(c) test trimming, trimmed-trained model"),
                 ("interaction", "(d) interaction"))
EFFECT_COLUMNS = ["encoder", "condition", "test_set", "pooling", "layer", "effect", "estimate",
                  "file_ci_low", "file_ci_high", "spk_ci_low", "spk_ci_high"]


def effects_by_layer_figure(effects: pd.DataFrame, condition: str = "trim40", ci: str = "spk",
                            height_cm: float = 11.0):
    """The test effect and the interaction by layer, with their intervals.

    ``effects``: ``effects.effects_ci`` of every layer. Rows = test sets (per
    encoder when the table holds two), columns = the two effects. Statistics
    pooling: solid line, filled circles, shifted to the left; mean pooling:
    dashed line, open squares, shifted to the right; the bars are the
    speaker-level (``ci="spk"``) or file-level intervals. The colour belongs
    to the effect; the y axis is shared within a row; one legend below the
    axes. Returns the figure.
    """
    from matplotlib.lines import Line2D

    _require(effects, EFFECT_COLUMNS, "effects by layer")
    low_column, high_column = INTERVAL_COLUMNS[ci]
    effects = effects[effects["condition"] == condition]
    encoders = list(pd.unique(effects["encoder"]))
    if not encoders:
        raise MissingRows(f"effects by layer: no rows for the condition {condition}")
    rows = [(encoder, test_set, label) for encoder in encoders for test_set, label in TEST_SET_PANELS]
    layers = sorted(int(k) for k in pd.unique(effects["layer"]))
    plt = pyplot()
    fig, axes = plt.subplots(len(rows), len(EFFECT_PANELS), sharex=True, sharey="row",
                             layout="constrained", squeeze=False,
                             figsize=(PRINT_WIDTH_IN, height_cm / 2.54 * len(encoders)))
    for i, (encoder, test_set, row_label) in enumerate(rows):
        for j, (effect, title, colour) in enumerate(EFFECT_PANELS):
            ax = axes[i][j]
            for pooling, style in POOLING_SERIES.items():
                part = effects[(effects["encoder"] == encoder) & (effects["test_set"] == test_set)
                               & (effects["pooling"] == pooling)
                               & (effects["effect"] == effect)].sort_values("layer")
                if sorted(part["layer"].astype(int)) != layers:
                    raise MissingRows(f"effects by layer: {encoder}, {test_set}, {pooling}, "
                                      f"{effect} does not hold every layer once")
                x = part["layer"].to_numpy(dtype=np.float64) + style["offset"]
                y = _finite(part["estimate"], "effects by layer")
                low = _finite(part[low_column], "effects by layer")
                high = _finite(part[high_column], "effects by layer")
                ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), color=colour,
                            linestyle=style["linestyle"], marker=style["marker"], markersize=3.0,
                            markerfacecolor=colour if pooling == "stats" else "white",
                            linewidth=1.0, elinewidth=0.7, capsize=1.5, label=style["label"])
            ax.axhline(0, color=DEEP_TEAL, linewidth=0.8, zorder=0)
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                prefix = row_label if len(encoders) == 1 else f"{row_label}, {encoder_label(encoder)}"
                ax.set_ylabel(f"{prefix}\nEER difference (pp)", fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel(f"{encoder_label(encoder)} layer" if len(encoders) == 1 else "layer",
                              fontsize=8)
            ax.set_xticks(layers)
            ax.tick_params(labelsize=7)
    handles = [Line2D([], [], color=DEEP_TEAL, linestyle=style["linestyle"], marker=style["marker"],
                      markersize=3.0, linewidth=1.0, label=style["label"],
                      markerfacecolor=DEEP_TEAL if pooling == "stats" else "white")
               for pooling, style in POOLING_SERIES.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=7,
               frameon=False, columnspacing=2.5, handlelength=2.6, title_fontsize=7,
               title=f"bars = 95 % {'speaker' if ci == 'spk' else 'file'}-level CI")
    return fig


def effects_forest_figure(effects: pd.DataFrame, condition: str = "trim40"):
    """Four effects of six configurations on the two test sets, with file- and
    speaker-level intervals.

    ``effects``: ``effects.effects_ci`` of the layers to show, one encoder.
    Rows = effects ((a) trimmed minus clean, (b) and (c) the test effect of
    the clean-trained and of the trimmed-trained model, (d) the interaction),
    columns = test sets. Circles: statistics pooling; squares: mean pooling.
    Light, open markers: file-level interval; dark, filled: speaker level.
    The red dashed line marks zero. Legend above the panels. Returns the figure.
    """
    from matplotlib.lines import Line2D

    _require(effects, EFFECT_COLUMNS, "effects forest")
    effects = effects[effects["condition"] == condition]
    if effects["encoder"].nunique() != 1:
        raise MissingRows("effects forest: one encoder and one condition per figure")
    configs = six_configs(effects, "effects forest")
    plt = pyplot()
    fig, axes = plt.subplots(len(FOREST_PANELS), 2, figsize=(PRINT_WIDTH_IN, 15.5 / 2.54),
                             sharey=True, layout="constrained")
    for r, (effect, name) in enumerate(FOREST_PANELS):
        for c, test_set in enumerate(("eval", "itw")):
            ax = axes[r][c]
            ax.axvline(0, color=FSM_RED, linestyle="--", linewidth=1.0, zorder=1)
            for i, (pooling, layer) in enumerate(configs):
                row = _one(effects, f"effects forest, {name}", test_set=test_set, effect=effect,
                           pooling=pooling, layer=layer)
                values = _finite([row[k] for k in ("estimate", "file_ci_low", "file_ci_high",
                                                   "spk_ci_low", "spk_ci_high")], "effects forest")
                estimate, file_low, file_high, spk_low, spk_high = values
                ax.errorbar(estimate, i + 0.15, xerr=[[estimate - file_low], [file_high - estimate]],
                            fmt=POOLING_MARKER[pooling], color=SKY_TEAL, markerfacecolor="white",
                            markeredgecolor=SKY_TEAL, markersize=3.5, capsize=3, linewidth=1.1,
                            zorder=2)
                ax.errorbar(estimate, i - 0.15, xerr=[[estimate - spk_low], [spk_high - estimate]],
                            fmt=POOLING_MARKER[pooling], color=FSM_TEAL, markerfacecolor=FSM_TEAL,
                            markeredgecolor=FSM_TEAL, markersize=3.5, capsize=4, linewidth=1.8,
                            zorder=3)
            ax.set_yticks(range(len(configs)))
            ax.set_yticklabels([f"{pooling} / L{layer}" for pooling, layer in configs])
            ax.invert_yaxis()
            ax.set_title(f"{name}\n{TEST_SET_NAMES[test_set]}", fontsize=7)
            ax.tick_params(labelsize=7)
            if r == len(FOREST_PANELS) - 1:
                ax.set_xlabel("EER difference (percentage points)", fontsize=7)
    handles = [
        Line2D([], [], color=SKY_TEAL, linewidth=1.4, marker="o", markerfacecolor="white",
               markeredgecolor=SKY_TEAL, label="file-level 95% CI"),
        Line2D([], [], color=FSM_TEAL, linewidth=2.2, marker="o", markerfacecolor=FSM_TEAL,
               label="speaker-level 95% CI"),
        Line2D([], [], color=DEEP_TEAL, linestyle="none", marker="o", label="circle = stats pooling"),
        Line2D([], [], color=DEEP_TEAL, linestyle="none", marker="s", label="square = mean pooling"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, fontsize=7, frameon=False,
               columnspacing=2.0)
    return fig


# Per attack -----------------------------------------------------------------------------------------

def per_attack_figure(per_attack: pd.DataFrame, condition: str = "trim40"):
    """Trimmed minus clean per evaluation attack for six configurations, with
    speaker-level intervals.

    ``per_attack``: ``effects.per_attack_effect_ci`` (condition, attack,
    pooling, layer, estimate, spk_ci_low, spk_ci_high). The colour is the
    layer; filled circles: statistics pooling; open squares: mean pooling.
    Legend below the axes. Returns the figure.
    """
    _require(per_attack, ["condition", "attack", "pooling", "layer", "estimate", "spk_ci_low",
                          "spk_ci_high"], "per-attack effect")
    table = per_attack[per_attack["condition"] == condition]
    configs = six_configs(table, "per-attack effect")
    attacks = sorted(table["attack"].unique())
    if attacks != list(cfg.SPOOF_SYSTEM_IDS["eval"]):
        raise MissingRows(f"per-attack effect: attacks {attacks}")
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(PRINT_WIDTH_IN, 7.2 / 2.54), layout="constrained")
    ax.axhline(0, color=SLATE_GRAY, linewidth=0.8)
    x = np.arange(len(attacks))
    for offset, (pooling, layer) in zip(np.linspace(-0.3, 0.3, len(configs)), configs):
        part = table[(table["pooling"] == pooling) & (table["layer"] == layer)].set_index("attack")
        if sorted(part.index) != attacks:
            raise MissingRows(f"per-attack effect: {pooling}/L{layer} does not hold every attack once")
        part = part.loc[attacks]
        estimate = _finite(part["estimate"], "per-attack effect")
        low = _finite(part["spk_ci_low"], "per-attack effect")
        high = _finite(part["spk_ci_high"], "per-attack effect")
        colour = LAYER_COLOR[layer]
        ax.errorbar(x + offset, estimate, yerr=[estimate - low, high - estimate], capsize=1.2,
                    linewidth=0.7, ecolor=colour, color=colour, marker=POOLING_MARKER[pooling],
                    markersize=3, linestyle="none", markeredgecolor=colour,
                    markerfacecolor=colour if pooling == "stats" else "white",
                    label=f"{pooling} pooling, L{layer}")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_xlabel("ASVspoof 2019 eval attack", fontsize=7.5)
    ax.set_ylabel("EER trimmed - clean (percentage points)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    fig.legend(loc="outside lower center", ncol=3, fontsize=7, frameon=False, columnspacing=2.0)
    return fig


# Padding deviation ------------------------------------------------------------------------------------

PADDING_COMPARISONS = {"no_mask": ("without attention mask", "-", "o"),
                       "mask": ("with attention mask", ":", "^")}


def padding_deviation_figure(per_layer: pd.DataFrame, sample: str = "random",
                             statistic: str = "mean", condition: str = "trim40",
                             encoder: str = "wavlm-base"):
    """Relative deviation of padded from unpadded features at each layer.

    ``per_layer``: the per-layer table of ``padding.padding_deviation``
    (condition, sample, comparison, layer, statistic, rel_mean_diff). Teal:
    clean; red: trimmed. Solid with circles: without an attention mask;
    dotted with triangles: with one. The vertical line marks layer 4. Legend
    below the axes. Returns the figure.
    """
    _require(per_layer, ["condition", "sample", "comparison", "layer", "statistic",
                         "rel_mean_diff"], "padding deviation")
    table = per_layer[(per_layer["sample"] == sample) & (per_layer["statistic"] == statistic)]
    colours = _condition_colours(condition)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(PRINT_WIDTH_IN, 8.0 / 2.54), layout="constrained")
    layers = None
    for cond in (BASE, condition):
        for comparison, (label, line, marker) in PADDING_COMPARISONS.items():
            part = table[(table["condition"] == cond)
                         & (table["comparison"] == comparison)].sort_values("layer")
            found = [int(k) for k in part["layer"]]
            if not found or (layers is not None and found != layers):
                raise MissingRows(f"padding deviation: layers {found} for {cond}, {comparison}")
            layers = found
            ax.plot(part["layer"], _finite(part["rel_mean_diff"], "padding deviation"),
                    color=colours[cond], linestyle=line, marker=marker, markersize=3.0,
                    linewidth=1.1, label=f"{CONDITION_LABELS[cond]}, {label}")
    ax.axvline(4, color=SLATE_GRAY, linewidth=0.6)
    ax.set_xticks(layers)
    ax.set_xlabel(f"{encoder_label(encoder)} layer", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylabel("relative mean difference", fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7, frameon=False,
               title="solid = no mask, dotted = mask", title_fontsize=7, columnspacing=2.0,
               handlelength=2.2)
    return fig


# Probes ---------------------------------------------------------------------------------------------------

PROBE_PANELS = (("acoustic targets", ("f0_mean_hz", "hnr_mean_db", "tilt_db_per_khz")),
                ("silence targets", ("win_lead_40", "pad", "length_in_window")))
PROBE_STYLE = {
    "f0_mean_hz": ("mean F0", FSM_TEAL, "-", "o"),
    "hnr_mean_db": ("mean HNR", FSM_RED, "--", "s"),
    "tilt_db_per_khz": ("spectral tilt", SKY_TEAL, "-.", "D"),
    "win_lead_40": ("leading silence in window", FSM_TEAL, "-", "o"),
    "pad": ("zero-padding", FSM_RED, "--", "s"),
    "length_in_window": ("utterance length in window", SKY_TEAL, "-.", "D"),
}


def probe_figure(r2: pd.DataFrame, main: str = "speaker", encoder: str = "wavlm-base"):
    """Probe R2 per layer: acoustic targets on the left, silence and length
    targets on the right.

    ``r2``: ``probes.probe_all_layers`` of one condition (cv, layer, target,
    control, r2_mean, r2_sd). Thick lines with markers and a band of one
    standard deviation over the folds: the ``main`` scheme; thin lines of the
    same colour and dash: the other scheme; grey dotted lines: the control
    with permuted targets (``main`` scheme). One legend below each panel. Not
    a print-width figure. Returns the figure.
    """
    _require(r2, ["cv", "layer", "target", "control", "r2_mean", "r2_sd"], "probes")
    panels = [(title, [name for name in names if name in set(r2["target"])])
              for title, names in PROBE_PANELS]
    panels = [(title, names) for title, names in panels if names]
    schemes = list(pd.unique(r2["cv"]))
    if main not in schemes or not panels:
        raise MissingRows(f"probes: no rows of the {main} scheme or of a known target")
    plt = pyplot(font_size=10, axes_linewidth=0.8)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels) + 0.4, 6.4), sharey=True,
                             squeeze=False)
    for ax, (title, names) in zip(axes[0], panels):
        for name in names:
            label, colour, line, marker = PROBE_STYLE[name]
            real = r2[(r2["target"] == name) & (r2["control"] == "real")]
            part = real[real["cv"] == main].sort_values("layer")
            x = part["layer"].to_numpy()
            mean = _finite(part["r2_mean"], "probes")
            sd = _finite(part["r2_sd"], "probes")
            ax.plot(x, mean, linewidth=1.7, markersize=4, color=colour, linestyle=line,
                    marker=marker, label=f"{label} ({main} CV)")
            ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.18, linewidth=0)
            for other in schemes:
                if other == main:
                    continue
                o = real[real["cv"] == other].sort_values("layer")
                ax.plot(o["layer"], _finite(o["r2_mean"], "probes"), color=colour, linestyle=line,
                        linewidth=0.8, alpha=0.8, label=f"{label} ({other} CV)")
            control = r2[(r2["target"] == name) & (r2["control"] == "shuffled")
                         & (r2["cv"] == main)].sort_values("layer")
            if control.empty:
                raise MissingRows(f"probes: no control rows for {name}")
            ax.plot(control["layer"], _finite(control["r2_mean"], "probes"), color=SLATE_GRAY,
                    linestyle=":", linewidth=0.9, marker=marker, markersize=2.5,
                    label="shuffled control" if name == names[0] else None)
        ax.axhline(0, color=DEEP_TEAL, linewidth=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{encoder_label(encoder)} layer")
        ax.set_xticks(sorted(int(k) for k in pd.unique(r2["layer"])))
        ax.legend(loc="upper center", bbox_to_anchor=(0.0, -0.13, 1.0, 0.0), mode="expand",
                  ncol=2, fontsize=6, frameon=False, borderaxespad=0.0)
    axes[0][0].set_ylabel("cross-validated R$^2$")
    fig.tight_layout()
    return fig
