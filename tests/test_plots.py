"""Figures: the drawn values are the values of the tables, and every figure
passes the checks of save_figure."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ssl_silence import config as cfg
from ssl_silence import plots as P

LAYERS = tuple(range(1, 13))
POOLINGS = ("stats", "mean")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    P.pyplot().close("all")


def summary_table(encoders=("wavlm-base",)) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [{"encoder": e, "condition": c, "pooling": p, "layer": k,
             "dev_eer_mean": rng.uniform(0.2, 3), "eval_eer_mean": rng.uniform(1, 16),
             "itw_eer_mean": rng.uniform(15, 30)}
            for e in encoders for c in ("clean", "trim40") for p in POOLINGS for k in LAYERS]
    return pd.DataFrame(rows)


def dev_table(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.sort_values(["dev_eer_mean", "layer"], kind="mergesort")
    return ordered.groupby(["encoder", "condition", "pooling"], sort=False).head(1)


def effects_table(encoders=("wavlm-base",), layers=LAYERS) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for e in encoders:
        for t in cfg.TEST_SETS:
            for p in POOLINGS:
                for k in layers:
                    for effect in cfg.EFFECTS:
                        estimate = rng.normal(0, 5)
                        rows.append({"encoder": e, "condition": "trim40", "test_set": t,
                                     "pooling": p, "layer": k, "effect": effect, "estimate": estimate,
                                     "file_ci_low": estimate - rng.uniform(0.1, 0.5),
                                     "file_ci_high": estimate + rng.uniform(0.1, 0.5),
                                     "spk_ci_low": estimate - rng.uniform(0.5, 2),
                                     "spk_ci_high": estimate + rng.uniform(0.5, 2)})
    return pd.DataFrame(rows)


def line(ax, label):
    found = [artist for artist in ax.lines if artist.get_label() == label]
    assert len(found) == 1, (label, [artist.get_label() for artist in ax.lines])
    return found[0]


def errorbars(ax):
    """(x, y, low, high) of every error-bar series of an axes, in drawing order."""
    out = []
    for container in ax.containers:
        data, _, bars = container.lines
        x, y = np.asarray(data.get_xdata(), float), np.asarray(data.get_ydata(), float)
        segments = np.asarray(bars[0].get_segments())            # (n, 2, 2)
        horizontal = np.allclose(segments[:, 0, 1], segments[:, 1, 1])
        ends = segments[:, :, 0] if horizontal else segments[:, :, 1]
        out.append((x, y, ends[:, 0], ends[:, 1], container.get_label()))
    return out


# Checks ------------------------------------------------------------------------------------

def test_palette_is_defined_once():
    assert P.PALETTE == {"primary": "#00687F", "accent": "#EE3342", "dark": "#0B3C49",
                         "light": "#6FB3C6", "neutral": "#6C757D", "bg": "#E6F0F3",
                         "bg-warm": "#FDE8EA"}
    assert (P.PRINT_WIDTH_CM, P.MIN_FONT_PT) == (13.0, 7.0)
    assert P.POOLING_LINE == {"stats": "-", "mean": "--"} and P.POOLING_MARKER == {"stats": "o", "mean": "s"}


def simple_figure(width_cm=13.0, colour=P.FSM_TEAL, fontsize=8):
    plt = P.pyplot()
    fig, ax = plt.subplots(figsize=(width_cm / 2.54, 2.0))
    ax.plot([0, 1], [0, 1], color=colour, label="series")
    ax.set_xlabel("x", fontsize=fontsize)
    ax.tick_params(labelsize=7)
    return fig, ax


def test_check_figure_rules(tmp_path):
    fig, _ = simple_figure()
    P.check_figure(fig)
    paths = P.save_figure(fig, str(tmp_path / "ok"))
    assert [os.path.basename(p) for p in paths] == ["ok.png", "ok.pdf"]
    assert all(0 < os.path.getsize(p) < P.MAX_FILE_BYTES for p in paths)
    with pytest.raises(ValueError, match="outside the palette"):
        P.check_figure(simple_figure(colour="#FF8800")[0])
    with pytest.raises(ValueError, match="outside the palette"):
        P.check_figure(simple_figure(colour="C0")[0])
    with pytest.raises(ValueError, match="cm wide"):
        P.check_figure(simple_figure(width_cm=12.0)[0])
    P.check_figure(simple_figure(width_cm=12.0)[0], print_width=False)
    with pytest.raises(ValueError, match="fonts below"):
        P.check_figure(simple_figure(fontsize=6)[0])
    P.check_figure(simple_figure(fontsize=6)[0], print_width=False)
    fig, _ = simple_figure()
    fig.suptitle("a title")
    with pytest.raises(ValueError, match="figure-level title"):
        P.check_figure(fig)
    with pytest.raises(ValueError):
        P.save_figure(simple_figure(colour="#123456")[0], str(tmp_path / "bad"))
    assert not os.path.exists(tmp_path / "bad.png")


def test_red_and_grey_series_need_another_difference():
    fig, ax = simple_figure(colour=P.FSM_RED)
    ax.plot([0, 1], [1, 0], color=P.SLATE_GRAY, label="control")
    with pytest.raises(ValueError, match="colour alone"):
        P.check_figure(fig)
    fig, ax = simple_figure(colour=P.FSM_RED)
    ax.plot([0, 1], [1, 0], color=P.SLATE_GRAY, linestyle=":", label="control")
    P.check_figure(fig)
    fig, ax = simple_figure(colour=P.FSM_RED)
    ax.axvline(0.5, color=P.SLATE_GRAY)                          # a reference line is no series
    P.check_figure(fig)


# Silence distribution ---------------------------------------------------------------------------

def test_ecdf_by_hand():
    x, f = P.ecdf(np.array([3.0, 1.0, 2.0, 2.0]))
    assert x.tolist() == [1, 2, 2, 3] and f.tolist() == [0.25, 0.5, 0.75, 1.0]
    tables = {"a": pd.DataFrame({"label": [0, 0, 1, 1], "lead_40": [0, 16_000, 32_000, 8_000]}),
              "b": pd.DataFrame({"label": [0, 1], "lead_40": [48_000, 0]})}
    assert P.ecdf_x_limit(tables, "lead_40", q=1.0) == 3.0
    assert P.ecdf_x_limit({"b": tables["b"].assign(lead_40=0)}, "lead_40") is None


def test_silence_ecdf_figure(tmp_path):
    rng = np.random.default_rng(2)
    tables = {name: pd.DataFrame({"label": (rng.random(400) < 0.7).astype(int),
                                  "lead_40": rng.integers(0, 30_000, 400)})
              for name in ("ASVspoof train", "ASVspoof dev", "ASVspoof eval", "In-the-Wild")}
    fig = P.silence_ecdf_figure(tables)
    assert [ax.get_title() for ax in fig.axes] == list(tables)
    table = tables["ASVspoof eval"]
    for label, name in ((0, "bonafide"), (1, "spoof")):
        drawn = line(fig.axes[2], name)
        x, f = P.ecdf(table.loc[table["label"] == label, "lead_40"].to_numpy() / 16_000)
        assert np.array_equal(drawn.get_xdata(), x) and np.array_equal(drawn.get_ydata(), f)
    assert {s[1:] for s in P.series_styles(fig)} == {("#00687F", "-", "o"), ("#EE3342", "--", "s")}
    assert fig.axes[0].get_xlim()[1] == pytest.approx(P.ecdf_x_limit(tables, "lead_40"))
    P.save_figure(fig, str(tmp_path / "silence_ecdf"), print_width=False)
    with pytest.raises(P.MissingRows):
        P.silence_ecdf_figure({"x": pd.DataFrame({"label": [0]})})


# Layers, cells and effects -----------------------------------------------------------------------

def test_layer_eer_figure_draws_the_table(tmp_path):
    summary = summary_table()
    dev = dev_table(summary)
    fig = P.layer_eer_figure(summary, dev)
    P.save_figure(fig, str(tmp_path / "layer_eer"))
    assert [ax.get_title() for ax in fig.axes] == ["(a) ASVspoof 2019 LA eval", "(b) In-the-Wild"]
    for ax, column in zip(fig.axes, ("eval_eer_mean", "itw_eer_mean")):
        for condition, name, colour in (("clean", "clean", "#00687F"), ("trim40", "trimmed", "#EE3342")):
            for pooling in POOLINGS:
                drawn = line(ax, f"{name}, {pooling} pooling")
                part = summary[(summary["condition"] == condition)
                               & (summary["pooling"] == pooling)].sort_values("layer")
                assert np.array_equal(drawn.get_ydata(), part[column].to_numpy())
                assert list(drawn.get_xdata()) == list(LAYERS)
                assert P._hex(drawn.get_color()) == colour
                assert drawn.get_linestyle() == P.POOLING_LINE[pooling]
        markers = [m for m in ax.lines if m.get_marker() in ("o", "s") and len(m.get_xdata()) == 1]
        assert len(markers) == 4
        for marker in markers:
            pooling = "stats" if marker.get_marker() == "o" else "mean"
            condition = "clean" if P._hex(marker.get_color()) == "#00687F" else "trim40"
            selected = dev[(dev["condition"] == condition) & (dev["pooling"] == pooling)]["layer"].iloc[0]
            assert marker.get_xdata()[0] == selected
            filled = P._hex(marker.get_markerfacecolor()) != "#FFFFFF"
            assert filled == (pooling == "stats")
    assert fig.axes[0].get_xlabel() == "WavLM layer"
    with pytest.raises(P.MissingRows):
        P.layer_eer_figure(summary[summary["pooling"] == "stats"], dev)
    with pytest.raises(P.MissingRows):
        P.layer_eer_figure(summary, dev.iloc[:-1])
    with pytest.raises(P.MissingRows):
        P.layer_eer_figure(summary.assign(eval_eer_mean=np.nan), dev)


def test_two_encoders_are_stacked_as_rows(tmp_path):
    summary = summary_table(("wavlm-base", "hubert-base"))
    fig = P.layer_eer_figure(summary, dev_table(summary))
    assert len(fig.axes) == 4 and fig.get_size_inches()[1] == pytest.approx(2 * 7.4 / 2.54)
    assert fig.axes[2].get_title() == "(a) ASVspoof 2019 LA eval, HuBERT"
    assert fig.axes[3].get_xlabel() == "HuBERT layer"
    P.save_figure(fig, str(tmp_path / "layer_eer_two"))
    effects = effects_table(("wavlm-base", "hubert-base"))
    fig = P.effects_by_layer_figure(effects)
    assert len(fig.axes) == 8
    hubert = effects[(effects["encoder"] == "hubert-base") & (effects["test_set"] == "itw")
                     & (effects["pooling"] == "mean") & (effects["effect"] == "interaction")]
    x, y, low, high, _ = errorbars(fig.axes[7])[1]
    assert np.allclose(y, hubert.sort_values("layer")["estimate"]) and np.allclose(x, np.array(LAYERS) + 0.12)
    P.save_figure(fig, str(tmp_path / "effects_two"))


def cells_table() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = [{"condition": "trim40", "pooling": p, "layer": k, "train_condition": tr,
             "test_condition": te, "eval_eer_mean": rng.uniform(1, 12),
             "itw_eer_mean": rng.uniform(14, 22)}
            for p in POOLINGS for k in (3, 4, 5) for tr in ("clean", "trim40")
            for te in ("clean", "trim40")]
    return pd.DataFrame(rows)


def test_cells_figure_draws_the_table(tmp_path):
    cells = cells_table()
    fig = P.cells_figure(cells)
    P.save_figure(fig, str(tmp_path / "cells_2x2"))
    titles = [ax.get_title() for ax in fig.axes[:6]]
    assert titles == ["L3, stats", "L3, mean", "L4, stats", "L4, mean", "L5, stats", "L5, mean"]
    ax = fig.axes[6 + 3]                                         # In-the-Wild, L4 mean
    part = cells[(cells["pooling"] == "mean") & (cells["layer"] == 4)].set_index(
        ["train_condition", "test_condition"])["itw_eer_mean"]
    assert line(ax, "trained clean").get_ydata().tolist() == [part[("clean", "clean")],
                                                              part[("clean", "trim40")]]
    trimmed = line(ax, "trained trimmed")
    assert trimmed.get_ydata().tolist() == [part[("trim40", "clean")], part[("trim40", "trim40")]]
    assert (P._hex(trimmed.get_color()), trimmed.get_linestyle(), trimmed.get_marker()) == (
        "#EE3342", "--", "s")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["C", "T"]
    with pytest.raises(P.MissingRows, match="6"):
        P.cells_figure(cells[cells["layer"] != 5])
    with pytest.raises(P.MissingRows):
        P.cells_figure(cells.iloc[1:])
    with pytest.raises(P.MissingRows):
        P.cells_figure(pd.concat([cells, cells.assign(condition="trim20")]))


def test_effects_by_layer_figure_draws_the_table(tmp_path):
    effects = effects_table()
    fig = P.effects_by_layer_figure(effects)
    P.save_figure(fig, str(tmp_path / "effects_by_layer"))
    assert fig.get_size_inches()[1] == pytest.approx(11.0 / 2.54)
    panels = [("eval", "delta_test"), ("eval", "interaction"), ("itw", "delta_test"),
              ("itw", "interaction")]
    for ax, (test_set, effect) in zip(fig.axes, panels):
        series = errorbars(ax)
        assert [s[4] for s in series] == ["statistics pooling", "mean pooling"]
        for (x, y, low, high, _), pooling, offset in zip(series, POOLINGS, (-0.12, 0.12)):
            part = effects[(effects["test_set"] == test_set) & (effects["pooling"] == pooling)
                           & (effects["effect"] == effect)].sort_values("layer")
            assert np.allclose(x, np.array(LAYERS) + offset, atol=1e-12)
            assert np.array_equal(y, part["estimate"].to_numpy())
            assert np.allclose(low, part["spk_ci_low"], atol=1e-12)
            assert np.allclose(high, part["spk_ci_high"], atol=1e-12)
        colour = "#00687F" if effect == "delta_test" else "#EE3342"
        data = ax.containers[0].lines[0]
        assert P._hex(data.get_color()) == colour and data.get_linestyle() == "-"
        assert ax.containers[1].lines[0].get_linestyle() == "--"
        assert P._hex(ax.containers[1].lines[0].get_markerfacecolor()) == "#FFFFFF"
    file_level = errorbars(P.effects_by_layer_figure(effects, ci="file").axes[0])[0]
    part = effects[(effects["test_set"] == "eval") & (effects["pooling"] == "stats")
                   & (effects["effect"] == "delta_test")].sort_values("layer")
    assert np.allclose(file_level[2], part["file_ci_low"], atol=1e-12)
    with pytest.raises(P.MissingRows):
        P.effects_by_layer_figure(effects[~((effects["layer"] == 7) & (effects["pooling"] == "mean"))])
    with pytest.raises(P.MissingRows):
        P.effects_by_layer_figure(effects.assign(spk_ci_low=np.nan))
    with pytest.raises(P.MissingRows):
        P.effects_by_layer_figure(effects, condition="trim20")


def test_effects_forest_figure_draws_the_table(tmp_path):
    effects = effects_table(layers=(3, 4, 5))
    fig = P.effects_forest_figure(effects)
    P.save_figure(fig, str(tmp_path / "effects_forest"))
    assert len(fig.axes) == 8
    assert fig.axes[5].get_title() == "(c) test trimming, trimmed-trained model\nIn-the-Wild"
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ["stats / L3", "mean / L3", "stats / L4", "mean / L4", "stats / L5", "mean / L5"]
    ax = fig.axes[5]
    row = effects[(effects["test_set"] == "itw") & (effects["effect"] == "delta_test_t")
                  & (effects["pooling"] == "mean") & (effects["layer"] == 4)].iloc[0]
    file_bar, speaker_bar = errorbars(ax)[6], errorbars(ax)[7]   # fourth configuration
    assert file_bar[0][0] == speaker_bar[0][0] == row["estimate"]
    assert (file_bar[1][0], speaker_bar[1][0]) == (3.15, 2.85)
    assert file_bar[2][0] == pytest.approx(row["file_ci_low"]) and file_bar[3][0] == pytest.approx(row["file_ci_high"])
    assert speaker_bar[2][0] == pytest.approx(row["spk_ci_low"])
    assert speaker_bar[3][0] == pytest.approx(row["spk_ci_high"])
    zero = [artist for artist in ax.lines if artist.get_linestyle() == "--"
            and P._hex(artist.get_color()) == "#EE3342"]
    assert len(zero) == 1 and set(zero[0].get_xdata()) == {0}
    with pytest.raises(P.MissingRows):
        P.effects_forest_figure(effects_table())                 # twelve layers are not six configurations
    with pytest.raises(P.MissingRows):
        P.effects_forest_figure(effects[effects["effect"] != "delta_total"])


# Per attack, padding and probes ---------------------------------------------------------------------

def attack_table() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows = []
    for attack in cfg.SPOOF_SYSTEM_IDS["eval"]:
        for p in POOLINGS:
            for k in (3, 4, 5):
                estimate = rng.normal(5, 6)
                rows.append({"condition": "trim40", "attack": attack, "pooling": p, "layer": k,
                             "estimate": estimate, "spk_ci_low": estimate - rng.uniform(0.5, 3),
                             "spk_ci_high": estimate + rng.uniform(0.5, 3)})
    return pd.DataFrame(rows)


def test_per_attack_figure_draws_the_table(tmp_path):
    table = attack_table()
    fig = P.per_attack_figure(table)
    P.save_figure(fig, str(tmp_path / "per_attack_effect"))
    ax = fig.axes[0]
    series = errorbars(ax)
    assert [s[4] for s in series] == ["stats pooling, L3", "mean pooling, L3", "stats pooling, L4",
                                      "mean pooling, L4", "stats pooling, L5", "mean pooling, L5"]
    offsets = np.linspace(-0.3, 0.3, 6)
    for (x, y, low, high, _), (pooling, layer), offset in zip(
            series, [(p, k) for k in (3, 4, 5) for p in POOLINGS], offsets):
        part = table[(table["pooling"] == pooling) & (table["layer"] == layer)].sort_values("attack")
        assert np.allclose(x, np.arange(13) + offset) and np.array_equal(y, part["estimate"].to_numpy())
        assert np.allclose(low, part["spk_ci_low"]) and np.allclose(high, part["spk_ci_high"])
    colours = [P._hex(c.lines[0].get_color()) for c in ax.containers]
    assert colours == ["#6FB3C6", "#6FB3C6", "#0B3C49", "#0B3C49", "#00687F", "#00687F"]
    assert [t.get_text() for t in ax.get_xticklabels()] == [f"A{k:02d}" for k in range(7, 20)]
    with pytest.raises(P.MissingRows):
        P.per_attack_figure(table[table["attack"] != "A19"])
    with pytest.raises(P.MissingRows):
        P.per_attack_figure(table.assign(spk_ci_high=np.nan))


def test_padding_deviation_figure_draws_the_table(tmp_path):
    rng = np.random.default_rng(5)
    rows = [{"condition": c, "sample": s, "comparison": m, "layer": k, "statistic": st,
             "rel_mean_diff": rng.uniform(0.1, 0.5)}
            for c in ("clean", "trim40") for s in ("random", "most_padded")
            for m in ("no_mask", "mask", "mask_minus") for k in LAYERS for st in ("mean", "std")]
    table = pd.DataFrame(rows)
    fig = P.padding_deviation_figure(table)
    P.save_figure(fig, str(tmp_path / "padding_deviation"))
    ax = fig.axes[0]
    for condition, name, colour in (("clean", "clean", "#00687F"), ("trim40", "trimmed", "#EE3342")):
        for comparison, label, style, marker in (("no_mask", "without attention mask", "-", "o"),
                                                 ("mask", "with attention mask", ":", "^")):
            drawn = line(ax, f"{name}, {label}")
            part = table[(table["condition"] == condition) & (table["sample"] == "random")
                         & (table["comparison"] == comparison) & (table["statistic"] == "mean")]
            assert np.array_equal(drawn.get_ydata(), part.sort_values("layer")["rel_mean_diff"].to_numpy())
            assert (P._hex(drawn.get_color()), drawn.get_linestyle(), drawn.get_marker()) == (
                colour, style, marker)
    other = P.padding_deviation_figure(table, sample="most_padded", statistic="std")
    part = table[(table["condition"] == "clean") & (table["sample"] == "most_padded")
                 & (table["comparison"] == "no_mask") & (table["statistic"] == "std")]
    assert np.array_equal(line(other.axes[0], "clean, without attention mask").get_ydata(),
                          part.sort_values("layer")["rel_mean_diff"].to_numpy())
    with pytest.raises(P.MissingRows):
        P.padding_deviation_figure(table[~((table["layer"] == 3) & (table["comparison"] == "mask"))])
    with pytest.raises(P.MissingRows):
        P.padding_deviation_figure(table, sample="first")


def test_probe_figure_draws_the_table(tmp_path):
    rng = np.random.default_rng(6)
    targets = ("f0_mean_hz", "hnr_mean_db", "tilt_db_per_khz", "win_lead_40", "pad", "length_in_window")
    rows = [{"cv": cv, "layer": k, "target": t, "control": control,
             "r2_mean": rng.uniform(0.3, 0.95) if control == "real" else rng.uniform(-0.02, 0),
             "r2_sd": rng.uniform(0.01, 0.1)}
            for t in targets for cv in ("random", "speaker") for k in LAYERS
            for control in ("real", "shuffled")]
    r2 = pd.DataFrame(rows)
    fig = P.probe_figure(r2)
    P.save_figure(fig, str(tmp_path / "probe_r2"), print_width=False)
    assert [ax.get_title() for ax in fig.axes] == ["acoustic targets", "silence targets"]

    def values(target, cv, control="real"):
        part = r2[(r2["target"] == target) & (r2["cv"] == cv) & (r2["control"] == control)]
        return part.sort_values("layer")["r2_mean"].to_numpy()

    thick = line(fig.axes[0], "mean HNR (speaker CV)")
    assert np.array_equal(thick.get_ydata(), values("hnr_mean_db", "speaker"))
    assert (P._hex(thick.get_color()), thick.get_linestyle(), thick.get_marker()) == ("#EE3342", "--", "s")
    thin = line(fig.axes[0], "mean HNR (random CV)")
    assert np.array_equal(thin.get_ydata(), values("hnr_mean_db", "random")) and thin.get_linewidth() < 1
    control = line(fig.axes[1], "shuffled control")
    assert np.array_equal(control.get_ydata(), values("win_lead_40", "speaker", "shuffled"))
    assert (P._hex(control.get_color()), control.get_linestyle()) == ("#6C757D", ":")
    assert np.array_equal(line(fig.axes[1], "utterance length in window (speaker CV)").get_ydata(),
                          values("length_in_window", "speaker"))
    trimmed = P.probe_figure(r2[r2["target"] != "win_lead_40"])   # the targets of a trimmed condition
    assert len(trimmed.axes[1].get_legend().get_texts()) == 5
    with pytest.raises(P.MissingRows):
        P.probe_figure(r2[r2["control"] == "real"])
    with pytest.raises(P.MissingRows):
        P.probe_figure(r2[r2["cv"] == "random"])
