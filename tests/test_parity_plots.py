"""The two figure functions that the original package holds, against this
package: the same drawn values and the same style. Skipped without the
original code. (The other figures of the manuscript were drawn by scripts
outside the original package.)"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ssl_silence import plots as P
from test_plots import effects_table, errorbars

OLD_EFFECTS = {"delta_test": ("test effect", "train=clean"), "interaction": ("interaction", "")}


def record(parity_record, name, differences):
    parity_record("plots", name, len(differences), max(differences))
    assert max(differences) == 0.0, f"{name}: max abs diff {max(differences)}"


def text_of(fig) -> list[str]:
    from matplotlib.text import Text

    return sorted(t.get_text() for t in fig.findobj(Text) if t.get_text().strip())


def test_empirical_distribution_and_its_figure(original, parity_record):
    O = original("plots")
    rng = np.random.default_rng(0)
    tables = {name: pd.DataFrame({"label": (rng.random(n) < 0.6).astype(int),
                                  "lead_40": rng.integers(0, 40_000, n),
                                  "pad": rng.integers(0, 2, n) * rng.integers(0, 30_000, n)})
              for name, n in (("ASVspoof train", 300), ("In-the-Wild", 77))}
    differences = []
    for table in tables.values():
        for column in ("lead_40", "pad"):
            new, old = P.ecdf(table[column].to_numpy()), O.ecdf(table[column].to_numpy())
            differences += [float(np.max(np.abs(new[0] - old[0]))), float(np.max(np.abs(new[1] - old[1])))]
    for column in ("lead_40", "pad"):
        for q in (0.5, 0.99, 1.0):
            differences.append(abs(P.ecdf_x_limit(tables, column, q) - O.ecdf_x_limit(tables, column, q)))
    record(parity_record, "ecdf and ecdf_x_limit", differences)

    new = P.silence_ecdf_figure(tables)
    plt = O._pyplot()
    old_axes = []
    original_savefig = plt.Figure.savefig
    plt.Figure.savefig = lambda self, *a, **k: old_axes.append(self)       # keep the figure, write nothing
    try:
        O.ecdf_by_class(tables, "lead_40", "unused", xlabel="leading silence in file (s)")
    finally:
        plt.Figure.savefig = original_savefig
    old = old_axes[0]
    differences = [0.0 if text_of(new) == text_of(old) else np.inf,
                   0.0 if list(new.get_size_inches()) == list(old.get_size_inches()) else np.inf,
                   0.0 if P.series_styles(new) == P.series_styles(old) else np.inf]
    for a, b in zip(new.axes, old.axes):
        differences.append(0.0 if a.get_xlim() == b.get_xlim() and a.get_ylim() == b.get_ylim() else np.inf)
        for x, y in zip(a.lines, b.lines):
            differences += [float(np.max(np.abs(x.get_xdata() - y.get_xdata()))),
                            float(np.max(np.abs(x.get_ydata() - y.get_ydata()))),
                            0.0 if list(x.get_markevery()) == list(y.get_markevery()) else np.inf]
    record(parity_record, "silence_ecdf_figure (drawn values, text, size and styles)", differences)


def test_effects_by_layer_figure(original, parity_record):
    O = original("plots")
    effects = effects_table()
    old_table = effects[effects["effect"].isin(OLD_EFFECTS)].copy()
    old_table["fixed"] = [OLD_EFFECTS[e][1] for e in old_table["effect"]]
    old_table["effect"] = [OLD_EFFECTS[e][0] for e in old_table["effect"]]
    old_table = old_table.rename(columns={"estimate": "observed", "file_ci_low": "ci_low",
                                          "file_ci_high": "ci_high"})
    differences = []
    for ci in ("spk", "file"):
        new = P.effects_by_layer_figure(effects, ci=ci)
        old, _ = O.effects_by_layer_figure(old_table, ci=ci)
        differences += [0.0 if text_of(new) == text_of(old) else np.inf,
                        0.0 if list(new.get_size_inches()) == list(old.get_size_inches()) else np.inf,
                        0.0 if P.figure_colours(new) == P.figure_colours(old) else np.inf]
        assert len(new.axes) == len(old.axes) == 4
        for a, b in zip(new.axes, old.axes):
            for (x1, y1, lo1, hi1, n1), (x2, y2, lo2, hi2, n2) in zip(errorbars(a), errorbars(b)):
                differences += [float(np.max(np.abs(x1 - x2))), float(np.max(np.abs(y1 - y2))),
                                float(np.max(np.abs(lo1 - lo2))), float(np.max(np.abs(hi1 - hi2))),
                                0.0 if n1 == n2 else np.inf]
            for c1, c2 in zip(a.containers, b.containers):
                l1, l2 = c1.lines[0], c2.lines[0]
                same = (l1.get_linestyle() == l2.get_linestyle() and l1.get_marker() == l2.get_marker()
                        and P._hex(l1.get_color()) == P._hex(l2.get_color())
                        and P._hex(l1.get_markerfacecolor()) == P._hex(l2.get_markerfacecolor()))
                differences.append(0.0 if same else np.inf)
    record(parity_record, "effects_by_layer_figure (drawn values, text, size and styles)", differences)
