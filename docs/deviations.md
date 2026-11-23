# Places where the code interprets the analysis plan

The analysis plan (`docs/analysis_plan_hubert.md`, committed at `1ecf732`)
is not edited. This file lists every place where its wording allows more
than one reading, or leaves a detail open, and the code had to choose. Each
entry quotes the plan, says what the code does, and says when the choice can
change a result. All entries were written before any HuBERT Base table
existed (the analysis notebook, `notebooks/08_analysis.ipynb`, had not been
run for HuBERT Base).

None of the entries changes a threshold, a unit or the list of units in the
rule (608 of 830).

## 1. The rule for Kendall's tau

Plan, Section 6:

> A WavLM Base unit is **replicated** with HuBERT Base if the HuBERT Base
> estimate has the same sign as the WavLM Base estimate and its speaker-level
> 95% interval excludes zero. For Kendall's τ the rule is that its
> speaker-level interval excludes zero.

Plan, Section 6, Reporting (for each row of the summary, the rows of
Kendall's τ included):

> - replicated: both exclude zero, same sign;
> - both exclude zero, opposite sign;

Plan, Section 6a, row F13, column "Type":

> effect (τ interval excludes zero); the comparison between the test sets is
> descriptive

Two readings. (a) The second sentence of Section 6 replaces the general
rule for τ: the interval alone decides, whatever the sign. (b) It restates
the interval condition for τ and the sign condition of the first sentence
still holds, as the list under "Reporting" says for every row.

The plan does not say that the sign is disregarded for τ. The code follows
reading (b), the general rule: a τ unit in the rule is replicated if the
HuBERT Base τ has the same sign as the WavLM Base τ and its speaker-level
percentile interval excludes zero (`replication.compare_units`; no quantity
is treated differently). Decision of the first author, 2026-11-20, before
the run. The commit `2841d28` had reading (a); it was never run on scores of
HuBERT Base.

When it matters: only if a HuBERT Base τ is negative with an interval that
excludes zero. All 12 τ units of WavLM Base are in the rule and positive.
Under (a) such a unit would be "replicated"; under (b) it is "not
replicated". In both readings it is counted as `both_exclude_opposite_sign`
in the 2×2 classification. `replication_comparison.csv` has the sign of both
estimates of every unit (`sign`, `replica_sign`) and `same_sign`, so the
reader can apply either reading.

The paired difference τ(40 dB) − τ(threshold) is an ordinary unit: the
general rule.

## 2. Absence findings (F7 and F16)

Plan, Section 6:

> The rule applies only to units in which the WavLM Base speaker-level
> percentile interval itself excludes zero. All other units are descriptive.

Plan, Section 6a:

> A finding stated for WavLM Base as the absence of an effect is an
> **absence finding**: it is reported descriptively with the 2×2 table and
> is not counted as replicated or failed.

The first sentence is about units, the second about findings. Three of the
24 units of F7 (Δtest, 40 dB, In-the-Wild) have a WavLM Base interval that
excludes zero, so they are among the 608 units in the rule; F16 has no unit
in the rule. The plan does not say whether these three units keep a
unit-level verdict.

What the code does:

- Unit level (`replication_comparison.csv`, `replication_outcome.csv` and
  the row "all units" of `replication_findings.csv`): the three units of F7
  are treated like every other unit in the rule and get the verdict
  "replicated" or "not replicated". The count of units in the rule stays
  608, as in the committed summary.
- Finding level (`replication_findings.csv`): the rows of F7 and F16 have
  the 2×2 classification and an empty `n_replicated`
  (`replication.findings_table`). Their units, with both estimates and
  intervals, are in `replication_absence_findings.csv`.

When it matters: the total of replicated units among the 608 includes up to
three units of an absence finding. A total without them is the row "all
units" minus the replicated units of F7 in `replication_comparison.csv`.

## 3. The row "remaining units at 20 and 15 dB" of the findings table

Plan, Section 6a, after the table:

> The remaining effect units at 20 and 15 dB (Δtrain, Δtrain^C, Δtest^T at
> layers 3–5, and Δtotal on In-the-Wild at layers 1–12) are in the file and
> in the rule where the WavLM Base interval excludes zero.

These 120 units belong to no numbered finding (72 units of Δtrain, Δtrain^C
and Δtest^T at layers 3–5; 48 units of Δtotal on In-the-Wild). The plan does
not say how to show them in a table per finding.

What the code does: `replication.FINDINGS` assigns every one of the 830
units to exactly one row. The 120 units form one row, "remaining units at 20
and 15 dB", of type "effect", after F26. It is a row of the table and not a
finding of the plan. The findings F14, F25 and F27 have no units in the file
and no row.

Where the description of two findings overlaps, the unit goes to the first
finding of the plan that names its quantity: Δtest at 40 dB on In-the-Wild
is F7 (the plan says so in row F10); Δtest on the > 4 s subset is F16 and
the other five effects on that subset are F17; all six effects on the ≤ 4 s
subset are F15; Kendall's τ and the τ difference at 20 and 15 dB are F24.
`tests/test_replication.py` checks the count of units per row against the
committed units file.

When it matters: never for a unit-level verdict or for the row "all units";
only for the grouping of the counts.

## 4. In-the-Wild by duration at 20 and 15 dB

Plan, Section 5, item 5, names the duration split at 40 dB. The duration
table without intervals of WavLM Base also has rows at 20 and 15 dB, as the
original analysis did.

What the code does: for HuBERT Base, `itw_by_duration.csv` has 40 dB only.
The rows at 20 and 15 dB are written to
`tables/hubert-base/exploratory/exploratory_itw_by_duration_thresholds.csv`
and are labelled exploratory in the notebook. No unit and no finding uses
them. Kept by decision of the first author, 2026-11-20.
