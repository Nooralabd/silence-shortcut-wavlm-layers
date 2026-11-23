# Pre-registration of the analysis plan

The analysis plan for the replication with HuBERT Base and for the padding
check on WavLM Large is `docs/analysis_plan_hubert.md`. This file records
when and in which state it was committed. The plan itself is not edited.

## The commit

- Commit: `1ecf7324dfd61369a1b941cc2ce32109784150d6`
  ("docs: analysis plan for HuBERT Base with the units of the replication
  rule (pre-registration)"), committed on 2026-11-19 (21:39 +03:00) and
  pushed to `origin/main` of the GitHub repository on the same day.
- It was committed and pushed before any HuBERT Base result existed
  (statement of the first author). In the repository: the cells that
  extract HuBERT Base were added in the next commit, `6a5b49e`, and no
  feature, score or table of HuBERT Base, and no padding measurement on
  WavLM Large, existed at `1ecf732`.

## The files at that commit

| File | sha256 |
|---|---|
| `docs/analysis_plan_hubert.md` | `8aeb9f75a706614347a995312577aeadbe589918dd0def22d4fa816b6e335b22` |
| `docs/replication_units.csv` | `4adfaf955d2ccac676aa2012a5c05e7741cdecf63f97e9020323adb10a95c279` |
| `docs/replication_units_summary.csv` | `c57ca3d644a39baed62b576d10f2402cece0cd2b20852489f0319610d8afb23b` |

The sha256 are those of the files as stored in the commit (line endings LF).
`tests/test_preregistration.py` compares the three files of the working
tree with this table, so a change to any of them fails the tests.

`docs/replication_units.csv` lists the 830 units of the replication rule,
608 of them in the rule. It was written from the WavLM Base tables by
`notebooks/08_analysis.ipynb` at commit `637b4b8`, after the reproduction
checks of WavLM Base had passed; its sha256 equals the one in its meta file
(`docs/replication_units.meta.json`).

## The header line of the plan

The committed plan begins with "Status: DRAFT, not committed. To be
committed unchanged before any HuBERT Base feature, score or table exists
and before any padding measurement on WavLM Large exists." This line is a
leftover from drafting. The content of the file committed at `1ecf732` is
the pre-registered plan; the line was left in place so that the file keeps
the sha256 above.
