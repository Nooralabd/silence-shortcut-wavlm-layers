"""datasets against the original protocol and metadata readers.
Skipped without the original code."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ssl_silence import datasets as D


def same(new, old) -> float:
    """0.0 if two values are identical, else inf."""
    if isinstance(new, pd.DataFrame):
        ok = isinstance(old, pd.DataFrame) and list(new.columns) == list(old.columns) \
            and list(new.dtypes) == list(old.dtypes) and new.equals(old)
    elif isinstance(new, np.ndarray):
        ok = isinstance(old, np.ndarray) and new.dtype == old.dtype and np.array_equal(new, old)
    elif isinstance(new, tuple):
        ok = isinstance(old, tuple) and len(new) == len(old) \
            and all(same(a, b) == 0.0 for a, b in zip(new, old))
    else:
        ok = type(new) is type(old) and new == old
    return 0.0 if ok else np.inf


def compare(parity_record, name, pairs):
    worst = max(same(new, old) for new, old in pairs)
    parity_record("datasets", name, len(pairs), worst)
    assert worst == 0.0, name


def protocols(folder) -> list[str]:
    """Protocol files: regular lines, blank and short lines, tabs, repeated and
    trailing spaces, a line with a sixth field, every attack id."""
    rng = np.random.default_rng(31)
    paths = []
    for k in range(6):
        lines = []
        for j in range(400):
            spoof = rng.random() < 0.9
            attack = f"A{int(rng.integers(1, 20)):02d}" if spoof else "-"
            fields = [f"LA_{int(rng.integers(0, 120)):04d}", f"LA_X_{k}{j:07d}", "-", attack,
                      "spoof" if spoof else "bonafide"]
            separator = ("\t", " ", "   ")[int(rng.integers(0, 3))]
            line = separator.join(fields) + ("  " if j % 7 == 0 else "")
            if j % 53 == 0:
                line = separator.join(fields[:4] + ["extra", fields[4]])
            lines.append(line)
            if j % 37 == 0:
                lines.append("")
            if j % 41 == 0:
                lines.append("LA_0000 incomplete -")
        path = folder / f"protocol_{k}.txt"
        path.write_text("\n".join(lines) + ("\n" if k % 2 else ""), encoding="utf-8", newline="\n")
        paths.append(str(path))
    return paths


def test_protocol_and_asvspoof_items(original, parity_record, tmp_path):
    O = original("features")
    parsed, tables = [], []
    for k, path in enumerate(protocols(tmp_path)):
        parsed.append((D.read_asvspoof_protocol(path), O.read_asvspoof_protocol(path)))
        split = ("train", "dev", "eval")[k % 3]
        audio_dir = str(tmp_path / "audio" / split)
        tables.append((D.asvspoof_items(split, protocol=path, audio_dir=audio_dir),
                       O.asvspoof_items(split, protocol=path, flac_dir=audio_dir)))
    compare(parity_record, "read_asvspoof_protocol", parsed)
    compare(parity_record, "asvspoof_items", tables)


def write_meta(folder, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    labels = ("bona-fide", "spoof", "Bona-fide", " spoof ", "SPOOF", " bona-fide")
    with open(folder / "meta.csv", "w", newline="", encoding="utf-8") as fh:
        fh.write("file,speaker,label\n")
        for j in range(n):
            pad = " " if j % 5 == 0 else ""
            speaker = f"Speaker Number {int(rng.integers(0, 40))}"
            fh.write(f"{pad}{j}.wav{pad},{pad}{speaker}{pad},{labels[int(rng.integers(0, 6))]}\n")


def test_itw_items(original, parity_record, tmp_path):
    O = original("features")
    cases = []
    for k, sub in enumerate(("release_in_the_wild", "a/b/release", ".")):
        root = tmp_path / f"itw_{k}"
        write_meta(root / sub, 300 + k, seed=40 + k)
        cases.append(root)
    write_meta(cases[0] / "zz_second_copy", 5, seed=50)  # a second meta.csv, later in sorted order
    meta = [(D.find_itw_meta(str(root)), O.find_itw_meta(str(root))) for root in cases]
    tables = [(D.itw_items(str(root)), O.itw_items(str(root))) for root in cases]
    compare(parity_record, "find_itw_meta", meta)
    compare(parity_record, "itw_items", tables)


def test_speaker_and_attack_maps(original, parity_record, tmp_path):
    """The original code read these maps from tables written from the file lists."""
    A = original("analysis")
    eval_items = D.asvspoof_items("eval", protocol=protocols(tmp_path)[2],
                                  audio_dir=str(tmp_path / "audio"))
    eval_items = eval_items.drop_duplicates("audio_id")
    write_meta(tmp_path / "itw", 250, seed=60)
    itw = D.itw_items(str(tmp_path / "itw"))
    tables = tmp_path / "tables"
    tables.mkdir()
    eval_items.drop(columns="path").to_csv(tables / "eval.csv", index=False)
    itw.drop(columns="path").to_csv(tables / "itw.csv", index=False)
    old_speakers = A.load_speaker_maps(("eval", "itw"), measurement_dir=str(tables))
    compare(parity_record, "speaker_map",
            [(D.speaker_map(eval_items), old_speakers["eval"]),
             (D.speaker_map(itw), old_speakers["itw"])])
    compare(parity_record, "attack_map",
            [(D.attack_map(eval_items), A.load_attack_map(str(tables / "eval.csv")))])
