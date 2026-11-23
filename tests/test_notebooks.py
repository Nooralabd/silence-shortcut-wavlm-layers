"""The notebooks are thin runners: static checks that can be made without Colab.

A notebook is saved without outputs, imports only what the package has, and
calls only functions that exist. What a notebook does on Drive is seen by
the first author before it is run.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import re

import pytest

from conftest import REPO
from ssl_silence import config as cfg

NOTEBOOKS = [REPO / name for name in cfg.NOTEBOOKS if (REPO / name).exists()]
MAGIC = re.compile(r"^\s*[!%]")


def cells(path, kind: str) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == kind]


def python_of(source: str) -> str:
    """A code cell without its shell and magic lines (kept as ``pass`` so that blocks stay valid)."""
    lines = []
    for line in source.splitlines():
        indent = line[:len(line) - len(line.lstrip())]
        lines.append(f"{indent}pass" if MAGIC.match(line) else line)
    return "\n".join(lines)


def test_the_first_two_notebooks_exist():
    found = [path.relative_to(REPO).as_posix() for path in NOTEBOOKS[:2]]
    assert found == ["notebooks/01_environment_check.ipynb",
                     "notebooks/02_migrate_wavlm_features.ipynb"] == list(cfg.NOTEBOOKS[:2])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_saved_without_outputs(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert cells(path, "markdown")[0].startswith("# ")
    assert path.stat().st_size < 200_000


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_parses_and_calls_what_the_package_has(path):
    modules: dict[str, object] = {}
    for source in cells(path, "code"):
        tree = ast.parse(python_of(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("ssl_silence"):
                package = importlib.import_module(node.module)
                for alias in node.names:
                    name = alias.asname or alias.name
                    modules[name] = (getattr(package, alias.name) if hasattr(package, alias.name)
                                     else importlib.import_module(f"{node.module}.{alias.name}"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in modules):
                assert hasattr(modules[node.value.id], node.attr), \
                    f"{path.name}: {node.value.id}.{node.attr} does not exist"
    assert {"cfg", "verify", "provenance"} <= set(modules)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs_at_a_fixed_commit_and_reads_the_mapping_from_drive(path):
    code = "\n".join(cells(path, "code"))
    assert "assert len(COMMIT) >= 7" in code and "HEAD = provenance.git_commit()" in code
    assert 'assert HEAD.startswith(COMMIT) and not HEAD.endswith("-dirty")' in code
    assert "drive.mount" in code and "os.path.ismount" in code
    assert "verify.load_map(MAP_PATH)" in code and "verify.write_check(" in code
    # No token is written into the notebook; it comes from the secrets of the runtime.
    assert 'userdata.get("GITHUB_TOKEN")' in code and "ghp_" not in code and "github_pat_" not in code
    # Nothing is deleted or moved, and nothing is written under the original folders.
    for word in ("shutil.rmtree", "os.remove", "os.rename", "shutil.move", "rm -", "to_csv(verify.legacy_path"):
        assert word not in code, word


def test_the_migration_writes_only_under_the_new_root():
    code = "\n".join(cells(REPO / "notebooks/02_migrate_wavlm_features.ipynb", "code"))
    targets = re.findall(r"(?:copy_files|copy_feature_dir|copy_padded_layer|copy_original_files)\(([^\n]*)", code)
    assert len(targets) == 4
    assert "cfg.require_free_space(quota=QUOTA)" in code
    assert code.index("QUOTA = cfg.drive_quota()") < code.index("cfg.require_free_space(quota=QUOTA)") \
        < code.index("copy_files(")
    assert all("cfg.ROOT" in call or "cfg.ROOT" in code[code.index(call):code.index(call) + 400]
               for call in targets)
    # Every source is read through the mapping file.
    assert code.count("verify.legacy_path(legacy,") >= 4


# Installation: torch, torchaudio and torchcodec are the builds of the original features (the
# pins, CUDA 13.0). Whatever is not as pinned is installed in one pass, followed by one restart;
# on the run after it nothing is installed.

FAMILY = {"torch": "2.11.0+cu130", "torchaudio": "2.11.0+cu130", "torchcodec": "0.17.0+cu130"}
HEAVY = ("torch", "torchaudio", "torchcodec", "transformers", "ssl_silence")


def install_cell(path) -> str:
    found = [source for source in cells(path, "code") if "pip install" in source]
    assert len(found) == 1, f"{path.name}: one cell installs, found {len(found)}"
    return found[0]


def pinned_versions() -> dict[str, str]:
    """The installed versions of a runtime in which everything is as pinned."""
    lines = [line.strip() for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    from importlib.metadata import version

    found = {}
    for line in lines:
        name = re.split(r"[=<>!~ ]", line, maxsplit=1)[0]
        found[name] = line.split("==")[1] if "==" in line else version(name)   # not pinned: as installed
    return {**found, **FAMILY, "ssl-silence": "0.1.0"}


def run_install_cell(path, tmp_path, monkeypatch, before: dict, after: dict | None = None,
                     hide_imports: bool = True) -> dict:
    """Run the Python of the install cell; the pip lines are skipped.

    ``before`` are the installed versions (None: not installed; a package
    that is not named has its real version); ``after`` are the versions once
    pip has run, that is, once the cell has written its constraint file
    (default: unchanged).
    """
    import importlib.metadata as metadata
    import sys

    workdir = tmp_path / "repo"
    workdir.mkdir(exist_ok=True)
    (workdir / "requirements.txt").write_text((REPO / "requirements.txt").read_text(encoding="utf-8"),
                                              encoding="utf-8")
    constraint = tmp_path / "torch_constraint.txt"

    real = metadata.version

    def version(name):
        versions = after if (after is not None and constraint.exists()) else before
        if name not in versions:
            return real(name)
        if versions[name] is None:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(metadata, "version", version)
    if hide_imports:        # a fresh kernel has imported none of them
        for name in HEAVY:
            monkeypatch.delitem(sys.modules, name, raising=False)
    names = {"os": importlib.import_module("os"), "WORKDIR": str(workdir)}
    exec(compile(python_of(install_cell(path)), path.name, "exec"), names)
    return names


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_the_install_cell_is_the_same_everywhere_and_installs_in_one_pass(path):
    source = install_cell(path)
    assert source == install_cell(NOTEBOOKS[0]), "the install cell is the same in every notebook"
    pip = [line for line in source.splitlines() if MAGIC.match(line)]
    assert [line.strip() for line in pip] == [
        "!pip uninstall -q -y torchvision",
        "!pip install -q {FAMILY_PINS} --index-url {TORCH_INDEX}",
        "!pip install -q -r {OTHERS} -c {CONSTRAINT}",
        "!pip install -q --no-deps -e ."]
    # The family is installed with its build in the pin: pip keeps torch 2.11.0+cpu for "torch==2.11.0".
    assert 'FAMILY_PINS = " ".join(f"{name}=={BUILDS[name]}" for name in TORCH_FAMILY)' in source
    assert "=={pins[name]}\" for name in TORCH_FAMILY" not in source
    assert 'TORCH_INDEX = "https://download.pytorch.org/whl/cu130"' in source
    assert 'TORCH_FAMILY = ("torch", "torchaudio", "torchcodec")' in source
    # Every pip line is inside the branch that ends in the restart; the imports come after it.
    assert all(line.startswith("    ") for line in pip)
    order = [source.index(text) for text in (
        "if different:", pip[0], pip[1], pip[2], pip[3], "raise RuntimeError(",
        "for name, module in LOADED.items():", "\nimport torch\n", "import torchaudio\n",
        "import transformers\n", "torch.version.cuda")]
    assert order == sorted(order)
    before_restart = source[:source.index("raise RuntimeError(")]
    assert not re.search(r"^\s*(import|from) (torch|torchaudio|torchcodec|transformers|ssl_silence)\b",
                         before_restart, flags=re.M)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.parametrize("change", [{"torch": "2.14.1+cu130"}, {"torch": "2.11.0+cu126"}, {"torch": "2.11.0"},
                                    {"torch": "2.11.0+cpu", "torchaudio": "2.11.0+cpu", "torchcodec": "0.17.0+cpu"},
                                    {"torchcodec": None}, {"torchaudio": "2.14.1+cu130"}, {"numpy": "2.0.2"},
                                    {"scipy": "1.18.1"}, {"ssl-silence": None}, {"pyyaml": None}])
def test_whatever_is_not_as_pinned_is_installed_in_one_pass_then_one_restart(path, tmp_path, monkeypatch, change):
    pinned = pinned_versions()
    before = {**pinned, **change}
    with pytest.raises(RuntimeError, match="Runtime > Restart session, then run the notebook again from the top"):
        run_install_cell(path, tmp_path, monkeypatch, before, after=pinned)
    assert (tmp_path / "installed_once.txt").exists()
    # The three builds are constraints of the step that installs the other requirements.
    assert (tmp_path / "torch_constraint.txt").read_text(encoding="utf-8").split() == [
        "torch==2.11.0+cu130", "torchaudio==2.11.0+cu130", "torchcodec==0.17.0+cu130"]
    others = (tmp_path / "requirements_without_torch.txt").read_text(encoding="utf-8").split()
    assert not [line for line in others if line.split("==")[0] in FAMILY]
    assert "numpy==2.1.3" in others and "scipy==1.16.3" in others and "soundfile==0.14.0" in others
    assert len(others) == len(pinned) - len(FAMILY) - 1


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_after_the_restart_nothing_is_installed(path, tmp_path, monkeypatch, capsys):
    (tmp_path / "installed_once.txt").write_text("installed", encoding="utf-8")
    names = run_install_cell(path, tmp_path, monkeypatch, pinned_versions(), hide_imports=False)
    assert names["different"] == {}
    assert not (tmp_path / "torch_constraint.txt").exists()
    assert not (tmp_path / "requirements_without_torch.txt").exists()
    printed = capsys.readouterr().out
    assert "CUDA of the build:" in printed and "torchaudio" in printed and "torchcodec 0.17.0+cu130" in printed
    assert {"torch", "torchaudio", "transformers"} <= set(names)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_there_is_no_second_restart(path, tmp_path, monkeypatch):
    pinned = pinned_versions()
    # pip did not install what is pinned: stop, and no restart is asked for.
    with pytest.raises(AssertionError, match="pip did not install these as pinned"):
        run_install_cell(path, tmp_path, monkeypatch, {**pinned, "torch": "2.14.1+cu130"})
    assert not (tmp_path / "installed_once.txt").exists()
    # After one installation and restart, a remaining difference stops without another installation.
    (tmp_path / "installed_once.txt").write_text("installed", encoding="utf-8")
    (tmp_path / "torch_constraint.txt").unlink()
    with pytest.raises(AssertionError, match="do not restart again"):
        run_install_cell(path, tmp_path, monkeypatch, {**pinned, "torch": "2.14.1+cu130"})
    assert not (tmp_path / "torch_constraint.txt").exists()


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_nothing_may_be_imported_before_the_installation(path, tmp_path, monkeypatch):
    import torch  # noqa: F401  (imported in this process, as in a kernel that ran another cell first)

    with pytest.raises(AssertionError, match="were imported before the installation"):
        run_install_cell(path, tmp_path, monkeypatch, {**pinned_versions(), "torch": "2.14.1+cu130"},
                         hide_imports=False)
    assert not (tmp_path / "torch_constraint.txt").exists()


# The three rules of the install cell (a kernel that imported numpy before pip replaced it
# kept the other numpy loaded, and scipy and pandas then failed).

@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_rule_1_the_kernel_never_continues_after_pip(path):
    """Every pip line is in one block whose last statement raises; nothing after the block installs."""
    source = install_cell(path)
    marked = "\n".join(f"{line[:len(line) - len(line.lstrip())]}PIP = {line.strip()!r}" if MAGIC.match(line) else line
                       for line in source.splitlines())
    tree = ast.parse(marked)
    is_pip = lambda node: (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                           and node.targets[0].id == "PIP")
    blocks = [node for node in tree.body if any(is_pip(inner) for inner in ast.walk(node))]
    assert len(blocks) == 1 and isinstance(blocks[0], ast.If)
    block = blocks[0]
    assert sum(is_pip(node) for node in block.body) == 4 and not block.orelse      # no pip in a nested branch
    assert isinstance(block.body[-1], ast.Raise)
    last_pip = max(k for k, node in enumerate(block.body) if is_pip(node))
    # Between the last pip line and the raise there is no branch, loop or import: only the checks.
    for node in block.body[last_pip + 1:-1]:
        assert isinstance(node, (ast.Expr, ast.Assign, ast.Assert, ast.With)), ast.dump(node)[:80]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.parametrize("change", [{"pyyaml": None}, {"tqdm": None}, {"ssl-silence": None}, {"pandas": "2.2.2"}])
def test_rule_1_whatever_was_installed_the_cell_stops_for_a_restart(path, tmp_path, monkeypatch, change):
    pinned = pinned_versions()
    with pytest.raises(RuntimeError, match="Runtime > Restart session, then run the notebook again from the top"):
        names = run_install_cell(path, tmp_path, monkeypatch, {**pinned, **change}, after=pinned)
        assert "LOADED" not in names


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.parametrize("module, name", [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
                                          ("sklearn", "scikit-learn"), ("torch", "torch")])
def test_rule_2_a_loaded_module_of_another_version_stops(path, tmp_path, monkeypatch, module, name):
    """The files on disk are as pinned, but the kernel has loaded another version."""
    loaded = importlib.import_module(module)
    assert loaded.__version__.split("+")[0] == pinned_versions()[name].split("+")[0]
    monkeypatch.setattr(loaded, "__version__", "2.4.6")
    with pytest.raises(AssertionError, match=f"{module} 2.4.6 is loaded in this kernel and the pin is .*"
                                             "Runtime > Restart session"):
        run_install_cell(path, tmp_path, monkeypatch, pinned_versions(), hide_imports=False)
    assert not (tmp_path / "torch_constraint.txt").exists()         # nothing was installed


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_rule_2_the_loaded_versions_are_checked_and_printed(path, tmp_path, monkeypatch, capsys):
    names = run_install_cell(path, tmp_path, monkeypatch, pinned_versions(), hide_imports=False)
    assert list(names["LOADED"].items())[:5] == [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
                                                 ("scikit-learn", "sklearn"), ("torch", "torch")]
    assert "loaded: {'numpy': '2.1.3', 'scipy': '1.16.3', 'pandas': '2.2.3'" in capsys.readouterr().out


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.parametrize("left", [{"numpy": "2.4.6"}, {"scipy": "1.18.1"}, {"torch": "2.14.1+cu130"},
                                  {"torchcodec": None}, {"soundfile": "0.13.1"}])
def test_rule_3_no_restart_unless_everything_on_disk_is_as_pinned(path, tmp_path, monkeypatch, left):
    """pip ran, but one package is still not as pinned: the cell stops without asking for a restart."""
    pinned = pinned_versions()
    with pytest.raises(AssertionError, match="pip did not install these as pinned") as failure:
        run_install_cell(path, tmp_path, monkeypatch, {**pinned, **left, "pandas": "2.2.2"},
                         after={**pinned, **left})
    assert list(left)[0] in str(failure.value) and "pandas" not in str(failure.value)
    assert "Restart session" not in str(failure.value)
    assert not (tmp_path / "installed_once.txt").exists()


def test_the_first_notebook_shows_the_output_of_pytest():
    code = "\n".join(cells(REPO / "notebooks/01_environment_check.ipynb", "code"))
    assert code.count("capture_output=True, text=True") >= 3          # git and both pytest runs
    assert code.count("print(result.stdout + result.stderr)") == 2
    order = [code.index(text) for text in (
        '"-m", "pytest", "-q", "-rfs"], env=env, capture_output=True, text=True)',
        'PYTEST_OUTPUT = "/content/pytest_output.txt"', 'record("test_suite"',
        'pd.read_csv("results/checks/parity_original_code.csv")')]
    assert order == sorted(order)


def test_the_migration_displays_the_commit_recorded_in_the_padded_shards():
    code = "\n".join(cells(REPO / "notebooks/02_migrate_wavlm_features.ipynb", "code"))
    assert '("padded", names["padded"]["folders"])' in code and '("unpadded", names["unpadded"])' in code
    assert 'record("original_shard_provenance_all", recorded, stop=False)' in code
    assert 'display(recorded[recorded["key"] == "git_commit"])' in code
    assert code.index("original_shard_provenance_all") < code.index("copy_feature_dir(")


# Clone: a short or full commit; a token is used if there is one and is never printed.

TOKEN = "secret-token-value"


class Secrets:
    """Stand-in for the secrets of the runtime."""

    def __init__(self, value):
        self.value = value

    def get(self, name):
        assert name == "GITHUB_TOKEN"
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def run_clone_cell(path, monkeypatch, commit: str, secret, returncode: int = 0) -> list[list[str]]:
    """Run the clone cell with a stand-in for git; returns the git commands."""
    import os
    import subprocess

    found = [source for source in cells(path, "code") if source.lstrip().startswith("import subprocess\n\nREPOSITORY")]
    assert len(found) == 1
    source = python_of(found[0]).replace('COMMIT = ""', f'COMMIT = "{commit}"')
    calls: list[list[str]] = []

    def run(command, **options):
        calls.append(list(command))
        text = f"fatal: could not read from https://{TOKEN}@github.com/x.git" if returncode else ""
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr=text)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(os.path, "isdir", lambda folder: False)
    try:
        exec(compile(source, path.name, "exec"), {"os": os, "userdata": Secrets(secret)})
    finally:
        run_clone_cell.calls = calls
    return calls


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_clone_needs_a_commit_before_anything_is_cloned(path, monkeypatch):
    for commit in ("", "abc12", "not-a-hash"):
        with pytest.raises(AssertionError, match="set COMMIT"):
            run_clone_cell(path, monkeypatch, commit, TOKEN)
        assert run_clone_cell.calls == []


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.parametrize("secret", [None, "", KeyError("no such secret")], ids=["none", "empty", "missing"])
def test_clone_of_a_public_repository_needs_no_token(path, monkeypatch, capsys, secret):
    """Without the secret the repository is cloned from its public address."""
    calls = run_clone_cell(path, monkeypatch, "abc1234", secret)
    assert calls[0][:3] == ["git", "clone", "-q"]
    address = calls[0][3]
    assert address.startswith("https://github.com/") and address.endswith(".git") and "@" not in address
    assert calls[1][3:6] == ["remote", "set-url", "origin"] and calls[1][-1] == address
    assert calls[2][3:] == ["checkout", "-q", "abc1234"]
    # What git prints is printed as it is: nothing is replaced when there is no token.
    with pytest.raises(AssertionError, match="git failed"):
        run_clone_cell(path, monkeypatch, "abc1234", secret, returncode=1)
    printed = capsys.readouterr().out
    assert "fatal: could not read from" in printed and "***" not in printed


def test_the_original_code_is_cloned_only_with_a_token():
    """The original analysis code is not public: the first notebook stops before its clone without a token."""
    code = "\n".join(cells(REPO / "notebooks/01_environment_check.ipynb", "code"))
    assert code.index('assert token, ("the original analysis code is not public') \
        < code.index('clone(original["repository"], ORIGINAL)')


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.parametrize("commit", ["abc1234", "abc1234" + "0" * 33])
def test_clone_checks_out_the_commit_and_drops_the_token(path, monkeypatch, capsys, commit):
    calls = run_clone_cell(path, monkeypatch, commit, TOKEN)
    assert [command[:2] for command in calls[:1]] == [["git", "clone"]]
    assert sum(TOKEN in " ".join(command) for command in calls) == 1          # the clone only
    assert calls[1][3:6] == ["remote", "set-url", "origin"] and TOKEN not in calls[1][-1]
    assert calls[2][3:] == ["checkout", "-q", commit]
    assert TOKEN not in capsys.readouterr().out


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_git_never_prints_the_token(path, monkeypatch, capsys):
    with pytest.raises(AssertionError, match="git failed") as failure:
        run_clone_cell(path, monkeypatch, "abc1234", TOKEN, returncode=1)
    printed = capsys.readouterr()
    assert "***@github.com" in printed.out and TOKEN not in printed.out + printed.err
    assert TOKEN not in str(failure.value)
    # No shell line of a notebook holds the token.
    for source in cells(path, "code"):
        assert not any(MAGIC.match(line) and "token" in line for line in source.splitlines())


def test_the_migration_compares_the_copied_tables_with_the_paper_archive():
    code = "\n".join(cells(REPO / "notebooks/02_migrate_wavlm_features.ipynb", "code"))
    assert 'ARCHIVE = os.path.join(cfg.migration_dir(), "paper_tables.zip")' in code
    order = [code.index(text) for text in (
        "assert os.path.exists(ARCHIVE)", "migration.archive_hashes(ARCHIVE)", "copy_files(",
        "migration.copy_original_files(",
        'record("original_tables_against_archive", migration.compare_with_archive(legacy, target, cfg.ROOT, ARCHIVE))')]
    assert order == sorted(order)


def test_the_first_notebook_says_when_to_upload_the_mapping_file():
    path = REPO / "notebooks/01_environment_check.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    k = next(i for i, source in enumerate(sources) if source.startswith("MAP_PATH = "))
    assert "On the first run this cell stops: upload `legacy_map.yaml` to\n`<root>/migration/`" in sources[k - 1]
    assert "cfg.makedirs(folder)" in sources[k - 2]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_free_space_is_the_drive_quota_and_the_local_disk_is_reported(path):
    code = "\n".join(cells(path, "code"))
    assert code.count("QUOTA = cfg.drive_quota()") == 1 and "cfg.require_free_space()" not in code
    assert code.index("QUOTA = cfg.drive_quota()") < code.index("cfg.require_free_space(quota=QUOTA)") \
        < code.index("cfg.local_free_space_gb()")
    assert "cfg.free_space_gb(" not in code and "disk_usage" not in code
    if path.name.startswith("01"):
        assert "provenance.environment_report(cfg.ROOT, quota=QUOTA)" in code


# Extraction notebook: re-extraction check of WavLM Base and extraction of HuBERT Base ------------

EXTRACT = REPO / "notebooks/03_extract_features.ipynb"


def test_the_extraction_notebook_extracts_one_condition_of_hubert_after_the_short_check():
    assert EXTRACT in NOTEBOOKS
    sources = cells(EXTRACT, "code")
    code = "\n".join(sources)
    # As saved, a run does the short check only: nothing is extracted and WavLM Base is not loaded.
    assert 'CONDITION = ""  ' in code and "WAVLM_CHECK = False  " in code
    assert 'ENCODER = "hubert-base"' in code and 'REFERENCE = "wavlm-base"' in code
    assert "assert CONDITION == \"\" or CONDITION in cfg.conditions()" in code
    assert "assert torch.cuda.is_available()" in code
    assert "datasets.ensure_itw(cfg.itw_zip(), sha256=cfg.ITW_ZIP_SHA256)" in code
    # Shards are written by extract_shards only, once, for the extracted encoder, unpadded.
    for word in ("write_shard(", "copy_files(", "copy_feature_dir(", "torch.save("):
        assert word not in code, word
    assert code.count("extract_shards(") == 1
    call = code[code.index("extraction.extract_shards("):]
    call = call[:call.index("\n        print(")]
    assert "encoder=ENCODER" in call and 'extraction="unpadded"' in call and "condition=CONDITION" in call
    assert "model=model" in call and "shard_size" not in call and "max_new_shards" not in call
    assert code.count("cfg.feature_dir(ENCODER,") == 1 and 'cfg.feature_dir(ENCODER, "unpadded", CONDITION)' in code
    order = [code.index(text) for text in (
        "datasets.ensure_asvspoof(", "extraction.reextraction_check(", "model = encoders.load_encoder(ENCODER, DEVICE)",
        'assert described["feat_extract_norm"] == "group" and described["n_layers"] == 12',
        "extraction.file_list_check(", "extraction.short_check(", "extraction.extraction_estimate(",
        'verify.require_ok(short, "short check of " + ENCODER)', "extraction.extract_shards(",
        "extraction.lengths_check(", "extraction.shard_meta_check(", 'print("extraction finished:"')]
    assert order == sorted(order)
    # The number of files, the generator stream and the tolerance are those of the package.
    for name in ("extraction.reextraction_check(", "extraction.short_check("):
        call = code[code.index(name):]
        call = call[:call.index("\n", call.index("device=DEVICE"))]
        assert " n=" not in call and "seed=" not in call and "tolerance=" not in call
    # Everything that writes shards or loads WavLM Base is inside its switch.
    for source in sources:
        for word, switch in (("extract_shards(", "if CONDITION:"), ("lengths_check(", "if CONDITION:"),
                             ("load_encoder(REFERENCE", "if WAVLM_CHECK:"),
                             ("reextraction_check(", "if WAVLM_CHECK:")):
            if word in source:
                line = next(line for line in source.splitlines() if word in line)
                assert line.startswith("    ") and switch in source[:source.index(word)]
    text = "\n".join(cells(EXTRACT, "markdown"))
    assert "T4" in text and "one condition per run" in text and "resumable" in text
    assert "No shard of WavLM Base is written, changed or deleted" in text


def run_the_check_cells(monkeypatch, failing: tuple = ()):
    """The comparison and summary cells of the re-extraction check with a stand-in for the check."""
    import types

    import pandas as pd

    from ssl_silence import verify

    calls = []

    def reextraction_check(items, folder, split, *, model, extraction, condition, device):
        assert model == "wavlm"
        calls.append((extraction, condition, split, folder, items))
        ok = (extraction, condition, split) not in failing
        return pd.DataFrame([{"split": split, "key": "mean", "max_abs_diff": 0.0 if ok else 1.0,
                              "exactly_zero": ok, "ok": ok}])

    checks = {"extraction_other": pd.DataFrame({"ok": [True]})}
    namespace = {
        "cfg": cfg, "verify": verify, "pd": pd, "checks": checks, "CHECKS": "checks", "reference_model": "wavlm",
        "REFERENCE": "wavlm-base", "WAVLM_CHECK": True, "DEVICE": "cuda", "reference_entry": {"revision": "r"},
        "display": lambda table: None,
        "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
        "legacy": {"features": {"wavlm-base": {"padded": {"folders": {"clean": "a", "trim40": "b"}}}}},
        "file_lists": {split: split for split in cfg.SPLITS},
        "extraction": types.SimpleNamespace(reextraction_check=reextraction_check),
        "record": lambda name, table, stop=True: checks.__setitem__(name, table) or stop and pytest.fail("stops early"),
    }
    sources = cells(EXTRACT, "code")
    k = next(i for i, source in enumerate(sources) if "extraction.reextraction_check(" in source)
    try:
        for source in sources[k:k + 2]:
            exec(compile(python_of(source), EXTRACT.name, "exec"), namespace)
    finally:
        namespace["calls"] = calls
    return namespace


def test_every_condition_and_split_is_checked(monkeypatch, capsys):
    namespace = run_the_check_cells(monkeypatch)
    calls, checks = namespace["calls"], namespace["checks"]
    expected = [("unpadded", c) for c in ("clean", "trim40", "trim20", "trim15")] + \
               [("padded", "clean"), ("padded", "trim40")]
    assert [call[:3] for call in calls] == [(kind, c, split) for kind, c in expected for split in cfg.SPLITS]
    assert all(call[3] == cfg.feature_dir("wavlm-base", call[0], call[1]) and call[4] == call[2] for call in calls)
    assert list(checks)[1:] == [f"reextraction_{kind}_{c}" for kind, c in expected]
    assert all(len(table) == 4 for table in list(checks.values())[1:])
    assert "re-extraction check passed: wavlm-base" in capsys.readouterr().out
    assert "reference_model" not in namespace               # released before HuBERT Base is loaded


def test_one_failing_row_stops_after_all_tables_were_written(monkeypatch, capsys):
    from ssl_silence import verify

    with pytest.raises(verify.CheckFailed):
        run_the_check_cells(monkeypatch, failing=(("unpadded", "trim40", "eval"),))
    assert "re-extraction check passed" not in capsys.readouterr().out


def test_without_the_switch_the_wavlm_cells_do_nothing(capsys):
    sources = cells(EXTRACT, "code")
    k = next(i for i, source in enumerate(sources) if "extraction.reextraction_check(" in source)
    namespace = {"WAVLM_CHECK": False}
    for source in sources[k:k + 2]:
        exec(compile(python_of(source), EXTRACT.name, "exec"), namespace)
    assert set(namespace) - {"__builtins__"} == {"WAVLM_CHECK"} and capsys.readouterr().out == ""


HUBERT: dict = {}


def run_the_hubert_cells(monkeypatch, tmp_path, condition="", failing=()):
    """The cells of HuBERT Base (load, short check, extraction, written shards) with stand-ins."""
    import os
    import types

    import pandas as pd

    from ssl_silence import extraction as real, shards as real_shards, verify

    calls = {"short": [], "lists": [], "extract": [], "lengths": [], "read": []}
    registry = cfg.encoder("hubert-base")

    def feature_dir(encoder, kind, condition_name):
        folder = tmp_path / encoder / kind / condition_name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "train_shard_0000.pt").write_bytes(b"0" * (2_000 if encoder == "wavlm-base" else 3_000))
        return str(folder)

    monkeypatch.setattr(cfg, "feature_dir", feature_dir)
    monkeypatch.setattr(cfg, "drive_quota", lambda: {"free_gb": 200.0})

    def file_list_check(items, reference, split):
        calls["lists"].append((reference, split))
        return pd.DataFrame([{"split": split, "n_files": len(items), "ok": "lists" not in failing}])

    def short_check(items, reference, split, *, model, condition, device):
        assert model == "hubert" and device == "cuda" and reference == ("lengths", feature_dir("wavlm-base", "unpadded", condition))
        calls["short"].append((condition, split))
        n = cfg.SHORT_CHECK_FILES
        return pd.DataFrame({"split": split, "condition": condition, "n_frames": [199] * n,
                             "conv_frames": [199] * n, "seconds": 0.5,
                             "ok": [True] * (n - 1) + ["short" not in failing]})

    def extract_shards(items, folder, split, **keywords):
        calls["extract"].append((folder, split, keywords))
        return ["a", "b"]

    def lengths_check(new, reference):
        calls["lengths"].append((new, reference))
        return pd.DataFrame({"split": list(cfg.SPLITS), "n_fallback": [1, 2, 3, 4], "ok": "lengths" not in failing})

    def shard_meta_check(table, encoder, condition_name):
        assert table == ("meta", feature_dir("hubert-base", "unpadded", condition_name)) and encoder == "hubert-base"
        return pd.DataFrame({"git_commit": ["c1", "c2"], "n_shards": [3, 5], "n_files": [10, 37 if "files" in failing else 36],
                             "ok": [True, "meta" not in failing]})

    checks: dict = {}
    namespace = {
        "cfg": cfg, "verify": verify, "pd": pd, "os": os, "checks": checks, "CHECKS": "checks", "HEAD": "abc",
        "ENCODER": "hubert-base", "REFERENCE": "wavlm-base", "CONDITION": condition, "DEVICE": "cuda",
        "display": lambda table: None,
        "file_lists": {split: list(range(k + 10)) for k, split in enumerate(cfg.SPLITS)},
        "provenance": types.SimpleNamespace(gpu_name=lambda device: "Tesla T4", versions=lambda: {"torch": "2.11.0+cu130"}),
        "encoders": types.SimpleNamespace(
            load_encoder=lambda name, device: {"hubert-base": "hubert"}[name],
            describe=lambda model: {key: registry[key] for key in ("feat_extract_norm", "n_layers", "hidden_size",
                                                                   "attn_implementation")}),
        "shards": types.SimpleNamespace(shard_lengths=lambda folder: ("lengths", folder), n_shards=real_shards.n_shards),
        "extraction": types.SimpleNamespace(
            file_list_check=file_list_check, short_check=short_check, extraction_estimate=real.extraction_estimate,
            extract_shards=extract_shards, lengths_check=lengths_check, shard_meta_check=shard_meta_check,
            shard_meta_table=lambda folder: ("meta", folder)),
        "record": lambda name, table, stop=True: checks.__setitem__(name, table) or stop and pytest.fail("stops early"),
    }
    sources = cells(EXTRACT, "code")
    k = next(i for i, source in enumerate(sources) if "encoders.load_encoder(ENCODER, DEVICE)" in source)
    assert k == len(sources) - 4
    HUBERT.clear()
    HUBERT.update(calls=calls, checks=checks)
    for source in sources[k:]:
        exec(compile(python_of(source), EXTRACT.name, "exec"), namespace)
    return namespace


def test_an_empty_condition_is_the_short_check_of_all_conditions_and_extracts_nothing(monkeypatch, tmp_path, capsys):
    namespace = run_the_hubert_cells(monkeypatch, tmp_path)
    calls, checks, out = HUBERT["calls"], HUBERT["checks"], capsys.readouterr().out
    conditions = ["clean", "trim40", "trim20", "trim15"]
    assert calls["short"] == [(c, split) for c in conditions for split in cfg.SPLITS]
    assert len(calls["lists"]) == 16 and not calls["extract"] and not calls["lengths"]
    assert list(checks) == [f"extraction_hubert-base_{name}_all" for name in ("encoder", "file_lists", "short_check",
                                                                           "estimate")]
    encoder = checks["extraction_hubert-base_encoder_all"].iloc[0]
    assert encoder["revision"] == cfg.encoder("hubert-base")["revision"] and encoder["git_commit"] == "abc"
    assert encoder["gpu"] == "Tesla T4" and encoder["torch"] == "2.11.0+cu130" and encoder["feat_extract_norm"] == "group"
    estimate = checks["extraction_hubert-base_estimate_all"]
    assert estimate["condition"].tolist() == conditions and set(estimate["n_files"]) == {10 + 11 + 12 + 13}
    assert estimate["hours"].tolist() == pytest.approx([46 * 0.5 / 3600] * 4)
    assert estimate["feature_gb"].tolist() == pytest.approx([46 * 12 * 768 * 8 / 1e9] * 4)
    assert estimate["reference_folder_gb"].tolist() == pytest.approx([2e-6] * 4)
    assert "feat_extract_norm: group | transformer layers: 12" in out
    assert "short check passed: hubert-base" in out and "files encoded: 400" in out
    assert out.count("expected for ") == 4 and "on Tesla T4" in out
    assert "CONDITION is empty: nothing is extracted in this run" in out and "extraction finished" not in out
    assert "FOLDER" not in namespace


def test_a_condition_is_checked_extracted_and_compared_with_the_wavlm_shards(monkeypatch, tmp_path, capsys):
    run_the_hubert_cells(monkeypatch, tmp_path, condition="trim20")
    calls, checks, out = HUBERT["calls"], HUBERT["checks"], capsys.readouterr().out
    assert calls["short"] == [("trim20", split) for split in cfg.SPLITS]
    folder = str(tmp_path / "hubert-base" / "unpadded" / "trim20")
    assert [(call[0], call[1]) for call in calls["extract"]] == [(folder, split) for split in cfg.SPLITS]
    assert all(call[2] == {"encoder": "hubert-base", "extraction": "unpadded", "condition": "trim20",
                           "device": "cuda", "model": "hubert"} for call in calls["extract"])
    reference = ("lengths", str(tmp_path / "wavlm-base" / "unpadded" / "trim20"))
    assert calls["lengths"] == [(("lengths", folder), reference)]
    assert list(checks) == [f"extraction_hubert-base_{name}_trim20" for name in (
        "encoder", "file_lists", "short_check", "estimate", "lengths", "shards")]
    assert "files encoded: 100" in out and out.count("expected for ") == 1
    assert ("extraction finished: hubert-base | condition: trim20 | files: 46 | shards: 8 "
            "| fallback files: 10 | folder: 0.0 GB") in out


@pytest.mark.parametrize("failing", ["short", "lists"])
def test_a_failed_short_check_stops_before_the_extraction(monkeypatch, tmp_path, capsys, failing):
    from ssl_silence import verify

    with pytest.raises(verify.CheckFailed):
        run_the_hubert_cells(monkeypatch, tmp_path, condition="clean", failing=(failing,))
    assert not HUBERT["calls"]["extract"] and "short check passed" not in capsys.readouterr().out
    assert len(HUBERT["checks"]) == 4                       # every table of the short check was written first


@pytest.mark.parametrize("failing, error", [("lengths", "CheckFailed"), ("meta", "CheckFailed"),
                                            ("files", "AssertionError")])
def test_written_shards_that_differ_are_not_a_finished_extraction(monkeypatch, tmp_path, capsys, failing, error):
    from ssl_silence import verify

    with pytest.raises(verify.CheckFailed if error == "CheckFailed" else AssertionError):
        run_the_hubert_cells(monkeypatch, tmp_path, condition="clean", failing=(failing,))
    assert len(HUBERT["calls"]["extract"]) == 4 and "extraction finished" not in capsys.readouterr().out
    assert {"extraction_hubert-base_lengths_clean", "extraction_hubert-base_shards_clean"} <= set(HUBERT["checks"])


# Training: a pilot and the grid of one encoder ------------------------------------------------

TRAIN = REPO / "notebooks/04_train.ipynb"


def test_the_training_notebook_runs_the_pilot_before_the_grid():
    assert TRAIN in NOTEBOOKS
    code = "\n".join(cells(TRAIN, "code"))
    order = [code.index(text) for text in (
        'ENCODER = "hubert-base"', "RUN_GRID = False",
        "assert torch.cuda.is_available()", "pilot = training.pilot_runs(planned[\"unpadded\"])",
        "file_lists = datasets.archive_file_lists(cfg.la_zip(), cfg.itw_zip())",
        "training.run_experiments(pilot,",
        'record("scores_pilot", verify.compare_runs(RUNS["unpadded"], legacy, run_ids=PILOT_IDS))',
        'check_runs("pilot", pilot)',
        'record(table_name("training", "estimate"), estimate, stop=False)',
        'training.run_experiments(planned["unpadded"],', 'finish("unpadded")',
        'training.run_experiments(planned["padded"],', 'finish("padded")')]
    assert order == sorted(order)
    # The grid and the padded runs are behind the switch; the padded runs need the configuration.
    assert "if RUN_GRID:\n    status = training.run_experiments(planned[\"unpadded\"]," in code
    assert 'if RUN_GRID and "padded" in KINDS:' in code and "if RUN_GRID:\n    finish(\"unpadded\")" in code
    # The tolerance is that of the package, and no existing score is replaced by hand.
    assert "tolerance=" not in code and "overwrite" not in code
    # No metric, analysis table or replication result is computed here.
    imported = set(re.findall(r"^from ssl_silence import (.+)$", code, flags=re.M))
    assert imported == {"config as cfg", "provenance, verify", "datasets, training"}
    for word in ("metrics.", "analysis.", "selection.", "replication.", "effects.", "bootstrap.", "eer", "dcf"):
        assert word not in code, word
    text = "\n".join(cells(TRAIN, "markdown"))
    assert "The notebook stops in this cell if\na pilot check fails" in text
    assert "every finished run is skipped" in text and "T4" in text
    assert "**This notebook computes no result.**" in text


TRAINING_CALLS: list = []          # (runs folder, run ids) of every call of the training loop


def run_the_training_cells(monkeypatch, encoder="wavlm-base", grid=True, failing=(), unfinished=(),
                           bad_records=()):
    """The cells of the training notebook after the mapping cell, with stand-ins for the
    training loop, the comparison with the original scores and the checks of the runs."""
    import pandas as pd
    import torch

    from ssl_silence import datasets, training, verify

    trained, compared, checks = TRAINING_CALLS, [], {}
    trained.clear()
    finished: set[str] = set()
    listed = []

    def run_experiments(planned, *, runs_dir, device, max_new_runs=None):
        assert device == "cuda"
        ids = [run.spec.run_id for run in planned]
        trained.append((runs_dir, ids))
        new = [i for i in ids if i not in finished and i not in unfinished]
        finished.update(new)
        return pd.DataFrame({"run_id": ids, "status": ["trained" if i in new else "skipped" for i in ids],
                             "seconds": [40.0 if i in new else float("nan") for i in ids]})

    def compare_runs(runs_dir, legacy, run_ids=None):
        compared.append((runs_dir, list(run_ids)))
        return pd.DataFrame([{"run_id": i, "score_set": "eval@clean", "max_abs_diff": 0.0,
                              "exactly_zero": True, "best_epoch_equal": True, "ok": i not in failing}
                             for i in run_ids])

    def score_set_check(planned, runs_dir, file_lists):
        assert file_lists is namespace["file_lists"] and runs_dir == cfg.runs_dir(encoder, "unpadded")
        return pd.DataFrame([{"run_id": run.spec.run_id, "score_set": name, "ok": run.spec.run_id not in failing}
                             for run in planned for name in (run.spec.selection_set,) + run.score_set_names])

    def run_record_check(planned, runs_dir, n_train):
        assert n_train == 3
        return pd.DataFrame([{"run_id": run.spec.run_id, "encoder": encoder, "encoder_revision": "r",
                              "git_commit": "c", "ok": run.spec.run_id not in bad_records} for run in planned])

    def archive_file_lists(la_zip, itw_zip):
        listed.append((la_zip, itw_zip))
        return {split: pd.DataFrame({"audio_id": ["a", "b", "c"], "label": [0, 1, 1]}) for split in cfg.SPLITS}

    def read_runs(runs_dir):
        return pd.DataFrame([{"run_id": i, "condition": i.split("_")[-4]} for i in sorted(finished)])

    def record(name, table, stop=True):
        checks[name] = table
        if stop and "ok" in table.columns:
            verify.require_ok(table, name)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(training, "run_experiments", run_experiments)
    monkeypatch.setattr(training, "finished_run_ids", lambda runs_dir: set(finished))
    monkeypatch.setattr(training, "read_runs", read_runs)
    monkeypatch.setattr(training, "score_set_check", score_set_check)
    monkeypatch.setattr(training, "run_record_check", run_record_check)
    monkeypatch.setattr(datasets, "archive_file_lists", archive_file_lists)
    monkeypatch.setattr(verify, "compare_runs", compare_runs)
    namespace = {"cfg": cfg, "verify": verify, "pd": pd, "checks": checks, "CHECKS": "checks", "legacy": {},
                 "display": lambda table: None, "record": record}
    sources = cells(TRAIN, "code")
    k = next(i for i, source in enumerate(sources) if source.startswith("ENCODER = "))
    assert 'ENCODER = "hubert-base" ' in sources[k] and "RUN_GRID = False " in sources[k]     # as saved
    sources[k] = sources[k].replace('ENCODER = "hubert-base" ', f'ENCODER = "{encoder}" ') \
        .replace("RUN_GRID = False ", f"RUN_GRID = {grid} ")
    try:
        for source in sources[k:]:
            exec(compile(python_of(source), TRAIN.name, "exec"), namespace)
    finally:
        namespace.update(trained=trained, compared=compared, listed=listed)
    return namespace


PILOT_IDS = ["wavlm-base_unpadded_clean_L04_stats_s42", "wavlm-base_unpadded_trim40_L04_stats_s42"]
HUBERT_PILOT = [name.replace("wavlm-base", "hubert-base") for name in PILOT_IDS]


def folders_and_counts(calls):
    return [(os.path.basename(folder), len(ids)) for folder, ids in calls]


def test_the_training_notebook_trains_the_pilot_the_grid_and_the_padded_runs(monkeypatch, capsys):
    namespace = run_the_training_cells(monkeypatch)
    trained, compared, checks = namespace["trained"], namespace["compared"], namespace["checks"]
    assert folders_and_counts(trained) == [("unpadded", 2), ("unpadded", 480), ("padded", 10)]
    assert trained[0][1] == PILOT_IDS == compared[0][1]
    assert folders_and_counts(compared) == [("unpadded", 2), ("unpadded", 480), ("padded", 10)]
    assert trained[0][0] == cfg.runs_dir("wavlm-base", "unpadded")
    # The tables of WavLM Base keep the names under which they are on Drive.
    assert list(checks) == ["training_pilot", "scores_pilot", "training_estimate", "training_unpadded",
                            "scores_unpadded", "training_padded", "scores_padded"]
    # The estimate: 478 runs of the grid are left after the pilot, at the mean time of the pilot runs.
    estimate = checks["training_estimate"].set_index(["extraction", "condition"])
    assert estimate.loc[("unpadded", "all"), ["planned", "finished", "to_train"]].tolist() == [480, 2, 478]
    assert estimate.loc[("unpadded", "all"), "seconds_per_run"] == 40.0
    assert estimate.loc[("unpadded", "all"), "hours_to_train"] == pytest.approx(478 * 40 / 3600)
    assert estimate.loc[("padded", "all"), "planned"] == 10
    assert not namespace["listed"]                 # no file lists: the original scores are the reference
    out = capsys.readouterr().out
    assert "pilot passed" in out and "gate A2 passed: wavlm-base | score sets compared: 492" in out


def test_a_failed_pilot_stops_before_the_grid(monkeypatch, capsys):
    from ssl_silence import verify

    with pytest.raises(verify.CheckFailed):
        run_the_training_cells(monkeypatch, failing=(PILOT_IDS[1],))
    out = capsys.readouterr().out
    assert "pilot passed" not in out and "gate A2 passed" not in out
    assert [ids for _, ids in TRAINING_CALLS] == [PILOT_IDS]          # the grid was never started


def test_an_incomplete_grid_or_a_failed_score_set_is_not_a_pass(monkeypatch, capsys):
    from ssl_silence import verify

    with pytest.raises(AssertionError, match="1 runs of the unpadded grid are not finished"):
        run_the_training_cells(monkeypatch, unfinished=("wavlm-base_unpadded_trim20_L07_mean_s44",))
    with pytest.raises(verify.CheckFailed):
        run_the_training_cells(monkeypatch, failing=("wavlm-base_unpadded_trim15_L12_mean_s46",))
    assert "gate A2 passed" not in capsys.readouterr().out


def test_as_saved_the_notebook_trains_the_pilot_of_hubert_and_stops(monkeypatch, capsys):
    namespace = run_the_training_cells(monkeypatch, encoder="hubert-base", grid=False)
    trained, checks = namespace["trained"], namespace["checks"]
    assert [ids for _, ids in trained] == [HUBERT_PILOT] and not namespace["compared"]
    assert trained[0][0] == cfg.runs_dir("hubert-base", "unpadded")
    assert namespace["KINDS"] == ["unpadded"] and len(namespace["listed"]) == 1
    # No table of WavLM Base is replaced: every name holds the encoder.
    assert list(checks) == ["training_hubert-base_pilot", "score_sets_hubert-base_pilot",
                            "run_records_hubert-base_pilot", "training_hubert-base_estimate"]
    # Untrimmed: 12 score sets; trimmed at 40 dB: its own condition and the untrimmed one.
    assert checks["score_sets_hubert-base_pilot"].groupby("run_id", sort=False).size().tolist() == [12, 6]
    estimate = checks["training_hubert-base_estimate"].set_index(["extraction", "condition"])
    assert estimate.loc[("unpadded", "all"), ["planned", "finished", "to_train"]].tolist() == [480, 2, 478]
    assert estimate.loc[("unpadded", "all"), "hours_to_train"] == pytest.approx(478 * 40 / 3600)
    assert estimate.index.get_level_values("extraction").unique().tolist() == ["unpadded"]
    out = capsys.readouterr().out
    assert "pilot passed" in out and "RUN_GRID is False: the grid is not started" in out
    assert "pilot finished: hubert-base | the grid was not started" in out
    assert "training finished" not in out and "gate A2" not in out


def test_the_grid_of_hubert_is_480_unpadded_runs_checked_without_original_scores(monkeypatch, capsys):
    namespace = run_the_training_cells(monkeypatch, encoder="hubert-base")
    trained, checks = namespace["trained"], namespace["checks"]
    assert folders_and_counts(trained) == [("unpadded", 2), ("unpadded", 480)]      # no padded runs
    assert not namespace["compared"]
    assert list(checks) == ["training_hubert-base_pilot", "score_sets_hubert-base_pilot",
                            "run_records_hubert-base_pilot", "training_hubert-base_estimate",
                            "training_hubert-base_unpadded", "score_sets_hubert-base_unpadded",
                            "run_records_hubert-base_unpadded"]
    # 120 untrimmed models with 12 score sets, 360 trimmed models with 6.
    assert len(checks["score_sets_hubert-base_unpadded"]) == 120 * 12 + 360 * 6 == 3600
    assert len(checks["run_records_hubert-base_unpadded"]) == 480
    out = capsys.readouterr().out
    assert ("training finished: hubert-base | runs: 480 | score sets checked: 3600 | revision: ['r'] "
            "| commits: ['c']") in out
    assert "gate A2" not in out


@pytest.mark.parametrize("kind", ["failing", "bad_records"])
def test_a_failed_pilot_check_of_hubert_stops_before_the_grid(monkeypatch, capsys, kind):
    from ssl_silence import verify

    with pytest.raises(verify.CheckFailed):
        run_the_training_cells(monkeypatch, encoder="hubert-base", **{kind: (HUBERT_PILOT[0],)})
    out = capsys.readouterr().out
    assert "pilot passed" not in out and "training finished" not in out
    assert [ids for _, ids in TRAINING_CALLS] == [HUBERT_PILOT]


def test_an_unfinished_run_or_a_failed_check_is_not_a_finished_training(monkeypatch, capsys):
    from ssl_silence import verify

    with pytest.raises(AssertionError, match="1 runs of the unpadded grid are not finished"):
        run_the_training_cells(monkeypatch, encoder="hubert-base",
                               unfinished=("hubert-base_unpadded_trim20_L07_mean_s44",))
    for kind in ("failing", "bad_records"):
        with pytest.raises(verify.CheckFailed):
            run_the_training_cells(monkeypatch, encoder="hubert-base",
                                   **{kind: ("hubert-base_unpadded_trim15_L12_mean_s46",)})
    assert "training finished" not in capsys.readouterr().out


# Analysis: the tables of one encoder; gate A3 for WavLM Base, the replication rule for HuBERT Base

ANALYSIS = REPO / "notebooks/08_analysis.ipynb"
TABLE_CALLS: list[str] = []
RESULT_ANALYSIS: dict = {}               # the check tables of the last run, also of one that stopped


def test_the_analysis_notebook_compares_the_quick_tables_first_and_stops_in_the_summary():
    assert ANALYSIS in NOTEBOOKS
    code = "\n".join(cells(ANALYSIS, "code"))
    order = [code.index(text) for text in (
        'ENCODER = "hubert-base" ', "replication.preregistration_check(WORKDIR", "analysis.planned_tables()",
        "training.finished_run_ids(RUNS[kind])", "silence.check_duration_subsets(durations)",
        "tables = analysis.ScoreTables(", "analysis.paired_resamples_check(reference, tables)",
        "tables.metrics()", 'compare("cells_2x2", tables.cells())',
        'compare("padded_vs_unpadded"', 'compare("fallback_counts"', 'compare("selected_configs_eer_ci"',
        'compare("itw_by_duration"', 'compare("effects_2x2", tables.effects())',
        "tables.rank_agreement()", 'compare("threshold_summary", tables.threshold_summary())',
        'compare("per_attack_effect", tables.per_attack_effect())',
        "record(STEPS, tables.steps_table(), stop=False)", "verify.require_ok(pd.concat(",
        '"gate A3 passed:"', "tables.matched_total_effects()", "tables.tau_differences()",
        "tables.itw_by_duration_ci()", "tables.replication_units()", "tables.rank_noise_ceiling()",
        "tables.selected_layer_tables()",
        "tables.encoder_differences(reference)", "tables.encoder_tau_differences(reference)",
        "tables.replication_outcome(REFERENCE_UNITS)", "tables.written_tables_check()",
        '"analysis finished:"')]
    assert order == sorted(order)
    # The steps of the commit at which gate A3 passed are read back; the hash is written in full.
    assert 'REUSE_FROM = ("88406e2d1f35ae39a51e033262c7e7db99774465",)' in code
    # The noise ceiling is not a table of the plan: it is computed for the reference encoder only.
    source = next(s for s in cells(ANALYSIS, "code") if "tables.rank_noise_ceiling()" in s)
    assert source.startswith("if not REPLICATION:") and code.count("rank_noise_ceiling()") == 1
    # While the tables of WavLM Base exist, no other table is computed: every other cell that calls
    # the tables of the encoder is behind TABLES, and the sha256 of the existing files are recorded
    # before any table and compared in the cell of the noise ceiling, as a hard stop.
    code_cells = cells(ANALYSIS, "code")
    k = next(i for i, text in enumerate(code_cells) if "\nENCODER = " in text)
    assert ('CEILING_ONLY = not REPLICATION and os.path.exists(os.path.join(cfg.tables_dir(REFERENCE), '
            '"replication_units.csv"))') in code_cells[k]
    assert "TABLES = not CEILING_ONLY " in code_cells[k]
    assert "GUARDED = [cfg.tables_dir(REFERENCE), CHECKS]" in code_cells[k]
    assert "BEFORE = verify.file_hashes(GUARDED, skip=OWN) if CEILING_ONLY else {}" in code_cells[k]
    for text in code_cells[k + 1:]:
        if "analysis.ScoreTables(" in text or "rank_noise_ceiling()" in text:
            continue                           # the inputs cell and the cell of the noise ceiling
        if "= tables." in text or "(tables." in text:
            assert text.startswith(("if TABLES:\n", "if REPLICATION:\n")), text[:60]
    assert "record(UNCHANGED, verify.unchanged_check(BEFORE, GUARDED, skip=OWN))      # a hard stop" in source
    assert source.index("tables.rank_noise_ceiling()") < source.index("verify.unchanged_check(")
    assert code.count("verify.unchanged_check(") == 1 and code.count("verify.file_hashes(") == 1
    assert "reuse_commits=() if REPLICATION else REUSE_FROM" in code
    # Every comparison is written without stopping; the only stop on a difference is in the summary.
    assert "record(check_name, check, stop=False)" in code
    assert code.count("verify.require_ok(") == 2                 # the shared helper and the summary
    assert "analysis.compare_with_original(name, table, legacy, ORIGINAL)" in code
    # The tolerance is that of the mapping file, and nothing is trained or extracted here.
    assert "tolerance" not in code and "run_experiments" not in code and "load_encoder" not in code
    assert "n_boot=" not in code and "seeds=" not in code and "layers=" not in code
    text = "\n".join(cells(ANALYSIS, "markdown"))
    assert "gate A3" in text and "hard stop if any row of any comparison fails" in text
    assert "**The written tables of WavLM Base are not overwritten.**" in text
    assert "No other table is\ncomputed, compared or written, and no step is read back." in text
    assert "a finished step is read back instead of computed again" in " ".join(text.split())


def test_the_analysis_notebook_separates_the_plan_from_what_is_exploratory():
    from ssl_silence import analysis, replication

    code_cells = cells(ANALYSIS, "code")
    code, text = "\n".join(code_cells), "\n".join(cells(ANALYSIS, "markdown"))
    # The pre-registered files are checked before any table of the second encoder, as a hard stop.
    k = next(i for i, source in enumerate(code_cells) if "replication.preregistration_check(" in source)
    assert "record(f\"preregistration_{ENCODER}\", replication.preregistration_check(WORKDIR, {" in code_cells[k]
    assert "stop=False" not in code_cells[k] and "tables." not in "\n".join(code_cells[:k + 1])
    assert 'os.path.join(cfg.tables_dir(REFERENCE), "replication_units.csv"): replication.PREREGISTERED_UNITS' \
        in code_cells[k]
    assert "REFERENCE_UNITS = os.path.join(WORKDIR, replication.PREREGISTERED_UNITS)" in code_cells[k]
    assert replication.PREREGISTERED_UNITS == "docs/replication_units.csv"
    assert "docs/preregistration.md" in text and "1ecf732" in text
    # The reference units are the committed file: no other units file is given to the rule.
    assert code.count("replication_outcome(") == 1 and code.count("REFERENCE_UNITS") == 3
    # Every cell that computes for the second encoder only is behind REPLICATION.
    for call in ("tables.dev_selected()", "tables.itw_by_duration_thresholds()", "tables.selected_layer_tables()",
                 "tables.encoder_differences(", "tables.replication_outcome(", "tables.written_tables_check()",
                 "analysis.paired_resamples_check("):
        source = next(s for s in code_cells if call in s)
        assert "if REPLICATION:" in source and source.index("if REPLICATION:") < source.index(call), call
    # The exploratory table: its own method, its own folder, the word in the file name, the cell and the text.
    assert "write_exploratory" not in code and code.count("tables.itw_by_duration_thresholds()") == 1
    source = next(s for s in code_cells if "itw_by_duration_thresholds()" in s)
    assert "# EXPLORATORY: not in the analysis plan" in source and '"EXPLORATORY, not in the analysis plan:"' in source
    assert "**EXPLORATORY (HuBERT Base only, not in the analysis plan):**" in text
    for name in analysis.EXPLORATORY_TABLES:
        assert f"`{analysis.EXPLORATORY}/{name}.csv`" in text
    # The tables of the rule are named in the text, and the rule is stated as in the plan.
    for name in ("replication_comparison", "replication_outcome", "replication_findings",
                 "replication_absence_findings", "encoder_difference_cells_effects",
                 "encoder_difference_rank_agreement"):
        assert f"`{name}.csv`" in text and name in analysis.PLAN_TABLES
    for words in ("608 of the 830 units", "has the same sign and its speaker-level percentile interval\n  excludes zero",
                  "Kendall's tau follows\n  the same rule", "`sign` and `replica_sign` in `replication_comparison.csv`",
                  "`docs/deviations.md`", "`replica_only` is the reverse count",
                  "neither as replicated nor as failed", "No\ncorrection for multiple comparisons"):
        assert words in text, words
    # No threshold, tolerance or selection of units in the notebook.
    assert "in_rule\"] &" not in code and "0.05" not in code and "alpha" not in code


def run_the_analysis_cells(monkeypatch, encoder="wavlm-base", failing=(), left_out=(), unfinished=(),
                           registered=True, paired=True, stray=(), root=None, disturb=None):
    """The cells of the analysis notebook from the settings cell on, with stand-ins
    for the tables and for the comparison with the original tables. The
    pre-registered units are the committed ones. ``root``: a folder that
    stands for Drive; the noise ceiling and the check tables are then written
    as files, and ``disturb`` is called while the noise ceiling is computed."""
    import pandas as pd

    from ssl_silence import analysis, datasets, replication, selection, silence, training, verify

    checks: dict = RESULT_ANALYSIS
    checks.clear()
    TABLE_CALLS.clear()
    names = analysis.original_table_names()
    kinds = [kind for kind in cfg.EXTRACTIONS if kind in cfg.experiments()["experiments"][encoder]]
    finished = {run.spec.run_id for name in {encoder, "wavlm-base"} for kind in cfg.EXTRACTIONS
                if kind in cfg.experiments()["experiments"][name]
                for run in training.plan_runs(name, kind)} - set(unfinished)
    committed = replication.read_units(str(REPO / replication.PREREGISTERED_UNITS))
    comparison = replication.compare_units(committed, committed)
    built = []

    class Tables:
        conditions, cross_layers = ("trim40", "trim20", "trim15"), {"trim40": (1, 2)}
        n_boot, seeds, detail_layers = 1000, (42,), (3, 4, 5)

        def __init__(self, name, runs_dir, out_dir, **inputs):
            self.encoder = name
            built.append(name)
            assert (runs_dir, out_dir) == (cfg.runs_dir(name, "unpadded"), cfg.tables_dir(name))
            files = {"speaker_maps", "attack_map", "duration_map", "asv"}
            if name == encoder:
                assert set(inputs) == files | {"padded_runs_dir", "reuse_commits"}
                assert inputs["reuse_commits"] == (("88406e2d1f35ae39a51e033262c7e7db99774465",)
                                                   if encoder == "wavlm-base" else ())
                assert inputs["padded_runs_dir"] == (cfg.runs_dir(name, "padded") if "padded" in kinds else None)
            else:
                assert name == "wavlm-base" and set(inputs) == files      # the reference: read only
            assert set(inputs["speaker_maps"]) == {"eval", "itw"} and inputs["attack_map"] == {"a": "A07"}
            assert inputs["duration_map"] == {"a": 1.0} and inputs["asv"] == {"dev": 1, "eval": 2}

        def __getattr__(self, name):
            def table(*arguments, **options):
                assert self.encoder == encoder, f"{name} was called on the tables of the reference encoder"
                TABLE_CALLS.append(name)
                one = pd.DataFrame({"table": [name], "n_seeds": [5]})
                if name == "fallback_counts":
                    assert arguments == ({c: cfg.feature_dir(encoder, "unpadded", c) for c in self.conditions},)
                if name == "selected_configs_eer_ci":
                    assert options == {"all_conditions": encoder != "wavlm-base"}
                if name == "itw_by_duration":
                    assert arguments == ((("trim40",),) if encoder != "wavlm-base" else ())
                if name in ("encoder_differences", "encoder_tau_differences"):
                    assert len(arguments) == 1 and arguments[0].encoder == "wavlm-base"
                if name == "replication_units":
                    return pd.DataFrame({"condition": ["trim40", "trim20"], "in_rule": [True, False]}), one
                if name == "selected_layers_outside_detail":
                    return (7,)
                if name == "selected_layer_tables":
                    return {"per_attack_effect_dev_selected_layers": one}
                if name == "replication_outcome":
                    assert arguments == (os.path.join(str(REPO), "docs/replication_units.csv"),)
                    return {"replication_comparison": comparison,
                            "replication_outcome": replication.outcome_table(comparison),
                            "replication_findings": replication.findings_table(comparison),
                            "replication_absence_findings": replication.absence_units(comparison)}
                if name == "written_tables_check":
                    listed = analysis.planned_tables()
                    listed["written"], listed["ok"] = True, True
                    extra = pd.DataFrame({"table": list(stray), "kind": "not listed", "plan": "",
                                          "written": True, "ok": False})
                    return pd.concat([listed, extra], ignore_index=True)
                if name == "rank_noise_ceiling" and root is not None:
                    for file in ("rank_noise_ceiling.csv", "rank_noise_ceiling.meta.json",
                                 "rank_noise_ceiling_splits.csv", "parts/rank_noise_ceiling_trim40.csv",
                                 "parts/rank_noise_ceiling_trim40.meta.json"):
                        target = os.path.join(cfg.tables_dir(encoder), file)
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        with open(target, "w", encoding="utf-8") as fh:
                            fh.write(f"written by this run, call {len(TABLE_CALLS)}\n")
                    if disturb is not None:
                        disturb()
                return (one, one) if name in ("rank_agreement", "itw_by_duration_ci",
                                              "rank_noise_ceiling") else one
            return table

        def path(self, name):
            return f"tables/{name}.csv"

        def exploratory_path(self, name):
            return f"tables/exploratory/{name}.csv"

        def steps_table(self):
            return pd.DataFrame({"step": ["effects_2x2_trim40"], "status": ["computed"]})

    def compare_with_original(name, table, legacy, copies):
        assert encoder == "wavlm-base", "a table of the second encoder was compared with an original table"
        assert copies == os.path.join(cfg.migration_dir(), "original") and len(table) == 1
        return {f"table_{original}": pd.DataFrame({"table": [original] * 2, "column": ["a", "b"],
                                                   "max_abs_diff": [0.0, 0.0],
                                                   "ok": [True, original not in failing]})
                for original in names[name] if original not in left_out}

    def preregistration_check(repo, copies):
        assert repo == str(REPO) and TABLE_CALLS == [] and built == []
        assert copies == {os.path.join(cfg.tables_dir("wavlm-base"), "replication_units.csv"):
                          "docs/replication_units.csv"}
        table = replication_check(repo)                            # the committed files, as they are
        drive = pd.DataFrame({"file": list(copies), "registered_as": list(copies.values()),
                              "recorded_sha256": ["x"], "sha256": ["x" if registered else "y"],
                              "ok": [registered]})
        return pd.concat([table, drive], ignore_index=True)

    def paired_resamples_check(reference, replica):
        assert (reference.encoder, replica.encoder) == ("wavlm-base", encoder) and TABLE_CALLS == []
        return pd.DataFrame({"test_set": ["eval", "itw"], "ok": [True, paired]})

    def record(name, table, stop=True):
        checks[name] = table
        if root is not None:                                       # a check table is a file with a meta file
            os.makedirs(cfg.checks_dir(), exist_ok=True)
            for suffix in (".csv", ".meta.json"):
                table.to_csv(os.path.join(cfg.checks_dir(), name + suffix), index=False)
        if stop and "ok" in table.columns:
            verify.require_ok(table, name)

    def duration_map(path):
        assert path.replace("\\", "/").endswith("migration/original/files/measurements_dir/itw.csv")
        return {"a": 1.0}

    file_list = pd.DataFrame({"audio_id": ["a"], "speaker": ["s"], "system_id": ["A07"]})
    replication_check = getattr(replication.preregistration_check, "real", replication.preregistration_check)
    preregistration_check.real = replication_check          # a second call in one test finds the real function
    monkeypatch.setattr(training, "finished_run_ids", lambda runs_dir: finished)
    monkeypatch.setattr(datasets, "ensure_asvspoof", lambda path: None)
    monkeypatch.setattr(datasets, "ensure_itw", lambda path, sha256: None)
    monkeypatch.setattr(datasets, "items", lambda split: file_list)
    monkeypatch.setattr(datasets, "check_items", lambda table, split: {"n_files": 1})
    monkeypatch.setattr(silence, "read_measurements", lambda path: path)
    monkeypatch.setattr(silence, "duration_map", duration_map)
    monkeypatch.setattr(silence, "check_duration_subsets", lambda durations: (1, 0))
    monkeypatch.setattr(selection, "load_asv", lambda folder: {"dev": 1, "eval": 2})
    monkeypatch.setattr(analysis, "ScoreTables", Tables)
    monkeypatch.setattr(analysis, "compare_with_original", compare_with_original)
    monkeypatch.setattr(analysis, "paired_resamples_check", paired_resamples_check)
    monkeypatch.setattr(replication, "preregistration_check", preregistration_check)
    if root is not None:
        monkeypatch.setattr(cfg, "ROOT", str(root))
    namespace = {"cfg": cfg, "verify": verify, "pd": pd, "os": os, "checks": checks,
                 "CHECKS": "checks" if root is None else cfg.checks_dir(),
                 "legacy": {}, "display": lambda table: None, "record": record, "WORKDIR": str(REPO)}
    sources = cells(ANALYSIS, "code")
    k = next(i for i, source in enumerate(sources) if "\nENCODER = " in source)
    assert 'ENCODER = "hubert-base" ' in sources[k]                # as saved
    sources[k] = sources[k].replace('ENCODER = "hubert-base" ', f'ENCODER = "{encoder}" ')
    for source in sources[k:]:
        exec(compile(python_of(source), ANALYSIS.name, "exec"), namespace)
    return namespace


A3_CALLS = ["metrics", "cells", "padded_vs_unpadded", "fallback_counts", "selected_configs_eer_ci",
            "itw_by_duration", "effects", "rank_agreement", "threshold_summary", "per_attack_effect"]
UNIT_CALLS = ["matched_total_effects", "tau_differences", "itw_by_duration_ci", "replication_units"]
PLAN_CALLS = ["metrics", "cells", "fallback_counts", "selected_configs_eer_ci", "dev_selected", "itw_by_duration",
              "itw_by_duration_thresholds", "effects", "rank_agreement", "threshold_summary", "per_attack_effect",
              *UNIT_CALLS, "selected_layers_outside_detail", "selected_layer_tables", "encoder_differences",
              "encoder_tau_differences", "replication_outcome", "written_tables_check"]
HUBERT_CHECKS = ["preregistration_hubert-base", "resamples_hubert-base_against_wavlm-base",
                 "analysis_steps_hubert-base", "analysis_tables_hubert-base"]


def all_original_tables():
    from ssl_silence import analysis

    return [f"table_{original}" for names in analysis.original_table_names().values() for original in names]


def test_the_analysis_notebook_compares_all_twenty_original_tables(monkeypatch, capsys):
    checks = run_the_analysis_cells(monkeypatch)["checks"]
    assert list(checks) == all_original_tables() + ["analysis_steps"] and len(checks) == 21
    assert TABLE_CALLS == A3_CALLS + UNIT_CALLS + ["rank_noise_ceiling"]
    out = capsys.readouterr().out
    assert "noise ceiling: 1 rows | 1 splits | tables/rank_noise_ceiling.csv" in out
    assert out.count("| equal") == 20 and "DIFFERENT" not in out
    assert "this run: wavlm-base | extractions: ['unpadded', 'padded'] | gate A3" in out
    assert "not a replication: the pre-registered files are not read" in out
    assert "gate A3 passed: wavlm-base | original tables compared: 20 | columns compared: 40" in out
    assert "units: 2 | in the rule: 1 | by condition: {'trim20': 1, 'trim40': 1}" in out
    assert "copy tables/replication_units.csv to docs/replication_units.csv" in out
    assert "analysis finished" not in out and "EXPLORATORY" not in out and "| written |" not in out


def test_a_different_table_stops_the_analysis_notebook_only_after_every_table_was_written(monkeypatch, capsys):
    from ssl_silence import verify

    with pytest.raises(verify.CheckFailed, match="1 of 40 rows failed"):
        run_the_analysis_cells(monkeypatch, failing=("cells_2x2_trim40",))
    # The first comparison differed; every later table was still computed and compared,
    # and no unit of the replication rule was computed from tables that did not pass.
    assert TABLE_CALLS == A3_CALLS
    out = capsys.readouterr().out
    assert "table_cells_2x2_trim40 | DIFFERENT" in out and out.count("| equal") == 19
    assert "gate A3 passed" not in out


def test_a_table_that_was_not_compared_or_an_unfinished_run_is_not_a_pass(monkeypatch, capsys):
    with pytest.raises(AssertionError, match=r"not compared: \['table_rank_agreement_trim15'\]"):
        run_the_analysis_cells(monkeypatch, left_out=("rank_agreement_trim15",))
    assert "gate A3 passed" not in capsys.readouterr().out and TABLE_CALLS == A3_CALLS
    with pytest.raises(AssertionError, match="1 padded runs are not finished"):
        run_the_analysis_cells(monkeypatch, unfinished=("wavlm-base_padded_all-frames_clean_L04_stats_s44",))
    assert TABLE_CALLS == []


def test_as_saved_the_analysis_notebook_computes_the_plan_and_the_rule_for_hubert(monkeypatch, capsys):
    namespace = run_the_analysis_cells(monkeypatch, encoder="hubert-base")
    checks = namespace["checks"]
    # Nothing is compared with an original table, and no padded table is computed.
    assert list(checks) == HUBERT_CHECKS and TABLE_CALLS == PLAN_CALLS
    assert checks["preregistration_hubert-base"]["ok"].all() and len(checks["preregistration_hubert-base"]) == 4
    assert checks["preregistration_hubert-base"]["registered_as"].tolist() == [
        "docs/analysis_plan_hubert.md", "docs/replication_units.csv", "docs/replication_units_summary.csv",
        "docs/replication_units.csv"]
    out = capsys.readouterr().out
    assert "this run: hubert-base | extractions: ['unpadded'] | the tables of the analysis plan and the " \
           "replication rule against wavlm-base" in out
    assert "the pre-registered files are unchanged | units of wavlm-base: 830 | in the rule: 608" in out
    assert "the resamples of hubert-base are those of wavlm-base" in out
    assert out.count("| written |") == 9 and "| equal" not in out and "gate A3 passed" not in out
    assert "no original tables for hubert-base: the tables above were written, not compared" in out
    assert "EXPLORATORY, not in the analysis plan: tables/exploratory/exploratory_itw_by_duration_thresholds.csv" in out
    assert "dev-selected layers outside (3, 4, 5) : [7] | tables: ['per_attack_effect_dev_selected_layers']" in out
    assert "copy tables/replication_units.csv" not in out and "noise ceiling" not in out
    # The stand-in for the second encoder is the reference itself: every unit of the rule is replicated.
    assert "units: 830 | in the rule: 608 | verdicts: {'replicated': 608, 'descriptive': 222} | reverse count: 0" in out
    assert "analysis finished: hubert-base | tables of the analysis plan: 28 | exploratory tables: 1" in out
    assert namespace["REFERENCE_UNITS"] == os.path.join(str(REPO), "docs/replication_units.csv")
    assert namespace["EXPECTED"] == {} and namespace["STEPS"] == "analysis_steps_hubert-base"


EXISTING = ["tables/wavlm-base/cells_2x2.csv", "tables/wavlm-base/cells_2x2.meta.json",
            "tables/wavlm-base/replication_units.csv", "tables/wavlm-base/parts/effects_2x2_trim40.csv",
            "tables/wavlm-base/probes/blocks/block_001.csv", "checks/table_cells_2x2_trim40.csv",
            "checks/analysis_steps.csv", "checks/analysis_steps.meta.json",
            "checks/analysis_tables_hubert-base.csv"]
EARLIER_RUN = ["tables/wavlm-base/rank_noise_ceiling.csv", "tables/wavlm-base/parts/rank_noise_ceiling_trim40.csv",
               "checks/analysis_steps_noise_ceiling.csv", "checks/analysis_files_unchanged_wavlm-base.csv",
               "checks/analysis_files_unchanged_wavlm-base.meta.json"]


def drive_with_the_tables_of_wavlm_base(root, files=EXISTING):
    for file in files:
        (root / file).parent.mkdir(parents=True, exist_ok=True)
        (root / file).write_bytes(f"verified: {file}\n".encode())
    return {file: (root / file).read_bytes() for file in files}


def test_with_its_tables_a_run_for_wavlm_base_computes_the_noise_ceiling_and_writes_no_other_table(
        monkeypatch, capsys, tmp_path):
    before = drive_with_the_tables_of_wavlm_base(tmp_path)
    namespace = run_the_analysis_cells(monkeypatch, root=tmp_path)
    assert namespace["CEILING_ONLY"] and not namespace["TABLES"] and namespace["EXPECTED"] == {}
    # No table of gate A3 and no unit is computed or compared; nothing is compared with an original table.
    assert TABLE_CALLS == ["rank_noise_ceiling"]
    checks = namespace["checks"]
    assert list(checks) == ["analysis_steps_noise_ceiling", "analysis_files_unchanged_wavlm-base"]
    unchanged = checks["analysis_files_unchanged_wavlm-base"]
    assert sorted(os.path.relpath(file, tmp_path).replace(os.sep, "/") for file in unchanged["file"]) \
        == sorted(EXISTING)
    assert unchanged["ok"].all() and set(unchanged["status"]) == {"unchanged"}
    assert (unchanged["sha256_at_start"] == unchanged["sha256_at_end"]).all()
    assert {file: (tmp_path / file).read_bytes() for file in EXISTING} == before
    assert (tmp_path / "tables/wavlm-base/rank_noise_ceiling.csv").exists()
    out = capsys.readouterr().out
    assert ("this run: wavlm-base | extractions: ['unpadded', 'padded'] | the noise ceiling of the rank "
            "agreement only; no other table is computed or written") in out
    assert "files of wavlm-base and check tables that exist: 9 | their sha256 is recorded and must not change" in out
    assert "original tables to compare: 0" in out
    assert ("noise ceiling run finished: wavlm-base | files that existed at the start: 9 unchanged "
            "| no other table was computed or written") in out
    assert "gate A3" not in out and "| equal" not in out and "units:" not in out and "tables written to" not in out

    # A second run finds the files of the first one: they are its own and may be written again.
    files = set(os.listdir(tmp_path / "checks")) | set(os.listdir(tmp_path / "tables/wavlm-base"))
    assert {"analysis_steps_noise_ceiling.csv", "analysis_files_unchanged_wavlm-base.meta.json",
            "rank_noise_ceiling_splits.csv", "rank_noise_ceiling.meta.json"} <= files
    drive_with_the_tables_of_wavlm_base(tmp_path, EARLIER_RUN)
    namespace = run_the_analysis_cells(monkeypatch, root=tmp_path)
    assert TABLE_CALLS == ["rank_noise_ceiling"] and len(namespace["BEFORE"]) == 9
    assert namespace["checks"]["analysis_files_unchanged_wavlm-base"]["ok"].all()
    assert {file: (tmp_path / file).read_bytes() for file in EXISTING} == before


@pytest.mark.parametrize("file, status", [
    ("tables/wavlm-base/cells_2x2.csv", "changed"), ("tables/wavlm-base/cells_2x2.meta.json", "changed"),
    ("tables/wavlm-base/parts/effects_2x2_trim40.csv", "changed"), ("checks/analysis_steps.csv", "changed"),
    ("checks/table_cells_2x2_trim40.csv", "removed"), ("tables/wavlm-base/effects_2x2.csv", "new"),
    ("checks/table_effects_2x2_trim40.meta.json", "new")])
def test_a_table_of_wavlm_base_that_changed_during_the_noise_ceiling_stops_the_notebook(
        monkeypatch, capsys, tmp_path, file, status):
    from ssl_silence import verify

    drive_with_the_tables_of_wavlm_base(tmp_path)

    def disturb():
        if status == "removed":
            os.remove(tmp_path / file)
        else:                                  # the same numbers written again with another line ending
            (tmp_path / file).write_bytes(f"verified: {file}\r\n".encode())

    with pytest.raises(verify.CheckFailed, match="analysis_files_unchanged_wavlm-base: 1 of"):
        run_the_analysis_cells(monkeypatch, root=tmp_path, disturb=disturb)
    failed = RESULT_ANALYSIS["analysis_files_unchanged_wavlm-base"]
    failed = failed[~failed["ok"]]
    assert [os.path.relpath(path, tmp_path).replace(os.sep, "/") for path in failed["file"]] == [file]
    assert failed["status"].tolist() == [status]
    assert "noise ceiling run finished" not in capsys.readouterr().out


def test_without_the_last_table_of_a_full_run_wavlm_base_runs_gate_a3_and_hubert_is_not_affected(
        monkeypatch, capsys, tmp_path):
    files = [file for file in EXISTING if not file.endswith("replication_units.csv")]
    drive_with_the_tables_of_wavlm_base(tmp_path, files)
    namespace = run_the_analysis_cells(monkeypatch, root=tmp_path)
    assert namespace["TABLES"] and not namespace["CEILING_ONLY"] and namespace["BEFORE"] == {}
    assert TABLE_CALLS == A3_CALLS + UNIT_CALLS + ["rank_noise_ceiling"]
    assert list(namespace["checks"]) == all_original_tables() + ["analysis_steps"]
    assert "gate A3 passed: wavlm-base" in capsys.readouterr().out
    # The replication: the tables of WavLM Base exist, and the run of HuBERT Base is the run of the plan.
    drive_with_the_tables_of_wavlm_base(tmp_path)
    namespace = run_the_analysis_cells(monkeypatch, encoder="hubert-base", root=tmp_path)
    assert namespace["TABLES"] and not namespace["CEILING_ONLY"] and namespace["BEFORE"] == {}
    assert TABLE_CALLS == PLAN_CALLS and list(namespace["checks"]) == HUBERT_CHECKS
    assert "analysis finished: hubert-base" in capsys.readouterr().out


@pytest.mark.parametrize("problem, check", [({"registered": False}, "preregistration_hubert-base"),
                                             ({"paired": False}, "resamples_hubert-base_against_wavlm-base")])
def test_changed_pre_registered_files_or_other_resamples_stop_before_any_table(monkeypatch, capsys, problem, check):
    from ssl_silence import verify

    with pytest.raises(verify.CheckFailed, match=check):
        run_the_analysis_cells(monkeypatch, encoder="hubert-base", **problem)
    assert TABLE_CALLS == []
    out = capsys.readouterr().out
    assert "analysis finished" not in out and "| written |" not in out
    assert ("the pre-registered files are unchanged" in out) == ("paired" in problem)


def test_a_table_that_is_not_listed_or_an_unfinished_run_is_not_a_finished_analysis(monkeypatch, capsys):
    from ssl_silence import verify

    with pytest.raises(verify.CheckFailed, match="analysis_tables_hubert-base"):
        run_the_analysis_cells(monkeypatch, encoder="hubert-base", stray=("something_else",))
    assert TABLE_CALLS == PLAN_CALLS and "analysis finished" not in capsys.readouterr().out
    with pytest.raises(AssertionError, match="1 unpadded runs are not finished"):
        run_the_analysis_cells(monkeypatch, encoder="hubert-base",
                               unfinished=("hubert-base_unpadded_trim15_L12_mean_s46",))
    assert TABLE_CALLS == []
    with pytest.raises(AssertionError, match="1 unpadded runs of wavlm-base are not finished"):
        run_the_analysis_cells(monkeypatch, encoder="hubert-base",
                               unfinished=("wavlm-base_unpadded_clean_L04_stats_s42",))
    assert TABLE_CALLS == []


# Padding deviation: gate A4 for WavLM Base, the intervals, the noise floor and the predictions ------

PADDING = REPO / "notebooks/06_padding_deviation.ipynb"
GATE_CHECKS = ["padding_encoder", "padding_samples", "table_padding_deviation",
               "table_padding_deviation_by_amount", "padding_printed_numbers"]
GATE_TABLES = ["padding_deviation.csv", "padding_deviation_files.csv", "padding_deviation_by_amount.csv"]
RUN_TABLES = ["padding_deviation_intervals.csv", "padding_mask_reduces.csv", "padding_noise_floor.csv",
              "padding_noise_floor_files.csv", "padding_noise_floor_intervals.csv"]


def test_the_padding_notebook_measures_the_files_of_the_original_check():
    assert PADDING in NOTEBOOKS
    code = "\n".join(cells(PADDING, "code"))
    order = [code.index(text) for text in (
        "assert torch.cuda.is_available()", 'datasets.items("train")', "padding_per_file.csv",
        "encoders.load_encoder(ENCODER, DEVICE)", "padding.random_items(items)",
        'padding.sample_ids(original_files, legacy["conditions"][condition], sample)',
        'record("padding_samples", pd.DataFrame(rows), stop=False)',
        "padding.padding_deviation(model, chosen, DEVICE, condition, sample)",
        'verify.compare_with_legacy(published, "padding_deviation", legacy, ORIGINAL)',
        'record("table_padding_deviation", deviation, stop=False)',
        'record("table_padding_deviation_by_amount", amount, stop=False)',
        "intext.compare_printed(new_tables, original_tables, intext.padding_entries())",
        'record("padding_printed_numbers", printed, stop=False)',
        'verify.require_ok(checks["padding_samples"], "padding_samples")',
        'verify.require_ok(gate, "rel_mean_diff of padding_deviation")',
        'verify.require_ok(printed, "padding numbers at the printed precision")')]
    assert order == sorted(order)
    # The printed numbers are read from the original tables; none is typed into the notebook.
    assert not re.search(r"0\.(31|44|15|34|20|94|95|80)\b", code) and "1.76" not in code
    assert 'gate = deviation[deviation["column"] == "rel_mean_diff"]' in code
    # The tolerance is that of the mapping file; the files are those of the original table.
    assert "tolerance=" not in code and "1e-4" not in code
    assert "samples[(condition, sample)] = padding.items_with_ids(items, ids)" in code
    text = "\n".join(cells(PADDING, "markdown"))
    assert "gate A4" in text and "at most 1e-4" in text and "do not decide the gate" in text


def test_the_padding_notebook_checks_the_plan_before_a_checkpoint_and_runs_one_encoder():
    code_cells = cells(PADDING, "code")
    code = "\n".join(code_cells)
    text = " ".join("\n".join(cells(PADDING, "markdown")).split())
    order = [code.index(piece) for piece in (
        'ENCODER = "hubert-base" ', "PREDICTIONS = ENCODER == padding.PREDICTED_ENCODER",
        "existing = [path for path in GATE_FILES if os.path.exists(path)]",
        "GATE = ENCODER == REFERENCE and not existing", "REUSE = ENCODER == REFERENCE and bool(existing)",
        "GATE_SHA256 = {path: provenance.sha256_file(path) for path in existing}",
        "gate_files = padding.gate_files_check(GATE_SHA256)",
        'print(os.path.basename(row.file), "| commit of its meta file:", row.git_commit)',
        'record(f"padding_gate_commit_{ENCODER}", gate_files)',
        'record(f"preregistration_padding_{ENCODER}", replication.preregistration_check(WORKDIR))',
        '"upper <= 10.0 * floor_upper" in PLAN and padding.FLOOR_FACTOR == 10.0',
        "assert not cfg.on_drive(HF_CACHE)",
        "encoders.load_encoder(ENCODER, DEVICE)", 'encoders.load_encoder(ENCODER, "cpu")',
        "padding.sample_ids(reference_files, condition, sample)",
        "padding.random_items(items), \"most_padded\": None}",
        "padding.stored_lengths(lengths)", "padding.lengths_from_audio(both, condition)",
        "padding.check_reference_sample(reference_files, condition, sample, known, lengths_from,",
        'record(f"padding_samples_{ENCODER}", pd.DataFrame(rows)) ',
        "padding.padding_deviation(model, timed, DEVICE, CONDITIONS[0], padding.TIMING)",
        'padding.noise_floor(model, DEVICE, cpu_model, "cpu", timed, CONDITIONS[0], padding.TIMING)',
        'record(f"padding_timing_{ENCODER}", timing, stop=False)',
        "padding.padding_deviation(model, chosen, DEVICE, condition, sample)",
        "padding.deviation_intervals(per_file)", "padding.mask_reduces_deviation(intervals)",
        'samples[(condition, "random")], condition, "random")',
        'padding.deviation_intervals(floor_file, comparisons=("floor",), difference=None)',
        "padding.floor_predictions(intervals, floor_intervals)",
        'padding.prediction_verdict(predictions, entry["n_layers"], CONDITIONS)',
        'record(f"padding_predictions_{ENCODER}", verdict, stop=False)',
        'record(f"padding_gate_tables_{ENCODER}", pd.DataFrame([', '"padding run finished:"')]
    assert order == sorted(order)
    # The tables of gate A4 are written by the gate alone: the deviation is measured and written
    # in one cell, which reads the tables instead when they exist; their sha256 is a hard stop.
    source = next(s for s in code_cells if "padding.padding_deviation(model, chosen," in s)
    assert source.startswith("if REUSE:\n    per_layer = pd.read_csv(") and "\nelse:\n" in source
    assert source.index("\nelse:\n") < source.index("padding.padding_deviation(model, chosen,")
    assert source.index("\nelse:\n") < source.index("provenance.write_table(")
    assert all(f'"{name}"' in source for name in padding_names()[0])
    source = next(s for s in code_cells if 'record(f"padding_gate_tables_{ENCODER}"' in s)
    assert "if not GATE:" in source and "stop=False" not in source
    assert source.index("sha256_at_end") < source.index('"padding run finished:"')
    # The pre-registered files and the place of the cache are hard stops, before any checkpoint.
    k = next(i for i, source in enumerate(code_cells) if "replication.preregistration_check(" in source)
    assert "stop=False" not in code_cells[k] and "load_encoder" not in "\n".join(code_cells[:k + 1])
    assert code.count("load_encoder(") == 2
    # What exists for the reference encoder only is behind GATE; the predictions behind PREDICTIONS.
    for call in ("verify.compare_with_legacy(", "intext.compare_printed(", '"gate A4 passed:"'):
        source = next(s for s in code_cells if call in s)
        assert source.startswith("if GATE:\n") and "\nelse:\n    print(" in source, call
    source = next(s for s in code_cells if "padding.floor_predictions(" in s)
    assert source.startswith("if PREDICTIONS:\n")
    # The verdict is reported either way; the samples of another encoder are a hard stop.
    assert 'record(f"padding_samples_{ENCODER}", pd.DataFrame(rows), stop=False)' not in code
    # Nothing but tables is written: no shard and no feature file.
    for word in ("write_shard", "save_shard", "torch.save", "np.save", "extract_split", "feature_dir(REFERENCE"):
        assert word not in code, word
    assert code.count("provenance.write_table(") == 4
    for piece in ("One run per encoder, in this order", '`ENCODER = "hubert-base"` (as saved)',
                  '`ENCODER = "wavlm-large"`', '`ENCODER = "wavlm-base"`',
                  "`tables/wavlm-base/padding_deviation_files.csv`", "`numpy.random.default_rng(0)`",
                  "computed in float64 as `upper <= 10.0 * floor_upper`",
                  "met at all 24 layers in both conditions", "reported either way",
                  "The most padded sample is not part of the predictions", "timed check of 25 files",
                  "`docs/preregistration.md`", "no feature file",
                  "**The tables of gate A4 are not overwritten.**", "the gate passed at commit `637b4b8`",
                  "they are read, not computed again and not written",
                  "runs only if its tables do not exist yet",
                  "the last cell stops unless every one is unchanged",
                  "it must record the commit `637b4b8`, at which the gate passed",
                  "before any checkpoint is loaded"):
        assert piece in text, piece


def padding_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The tables and the check tables of gate A4, as the package names them."""
    from ssl_silence import padding

    assert [f"{name}.csv" for name in padding.GATE_TABLES] == GATE_TABLES
    assert list(padding.GATE_CHECKS) == GATE_CHECKS
    return padding.GATE_TABLES, padding.GATE_CHECKS


def run_the_padding_cells(monkeypatch, capsys, different_sample=False, difference=0.0, other=0.0,
                          missing_rows=0, printed_differs=False, encoder="wavlm-base", registered=True,
                          cache="/root/.cache/huggingface/hub", other_random=False, other_most_padded=False,
                          other_length=False, normalisation=None, a_fails=(), b_fails=(),
                          gate_tables=None, changed_table=None, other_commit=None):
    """The cells of the padding notebook from the settings cell on, with stand-ins
    for the encoder, for the measurements and for the comparison with the
    original tables. The check of the pre-registered files is the real one,
    on the files of this repository, unless ``registered`` is false.

    ``gate_tables``: how many of the eight tables of gate A4 exist (None: all
    of them, except for WavLM Base, for which none exists). ``changed_table``:
    a table of the gate whose sha256 differs at the end of the run.
    ``other_commit``: a table of the gate whose meta file names another commit."""
    import pandas as pd
    import torch

    from ssl_silence import datasets, encoders, padding, provenance, replication, shards, verify
    from ssl_silence.paper import intext

    checks: dict = {}
    measured, written, loaded, floors, compared = [], [], [], [], []
    gate_files = [os.path.join(cfg.tables_dir("wavlm-base"), f"{name}.csv") for name in padding.GATE_TABLES] \
        + [os.path.join("checks", f"{name}.csv") for name in padding.GATE_CHECKS]
    if gate_tables is None:
        gate_tables = 0 if encoder == "wavlm-base" else len(gate_files)
    present = gate_files[:gate_tables]
    gate = encoder == "wavlm-base" and not present          # a run of gate A4
    reuse = encoder == "wavlm-base" and bool(present)       # WavLM Base with its tables
    conditions = ("clean", "trim40")
    hashed: list = []

    def sha256_file(path):
        assert path in present
        hashed.append(path)
        again = hashed.count(path) > 1
        return f"sha256 of {os.path.basename(path)}" + (" (changed)" if again and changed_table == os.path.basename(path) else "")

    def read_legacy(name, legacy, copies):
        assert copies == os.path.join(cfg.migration_dir(), "original")
        return pd.DataFrame({"table": [name]})

    def compare_printed(new, reference, entries):
        compared.append("printed")
        assert [entry.key for entry in entries] == [
            "padding_deviation_layer4", "padding_median", "padding_by_amount", "padding_most_padded_layer4"]
        names = [intext.PADDING_DEVIATION, intext.PADDING_BY_AMOUNT, intext.PADDING_PER_FILE]
        assert [new[name] is not None and reference[name] is not None for name in names] == [True] * 3
        assert reference[intext.PADDING_DEVIATION]["table"].tolist() == ["padding_deviation"]
        # The original per-file table is given with the condition names of this package.
        assert set(reference[intext.PADDING_PER_FILE]["condition"]) == {"clean", "trim40"}
        assert len(new[intext.PADDING_PER_FILE]) == 12
        return pd.DataFrame({"key": ["k"] * 10, "position": range(10), "printed": 0.3,
                             "reference_printed": [0.3] * 9 + [0.31 if printed_differs else 0.3],
                             "ok": [True] * 9 + [not printed_differs]})
    label = {"clean": "clean", "trim40": "cut"}
    ids = {(c, s): [f"{c}-{s}-{k}" for k in range(3)] for c in ("clean", "cut") for s in padding.SAMPLES}
    original_files = pd.DataFrame([{"audio_id": a, "condition": c, "sample": s}
                                   for (c, s), names in ids.items() for a in names])
    # The per-file table of the reference encoder: the same files under the condition names of this package.
    reference_files = pd.DataFrame([{"audio_id": a, "condition": c, "sample": s, "n_real": 1_000 + 10 * k}
                                    for c in conditions for s in padding.SAMPLES
                                    for k, a in enumerate(ids[(label[c], s)])])
    items = pd.DataFrame({"audio_id": [a for names in ids.values() for a in names]})
    items["path"] = "p/" + items["audio_id"]

    def drawn(condition, sample):
        names = ids[(label[condition], sample)]
        return items[items["audio_id"].isin(names[1:] + ["other"] if different_sample else names)]

    state = {"condition": None, "random": iter(conditions)}

    def length(audio_id, condition):
        part = reference_files[(reference_files["audio_id"] == audio_id) & (reference_files["condition"] == condition)]
        return int(part["n_real"].iloc[0]) + (1 if other_length and audio_id.endswith("random-1") else 0)

    def shard_lengths(folder, splits):
        assert splits == ("train",) and encoder != "wavlm-large"         # WavLM Large has no shards
        state["condition"] = "clean" if folder == cfg.feature_dir(encoder, "unpadded", "clean") else "trim40"
        assert folder == cfg.feature_dir(encoder, "unpadded", state["condition"])
        if gate:
            return pd.DataFrame({"audio_id": ["a"], "n_real": [1]})
        names = [a for s in padding.SAMPLES for a in ids[(label[state["condition"]], s)]]
        return pd.DataFrame({"audio_id": names, "n_real": [length(a, state["condition"]) for a in names]})

    def random_items(table):
        if not gate:                                                     # drawn before the lengths are read
            state["condition"] = next(state["random"])
            if other_random:
                return drawn(state["condition"], "random").iloc[::-1]
        return drawn(state["condition"], "random")

    def most_padded_items(table, pads):
        chosen = drawn(state["condition"], "most_padded")
        return chosen.iloc[::-1] if other_most_padded and not gate else chosen

    def lengths_from_audio(chosen, condition):
        assert encoder == "wavlm-large" and condition == state["condition"]
        assert sorted(chosen["audio_id"]) == sorted(a for s in padding.SAMPLES for a in ids[(label[condition], s)])
        return {a: length(a, condition) for a in chosen["audio_id"]}

    def gate_files_check(sha256):
        assert list(sha256) == present and hashed == present and not loaded       # before any checkpoint
        commit = {path: ("0043fff0ec4d4816f21ef8c28221c028c3252bf5" if os.path.basename(path) == other_commit
                         else "637b4b89b1fa8eb551f178c90a520df58a65a1ac") for path in sha256}
        return pd.DataFrame({"file": list(sha256), "git_commit": list(commit.values()),
                             "recorded_sha256": list(sha256.values()), "sha256": list(sha256.values()),
                             "ok": [value.startswith(padding.GATE_COMMIT) for value in commit.values()]})

    def padding_deviation(model, chosen, device, condition, sample):
        assert (model, device) == ("model on cuda", "cuda") and not reuse      # the tables of the gate are read
        if sample == padding.TIMING:
            assert (condition, len(chosen)) == ("clean", 3) and not measured     # before the long part
            return None, None
        assert chosen["audio_id"].tolist() == ids[(label[condition], sample)]
        measured.append((condition, sample))
        rows = [{"condition": condition, "sample": sample, "comparison": comparison, "statistic": statistic,
                 "layer": layer, "rel_mean_diff": 0.3, "n_skipped": 0}
                for comparison in padding.COMPARISONS for statistic in padding.STATISTICS for layer in (1, 4)]
        return pd.DataFrame(rows), pd.DataFrame({"audio_id": chosen["audio_id"], "pad_s": 1.0,
                                                 "condition": condition, "sample": sample})

    def noise_floor(reference_model, reference_device, other_model, other_device, chosen, condition, sample):
        assert (reference_model, reference_device) == ("model on cuda", "cuda")      # the GPU is the reference
        assert (other_model, other_device) == ("model on cpu", "cpu")
        if sample == padding.TIMING:
            assert (condition, len(chosen)) == ("clean", 3) and not floors
            return None, None
        assert sample == "random" and chosen["audio_id"].tolist() == ids[(label[condition], "random")]
        assert len(measured) == (0 if reuse else 4)                      # after the deviation
        floors.append(condition)
        rows = [{"condition": condition, "sample": sample, "comparison": "floor", "statistic": statistic,
                 "layer": layer, "rel_mean_diff": 1e-6, "n_skipped": 0}
                for statistic in padding.STATISTICS for layer in (1, 4)]
        return pd.DataFrame(rows), pd.DataFrame({"audio_id": chosen["audio_id"], "condition": condition,
                                                 "sample": sample})

    def deviation_intervals(per_file, comparisons=("no_mask", "mask"), difference=("mask", "no_mask")):
        quantities = list(comparisons) + ([f"{difference[0]}_minus_{difference[1]}"] if difference else [])
        assert quantities in (["no_mask", "mask", "mask_minus_no_mask"], ["floor"])
        return pd.DataFrame([{"condition": c, "sample": "random", "statistic": "mean", "layer": layer,
                              "quantity": q, "estimate": 0.1, "ci_low": 0.05, "ci_high": 0.2}
                             for c in per_file["condition"].unique() for layer in (1, 4) for q in quantities])

    def floor_predictions(intervals, floor):
        assert encoder == "wavlm-large" and set(floor["quantity"]) == {"floor"}
        assert set(intervals["quantity"]) == {"no_mask", "mask", "mask_minus_no_mask"}
        return pd.DataFrame([{"condition": c, "sample": "random", "layer": layer,
                              "a_holds": (c, layer) not in a_fails, "b_holds": (c, layer) not in b_fails}
                             for c in conditions for layer in range(1, 25)])

    def compare_with_legacy(new, name, legacy, copies):
        compared.append(name)
        assert copies == os.path.join(cfg.migration_dir(), "original")
        assert set(new["comparison"]) == set(padding.PUBLISHED_COMPARISONS)
        if name == "padding_deviation":
            assert set(new["statistic"]) == {"mean"} and len(new) == 4 * 2 * 2
            return pd.DataFrame({"table": name, "column": ["rel_mean_diff", "rel_max_diff"],
                                 "n_matched": 2 * 2 * 2 * 12 - missing_rows,
                                 "max_abs_diff": [difference, other], "tolerance": 1e-4,
                                 "ok": [difference <= 1e-4 and not missing_rows, other <= 1e-4]})
        assert name == "padding_deviation_by_amount"
        return pd.DataFrame({"table": name, "column": ["mean_rel_L4"], "n_matched": [8],
                             "max_abs_diff": [0.0], "tolerance": 1e-4, "ok": [True]})

    def record(name, table, stop=True):
        checks[name] = table
        if stop and "ok" in table.columns:
            verify.require_ok(table, name)

    real_read_csv, real_exists = pd.read_csv, os.path.exists
    reference_path = os.path.join(cfg.tables_dir("wavlm-base"), "padding_deviation_files.csv")
    read: list = []

    def read_csv(path, **options):
        if str(path).replace("\\", "/").endswith("migration/original/files/padding_per_file.csv"):
            assert options == {"dtype": {"audio_id": str}} and gate
            return original_files
        if str(path) == reference_path:
            assert options == {"dtype": {"audio_id": str}} and not gate and path in present
            read.append(os.path.basename(path))
            return reference_files
        if str(path) == os.path.join(cfg.tables_dir("wavlm-base"), "padding_deviation.csv"):
            assert reuse and not options
            read.append(os.path.basename(path))
            return pd.DataFrame({"layer": [1, 4]})
        return real_read_csv(path, **options)

    def load_encoder(name, device):
        assert name == encoder and list(checks) == (
            [] if gate else [f"padding_gate_commit_{encoder}"]) + [f"preregistration_padding_{encoder}"]
        loaded.append(device)
        return f"model on {device}"

    def unregistered(repo):
        return pd.DataFrame({"file": ["docs/analysis_plan_hubert.md"], "ok": [False]})

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pd, "read_csv", read_csv)
    monkeypatch.setattr(os.path, "exists", lambda path: (
        str(path) in present if str(path) in gate_files else real_exists(path)))
    monkeypatch.setattr(provenance, "sha256_file", sha256_file)
    monkeypatch.setattr(padding, "gate_files_check", gate_files_check)
    monkeypatch.setattr(datasets, "ensure_asvspoof", lambda path: None)
    monkeypatch.setattr(datasets, "items", lambda split: items)
    monkeypatch.setattr(datasets, "check_items", lambda table, split: {"n_files": len(table)})
    monkeypatch.setattr(encoders, "load_encoder", load_encoder)
    monkeypatch.setattr(encoders, "cache_dir", lambda: cache)
    monkeypatch.setattr(encoders, "describe", lambda model: {
        "feat_extract_norm": normalisation or cfg.encoder(encoder)["feat_extract_norm"]})
    # The real check is kept on the stand-in, so that a later call of this function finds it again.
    real_check = getattr(replication.preregistration_check, "real", replication.preregistration_check)
    unregistered.real = real_check
    monkeypatch.setattr(replication, "preregistration_check", real_check if registered else unregistered)
    monkeypatch.setattr(provenance, "gpu_name", lambda device: "a GPU")
    monkeypatch.setattr(provenance, "write_table", lambda table, path, **extra: written.append(
        (os.path.basename(path), len(table), sorted(extra))))
    monkeypatch.setattr(shards, "shard_lengths", shard_lengths)
    monkeypatch.setattr(padding, "random_items", random_items)
    monkeypatch.setattr(padding, "most_padded_items", most_padded_items)
    monkeypatch.setattr(padding, "padding_from_lengths", lambda lengths: {})
    monkeypatch.setattr(padding, "lengths_from_audio", lengths_from_audio)
    monkeypatch.setattr(padding, "padding_deviation", padding_deviation)
    monkeypatch.setattr(padding, "noise_floor", noise_floor)
    monkeypatch.setattr(padding, "deviation_intervals", deviation_intervals)
    monkeypatch.setattr(padding, "mask_reduces_deviation", lambda intervals: pd.DataFrame(
        {"condition": list(conditions), "mask_reduces": True}))
    monkeypatch.setattr(padding, "floor_predictions", floor_predictions)
    monkeypatch.setattr(padding, "deviation_by_padding", lambda per_file, per_layer: pd.DataFrame(
        {"comparison": list(padding.COMPARISONS), "mean_rel_L4": 0.1}))
    monkeypatch.setattr(verify, "compare_with_legacy", compare_with_legacy)
    monkeypatch.setattr(verify, "read_legacy", read_legacy)
    monkeypatch.setattr(intext, "compare_printed", compare_printed)
    namespace = {"cfg": cfg, "verify": verify, "provenance": provenance, "pd": pd, "os": os, "checks": checks,
                 "CHECKS": "checks", "legacy": {"conditions": {"clean": "clean", "trim40": "cut"}},
                 "display": lambda table: None, "record": record, "WORKDIR": str(REPO)}
    sources = cells(PADDING, "code")
    k = next(i for i, source in enumerate(sources) if "\nENCODER = " in source)
    assert 'ENCODER = "hubert-base" ' in sources[k]                # as saved
    sources[k] = sources[k].replace('ENCODER = "hubert-base" ', f'ENCODER = "{encoder}" ')
    try:
        for source in sources[k:]:
            exec(compile(python_of(source), PADDING.name, "exec"), namespace)
    finally:
        namespace.update(measured=measured, written=written, loaded=loaded, floors=floors, compared=compared,
                         read=read, hashed=hashed, gate_files=gate_files, out=capsys.readouterr().out)
        RESULT.clear()
        RESULT.update(namespace)
    return namespace


RESULT: dict = {}


def intext_name() -> str:
    """The per-file table has the file name that the in-text registry reads."""
    from ssl_silence.paper import intext

    return intext.PADDING_PER_FILE


def test_the_padding_notebook_passes_on_the_rel_mean_diff_of_the_same_files(monkeypatch, capsys):
    namespace = run_the_padding_cells(monkeypatch, capsys, difference=9e-5, other=3e-4)
    assert namespace["measured"] == [("clean", "random"), ("clean", "most_padded"),
                                     ("trim40", "random"), ("trim40", "most_padded")]
    # The checks of gate A4 are the same tables under the same names, in the same order.
    assert [name for name in namespace["checks"] if name in GATE_CHECKS] == GATE_CHECKS
    assert list(namespace["checks"]) == ["preregistration_padding_wavlm-base", "padding_encoder", "padding_samples",
                                         "padding_timing_wavlm-base", *GATE_CHECKS[2:]]
    assert namespace["checks"]["padding_samples"]["ok"].all() and len(namespace["checks"]["padding_samples"]) == 4
    assert list(namespace["checks"]["padding_samples"].columns) == [
        "condition", "sample", "n_files", "n_reference", "n_common", "same_order", "ok"]
    assert [name for name, _, _ in namespace["written"]][:3] == GATE_TABLES
    assert namespace["GATE"] and not namespace["REUSE"] and namespace["hashed"] == []
    assert "tables of gate A4: none yet" in namespace["out"]
    assert "| tables of gate A4: written by this run |" in namespace["out"]
    assert namespace["written"][1][0] == intext_name()
    assert namespace["written"][0][1:] == (4 * 3 * 2 * 2, ["encoder", "revision"])
    # Another column is beyond its tolerance: it is reported and does not decide the gate.
    assert ("gate A4 passed: wavlm-base | rows compared: 96 | largest difference of rel_mean_diff: 9e-05 "
            "| printed numbers reproduced: 10 "
            "| every other compared column within its tolerance: False") in namespace["out"]
    # After the gate: the intervals and the noise floor of WavLM Base, and no prediction.
    assert [name for name, _, _ in namespace["written"]][3:] == RUN_TABLES
    assert namespace["floors"] == ["clean", "trim40"] and namespace["loaded"] == ["cuda", "cpu"]
    assert namespace["out"].index("gate A4 passed") < namespace["out"].index("padding run finished: wavlm-base")
    assert "no prediction is stated for wavlm-base" in namespace["out"]
    assert "predictions: none stated for this encoder" in namespace["out"]


def test_the_padding_notebook_stops_on_rel_mean_diff_other_files_or_missing_rows(monkeypatch, capsys):
    from ssl_silence import verify

    with pytest.raises(verify.CheckFailed, match="rel_mean_diff of padding_deviation"):
        run_the_padding_cells(monkeypatch, capsys, difference=2e-4)
    # Everything was measured, compared and written before the stop.
    assert len(RESULT["measured"]) == 4 and len(RESULT["written"]) == 3
    assert "table_padding_deviation_by_amount" in RESULT["checks"] and "gate A4 passed" not in RESULT["out"]
    # The noise floor is not measured for a definition that the gate did not reproduce.
    assert RESULT["floors"] == [] and "padding run finished" not in RESULT["out"]
    with pytest.raises(verify.CheckFailed, match="padding_samples"):
        run_the_padding_cells(monkeypatch, capsys, different_sample=True)
    assert len(RESULT["measured"]) == 4 and "gate A4 passed" not in RESULT["out"]
    assert not RESULT["checks"]["padding_samples"]["ok"].any()
    with pytest.raises(AssertionError):
        run_the_padding_cells(monkeypatch, capsys, missing_rows=12)
    assert "gate A4 passed" not in RESULT["out"]
    # A printed number that is not reproduced stops the notebook, after its table was written.
    with pytest.raises(verify.CheckFailed, match="padding numbers at the printed precision"):
        run_the_padding_cells(monkeypatch, capsys, printed_differs=True)
    assert not RESULT["checks"]["padding_printed_numbers"]["ok"].all() and "gate A4 passed" not in RESULT["out"]


def test_the_padding_notebook_measures_hubert_base_on_the_samples_of_wavlm_base(monkeypatch, capsys):
    namespace = run_the_padding_cells(monkeypatch, capsys, encoder="hubert-base")
    assert namespace["measured"] == [("clean", "random"), ("clean", "most_padded"),
                                     ("trim40", "random"), ("trim40", "most_padded")]
    assert list(namespace["checks"]) == ["padding_gate_commit_hubert-base",
                                         "preregistration_padding_hubert-base", "padding_encoder_hubert-base",
                                         "padding_samples_hubert-base", "padding_timing_hubert-base",
                                         "padding_gate_tables_hubert-base"]
    # The eight tables of gate A4 are hashed at the start and at the end, and none is written.
    unchanged = namespace["checks"]["padding_gate_tables_hubert-base"]
    assert unchanged["file"].tolist() == namespace["gate_files"] and unchanged["ok"].all()
    assert namespace["hashed"] == namespace["gate_files"] * 2 and namespace["read"] == ["padding_deviation_files.csv"]
    assert not namespace["GATE"] and not namespace["REUSE"]
    # The real check of the pre-registered files, on the files of this repository.
    registered = namespace["checks"]["preregistration_padding_hubert-base"]
    assert registered["ok"].all() and registered["file"].tolist() == [
        "docs/analysis_plan_hubert.md", "docs/replication_units.csv", "docs/replication_units_summary.csv"]
    samples = namespace["checks"]["padding_samples_hubert-base"]
    assert samples["ok"].all() and len(samples) == 4 and set(samples["n_lengths_equal"]) == {3}
    assert set(samples["lengths_from"]) == {"training shards of hubert-base"}
    assert set(samples["redrawn"]) == {"same files in the same order"}        # both samples, drawn again
    timing = namespace["checks"]["padding_timing_hubert-base"]
    assert timing["step"].tolist() == ["deviation", "noise floor"] and timing["files_timed"].tolist() == [3, 3]
    assert timing["files"].tolist() == [12, 6] and (timing["estimated_minutes"] >= 0).all()
    assert namespace["out"].index("estimate for hubert-base:") < namespace["out"].index("clean random | files: 3")
    # Nothing is compared with an original table, and gate A4 is not claimed.
    assert namespace["compared"] == [] and "gate A4 passed" not in namespace["out"]
    assert "not a run of gate A4: the gate is not evaluated" in namespace["out"]
    assert [name for name, _, _ in namespace["written"]] == GATE_TABLES + RUN_TABLES
    assert all(extra == ["encoder", "revision"] for _, _, extra in namespace["written"])
    assert namespace["floors"] == ["clean", "trim40"] and namespace["loaded"] == ["cuda", "cpu"]
    assert "no prediction is stated for hubert-base" in namespace["out"]
    assert ("padding run finished: hubert-base | files measured: 12 | files of the noise floor: 6 "
            "| tables of gate A4: 8 unchanged | predictions: none stated for this encoder") in namespace["out"]


def test_the_padding_notebook_reports_the_predictions_of_wavlm_large_either_way(monkeypatch, capsys):
    namespace = run_the_padding_cells(monkeypatch, capsys, encoder="wavlm-large")
    assert list(namespace["checks"]) == ["padding_gate_commit_wavlm-large",
                                         "preregistration_padding_wavlm-large", "padding_encoder_wavlm-large",
                                         "padding_samples_wavlm-large", "padding_timing_wavlm-large",
                                         "padding_predictions_wavlm-large", "padding_gate_tables_wavlm-large"]
    assert namespace["checks"]["padding_gate_tables_wavlm-large"]["ok"].all()
    assert namespace["checks"]["padding_encoder_wavlm-large"]["feat_extract_norm"].tolist() == ["layer"]
    # No shards: the lengths come from the audio, and the most padded files are not chosen again.
    samples = namespace["checks"]["padding_samples_wavlm-large"].set_index(["condition", "sample"])
    assert samples["ok"].all() and set(samples["lengths_from"]) == {"audio"}
    assert samples.loc[("clean", "random"), "redrawn"] == "same files in the same order"
    assert samples.loc[("trim40", "most_padded"), "redrawn"] == "not redrawn"
    assert [name for name, _, _ in namespace["written"]] == GATE_TABLES + RUN_TABLES + ["padding_floor_predictions.csv"]
    assert namespace["written"][-1][1] == 2 * 24
    verdict = namespace["checks"]["padding_predictions_wavlm-large"]
    assert verdict["prediction"].tolist() == ["a", "b"] and verdict["holds"].tolist() == [True, True]
    assert verdict["n_cells"].tolist() == [48, 48] and "ok" not in verdict.columns
    assert "prediction (a) holds | met in 48 of 48 layers x conditions" in namespace["out"]
    assert "predictions: {'a': True, 'b': True}" in namespace["out"] and "gate A4 passed" not in namespace["out"]
    # A prediction that fails at one layer of one condition does not hold; the run still finishes.
    namespace = run_the_padding_cells(monkeypatch, capsys, encoder="wavlm-large", a_fails={("trim40", 7)})
    verdict = namespace["checks"]["padding_predictions_wavlm-large"]
    assert verdict["n_met"].tolist() == [47, 48] and verdict["holds"].tolist() == [False, True]
    assert "prediction (a) does not hold | met in 47 of 48 layers x conditions" in namespace["out"]
    assert "prediction (b) holds | met in 48 of 48 layers x conditions" in namespace["out"]
    assert "padding run finished: wavlm-large" in namespace["out"]
    assert "predictions: {'a': False, 'b': True}" in namespace["out"]
    assert namespace["written"][-1][0] == "padding_floor_predictions.csv"


def test_the_padding_notebook_stops_before_a_checkpoint_or_a_measurement(monkeypatch, capsys):
    from ssl_silence import verify

    # A pre-registered file that changed: nothing is loaded.
    for encoder in ("hubert-base", "wavlm-large", "wavlm-base"):
        with pytest.raises(verify.CheckFailed, match=f"preregistration_padding_{encoder}"):
            run_the_padding_cells(monkeypatch, capsys, encoder=encoder, registered=False)
        assert RESULT["loaded"] == [] and RESULT["measured"] == [] and RESULT["written"] == []
    # A cache on Drive: nothing is downloaded.
    with pytest.raises(AssertionError, match="the Hugging Face cache is on Drive"):
        run_the_padding_cells(monkeypatch, capsys, encoder="wavlm-large",
                              cache="/content/drive/MyDrive/cache/hub")
    assert RESULT["loaded"] == []
    # WavLM Large must have layer normalisation in its convolutional encoder.
    with pytest.raises(AssertionError):
        run_the_padding_cells(monkeypatch, capsys, encoder="wavlm-large", normalisation="group")
    assert RESULT["measured"] == [] and "padding_samples_wavlm-large" not in RESULT["checks"]
    # The samples of another encoder are a hard stop, before the timed check and the deviation.
    for encoder, options in (("hubert-base", {"other_random": True}), ("wavlm-large", {"other_random": True}),
                             ("hubert-base", {"other_most_padded": True}),
                             ("hubert-base", {"other_length": True}), ("wavlm-large", {"other_length": True})):
        with pytest.raises(verify.CheckFailed, match=f"padding_samples_{encoder}"):
            run_the_padding_cells(monkeypatch, capsys, encoder=encoder, **options)
        assert RESULT["measured"] == [] and RESULT["floors"] == [] and RESULT["written"] == [], (encoder, options)
        assert f"padding_timing_{encoder}" not in RESULT["checks"]
        failed = RESULT["checks"][f"padding_samples_{encoder}"]
        assert 0 < int((~failed["ok"]).sum()) < len(failed)
    # WavLM Large has no shards, so its most padded files cannot be chosen again: the option changes nothing.
    namespace = run_the_padding_cells(monkeypatch, capsys, encoder="wavlm-large", other_most_padded=True)
    assert namespace["checks"]["padding_samples_wavlm-large"]["ok"].all()


def test_the_padding_notebook_does_not_rewrite_the_tables_of_gate_a4(monkeypatch, capsys):
    from ssl_silence import verify

    # WavLM Base with its tables: the gate is not run again and its tables are read.
    namespace = run_the_padding_cells(monkeypatch, capsys, encoder="wavlm-base", gate_tables=8)
    assert namespace["REUSE"] and not namespace["GATE"]
    assert namespace["measured"] == [] and namespace["compared"] == []
    assert namespace["read"] == ["padding_deviation_files.csv", "padding_deviation.csv",
                                 "padding_deviation_files.csv"]
    # Nothing of the gate is written: neither its three tables nor its five check tables.
    assert [name for name, _, _ in namespace["written"]] == RUN_TABLES
    assert not set(namespace["checks"]) & set(GATE_CHECKS)
    assert list(namespace["checks"]) == ["padding_gate_commit_wavlm-base",
                                         "preregistration_padding_wavlm-base", "padding_encoder_wavlm-base",
                                         "padding_samples_wavlm-base", "padding_timing_wavlm-base",
                                         "padding_gate_tables_wavlm-base"]
    # Every table of the gate has the commit of the gate in its meta file, and the notebook prints it.
    commits = namespace["checks"]["padding_gate_commit_wavlm-base"]
    assert commits["file"].tolist() == namespace["gate_files"] and commits["ok"].all()
    assert set(commits["git_commit"]) == {"637b4b89b1fa8eb551f178c90a520df58a65a1ac"}
    assert namespace["out"].count("| commit of its meta file: 637b4b89b1fa8eb551f178c90a520df58a65a1ac") == 8
    assert "padding_deviation_files.csv | commit of its meta file: 637b4b8" in namespace["out"]
    # Its samples are checked like those of any other encoder, against its own shards.
    samples = namespace["checks"]["padding_samples_wavlm-base"]
    assert samples["ok"].all() and set(samples["lengths_from"]) == {"training shards of wavlm-base"}
    assert set(samples["redrawn"]) == {"same files in the same order"}
    # Only the noise floor is timed and measured.
    assert namespace["checks"]["padding_timing_wavlm-base"]["step"].tolist() == ["noise floor"]
    assert namespace["floors"] == ["clean", "trim40"] and namespace["loaded"] == ["cuda", "cpu"]
    unchanged = namespace["checks"]["padding_gate_tables_wavlm-base"]
    assert unchanged["file"].tolist() == namespace["gate_files"] and unchanged["ok"].all()
    assert (unchanged["sha256_at_start"] == unchanged["sha256_at_end"]).all()
    assert namespace["hashed"] == namespace["gate_files"] * 2
    out = namespace["out"]
    assert "the deviation is read from the tables of gate A4" in out and "gate A4 passed" not in out
    assert "tables of gate A4: 8 exist, written at commit 637b4b8; their sha256 is recorded and must not change" in out
    assert "read from the tables of gate A4, not measured" in out
    assert "not a run of gate A4: the gate is not evaluated" in out
    assert ("padding run finished: wavlm-base | files measured: none (read) | files of the noise floor: 6 "
            "| tables of gate A4: 8 unchanged | predictions: none stated for this encoder") in out

    # A table of the gate that changed during the run stops the notebook at the end, for every encoder.
    for encoder, table in (("wavlm-base", "padding_deviation_files.csv"), ("hubert-base", "padding_deviation.csv"),
                           ("wavlm-large", "padding_samples.csv")):
        with pytest.raises(verify.CheckFailed, match=f"padding_gate_tables_{encoder}"):
            run_the_padding_cells(monkeypatch, capsys, encoder=encoder, gate_tables=8, changed_table=table)
        failed = RESULT["checks"][f"padding_gate_tables_{encoder}"]
        assert [os.path.basename(path) for path in failed[~failed["ok"]]["file"]] == [table]
        assert "padding run finished" not in RESULT["out"]

    # A table of the gate whose meta file names another commit: nothing is loaded, for every encoder.
    for encoder, table in (("wavlm-base", "padding_deviation_files.csv"), ("hubert-base", "padding_samples.csv"),
                           ("wavlm-large", "padding_deviation_files.csv")):
        with pytest.raises(verify.CheckFailed, match=f"padding_gate_commit_{encoder}"):
            run_the_padding_cells(monkeypatch, capsys, encoder=encoder, gate_tables=8, other_commit=table)
        assert RESULT["loaded"] == [] and RESULT["measured"] == [] and RESULT["written"] == []
        assert list(RESULT["checks"]) == [f"padding_gate_commit_{encoder}"]        # before the pre-registered files
        failed = RESULT["checks"][f"padding_gate_commit_{encoder}"]
        assert [os.path.basename(path) for path in failed[~failed["ok"]]["file"]] == [table]
        assert f"{table} | commit of its meta file: 0043fff" in RESULT["out"]       # printed before the stop
    # Some tables of the gate and not the others: nothing is loaded, measured or written.
    with pytest.raises(AssertionError, match="only 3 of the 8 tables of gate A4 exist"):
        run_the_padding_cells(monkeypatch, capsys, encoder="wavlm-base", gate_tables=3)
    assert RESULT["loaded"] == [] and RESULT["written"] == [] and RESULT["checks"] == {}
    # Another encoder without the tables of the gate: WavLM Base comes first.
    for encoder in ("hubert-base", "wavlm-large"):
        with pytest.raises(AssertionError, match="the tables of gate A4 do not exist"):
            run_the_padding_cells(monkeypatch, capsys, encoder=encoder, gate_tables=0)
        assert RESULT["loaded"] == [] and RESULT["written"] == []

    # Without the tables, WavLM Base runs the gate as before and writes them.
    namespace = run_the_padding_cells(monkeypatch, capsys, encoder="wavlm-base", gate_tables=0)
    assert namespace["GATE"] and not namespace["REUSE"] and len(namespace["measured"]) == 4
    assert [name for name, _, _ in namespace["written"]] == GATE_TABLES + RUN_TABLES
    assert [name for name in namespace["checks"] if name in GATE_CHECKS] == GATE_CHECKS
    assert namespace["compared"] == ["padding_deviation", "padding_deviation_by_amount", "printed"]
    assert "padding_gate_tables_wavlm-base" not in namespace["checks"] and namespace["hashed"] == []
    assert "gate A4 passed: wavlm-base" in namespace["out"]



# The silence measurements (gate A5) ---------------------------------------------------------------

SILENCE = REPO / "notebooks/05_silence_measurements.ipynb"
SILENCE_TABLES = ["silence_summary", "residual_silence_summary", "trim_amount", "padding_only_baseline",
                  "padding_by_label", "silence_only_baseline", "residual_silence_baseline"]
SILENCE_GATE = ["silence_per_file", *[f"table_{name}" for name in SILENCE_TABLES], "silence_fallback_counts",
                "silence_trim_lengths", "silence_itw_durations", "silence_printed_numbers", "silence_quiet_files"]
SILENCE_RESULT: dict = {}


def test_the_silence_notebook_measures_every_file_and_stops_in_the_gate_cell():
    from ssl_silence import silence

    assert SILENCE in NOTEBOOKS and list(silence.GATE_CHECKS) == SILENCE_GATE
    assert list(silence.GATE_TABLES) == SILENCE_TABLES
    code_cells = cells(SILENCE, "code")
    code = "\n".join(code_cells)
    order = [code.index(piece) for piece in (
        "LOCAL = cfg.LOCAL_MEASUREMENTS", "assert not cfg.on_drive(LOCAL)",
        'record("silence_original_files", pd.DataFrame({',
        "datasets.ensure_asvspoof(cfg.la_zip())", "datasets.ensure_itw(cfg.itw_zip(), sha256=cfg.ITW_ZIP_SHA256)",
        "datasets.check_items(file_list, split)",
        "timed = file_lists[split].head(silence.TIMING_FILES)", "silence.measure_file(path)",
        'record("silence_timing", timing, stop=False)',
        "wide[split] = silence.measure_items(file_lists[split], LOCAL, split)",
        'provenance.write_table(wide[split], os.path.join(OUT, f"{split}.csv"))',
        'silence.compare_with_original_files(wide[split], split, os.path.join(ORIGINAL, "files"))',
        'record("silence_per_file", per_file, stop=False)',
        'shards.shard_lengths(cfg.feature_dir(SHARDS_OF, "unpadded", condition))',
        "silence.compare_with_shards(long, lengths, cfg.trim_top_db(condition))",
        'record("silence_fallback_counts", pd.concat(counts, ignore_index=True), stop=False)',
        'record("silence_trim_lengths", pd.concat(same, ignore_index=True), stop=False)',
        'record("silence_itw_durations", silence.duration_checks(raw["itw"]), stop=False)',
        "tables = silence.summary_tables(wide)", "silence.silence_only_baselines(measured,",
        'record(f"table_{name}", verify.compare_with_legacy(table, name, legacy, ORIGINAL), stop=False)',
        "share = silence.remaining_silence(residual)",
        "intext.compare_stated(intext.Tables.of({intext.RESIDUAL_SHARE: share}),",
        'record("silence_printed_numbers", stated, stop=False)',
        'record("silence_earlier_counts", silence.earlier_counts_report(share), stop=False)',
        "quiet = silence.quiet_files(raw)", "quiet_check = intext.compare_quiet(quiet)",
        'record("silence_quiet_files", quiet_check, stop=False)',
        "plots.silence_ecdf_figure(", "gate = {name: checks[name] for name in silence.GATE_CHECKS}",
        "for name, table in gate.items():", '"gate A5 passed | files measured:"')]
    assert order == sorted(order)
    # Nothing stops before the gate cell: every check table of the gate is written first.
    assert code.count("stop=False") == code.count("record(") - 1                 # all but the definition of record
    assert code.count("verify.require_ok(") == 2                                  # in record and in the gate cell
    # No tolerance, no number of the manuscript and no count is typed into the notebook.
    assert "tolerance" not in code and "1e-" not in code
    assert not re.search(r"\b(3\.1|4\.2|2\.2|4\.0|6\.7|3\.0|0\.16|0\.23)\b", code)
    assert not re.search(r"\b(1_?046|972|2_?382|938|19_?387|12_?392)\b", code)
    # The chunks are on the local disk; no encoder is loaded and no GPU is required.
    assert code.count("measure_items(") == 1 and "measure_items(file_lists[split], OUT" not in code
    for word in ("load_encoder", "torch.cuda", "write_shard", "torch.save", "extract_split"):
        assert word not in code, word
    # Only the scores of the silence-only classifiers are written again by a second run.
    assert code.count("overwrite=True") == 1
    text = " ".join("\n".join(cells(SILENCE, "markdown")).split())
    for piece in ("(gate A5)", "No GPU is needed and no encoder is loaded", "153,240 files",
                  "The chunks of the measurement are written to the local disk of the runtime only",
                  "Gate A5 is not part of the pre-registered analysis plan",
                  "written in `docs/decisions.md` (Section 1), before any run",
                  "The tolerance is 1e-9 (`verify.SECONDS_TOLERANCE`)", "A row that only one table has fails",
                  "19,387 files are at most 4 s long and 12,392 are longer",
                  "each rounded once to the digits the manuscript prints",
                  "Every file is counted, also a file that the fallback rule kept untrimmed",
                  "none is typed into this notebook", "`intext.STATED_QUIET`",
                  "No tolerance is loosened: a comparison that fails at 1e-9 stops the notebook",
                  "Reported and not part of the gate", "the timed check of 100 files per set"):
        assert piece in text, piece


def silence_tables(itw_window: int = 21, quiet_labels=(0, 0, 0, 0, 0)):
    """Wide tables of four small sets whose remaining silence gives the numbers
    that the manuscript prints: per set the files, the bonafide files, the
    files with more than 50 ms in the whole file and inside the window (per
    class), and the remaining length of those files."""
    import numpy as np
    import pandas as pd

    plan = {"train": (1_000, 100, {0: (10, 10), 1: (32, 30)}, 3_200),
            "dev": (1_000, 100, {0: (9, 9), 1: (30, 28)}, 3_680),
            "eval": (10_000, 1_000, {0: (67, 65), 1: (270, 250)}, 3_520),
            "itw": (1_000, 600, {0: (20, itw_window - 8), 1: (11, 8)}, 2_560)}
    wide = {}
    for split, (n, n_bonafide, over, lead_samples) in plan.items():
        label = np.array([0] * n_bonafide + [1] * (n - n_bonafide))
        lead, win_lead = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
        for value, (n_lead, n_window) in over.items():
            first = np.flatnonzero(label == value)
            lead[first[:n_lead]], win_lead[first[:n_window]] = lead_samples, lead_samples
        n_samples = np.where(np.arange(n) % 5 < 3, 48_000, 80_000)             # 60% of at most 4 s
        peak = np.full(n, 0.1)
        if split == "itw":
            for k, value in enumerate(quiet_labels):
                peak[np.flatnonzero(label == value)[-1 - k]] = 5e-4
        table = pd.DataFrame({"dataset": "x", "split": split, "audio_id": [f"{split}-{k}" for k in range(n)],
                              "label": label, "system_id": "-", "speaker": "s", "sr_orig": 16_000,
                              "n_samples": n_samples, "all_zero": 0, "peak_frame_rms": peak,
                              "residual_fallback_trim40": (np.arange(n) % 250 == 0).astype(int),
                              "residual_lead_40": lead, "residual_win_lead_40": win_lead,
                              "amount_n_samples_orig": n_samples, "amount_orig_s": n_samples / 16_000})
        for d, removed in ((40, 8_000), (20, 12_000), (15, 16_000)):
            fallback = (np.arange(n) % (10 * d) == 0).astype(int)
            table[f"amount_n_samples_trimmed_{d}"] = np.where(fallback == 1, n_samples, n_samples - removed)
            table[f"amount_removed_lead_s_{d}"] = np.where(fallback == 1, 0.0, removed / 16_000)
            table[f"amount_removed_trail_s_{d}"] = 0.0
            table[f"amount_removed_total_s_{d}"] = table[f"amount_removed_lead_s_{d}"]
            table[f"amount_fallback_{d}"] = fallback
            table[f"amount_too_short_{d}"] = 0
        wide[split] = table
    return wide


def run_the_silence_cells(monkeypatch, capsys, per_file_differs=False, table_differs=None, other_length=False,
                          other_count=False, other_rate=False, itw_window=21, quiet_labels=(0, 0, 0, 0, 0),
                          missing_original=False, local="/content/measurements"):
    """The cells of the silence notebook from the settings cell on. The
    measurement, the comparison with the original tables, the shards, the
    summary tables, the classifiers and the figure are stand-ins; the
    remaining silence, the printed numbers, the quiet files, the comparison
    with the shards and the duration subsets are the real functions on the
    tables of ``silence_tables``."""
    import numpy as np
    import pandas as pd

    from ssl_silence import datasets, plots, provenance, shards, silence, verify

    wide = silence_tables(itw_window, quiet_labels)
    if other_rate:
        wide["itw"].loc[3, "sr_orig"] = 44_100
    original = os.path.join(cfg.migration_dir(), "original")
    checks: dict = {}
    calls: list = []
    written, timed, baselines, figures = [], [], [], []

    def exists(path):
        if str(path).startswith(original):
            return not (missing_original and str(path).endswith(os.path.join("tables", "trim_amount.csv")))
        return real_exists(path)

    def items(split):
        table = wide[split][silence.ITEM_COLUMNS].copy()
        table["path"] = "audio/" + table["audio_id"]
        return table

    def measure_file(path):
        assert not any(call[0] == "measure_items" for call in calls)      # the timed check comes first
        timed.append(path)
        return {}

    def measure_items(file_list, out_dir, name):
        assert out_dir == local and not cfg.on_drive(out_dir)             # the chunks stay on the local disk
        assert list(file_list["audio_id"]) == list(wide[name]["audio_id"])
        calls.append(("measure_items", name))
        return wide[name]

    def compare_with_original_files(table, split, original_dir):
        assert table is wide[split] and original_dir == os.path.join(original, "files")
        calls.append(("per_file", split))
        return pd.DataFrame({"part": list(silence.ORIGINAL_PARTS), "split": split, "column": "lead_40",
                             "n_matched": len(table), "max_abs_diff": [0.0, 1.0 if per_file_differs else 0.0, 0.0],
                             "tolerance": verify.SECONDS_TOLERANCE,
                             "ok": [True, not (per_file_differs and split == "eval"), True]})

    def shard_lengths(folder):
        condition = next(c for c in ("trim40", "trim20", "trim15")
                         if folder == cfg.feature_dir("wavlm-base", "unpadded", c))
        d = cfg.trim_top_db(condition)
        calls.append(("shard_lengths", condition))
        parts = []
        for split, table in wide.items():
            parts.append(pd.DataFrame({
                "split": split, "audio_id": table["audio_id"], "label": table["label"],
                "n_real": np.minimum(table[f"amount_n_samples_trimmed_{d:g}"], cfg.WINDOW),
                "trim_fallback": table[f"amount_fallback_{d:g}"].astype(bool), "too_short": False}))
        lengths = pd.concat(parts, ignore_index=True)
        if other_length and condition == "trim20":
            lengths.loc[7, "n_real"] += 160
        if other_count and condition == "trim15":
            lengths.loc[1, "trim_fallback"] = True
        return lengths

    def summary_tables(tables):
        assert tables is namespace["wide"]
        return {name: pd.DataFrame({"table": [name]}) for name in silence.SUMMARY_TABLES}

    def silence_only_baselines(tables, top_dbs, condition, scores_dir, overwrite=False):
        part = namespace["raw"] if condition == "clean" else namespace["residual"]
        assert tables is part and list(tables) == list(cfg.SPLITS)
        baselines.append((condition, tuple(top_dbs), scores_dir, overwrite))
        return pd.DataFrame({"condition": condition, "top_db": list(top_dbs)})

    def compare_with_legacy(table, name, legacy, copies):
        assert copies == original and legacy is namespace["legacy"]
        calls.append(("legacy", name))
        return pd.DataFrame({"table": name, "column": ["a", "b"], "max_abs_diff": 0.0, "tolerance": 1e-9,
                             "ok": [True, name != table_differs]})

    def duration_checks(table):
        seconds = table["n_samples"] / 16_000
        return real_duration_checks(table, expected=(int((seconds <= 4).sum()), int((seconds > 4).sum())))

    def record(name, table, stop=True):
        checks[name] = table
        if stop and "ok" in table.columns:
            verify.require_ok(table, name)

    real_exists, real_duration_checks = os.path.exists, silence.duration_checks
    monkeypatch.setattr(cfg, "LOCAL_MEASUREMENTS", local)
    monkeypatch.setattr(os.path, "exists", exists)
    monkeypatch.setattr(provenance, "sha256_file", lambda path: f"sha256 of {os.path.basename(path)}")
    monkeypatch.setattr(provenance, "write_table", lambda table, path, **extra: written.append(
        (os.path.relpath(path, cfg.measurements_dir()).replace(os.sep, "/"), len(table))))
    monkeypatch.setattr(datasets, "ensure_asvspoof", lambda path: calls.append(("unpack", path)))
    monkeypatch.setattr(datasets, "ensure_itw", lambda path, sha256: calls.append(("unpack", path, sha256)))
    monkeypatch.setattr(datasets, "items", items)
    monkeypatch.setattr(datasets, "check_items", lambda table, split: {"n_files": len(table)})
    monkeypatch.setattr(silence, "measure_file", measure_file)
    monkeypatch.setattr(silence, "measure_items", measure_items)
    monkeypatch.setattr(silence, "compare_with_original_files", compare_with_original_files)
    monkeypatch.setattr(silence, "summary_tables", summary_tables)
    monkeypatch.setattr(silence, "silence_only_baselines", silence_only_baselines)
    monkeypatch.setattr(silence, "duration_checks", duration_checks)
    monkeypatch.setattr(shards, "shard_lengths", shard_lengths)
    monkeypatch.setattr(verify, "compare_with_legacy", compare_with_legacy)
    monkeypatch.setattr(plots, "silence_ecdf_figure", lambda tables: figures.append(list(tables)) or "figure")
    monkeypatch.setattr(plots, "save_figure", lambda figure, path, print_width: [path + ".pdf"])
    namespace = {"cfg": cfg, "verify": verify, "provenance": provenance, "pd": pd, "os": os, "checks": checks,
                 "CHECKS": "checks", "legacy": {"tables": {}}, "display": lambda table: None, "record": record}
    sources = cells(SILENCE, "code")
    k = next(i for i, source in enumerate(sources) if "from ssl_silence import datasets, plots, shards, silence" in source)
    try:
        for source in sources[k:]:
            exec(compile(python_of(source), SILENCE.name, "exec"), namespace)
    finally:
        namespace.update(calls=calls, written=written, timed=timed, baselines=baselines, figures=figures,
                         out=capsys.readouterr().out)
        SILENCE_RESULT.clear()
        SILENCE_RESULT.update(namespace)
    return namespace


def test_the_silence_notebook_passes_gate_a5_on_tables_that_give_the_printed_numbers(monkeypatch, capsys):
    from ssl_silence import silence
    from ssl_silence.paper import intext

    namespace = run_the_silence_cells(monkeypatch, capsys)
    checks = namespace["checks"]
    assert list(checks) == ["silence_original_files", "silence_timing", "silence_per_file",
                            "silence_fallback_counts", "silence_trim_lengths", "silence_itw_durations",
                            *[f"table_{name}" for name in SILENCE_TABLES], "silence_printed_numbers",
                            "silence_earlier_counts", "silence_quiet_files"]
    # The originals: twelve per-file tables and seven tables, with their sha256.
    files = checks["silence_original_files"]
    assert len(files) == 3 * 4 + 7 and files["file"].iloc[0] == "files/measurements_dir/train.csv"
    assert files["file"].iloc[-1] == "tables/residual_silence_baseline.csv"
    # 100 files per set are timed before anything is measured; then every set, on the local disk.
    assert len(namespace["timed"]) == 4 * silence.TIMING_FILES == 400
    assert checks["silence_timing"]["files"].tolist() == [1_000, 1_000, 10_000, 1_000]
    order = [name for name, *_ in namespace["calls"]]
    assert order == ["unpack"] * 2 + ["measure_items"] * 4 + ["per_file"] * 4 + ["shard_lengths"] * 3 + ["legacy"] * 7
    assert [call[1] for call in namespace["calls"] if call[0] == "measure_items"] == list(cfg.SPLITS)
    assert [call[1] for call in namespace["calls"] if call[0] == "legacy"] == SILENCE_TABLES
    # Drive receives the finished tables only: the per-file tables, the seven tables, the share, the quiet files.
    assert [name for name, _ in namespace["written"]] == [
        "train.csv", "dev.csv", "eval.csv", "itw.csv", *[f"{name}.csv" for name in SILENCE_TABLES],
        "residual_silence_share.csv", "quiet_file_counts.csv", "quiet_files.csv"]
    assert namespace["written"][2] == ("eval.csv", 10_000)
    assert os.path.basename(intext.RESIDUAL_SHARE) == "residual_silence_share.csv"
    # The classifiers at the thresholds of the original tables; their scores go to the baselines folder.
    scores = os.path.join(cfg.baselines_dir(), "scores")
    assert namespace["baselines"] == [("clean", (30, 40, 50), scores, True), ("trim40", (15, 20, 30), scores, True)]
    # The shards: three thresholds, every set, the counts per class and the length of every file.
    assert checks["silence_trim_lengths"]["n_files"].sum() == 3 * 13_000 and checks["silence_trim_lengths"]["ok"].all()
    assert checks["silence_fallback_counts"]["ok"].all() and len(checks["silence_fallback_counts"]) == 3 * 4 * 3
    assert checks["silence_fallback_counts"]["condition"].unique().tolist() == ["trim40", "trim20", "trim15"]
    assert checks["silence_itw_durations"]["ok"].all()
    # The printed numbers, computed by the real functions and rounded once.
    printed = checks["silence_printed_numbers"]
    assert printed["printed"].tolist() == [3.1, 4.2, 2.1, 4.0, 6.7, 3.0, 0.16, 0.23] and printed["ok"].all()
    assert printed["stated"].tolist() == [v for values in intext.STATED.values() for v in values]
    # The earlier counts are reported and decide nothing.
    assert "ok" not in checks["silence_earlier_counts"].columns
    assert "silence_earlier_counts" not in silence.GATE_CHECKS and "silence_timing" not in silence.GATE_CHECKS
    quiet = checks["silence_quiet_files"].iloc[0]
    assert (quiet["n_files"], quiet["n_bonafide"], quiet["n_spoof"], quiet["ok"]) == (5, 5, 0, True)
    assert namespace["quiet"]["table"].tolist() == ["itw"] * 5
    assert namespace["figures"] == [["ASVspoof train", "ASVspoof dev", "ASVspoof eval", "In-the-Wild"]]
    assert ("gate A5 passed | files measured: 13000 | columns compared file by file: 12 "
            "| original tables reproduced: 7 | printed numbers reproduced: 8 "
            "| quiet In-the-Wild files: 5") in namespace["out"]


@pytest.mark.parametrize("problem, check", [
    ({"per_file_differs": True}, "silence_per_file"), ({"table_differs": "trim_amount"}, "table_trim_amount"),
    ({"table_differs": "residual_silence_baseline"}, "table_residual_silence_baseline"),
    ({"other_count": True}, "silence_fallback_counts"), ({"other_length": True}, "silence_trim_lengths"),
    ({"other_rate": True}, "silence_itw_durations"), ({"itw_window": 22}, "silence_printed_numbers"),
    ({"quiet_labels": (0, 0, 0, 0)}, "silence_quiet_files"), ({"quiet_labels": (0, 0, 0, 0, 1)}, "silence_quiet_files"),
    ({"quiet_labels": (0,) * 6}, "silence_quiet_files")])
def test_a_failed_check_stops_the_silence_notebook_in_the_gate_cell_after_everything_was_written(
        monkeypatch, capsys, problem, check):
    from ssl_silence import verify

    with pytest.raises(verify.CheckFailed, match=check):
        run_the_silence_cells(monkeypatch, capsys, **problem)
    # Every check table, every table and the figure exist; the gate did not pass.
    assert [name for name in SILENCE_RESULT["checks"] if name in SILENCE_GATE] == [
        "silence_per_file", "silence_fallback_counts", "silence_trim_lengths", "silence_itw_durations",
        *SILENCE_GATE[1:8], "silence_printed_numbers", "silence_quiet_files"]
    assert not SILENCE_RESULT["checks"][check]["ok"].all()
    assert len(SILENCE_RESULT["written"]) == 4 + 7 + 3 and len(SILENCE_RESULT["figures"]) == 1
    assert "gate A5 passed" not in SILENCE_RESULT["out"]
    if "itw_window" in problem:
        # 22 of 1,000 files inside the window: 2.2 is printed and the stated number is 2.1.
        row = SILENCE_RESULT["checks"][check].iloc[2]
        assert (row["printed"], row["stated"], row["ok"]) == (2.2, 2.1, False)


def test_the_silence_notebook_stops_before_the_measurement_without_the_originals_or_with_chunks_on_drive(
        monkeypatch, capsys):
    with pytest.raises(AssertionError, match="notebooks/02_migrate_wavlm_features.ipynb first"):
        run_the_silence_cells(monkeypatch, capsys, missing_original=True)
    assert SILENCE_RESULT["calls"] == [] and SILENCE_RESULT["timed"] == [] and SILENCE_RESULT["checks"] == {}
    with pytest.raises(AssertionError, match="would be written to Drive"):
        run_the_silence_cells(monkeypatch, capsys, local="/content/drive/MyDrive/measurements")
    assert SILENCE_RESULT["calls"] == [] and SILENCE_RESULT["written"] == []


PROBES = REPO / "notebooks/07_probes.ipynb"
PROBE_GATE = ["probe_sample", "probe_speakers", "probe_targets", "probe_r2_original_targets",
              "probe_r2_new_targets", "probe_table_cells", "probe_printed_numbers"]
PROBE_RESULT: dict = {}


def test_the_probe_notebook_checks_the_sample_and_the_targets_before_the_long_part():
    from ssl_silence import probes

    assert PROBES in NOTEBOOKS and list(probes.GATE_CHECKS) == PROBE_GATE
    code_cells = cells(PROBES, "code")
    code = "\n".join(code_cells)
    order = [code.index(piece) for piece in (
        "N_JOBS = os.cpu_count()", "assert not cfg.on_drive(LOCAL)",
        'f"run notebooks/02_migrate_wavlm_features.ipynb first: {missing} are missing"',
        'f"run notebooks/05_silence_measurements.ipynb first: {MEASUREMENTS} is missing"',
        'record("probe_original_files", pd.DataFrame({', '"cores of the runtime:", N_JOBS',
        "datasets.ensure_asvspoof(cfg.la_zip())", "file_list = datasets.items(SPLIT)",
        "sample = probes.stratified_sample(file_list)", 'record("probe_sample", probes.compare_sample(',
        'record("probe_speakers", probes.speaker_check(sample))',
        'groups = sample["speaker"].astype(str).to_numpy()',
        "silence.silence_table(silence.read_measurements(MEASUREMENTS))",
        'shards.build_layer_cache(cfg.feature_dir(ENCODER, "unpadded", condition), SPLIT, CACHE[condition])',
        "probes.load_sample_features(",
        "probes.compute_targets(sample, os.path.join(LOCAL, condition), condition)",
        "probes.in_sample_order(", "probes.compare_targets(new_targets[condition], original_targets[condition]",
        'record("probe_targets", target_check)', "rerun = probes.rerun_table(target_check)",
        'record("probe_rerun", rerun, stop=False)', "probes.time_probes(",
        'record("probe_timing", timing, stop=False)', '"| estimate of the probes:"',
        "probes.attach_silence_targets(original_targets[condition], measurements, condition,",
        'f"{condition}__original_targets", n_jobs=N_JOBS)',
        "probes.compare_r2(r2_original[condition], condition, legacy, ORIGINAL)",
        'record("probe_r2_original_targets", on_original)', "if again[condition]:",
        "probes.attach_silence_targets(new_targets[condition], measurements, condition,",
        'f"{condition}__new_targets", n_jobs=N_JOBS)',
        "probes.merge_reruns(r2_original[condition], redone, probes.TARGETS[condition])",
        "probe_r2 = probes.assemble(r2, ENCODER)",
        "os.path.join(cfg.tables_dir(ENCODER), paper_tables.PROBE_R2)",
        'record("probe_r2_new_targets", on_new)',
        'record("probe_table_cells", probes.selected_cells_check(probe_r2, original_table))',
        "intext.compare_stated(intext.Tables.of({paper_tables.PROBE_R2: probe_r2}), intext.probe_entries(),",
        'record("probe_printed_numbers", stated)', "plots.probe_figure(",
        "gate = {name: checks[name] for name in probes.GATE_CHECKS}", '"gate A6 passed | files sampled:"')]
    assert order == sorted(order)
    # Every check of the gate stops where it fails; only the reported tables do not stop.
    assert code.count("stop=False") == 3 and code.count("record(") == 1 + 3 + len(PROBE_GATE)
    # No tolerance, no number of the manuscript, no number of files or speakers is typed into the notebook.
    assert "tolerance" not in code.lower() and "1e-" not in code
    assert not re.search(r"\b(0\.74|0\.9|0\.90|0\.96|0\.88|768|5_?000|20)\b", code)
    # The probes use every core; the blocks are on Drive, the chunks and the cache on the local disk.
    assert code.count("n_jobs=N_JOBS") == 2 and code.count("probes.probe_blocks(") == 2
    assert code.count("BLOCKS") == 3 and 'BLOCKS = os.path.join(OUT, "blocks")' in code
    assert "probes.compute_targets(sample, OUT" not in code and "probes.run_probes(" not in code
    for word in ("load_encoder", "torch.cuda", "write_shard", "torch.save", "extract_split"):
        assert word not in code, word
    text = " ".join("\n".join(cells(PROBES, "markdown")).split())
    for piece in ("(gate A6)", "No GPU is needed and no encoder is loaded", "Choose a high-RAM runtime",
                  "the probes use every core of the runtime",
                  "prints the number of cores and a timed estimate before the long part",
                  "Gate A6 is not part of the pre-registered analysis plan",
                  "written in `docs/decisions.md` (Section 2), before any run",
                  "official ASVspoof 2019 LA training protocol", "`probes.N_SPEAKERS`",
                  "`verify.TARGET_TOLERANCE`", "`probes.R2_TOLERANCE`",
                  "the median alpha, the number of files and the number of speakers must be equal",
                  "whose last digits depend on the CPU and on the number of threads",
                  "the machine that wrote the original tables is not recorded",
                  "The largest difference is printed, and whether it is exactly 0.0",
                  "By rule, not by a list", "every target with any difference, however small, is probed again",
                  "states a lower bound and is cut off (`intext.lower_bound`)",
                  "None is typed into this notebook",
                  "The four targets that only the original table has are listed in the check table as not ported",
                  "a run that starts again reads the finished blocks instead of computing them",
                  "only read if it was computed from the same features, targets and speaker ids"):
        assert piece in text, piece


def probe_spec():
    columns = {name: name for name in ("cv", "layer", "target", "control", "n_files", "n_groups", "r2_mean",
                                       "r2_sd", "alpha_median", *[f"r2_fold_{i}" for i in range(1, 6)])}
    keys = ["condition", "cv", "layer", "target", "control"]
    return {"tables": {
        "probe_r2_clean": {"file": "-", "columns": columns, "constants": {"condition": "clean"}, "keys": keys},
        "probe_r2_trim40": {"file": "-", "columns": columns, "constants": {"condition": "trim40"}, "keys": keys,
                            "where": {"target": ["f0_mean_hz", "hnr_mean_db", "tilt_db_per_khz", "pad",
                                                 "length_in_window"]}}}}


def run_the_probe_cells(monkeypatch, capsys, tmp_path, other_sample=False, n_speakers=20, target_shift=None,
                        original_shift=None, trimmed_floor=0.886956, missing=None, local=None):
    """The cells of the probe notebook from the settings cell on, on a small
    training set under a temporary root. The datasets, the acoustic targets,
    the stored features, the fits and the figure are stand-ins; the sample,
    the comparisons, the rule of check 4, the blocks, the tables and the
    printed numbers are the real functions on real files.

    ``target_shift``: (condition, target, amount) added to one new target
    value. ``original_shift``: (column, amount) added to one row of the
    original clean R2 table. The stand-in of a fit returns fixed values plus
    the change of the sum of its target, so other targets give other R2."""
    import numpy as np
    import pandas as pd

    from ssl_silence import datasets, plots, probes, provenance, shards, silence, verify

    root, local_root = str(tmp_path / "root"), local or str(tmp_path / "local")
    monkeypatch.setattr(cfg, "ROOT", root)
    monkeypatch.setattr(cfg, "LOCAL_CACHE", local_root)
    original = os.path.join(cfg.migration_dir(), "original")
    n = 400
    label = np.array([0] * 40 + [1] * 360)
    file_list = pd.DataFrame({
        "dataset": "asvspoof19", "split": "train", "audio_id": [f"LA_T_{k:04d}" for k in range(n)],
        "path": [f"audio/LA_T_{k:04d}.flac" for k in range(n)], "label": label,
        "system_id": ["-" if k < 40 else f"A0{1 + k % 6}" for k in range(n)],
        "speaker": [f"LA_{k % n_speakers:04d}" for k in range(n)]})
    real_sample = getattr(probes.stratified_sample, "real", probes.stratified_sample)   # also in a second run
    sample = real_sample(file_list, n=100)
    order = np.arange(100)
    base = pd.DataFrame({"audio_id": sample["audio_id"].to_numpy(), "label": sample["label"].to_numpy(),
                         "system_id": sample["system_id"].to_numpy(), "f0_mean_hz": 100.0 + order,
                         "hnr_mean_db": 10.0 + 0.125 * order, "tilt_db_per_khz": -1.0 - 0.0625 * order,
                         "n_voiced_frames": 50 + order, "n_hnr_frames": 60 + order})
    measured = pd.DataFrame({"audio_id": file_list["audio_id"], "win_lead_40": np.arange(n) % 7 * 160.0,
                             "pad": np.arange(n) % 5 * 8_000.0, "pad_trimmed_40": np.arange(n) % 4 * 8_000.0,
                             "trim_fallback_40": (np.arange(n) % 50 == 0).astype(int)})
    layers = list(range(1, 13))
    lengths = {"clean": np.linspace(0.9617, 0.9003, 12), "trim40": np.linspace(0.9628, trimmed_floor, 12)}

    def lookup(condition, name, layer):
        if name in ("pad", "length_in_window"):
            return float(lengths[condition][layer - 1])
        return 0.7365 if name == "win_lead_40" else 0.8 - 0.01 * layer - 0.02 * len(name) / 20

    def full_targets(table, condition):
        names = [name for name in probes.TARGETS[condition] if name in probes.SILENCE_TARGETS]
        return probes.attach_silence_targets(table, measured, condition, names)

    sums = {condition: {name: float(np.sum(full_targets(base, condition)[name].to_numpy(dtype=np.float64)))
                        for name in probes.TARGETS[condition]} for condition in probes.GATE_CONDITIONS}
    calls, fits, written, figures = [], [], [], []

    def rows_of(condition, names, schemes, change=None):
        rows = []
        for name in names:
            delta = 0.0 if change is None else change[name]
            for cv in schemes:
                for layer in layers:
                    for control in ("real", "shuffled"):
                        value = lookup(condition, name, layer) + delta if control == "real" else -0.01
                        row = {"cv": cv, "layer": layer, "target": name, "control": control, "n_files": 5000,
                               "n_groups": 20, "r2_mean": value, "r2_sd": 0.01,
                               "alpha_median": 10.0}
                        row.update({f"r2_fold_{i}": value for i in range(1, 6)})
                        rows.append(row)
        return pd.DataFrame(rows)

    def probe_all_layers(features, targets, target_names, groups=None, cv_schemes=(), layers=None, n_jobs=1,
                         progress=True):
        condition = "clean" if features[0, 0, 0] == 0 else "trim40"
        assert n_jobs == os.cpu_count() and len(groups) == len(targets) == 100
        change = {name: float(np.sum(targets[name].to_numpy(dtype=np.float64))) - sums[condition][name]
                  for name in target_names}
        fits.append((condition, tuple(target_names), tuple(cv_schemes)))
        return rows_of(condition, target_names, cv_schemes, change)

    def compute_targets(table, out_dir, condition):
        assert out_dir == os.path.join(local_root, "probes", condition) and not cfg.on_drive(out_dir)
        assert list(table["audio_id"]) == list(sample["audio_id"])
        calls.append(("targets", condition))
        out = base.copy()
        if target_shift and target_shift[0] == condition:
            out.loc[3, target_shift[1]] += target_shift[2]
        return out

    def load_sample_features(cache_dir, split, audio_ids):
        condition = os.path.basename(cache_dir)
        assert list(audio_ids) == list(sample["audio_id"]) and split == "train"
        calls.append(("features", condition))
        return np.full((100, 12, 3), 0.0 if condition == "clean" else 1.0, dtype=np.float32), layers

    def time_probes(features, targets, name, groups, n_probes, n_jobs):
        calls.append(("timing", n_probes, n_jobs))
        return pd.DataFrame({"cv": list(probes.CV_SCHEMES), "probes_timed": n_jobs, "seconds": [1.0, 2.0],
                             "n_probes": n_probes, "n_jobs": n_jobs, "estimated_minutes": [3.0, 4.5]})

    def record(name, table, stop=True):
        checks[name] = table
        if stop and "ok" in table.columns:
            verify.require_ok(table, name)

    # The original files and tables, written as the migration notebook leaves them.
    os.makedirs(os.path.join(original, "files"), exist_ok=True)
    os.makedirs(os.path.join(original, "tables"), exist_ok=True)
    kept = base[["audio_id", "label", "system_id"]]
    (kept.iloc[::-1] if other_sample else kept).to_csv(os.path.join(original, "files", "probe_sample.csv"),
                                                       index=False)
    for condition in probes.GATE_CONDITIONS:
        base.iloc[::-1].to_csv(os.path.join(original, "files", f"probe_targets_{condition}.csv"), index=False)
        table = rows_of(condition, probes.TARGETS[condition], probes.CV_SCHEMES)
        if condition == "trim40":
            others = rows_of("clean", ["win_lead_40"], probes.CV_SCHEMES)
            table = pd.concat([table] + [others.assign(target=name) for name in (
                "win_lead_40", "win_trail_40", "speech_fraction", "length_residual")], ignore_index=True)
        elif original_shift:
            table.loc[5, original_shift[0]] += original_shift[1]
        table.to_csv(os.path.join(original, "tables", f"probe_r2_{condition}.csv"), index=False)
    os.makedirs(cfg.measurements_dir(), exist_ok=True)
    open(os.path.join(cfg.measurements_dir(), "train.csv"), "w").close()
    for name in ([missing] if missing else []):
        os.remove(os.path.join(root, *name.split("/")))

    checks: dict = {}
    real_write = getattr(provenance.write_table, "real", provenance.write_table)

    def write_table(table, path, **extra):
        written.append(os.path.relpath(path, root).replace(os.sep, "/"))
        return real_write(table, path, **extra)

    def stratified_sample(table):
        return real_sample(table, n=100)

    write_table.real, stratified_sample.real = real_write, real_sample
    monkeypatch.setattr(provenance, "write_table", write_table)
    monkeypatch.setattr(datasets, "ensure_asvspoof", lambda path: calls.append(("unpack", path)))
    monkeypatch.setattr(datasets, "items", lambda split: file_list)
    monkeypatch.setattr(datasets, "check_items", lambda table, split: {"n_files": len(table)})
    monkeypatch.setattr(probes, "stratified_sample", stratified_sample)
    monkeypatch.setattr(probes, "require_praat", lambda: {"parselmouth": "0.4.7", "praat": "6.1.38"})
    monkeypatch.setattr(probes, "praat_versions", lambda: {"parselmouth": "0.4.7", "praat": "6.1.38"})
    monkeypatch.setattr(probes, "compute_targets", compute_targets)
    monkeypatch.setattr(probes, "load_sample_features", load_sample_features)
    monkeypatch.setattr(probes, "probe_all_layers", probe_all_layers)
    monkeypatch.setattr(probes, "time_probes", time_probes)
    monkeypatch.setattr(silence, "read_measurements", lambda path: calls.append(("measurements", path)) or "wide")
    monkeypatch.setattr(silence, "silence_table", lambda wide: measured)
    monkeypatch.setattr(shards, "build_layer_cache", lambda feature_dir, split, cache_dir: calls.append(
        ("cache", feature_dir, cache_dir)))
    monkeypatch.setattr(plots, "probe_figure", lambda r2, encoder: figures.append(len(r2)) or "figure")
    monkeypatch.setattr(plots, "save_figure", lambda figure, path, print_width: [path + ".pdf"])
    namespace = {"cfg": cfg, "verify": verify, "provenance": provenance, "pd": pd, "os": os, "checks": checks,
                 "CHECKS": "checks", "legacy": probe_spec(), "display": lambda table: None, "record": record}
    sources = cells(PROBES, "code")
    k = next(i for i, source in enumerate(sources)
             if "from ssl_silence import datasets, plots, probes, shards, silence" in source)
    try:
        for source in sources[k:]:
            exec(compile(python_of(source), PROBES.name, "exec"), namespace)
    finally:
        blocks = os.path.join(cfg.tables_dir("wavlm-base"), "probes", "blocks")
        namespace.update(calls=calls, fits=fits, written=written, figures=figures, root=root,
                         blocks=sorted(os.listdir(blocks)) if os.path.isdir(blocks) else [],
                         out=capsys.readouterr().out)
        PROBE_RESULT.clear()
        PROBE_RESULT.update(namespace)
    return namespace


def test_the_probe_notebook_passes_gate_a6_and_a_second_run_reads_the_blocks(monkeypatch, capsys, tmp_path):
    from ssl_silence import probes

    namespace = run_the_probe_cells(monkeypatch, capsys, tmp_path)
    checks = namespace["checks"]
    assert list(checks) == ["probe_original_files", "probe_sample", "probe_speakers", "probe_targets",
                            "probe_rerun", "probe_timing", "probe_r2_original_targets", "probe_r2_new_targets",
                            "probe_table_cells", "probe_printed_numbers"]
    assert checks["probe_original_files"]["file"].tolist() == [
        "files/probe_sample.csv", "files/probe_targets_clean.csv", "files/probe_targets_trim40.csv",
        "tables/probe_r2_clean.csv", "tables/probe_r2_trim40.csv"]
    # The sample, its speakers, the cache, the targets and the timed check come before any fit.
    order = [call[0] for call in namespace["calls"]]
    assert order == ["unpack", "measurements", "cache", "features", "cache", "features", "targets", "targets",
                     "timing"]
    assert [call[1] for call in namespace["calls"] if call[0] == "cache"] == [
        cfg.feature_dir("wavlm-base", "unpadded", condition) for condition in ("clean", "trim40")]
    assert all(not cfg.on_drive(call[2]) for call in namespace["calls"] if call[0] == "cache")
    # Identical targets: nothing is probed again, and the estimate counts one pass of 11 targets and 12 layers.
    assert not checks["probe_rerun"]["probed_again"].any() and len(checks["probe_rerun"]) == 6 + 5
    assert namespace["again"] == {"clean": [], "trim40": []}
    assert namespace["calls"][-1] == ("timing", (6 + 5) * 12 * 2, os.cpu_count())
    assert "| probes per scheme: 264 | estimate of the probes: 7.5 minutes" in namespace["out"]
    assert f"cores of the runtime: {os.cpu_count()}" in namespace["out"]
    # One fit per target and scheme of each condition, each written as a block with its meta file.
    assert namespace["fits"] == [(condition, (name,), (cv,)) for condition in ("clean", "trim40")
                                 for name in probes.TARGETS[condition] for cv in probes.CV_SCHEMES]
    assert len(namespace["blocks"]) == 2 * 22 and all("__original_targets__" in name for name in namespace["blocks"])
    # Both conditions are compared with the original tables; the four original-only targets are listed.
    for name in ("probe_r2_original_targets", "probe_r2_new_targets"):
        table = checks[name]
        assert table["ok"].all() and table[table["note"] == ""]["identical"].all()
        assert table[table["note"] != ""]["column"].tolist() == [
            "target win_lead_40", "target win_trail_40", "target speech_fraction", "target length_residual"]
        assert set(table[table["note"] != ""]["table"]) == {"probe_r2_trim40"}
    assert namespace["out"].count("largest difference from the original tables: 0.0 | exactly 0.0: True") == 2
    # Drive: the targets and the table of each condition, one R2 table of both, the selected layers.
    assert namespace["written"] == [
        "tables/wavlm-base/probes/clean/probe_targets.csv", "tables/wavlm-base/probes/trim40/probe_targets.csv",
        "tables/wavlm-base/probes/clean/probe_r2_table.csv", "tables/wavlm-base/probes/trim40/probe_r2_table.csv",
        "tables/wavlm-base/probe_r2.csv", "tables/wavlm-base/probes/clean/probe_r2_selected.csv"]
    probe_r2 = namespace["probe_r2"]
    assert len(probe_r2) == (6 + 5) * 2 * 12 * 2 and probe_r2["condition"].unique().tolist() == ["clean", "trim40"]
    assert len(checks["probe_table_cells"]) == 20 and checks["probe_table_cells"]["ok"].all()
    printed = checks["probe_printed_numbers"]
    assert printed["printed"].tolist() == [768, 5000, 20, 0.74, 0.9, 0.96, 0.88] and printed["ok"].all()
    assert namespace["figures"] == [6 * 2 * 12 * 2]
    assert ("gate A6 passed | files sampled: 100 | speakers: 20 | targets probed again: 0 "
            "| largest difference on the original targets: 0.0 | on the new targets: 0.0 "
            "| printed numbers reproduced: 7") in namespace["out"]
    # A run that starts again fits nothing: every block is read from Drive.
    again = run_the_probe_cells(monkeypatch, capsys, tmp_path)
    assert again["fits"] == [] and again["blocks"] == namespace["blocks"]
    assert again["checks"]["probe_r2_new_targets"]["ok"].all() and "gate A6 passed" in again["out"]
    assert again["out"].count("read from ") == 22


def test_a_target_that_differs_in_the_last_digit_is_probed_again_and_the_others_are_not(monkeypatch, capsys,
                                                                                       tmp_path):
    namespace = run_the_probe_cells(monkeypatch, capsys, tmp_path, target_shift=("clean", "f0_mean_hz", 1e-11))
    checks = namespace["checks"]
    targets = checks["probe_targets"].set_index(["condition", "column"])
    assert targets["ok"].all() and not targets.loc[("clean", "f0_mean_hz"), "identical"]
    assert targets["identical"].sum() == len(targets) - 1
    assert namespace["again"] == {"clean": ["f0_mean_hz"], "trim40": []}
    assert "probed again with the new targets (check 4): {'clean': ['f0_mean_hz'], 'trim40': []}" in namespace["out"]
    assert namespace["calls"][-1][1] == (6 + 5 + 1) * 12 * 2
    # 22 blocks on the original targets, then the two blocks of the one target on the new values.
    assert len(namespace["fits"]) == 22 + 2 and namespace["fits"][-2:] == [
        ("clean", ("f0_mean_hz",), ("random",)), ("clean", ("f0_mean_hz",), ("speaker",))]
    new_blocks = [name for name in namespace["blocks"] if "__new_targets__" in name]
    assert new_blocks == ["clean__new_targets__f0_mean_hz__random.csv",
                          "clean__new_targets__f0_mean_hz__random.meta.json",
                          "clean__new_targets__f0_mean_hz__speaker.csv",
                          "clean__new_targets__f0_mean_hz__speaker.meta.json"]
    # The difference is reported, is inside the tolerance and is not 0.0.
    on_new = checks["probe_r2_new_targets"]
    assert on_new["ok"].all() and not on_new.set_index(["table", "column"]).loc[("probe_r2_clean", "r2_mean"),
                                                                              "identical"]
    assert checks["probe_r2_original_targets"][checks["probe_r2_original_targets"]["note"] == ""]["identical"].all()
    assert "| exactly 0.0: False" in namespace["out"] and "| targets probed again: 1 |" in namespace["out"]
    assert "gate A6 passed" in namespace["out"]


@pytest.mark.parametrize("problem, check, n_fits", [
    ({"other_sample": True}, "probe_sample", 0), ({"n_speakers": 19}, "probe_speakers", 0),
    ({"target_shift": ("trim40", "hnr_mean_db", 1e-3)}, "probe_targets", 0),
    ({"original_shift": ("r2_mean", 1e-6)}, "probe_r2_original_targets", 22),
    ({"original_shift": ("alpha_median", 1e-12)}, "probe_r2_original_targets", 22),
    ({"target_shift": ("trim40", "tilt_db_per_khz", 1e-7)}, "probe_r2_new_targets", 24),
    ({"trimmed_floor": 0.8799}, "probe_printed_numbers", 22)])
def test_a_failed_check_stops_the_probe_notebook_where_it_fails(monkeypatch, capsys, tmp_path, problem, check,
                                                                n_fits):
    from ssl_silence import verify

    with pytest.raises(verify.CheckFailed, match=check):
        run_the_probe_cells(monkeypatch, capsys, tmp_path, **problem)
    assert list(PROBE_RESULT["checks"])[-1] == check and not PROBE_RESULT["checks"][check]["ok"].all()
    assert len(PROBE_RESULT["fits"]) == n_fits and "gate A6 passed" not in PROBE_RESULT["out"]
    if n_fits == 0:
        # Nothing of the long part ran: no timed check, no fit, no block.
        assert not any(call[0] == "timing" for call in PROBE_RESULT["calls"]) and PROBE_RESULT["blocks"] == []
    if check == "probe_r2_original_targets":
        assert "tables/wavlm-base/probe_r2.csv" not in PROBE_RESULT["written"]
    if check == "probe_printed_numbers":
        row = PROBE_RESULT["checks"][check].iloc[-1]
        assert (row["printed"], row["stated"], row["ok"]) == (0.87, 0.88, False)
        assert PROBE_RESULT["figures"] == []


def test_the_probe_notebook_stops_before_anything_without_its_inputs_or_with_chunks_on_drive(monkeypatch, capsys,
                                                                                            tmp_path):
    with pytest.raises(AssertionError, match="notebooks/02_migrate_wavlm_features.ipynb first"):
        run_the_probe_cells(monkeypatch, capsys, tmp_path, missing="migration/original/tables/probe_r2_trim40.csv")
    assert PROBE_RESULT["calls"] == [] and PROBE_RESULT["checks"] == {}
    with pytest.raises(AssertionError, match="notebooks/05_silence_measurements.ipynb first"):
        run_the_probe_cells(monkeypatch, capsys, tmp_path, missing="measurements/train.csv")
    assert PROBE_RESULT["calls"] == [] and PROBE_RESULT["checks"] == {}
    with pytest.raises(AssertionError, match="would be written to Drive"):
        run_the_probe_cells(monkeypatch, capsys, tmp_path, local="/content/drive/MyDrive/cache")
    assert PROBE_RESULT["calls"] == [] and PROBE_RESULT["written"] == []


# The supplementary condition trimmed by voice activity detection -----------------------------------

VAD_NOTEBOOK = REPO / "notebooks/10_vad_condition.ipynb"
VAD_EXISTING = ["runs/wavlm-base/unpadded/runs.csv",
                "runs/wavlm-base/unpadded/scores/wavlm-base_unpadded_clean_L03_mean_s42.npz",
                "runs/hubert-base/unpadded/runs.csv", "tables/wavlm-base/effects_2x2.csv",
                "tables/wavlm-base/parts/effects_2x2_trim40.csv", "tables/hubert-base/effects_2x2.csv",
                "checks/analysis_steps.csv", "checks/table_effects_2x2_trim40.csv"]
VAD_CHECKS_SHORT = ["vad_file_lists", "vad_encoder", "vad_short_check", "vad_estimate"]
VAD_CHECKS = VAD_CHECKS_SHORT + ["vad_shards", "vad_shard_meta", "vad_training_clean", "vad_clean_scores",
                                 "vad_training", "vad_score_sets", "vad_run_records", "vad_steps"]
VAD_TABLES = ["cells_2x2", "effects_vad_and_trim40", "removed_amount"]
VAD_RESULT: dict = {}


def test_the_vad_notebook_is_supplementary_and_writes_only_to_its_own_folders():
    assert VAD_NOTEBOOK in NOTEBOOKS
    code_cells = cells(VAD_NOTEBOOK, "code")
    code, text = "\n".join(code_cells), "\n".join(cells(VAD_NOTEBOOK, "markdown"))
    assert "**Not in the analysis plan and not pre-registered** (supplementary analysis)" in text
    assert "`docs/decisions.md`, Section 4" in text
    order = [code.index(part) for part in (
        "EXTRACT = False ", "BEFORE = verify.file_hashes(GUARDED, skip=OWN, skip_folders=OWN_FOLDERS)",
        'record("vad_file_lists"', "encoders.load_encoder(ENCODER, DEVICE)", "vad.short_check(",
        'verify.require_ok(short, ', "extraction.extract_shards(", "shards.kept_intervals(FEATURES)",
        "vad.shards_check(kept, reference, RULE)", "training.run_experiments(clean_runs, runs_dir=RUNS",
        'record("vad_clean_scores", clean_scores)          # a hard stop',
        "training.run_experiments(vad_runs, runs_dir=RUNS", "training.score_set_check(planned, RUNS, file_lists)",
        "analysis.ScoreTables(", "tables.effects()", "vad.side_by_side(effects, reference_effects)",
        "tables.per_attack_effect()",
        "verify.unchanged_check(BEFORE, GUARDED, skip=OWN, skip_folders=OWN_FOLDERS)",
        'record("vad_files_unchanged", unchanged)          # a hard stop')]
    assert order == sorted(order)
    # The runs have their own folder, the existing runs are only read, and the rule is not set here.
    assert 'RUNS = cfg.vad_runs_dir(ENCODER, "unpadded")' in code
    assert code.count("training.run_experiments(") == 2 and code.count("runs_dir=RUNS") == 2
    assert code.count("EXISTING_RUNS") == 3 and "base_runs_dir=EXISTING_RUNS" in code
    assert "vad.clean_scores_check(planned, RUNS, EXISTING_RUNS)" in code
    assert 'GUARDED = [os.path.join(cfg.ROOT, "runs"), os.path.join(cfg.ROOT, "tables"), CHECKS]' in code
    for word in ("aggressiveness", "frame_ms", "run_frames", "webrtcvad.", "tolerance", "n_boot=", "seeds="):
        assert word not in code, word
    assert "RULE = cfg.vad_rule(CONDITION)" in code and "condition=CONDITION" in code
    # Everything after the short check is behind EXTRACT, except the last cell.
    k = next(i for i, source in enumerate(code_cells) if "extraction.extract_shards(" in source)
    assert all(source.startswith("if EXTRACT:\n") for source in code_cells[k:-1])
    assert "EXTRACT" not in code_cells[-1].split("print(")[0]
    # Every check table of the notebook has a name that the comparison leaves out, and no other.
    names = re.findall(r'record\("([^"]+)"', code)
    assert names == VAD_CHECKS + ["vad_files_unchanged"] and 'OWN = ("vad_",)' in code


def run_the_vad_cells(monkeypatch, capsys, tmp_path, extract=True, clean_ok=True, disturb=None,
                      short_ok=True, files_ok=True):
    """The cells of the notebook from its settings cell on, on a folder that stands for Drive,
    with stand-ins for the datasets, the encoder, the extraction, the training and the tables."""
    import pandas as pd
    import torch

    from ssl_silence import analysis, datasets, encoders, extraction, provenance, shards, training, vad, verify

    root = tmp_path / "drive"
    for file in VAD_EXISTING:
        (root / file).parent.mkdir(parents=True, exist_ok=True)
        (root / file).write_bytes(f"verified: {file}\n".encode())
    before = {file: (root / file).read_bytes() for file in VAD_EXISTING}
    monkeypatch.setattr(cfg, "ROOT", str(root))
    events, checks, written = [], {}, []
    VAD_RESULT.clear()
    VAD_RESULT.update(events=events, checks=checks, written=written, root=root, before=before)
    sizes = {"train": 30, "dev": 20, "eval": 40, "itw": 10}
    planned = vad.planned_runs()
    features = cfg.feature_dir("wavlm-base", "unpadded", "vad")
    runs_dir = cfg.vad_runs_dir("wavlm-base", "unpadded")

    def record(name, table, stop=True):
        checks[name] = table
        os.makedirs(cfg.checks_dir(), exist_ok=True)
        for suffix in (".csv", ".meta.json"):
            table.to_csv(os.path.join(cfg.checks_dir(), name + suffix), index=False)
        if stop and "ok" in table.columns:
            verify.require_ok(table, name)

    def items(split):
        return pd.DataFrame({"audio_id": [f"{split}_{k}" for k in range(sizes[split])], "label": 0,
                             "speaker": "s", "system_id": "A07", "path": "x"})

    def shard_lengths(folder):
        assert folder == cfg.feature_dir("wavlm-base", "unpadded", "clean")
        return pd.DataFrame({"reference": [1]})

    def file_list_check(file_list, reference, split):
        assert list(reference.columns) == ["reference"] and len(file_list) == sizes[split]
        return pd.DataFrame({"split": [split], "n_files": [sizes[split]], "ok": [files_ok or split != "itw"]})

    def short_check(file_list, split, *, model, device):
        assert (model, device) == ("the encoder", "cuda") and len(file_list) == sizes[split]
        events.append(f"short check {split}")
        n = cfg.SHORT_CHECK_FILES
        return pd.DataFrame({"split": split, "condition": "vad", "label": [0, 1] * (n // 2) + [0] * (n % 2),
                             "n_samples": 32_000, "kept_start": 1_600, "kept_end": 30_400,
                             "trim_fallback": [True] + [False] * (n - 1), "seconds": 0.01,
                             "ok": [short_ok or split != "dev"] * n})

    def extract_shards(file_list, folder, split, *, encoder, extraction, condition, device, model):
        assert (folder, encoder, extraction, condition, device, model) == (
            features, "wavlm-base", "unpadded", "vad", "cuda", "the encoder")
        events.append(f"extract {split}")
        return [f"{split}_shard_0000.pt"]

    def kept_intervals(folder):
        assert folder == features
        return pd.DataFrame({"kept": [1]})

    def shards_check(kept, reference, rule):
        assert list(kept.columns) == ["kept"] and list(reference.columns) == ["reference"]
        assert rule == cfg.vad_rule("vad")
        events.append("shards check")
        return pd.DataFrame({"split": list(sizes), "n_fallback": [1, 0, 2, 0], "ok": True})

    def shard_meta_check(table, encoder, condition):
        assert (table, encoder, condition) == ("meta of " + features, "wavlm-base", "vad")
        return pd.DataFrame({"n_shards": [4], "n_files": [sum(sizes.values())], "ok": [True]})

    def run_experiments(runs, *, runs_dir, device):
        assert runs_dir == cfg.vad_runs_dir("wavlm-base", "unpadded") and device == "cuda"
        events.append(("trained", tuple(run.spec.run_id for run in runs)))
        os.makedirs(os.path.join(runs_dir, "scores"), exist_ok=True)
        for run in runs:
            with open(os.path.join(runs_dir, "scores", run.spec.run_id + ".npz"), "w") as fh:
                fh.write("scores\n")
        if disturb is not None:
            disturb(root)
        return pd.DataFrame({"run_id": [run.spec.run_id for run in runs], "status": "trained", "seconds": 20.0})

    def clean_scores_check(runs, folder, existing):
        assert runs == planned and folder == runs_dir and existing == cfg.runs_dir("wavlm-base", "unpadded")
        events.append("clean scores check")
        return pd.DataFrame({"run_id": ["a", "b"], "exactly_zero": [True, clean_ok], "max_abs_diff": [0.0, 0.5],
                             "ok": [True, clean_ok]})

    def score_set_check(runs, folder, file_lists):
        assert runs == planned and folder == runs_dir and set(file_lists) == set(sizes)
        return pd.DataFrame({"run_id": ["a"], "ok": [True]})

    def run_record_check(runs, folder, n_train):
        assert runs == planned and folder == runs_dir and n_train == sizes["train"]
        return pd.DataFrame({"run_id": ["a"] * 60, "git_commit": "c" * 40, "ok": True})

    class Tables:
        def __init__(self, encoder, runs, out, **options):
            assert (encoder, runs, out) == ("wavlm-base", runs_dir, cfg.vad_tables_dir("wavlm-base"))
            assert set(options) == {"speaker_maps", "attack_map", "conditions", "layers", "cross_layers",
                                    "detail_layers", "base_runs_dir"}
            assert options["conditions"] == ("vad",) and options["layers"] == (3, 4, 5) == options["detail_layers"]
            assert options["cross_layers"] == {"vad": (3, 4, 5)}
            assert options["base_runs_dir"] == cfg.runs_dir("wavlm-base", "unpadded")
            assert set(options["speaker_maps"]) == {"eval", "itw"} and options["attack_map"]["eval_0"] == "A07"
            events.append("tables")

        def _write(self, name):
            written.append(name)
            os.makedirs(cfg.vad_tables_dir("wavlm-base"), exist_ok=True)
            with open(os.path.join(cfg.vad_tables_dir("wavlm-base"), name + ".csv"), "w") as fh:
                fh.write("a table of this condition\n")
            return pd.DataFrame({"table": [name]})

        def effects(self):
            return self._write("effects_2x2")

        def write(self, name, table):
            assert table["made_by"].iloc[0] == {"cells_2x2": "cells_from_effects", "removed_amount":
                                                "summarise_removed", "effects_vad_and_trim40": "side_by_side"}[name]
            return self._write(name)

        def fallback_counts(self, folders):
            assert folders == {"vad": features}
            return self._write("fallback_counts")

        def per_attack_effect(self):
            return self._write("per_attack_effect")

        def steps_table(self):
            return pd.DataFrame({"step": ["effects_2x2_vad"], "status": ["computed"]})

    def read_table(path):
        assert path == os.path.join(cfg.tables_dir("wavlm-base"), "effects_2x2.csv")
        return pd.DataFrame({"table": ["the effects of the analysis"]})

    def made_by(name, *expected):
        def stand_in(*arguments):
            assert [list(a.columns)[0] for a in arguments] == list(expected), name
            return pd.DataFrame({"made_by": [name]})
        return stand_in

    monkeypatch.setattr(datasets, "ensure_asvspoof", lambda path: None)
    monkeypatch.setattr(datasets, "ensure_itw", lambda path, sha256: None)
    monkeypatch.setattr(datasets, "items", items)
    monkeypatch.setattr(datasets, "check_items", lambda table, split: {"n_files": len(table)})
    monkeypatch.setattr(shards, "shard_lengths", shard_lengths)
    monkeypatch.setattr(shards, "kept_intervals", kept_intervals)
    monkeypatch.setattr(extraction, "file_list_check", file_list_check)
    monkeypatch.setattr(extraction, "extract_shards", extract_shards)
    monkeypatch.setattr(extraction, "shard_meta_table", lambda folder: "meta of " + folder)
    monkeypatch.setattr(extraction, "shard_meta_check", shard_meta_check)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(encoders, "load_encoder", lambda encoder, device: "the encoder")
    monkeypatch.setattr(encoders, "describe", lambda model: {"n_layers": 12, "hidden_size": 768})
    monkeypatch.setattr(provenance, "gpu_name", lambda device: "Tesla T4")
    monkeypatch.setattr(cfg, "drive_quota", lambda: {"free_gb": 200.0})
    monkeypatch.setattr(vad, "short_check", short_check)
    monkeypatch.setattr(vad, "shards_check", shards_check)
    monkeypatch.setattr(vad, "clean_scores_check", clean_scores_check)
    monkeypatch.setattr(vad, "cells_from_effects", made_by("cells_from_effects", "table"))
    monkeypatch.setattr(vad, "side_by_side", made_by("side_by_side", "table", "table"))
    monkeypatch.setattr(vad, "removed_per_file", made_by("removed_per_file", "kept"))
    monkeypatch.setattr(vad, "summarise_removed", made_by("summarise_removed", "made_by"))
    monkeypatch.setattr(training, "run_experiments", run_experiments)
    monkeypatch.setattr(training, "score_set_check", score_set_check)
    monkeypatch.setattr(training, "run_record_check", run_record_check)
    monkeypatch.setattr(analysis, "ScoreTables", Tables)
    monkeypatch.setattr(analysis, "read_table", read_table)
    namespace = {"cfg": cfg, "verify": verify, "pd": pd, "os": os, "provenance": provenance, "checks": checks,
                 "CHECKS": cfg.checks_dir(), "HEAD": "c" * 40, "display": lambda table: None, "record": record}
    sources = cells(VAD_NOTEBOOK, "code")
    k = next(i for i, source in enumerate(sources) if "\nEXTRACT = " in source)
    assert "EXTRACT = False " in sources[k]                       # as saved
    sources[k] = sources[k].replace("EXTRACT = False ", f"EXTRACT = {extract} ")
    try:
        for source in sources[k:]:
            exec(compile(python_of(source), VAD_NOTEBOOK.name, "exec"), namespace)
    finally:
        VAD_RESULT["out"] = capsys.readouterr().out
    return namespace


def vad_files(root):
    return {file: (root / file).read_bytes() for file in VAD_EXISTING}


def test_as_saved_the_vad_notebook_stops_after_the_short_check(monkeypatch, capsys, tmp_path):
    namespace = run_the_vad_cells(monkeypatch, capsys, tmp_path, extract=False)
    assert VAD_RESULT["events"] == [f"short check {split}" for split in cfg.SPLITS]
    assert list(namespace["checks"]) == VAD_CHECKS_SHORT + ["vad_files_unchanged"] and VAD_RESULT["written"] == []
    assert len(namespace["BEFORE"]) == len(VAD_EXISTING) and namespace["LAYERS"] == (3, 4, 5)
    encoder = namespace["checks"]["vad_encoder"].iloc[0]
    assert (encoder["vad_frame_ms"], encoder["vad_aggressiveness"], encoder["vad_run_frames"]) == (10, 3, 3)
    assert encoder["webrtcvad-wheels"] == "2.0.14.post1" and encoder["git_commit"] == "c" * 40
    estimate = namespace["checks"]["vad_estimate"]
    assert estimate["n_files"].tolist() == [100] and estimate["condition"].tolist() == ["vad"]
    out = VAD_RESULT["out"]
    assert ("this run: wavlm-base | condition: vad | rule: {'frame_ms': 10, 'aggressiveness': 3, 'run_frames': 3} "
            "| layers: (3, 4, 5) | runs planned: 60 | extraction, runs and tables: False") in out
    assert "existing files under runs/, tables/ and checks/: 8 | their sha256 is recorded and must not change" in out
    assert "short check passed: vad | files encoded: 100 | fallback files: 4" in out
    assert "EXTRACT is False: nothing is extracted or trained in this run" in out
    assert "vad run finished: wavlm-base | extraction, runs and tables: False | files that existed at the start: 8 unchanged" in out
    assert not (VAD_RESULT["root"] / "runs_vad").exists() and not (VAD_RESULT["root"] / "features").exists()
    assert vad_files(VAD_RESULT["root"]) == VAD_RESULT["before"]


def test_the_vad_notebook_extracts_trains_the_clean_models_first_and_writes_its_tables(monkeypatch, capsys,
                                                                                    tmp_path):
    from ssl_silence import vad

    namespace = run_the_vad_cells(monkeypatch, capsys, tmp_path)
    planned = vad.planned_runs()
    clean = tuple(run.spec.run_id for run in planned[:30])
    new = tuple(run.spec.run_id for run in planned[30:])
    assert all("_clean_" in run_id for run_id in clean) and all("_vad_" in run_id for run_id in new)
    assert VAD_RESULT["events"] == [f"short check {split}" for split in cfg.SPLITS] + [
        f"extract {split}" for split in cfg.SPLITS] + [
        "shards check", ("trained", clean), "clean scores check", ("trained", new), "tables"]
    assert list(namespace["checks"]) == VAD_CHECKS + ["vad_files_unchanged"]
    assert VAD_RESULT["written"] == ["effects_2x2"] + VAD_TABLES + ["fallback_counts", "per_attack_effect"]
    root = VAD_RESULT["root"]
    # The runs are in their own folder, the tables in the folder of the condition; nothing else changed.
    assert len(os.listdir(root / "runs_vad" / "wavlm-base" / "unpadded" / "scores")) == 60
    assert sorted(os.listdir(root / "tables" / "wavlm-base" / "vad")) == sorted(
        name + ".csv" for name in VAD_RESULT["written"])
    assert sorted(os.listdir(root / "runs" / "wavlm-base" / "unpadded")) == ["runs.csv", "scores"]
    assert vad_files(root) == VAD_RESULT["before"]
    unchanged = namespace["checks"]["vad_files_unchanged"]
    assert len(unchanged) == len(VAD_EXISTING) and unchanged["ok"].all()
    out = VAD_RESULT["out"]
    assert "extraction finished: vad | files: 100 | shards: 4 | fallback files: 3" in out
    assert ("the clean models trained again have the scores of the existing runs | score sets: 2 "
            "| exactly equal: 2 | largest difference: 0.5") in out
    assert "training finished: vad | runs: 60 | score sets checked: 1" in out
    assert "| effects: 1 rows | side by side: 1 rows | per attack: 1 rows" in out
    assert "vad run finished: wavlm-base | extraction, runs and tables: True | files that existed at the start: 8 unchanged" in out

    # A second run (after a disconnect) finds its own check tables and tables: they are not compared.
    namespace = run_the_vad_cells(monkeypatch, capsys, tmp_path)
    assert len(namespace["BEFORE"]) == len(VAD_EXISTING) and namespace["checks"]["vad_files_unchanged"]["ok"].all()


def test_clean_scores_that_differ_stop_the_vad_notebook_before_the_new_models(monkeypatch, capsys, tmp_path):
    from ssl_silence import verify

    with pytest.raises(verify.CheckFailed, match="vad_clean_scores: 1 of 2 rows failed"):
        run_the_vad_cells(monkeypatch, capsys, tmp_path, clean_ok=False)
    trained = [event for event in VAD_RESULT["events"] if event[0] == "trained"]
    assert len(trained) == 1 and all("_clean_" in run_id for run_id in trained[0][1])
    assert VAD_RESULT["events"][-1] == "clean scores check" and VAD_RESULT["written"] == []
    assert "training finished" not in VAD_RESULT["out"] and "vad run finished" not in VAD_RESULT["out"]
    assert list(VAD_RESULT["checks"])[-1] == "vad_clean_scores"            # written before the stop


@pytest.mark.parametrize("problem, check, last", [({"files_ok": False}, "vad_file_lists", None),
                                                 ({"short_ok": False}, "short check of the condition vad",
                                                  "short check itw")])
def test_a_failed_check_stops_the_vad_notebook_before_the_extraction(monkeypatch, capsys, tmp_path, problem,
                                                                     check, last):
    from ssl_silence import verify

    with pytest.raises(verify.CheckFailed, match=check):
        run_the_vad_cells(monkeypatch, capsys, tmp_path, **problem)
    assert VAD_RESULT["events"] == ([] if last is None else [f"short check {split}" for split in cfg.SPLITS])
    assert not (VAD_RESULT["root"] / "runs_vad").exists() and VAD_RESULT["written"] == []
    if last is not None:
        assert list(VAD_RESULT["checks"]) == VAD_CHECKS_SHORT              # both tables written before the stop


@pytest.mark.parametrize("file, status", [
    ("runs/wavlm-base/unpadded/scores/wavlm-base_unpadded_clean_L03_mean_s42.npz", "changed"),
    ("runs/wavlm-base/unpadded/runs.csv", "changed"), ("tables/wavlm-base/effects_2x2.csv", "changed"),
    ("checks/analysis_steps.csv", "changed"), ("tables/hubert-base/effects_2x2.csv", "removed"),
    ("runs/wavlm-base/unpadded/scores/wavlm-base_unpadded_vad_L03_mean_s42.npz", "new"),
    ("tables/wavlm-base/cells_2x2.csv", "new"), ("checks/table_cells_2x2_vad.csv", "new")])
def test_an_existing_file_that_changed_stops_the_vad_notebook_at_the_end(monkeypatch, capsys, tmp_path, file,
                                                                         status):
    from ssl_silence import verify

    def disturb(root):
        if status == "removed":
            if (root / file).exists():
                os.remove(root / file)
        else:
            (root / file).write_bytes(b"written by this run\r\n")

    with pytest.raises(verify.CheckFailed, match="vad_files_unchanged: 1 of"):
        run_the_vad_cells(monkeypatch, capsys, tmp_path, disturb=disturb)
    failed = VAD_RESULT["checks"]["vad_files_unchanged"]
    failed = failed[~failed["ok"]]
    assert [os.path.relpath(path, VAD_RESULT["root"]).replace(os.sep, "/") for path in failed["file"]] == [file]
    assert failed["status"].tolist() == [status] and "vad run finished" not in VAD_RESULT["out"]
    assert VAD_RESULT["events"][-1] == "tables"                            # everything ran; the stop is the last cell
