"""What produced a file: library versions, device, repository commit, checksums."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import torch

VERSIONED_PACKAGES = ("torch", "torchaudio", "transformers", "librosa", "numpy", "soundfile",
                      "torchcodec", "scikit-learn", "pandas", "scipy")
NOT_INSTALLED = "not installed"


def installed_version(package: str) -> str:
    """Installed version of a distribution, or "not installed". Nothing is assumed."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return NOT_INSTALLED


def versions(packages: tuple[str, ...] = VERSIONED_PACKAGES) -> dict[str, str]:
    """Python and package versions of this runtime."""
    out = {"python": platform.python_version()}
    out.update({package: installed_version(package) for package in packages})
    return out


def gpu_name(device: str | torch.device | None = None) -> str:
    """Name of the GPU behind ``device``, or "cpu"."""
    device = torch.device(device if device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        return "cpu"
    return torch.cuda.get_device_name(device.index if device.index is not None else 0)


def git_commit(repo: str | Path | None = None) -> str:
    """Commit of the repository that holds this package, with "-dirty" if the
    working tree has uncommitted changes; "unknown" outside a git checkout."""
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                              check=True).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """sha256 of a file as a hexadecimal string."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_table(table, path: str | Path, **extra) -> Path:
    """Write a result table as CSV with a ``<name>.meta.json`` next to it.

    The meta file records the library versions of the runtime (scipy among
    them), the repository commit, the number of rows and ``extra``.

    Returns the path of the meta file.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")
    meta = {"table": path.name, "rows": int(len(table)), "columns": list(map(str, table.columns)),
            "sha256": sha256_file(path), "versions": versions(), "git_commit": git_commit(), **extra}
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta_path


def environment_report(root: str | None = None, quota: dict | None = None):
    """The runtime as a table (item, value): Python and package versions, the
    Parselmouth and Praat versions, the default text encoding, the device and
    GPU, the repository commit and the free space of the local disk. With
    ``root``: for a root on Drive the limit, usage and free space of the
    account quota (``quota``, from ``config.drive_quota``; required), for
    any other root the free space of its file system. All in GB."""
    import locale

    import pandas as pd

    from . import config as cfg

    rows = dict(versions())
    for package in ("praat-parselmouth", "huggingface_hub", "matplotlib", "pyyaml"):
        rows[package] = installed_version(package)
    try:
        import parselmouth

        rows["praat"] = str(parselmouth.PRAAT_VERSION)
    except ImportError:
        rows["praat"] = NOT_INSTALLED
    rows["text_encoding"] = locale.getpreferredencoding(False)
    rows["platform"] = platform.platform()
    rows["cuda_available"] = str(torch.cuda.is_available())
    rows["cuda_version"] = str(torch.version.cuda)
    rows["gpu_name"] = gpu_name()
    rows["git_commit"] = git_commit()
    rows["local_disk_free_gb"] = f"{cfg.local_free_space_gb():.1f}"
    if root is not None and cfg.on_drive(root):
        if quota is None:
            raise ValueError("a root on Drive needs the quota of the account (config.drive_quota)")
        for key in ("limit_gb", "usage_gb", "free_gb"):
            rows[f"drive_{key}"] = f"{quota[key]:.1f}"
    elif root is not None:
        rows["free_space_gb"] = f"{cfg.free_space_gb(root):.1f}"
    return pd.DataFrame({"item": list(rows), "value": [str(v) for v in rows.values()]})


def dataset_revision(repository: str) -> str:
    """Current commit of a dataset repository on Hugging Face (read only).

    The commit is read so that it can be pinned in ``config``; nothing is
    downloaded.
    """
    from huggingface_hub import HfApi

    return str(HfApi().dataset_info(repository).sha)
