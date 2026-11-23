"""Configuration, paths and the Drive guard."""

from __future__ import annotations

import pytest

from ssl_silence import config as cfg


def test_encoder_registry():
    registry = cfg.encoders()
    assert set(registry) == {"wavlm-base", "hubert-base", "wavlm-large"}
    for entry in registry.values():
        assert set(entry) == {"hf_id", "revision", "model_class", "attn_implementation",
                              "n_layers", "hidden_size", "feat_extract_norm", "conv_kernel",
                              "conv_stride"}
        assert len(entry["revision"]) == 40
    for name in ("wavlm-base", "hubert-base"):
        entry = registry[name]
        assert entry["n_layers"] == 12 and entry["hidden_size"] == 768
        assert entry["feat_extract_norm"] == "group"
    large = registry["wavlm-large"]
    assert large["n_layers"] == 24 and large["hidden_size"] == 1024
    assert large["feat_extract_norm"] == "layer"
    with pytest.raises(KeyError):
        cfg.encoder("unknown")


def test_conditions_and_grid_constants():
    assert cfg.conditions() == {"clean": None, "trim40": 40, "trim20": 20, "trim15": 15}
    assert cfg.trim_top_db("clean") is None and cfg.trim_top_db("trim15") == 15
    assert cfg.SEEDS == (42, 43, 44, 45, 46)
    assert cfg.POOLINGS == ("mean", "stats")
    with pytest.raises(KeyError):
        cfg.trim_top_db("trimmed")


def test_paths_contain_the_encoder_name():
    root = "/data"
    for name in cfg.encoders():
        paths = [cfg.feature_dir(name, "unpadded", "trim40", root), cfg.runs_dir(name, "unpadded", root),
                 cfg.tables_dir(name, root)]
        assert all(name in p.replace("\\", "/").split("/") for p in paths)
    assert cfg.feature_dir("hubert-base", "unpadded", "clean", root).replace("\\", "/") == \
        "/data/features/hubert-base/unpadded/clean"
    with pytest.raises(KeyError):
        cfg.feature_dir("wavlm-base", "masked", "clean", root)


def test_drive_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg.os.path, "ismount", lambda p: False)
    with pytest.raises(RuntimeError):
        cfg.require_drive("/content/drive/MyDrive/ssl_silence/tables")
    cfg.require_drive(str(tmp_path))
    cfg.makedirs(str(tmp_path / "a" / "b"))
    assert (tmp_path / "a" / "b").is_dir()


def test_training_groups():
    groups = cfg.experiments()["experiments"]
    assert "wavlm-large" not in groups
    hubert = groups["hubert-base"]["unpadded"]
    assert [g["train"] for g in hubert] == ["clean", "trim40", "trim20", "trim15"]
    assert all(g["layers"] == "all" for g in hubert)
    assert hubert[0]["test"] == ["clean", "trim40", "trim20", "trim15"]
    assert all(g["test"] == [g["train"], "clean"] for g in hubert[1:])


def test_free_space_check(tmp_path, monkeypatch):
    usage = cfg.shutil.disk_usage(str(tmp_path))
    monkeypatch.setattr(cfg.shutil, "disk_usage", lambda p: usage._replace(free=160 * 10**9))
    assert cfg.MIN_FREE_GB == 150
    assert cfg.free_space_gb(str(tmp_path / "not" / "created")) == pytest.approx(160.0)
    assert cfg.require_free_space(str(tmp_path)) == pytest.approx(160.0)
    monkeypatch.setattr(cfg.shutil, "disk_usage", lambda p: usage._replace(free=149 * 10**9))
    with pytest.raises(RuntimeError):
        cfg.require_free_space(str(tmp_path))


# Free space on Drive: the quota of the account from the Drive API ------------------------------

class DriveApi:
    """Stand-in for the Drive API: about().get(fields="storageQuota").execute()."""

    def __init__(self, answer=None, error: Exception | None = None):
        self.answer, self.error, self.fields = answer, error, []

    def about(self):
        return self

    def get(self, fields):
        self.fields.append(fields)
        return self

    def execute(self):
        if self.error is not None:
            raise self.error
        return self.answer


GIB = 2 ** 30
# The account of the first Colab runs: 141.2 GB used of 400 GB (as Google shows them, in 2**30 bytes).
ACCOUNT = {"storageQuota": {"limit": str(400 * GIB), "usage": str(int(141.2 * GIB)),
                            "usageInDrive": str(140 * GIB), "usageInDriveTrash": "0"}}
DRIVE_ROOT = "/content/drive/MyDrive/ssl_silence"


def test_drive_quota_is_limit_minus_usage():
    api = DriveApi(ACCOUNT)
    quota = cfg.drive_quota(api)
    assert api.fields == ["storageQuota"]
    assert quota["limit_gb"] == 400 * GIB / 1e9 and quota["usage_gb"] == int(141.2 * GIB) / 1e9
    assert quota["free_gb"] == (400 * GIB - int(141.2 * GIB)) / 1e9
    assert quota["free_gb"] == pytest.approx(258.8 * GIB / 1e9, abs=1e-6)          # about 277.9 in 10**9 bytes
    assert set(quota) == {"limit_gb", "usage_gb", "free_gb"}


def test_free_space_on_drive_is_the_quota_and_never_the_file_system(monkeypatch):
    monkeypatch.setattr(cfg.os.path, "ismount", lambda path: path == cfg.DRIVE_MOUNT)
    monkeypatch.setattr(cfg.shutil, "disk_usage", lambda path: pytest.fail("the file system was asked"))
    assert cfg.on_drive(DRIVE_ROOT) and cfg.on_drive(cfg.DRIVE_MOUNT) and not cfg.on_drive("/content/local")
    quota = cfg.drive_quota(DriveApi(ACCOUNT))
    assert cfg.require_free_space(DRIVE_ROOT, quota=quota) == quota["free_gb"]
    # Without a quota the Drive API is asked.
    monkeypatch.setattr(cfg, "drive_service", lambda: DriveApi(ACCOUNT))
    assert cfg.require_free_space(DRIVE_ROOT) == quota["free_gb"]
    # The bound passes, less fails.
    at_bound = {"limit_gb": 400.0, "usage_gb": 250.0, "free_gb": 150.0}
    assert cfg.require_free_space(DRIVE_ROOT, quota=at_bound) == 150.0
    with pytest.raises(RuntimeError, match="149.9 GB free, 150 GB required"):
        cfg.require_free_space(DRIVE_ROOT, quota={**at_bound, "free_gb": 149.9})


@pytest.mark.parametrize("api", [
    DriveApi(error=ConnectionError("no network")),                    # the call fails
    DriveApi({"storageQuota": {"usage": "5"}}),                         # an account without a limit in the answer
    DriveApi({}),                                                       # no quota in the answer
    DriveApi({"storageQuota": {"limit": "many", "usage": "5"}}),
], ids=["call fails", "no limit", "no quota", "not a number"])
def test_a_failed_quota_call_stops_without_the_file_system_number(monkeypatch, api):
    monkeypatch.setattr(cfg.os.path, "ismount", lambda path: path == cfg.DRIVE_MOUNT)
    monkeypatch.setattr(cfg.shutil, "disk_usage", lambda path: pytest.fail("the file system was asked"))
    with pytest.raises(RuntimeError, match="could not be read from the Drive API"):
        cfg.drive_quota(api)
    monkeypatch.setattr(cfg, "drive_service", lambda: api)
    with pytest.raises(RuntimeError, match="is not used instead"):
        cfg.require_free_space(DRIVE_ROOT)


def test_no_authorisation_stops_too(monkeypatch):
    monkeypatch.setattr(cfg.os.path, "ismount", lambda path: path == cfg.DRIVE_MOUNT)

    def refused():
        raise PermissionError("authorisation was refused")

    monkeypatch.setattr(cfg, "drive_service", refused)
    with pytest.raises(RuntimeError, match="could not be read from the Drive API .*PermissionError"):
        cfg.require_free_space(DRIVE_ROOT)


def test_a_quota_belongs_to_a_drive_path_only(tmp_path):
    with pytest.raises(ValueError, match="not on Drive"):
        cfg.require_free_space(str(tmp_path), quota={"limit_gb": 1.0, "usage_gb": 0.0, "free_gb": 1.0})
    assert cfg.local_free_space_gb() > 0
