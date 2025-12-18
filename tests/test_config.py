"""Basic configuration tests."""

from pathlib import Path

from transcribe_tool import config as config_module
from transcribe_tool.config import AppConfig


def test_enrollment_path(tmp_path: Path) -> None:
    config = AppConfig()
    config.paths.enrollment_dir = tmp_path
    expected = tmp_path / "alice.json"
    path = config.enrollment_path_for("alice")
    assert path == expected
    assert path.parent.exists()


def test_config_persistence(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / ".transcribe.config"
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path, raising=False)

    config = AppConfig(device="cpu", huggingface_token="hf_test_token")
    config.paths.enrollment_dir = tmp_path / "enrollments"
    config.save()

    loaded = AppConfig.load()
    assert loaded.device == "cpu"
    assert loaded.huggingface_token == "hf_test_token"
    assert loaded.paths.enrollment_dir == config.paths.enrollment_dir
