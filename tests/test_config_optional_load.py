from __future__ import annotations

from pathlib import Path

from src.config import JugemuConfig, load_optional_config


def test_load_optional_config_none_when_missing(tmp_path: Path) -> None:
    cwd = Path.cwd()
    try:
        # simulate a project dir with no config.toml
        import os

        os.chdir(tmp_path)
        assert load_optional_config(None) is None
    finally:
        import os

        os.chdir(cwd)


def test_load_optional_config_loads_when_present(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text("[x]\ny=1\n", encoding="utf-8")

    cfg = load_optional_config(p)
    assert isinstance(cfg, JugemuConfig)
    assert cfg.get("x", "y") == 1
