from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class JugemuConfig:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "JugemuConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        data = _load_toml(p)
        if not isinstance(data, dict):
            raise ValueError("config.toml must parse to a table")
        return JugemuConfig(raw=data)

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur


def _load_toml(path: Path) -> Dict[str, Any]:
    # Python 3.11+: tomllib is in stdlib.
    try:
        import tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        # Shouldn't happen on our pinned Python, but keeps things graceful.
        raise RuntimeError("tomllib not available; use Python 3.11+")

    return tomllib.loads(path.read_text(encoding="utf-8"))
