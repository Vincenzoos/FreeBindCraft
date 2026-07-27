"""Repository path constants and resolution helpers for IPSAE additions."""

from __future__ import annotations

from pathlib import Path

ADDITIONS_DIR = Path(__file__).resolve().parent
IPSAE_DIR = ADDITIONS_DIR.parent
ROOT = IPSAE_DIR.parent.parent
IPSAE_SCRIPT = IPSAE_DIR / "ipsae.py"
DEFAULT_EVALS_DIR = ROOT / "ipsae_evals"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def rel_repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
