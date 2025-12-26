from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"


def get_paths() -> Paths:
    project_root = Path(__file__).resolve().parents[2]
    return Paths(project_root=project_root)
