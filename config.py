from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse


@dataclass(frozen=True, slots=True)
class Paths:
    root: Path
    category: str

    @property
    def category_root(self) -> Path:
        return self.root / self.category.replace(" ", "")

    @property
    def train(self) -> Path:
        return self.category_root / "train" / "good"

    @property
    def validation(self) -> Path:
        return self.category_root / "validation"

    @property
    def test(self) -> Path:
        return self.category_root / "test_public"


@dataclass(slots=True)
class CliArgs:
    root: Path
    category: str
    threshold: float
    auto_threshold: bool

    @classmethod
    def parse(cls) -> "CliArgs":
        parser = argparse.ArgumentParser(description="Run MVTec AD2 anomaly detection pipeline.")
        parser.add_argument("--root", type=Path, required=True, help="Path to MVTec AD2 root directory")
        parser.add_argument("--category", type=str, required=True, help="Category name (e.g. 'vial')")
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--thresh", type=float, dest="threshold", help="Fixed threshold to use")
        group.add_argument("--auto-thresh", action="store_true", dest="auto_threshold", help="Automatically search for best threshold")
        ns = parser.parse_args()
        return cls(
            root=ns.root.expanduser(),
            category=ns.category,
            threshold=ns.threshold or 0.6,
            auto_threshold=ns.auto_threshold,
        )
