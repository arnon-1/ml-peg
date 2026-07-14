"""Tests for the distillation head stripping script."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import time

import pytest
import torch

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "strip_distillation_heads.py"
_spec = importlib.util.spec_from_file_location("strip_distillation_heads", SCRIPT_PATH)
strip_heads = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strip_heads)

OLD = time.time() - 3600


def make_models(models_dir: Path) -> None:
    """
    Write one checkpoint with distillation heads and one without.

    Both are backdated an hour so the mid-copy freshness guard passes.

    Parameters
    ----------
    models_dir
        Directory to write the checkpoints into.
    """
    plain = torch.nn.Linear(4, 4)
    torch.save(plain, models_dir / "plain.model")

    distil = torch.nn.Linear(4, 4)
    distil.distillation_heads = torch.nn.ModuleDict({"omat": torch.nn.Linear(4, 1)})
    torch.save(distil, models_dir / "distil.model")

    for name in ("plain.model", "distil.model"):
        os.utime(models_dir / name, (OLD, OLD))


def test_strips_and_copies(tmp_path):
    """Test head-ful models are stripped and head-less models copied."""
    make_models(tmp_path)
    strip_heads.main(tmp_path)

    # Head-less: byte-identical copy
    assert (tmp_path / "plain_str.model").read_bytes() == (
        tmp_path / "plain.model"
    ).read_bytes()

    # Head-ful: heads removed, rest of the model intact
    stripped = torch.load(
        tmp_path / "distil_str.model", map_location="cpu", weights_only=False
    )
    assert not hasattr(stripped, "distillation_heads")
    original = torch.load(
        tmp_path / "distil.model", map_location="cpu", weights_only=False
    )
    assert torch.equal(stripped.weight, original.weight)

    # Outputs inherit the source mtime, so the job script's YAML generation
    # does not mistake freshly stripped models for mid-copy uploads
    for name in ("plain_str.model", "distil_str.model"):
        assert (tmp_path / name).stat().st_mtime == pytest.approx(OLD)


def test_existing_outputs_never_regenerated(tmp_path):
    """Test re-copied sources with fresh mtimes do not rewrite outputs."""
    make_models(tmp_path)
    strip_heads.main(tmp_path)
    before = {
        name: (tmp_path / name).read_bytes()
        for name in ("plain_str.model", "distil_str.model")
    }

    # Simulate re-staging byte-identical raw models with plain cp: mtime bumps
    # (backdated past the freshness guard, as on the cluster after 5 minutes)
    for name in ("plain.model", "distil.model"):
        os.utime(tmp_path / name, (OLD + 60, OLD + 60))
    strip_heads.main(tmp_path)

    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content


def test_fresh_sources_skipped(tmp_path):
    """Test sources modified moments ago are treated as mid-copy."""
    torch.save(torch.nn.Linear(4, 4), tmp_path / "new.model")
    strip_heads.main(tmp_path)
    assert not (tmp_path / "new_str.model").exists()
