"""
Strip distillation heads so raw MACE checkpoints load with stock mace.

Models trained with the distillation fork of MACE pickle a ``DistillationHead``
class that stock mace does not have, so ``torch.load`` fails with an
AttributeError. The head lives in a separate ``distillation_heads`` ModuleDict
that inference never touches, so it can simply be deleted.

For every ``<name>.model`` in the given directory tree (subdirectories
included) this script writes a ``<name>_str.model`` sibling: the model with
its distillation heads removed,
or a plain copy if it has none. The benchmark job only evaluates the
``*_str.model`` files. The script is idempotent and safe under concurrent
invocation (atomic writes; up-to-date outputs are skipped). Each output
inherits its source's mtime, so a re-uploaded source is re-stripped and the
job script's fresh-mtime (mid-copy) guard keeps working.

Usage: python strip_distillation_heads.py <models_dir>
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import time

import torch

# Sources modified more recently than this may still be mid-copy
FRESH_SECONDS = 300


class DistillationHead(torch.nn.Module):
    """Stub standing in for the distillation fork's class during unpickling."""


def _load(path: Path) -> torch.nn.Module:
    """
    Load a checkpoint, injecting the stub class if stock mace lacks it.

    Parameters
    ----------
    path
        Checkpoint file to load.

    Returns
    -------
    torch.nn.Module
        The unpickled model.
    """
    import mace.modules.blocks as blocks

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except AttributeError:
        if hasattr(blocks, "DistillationHead"):
            raise
    blocks.DistillationHead = DistillationHead
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        del blocks.DistillationHead


def strip_model(src: Path, dst: Path) -> str:
    """
    Write the evaluation-ready version of one checkpoint.

    Parameters
    ----------
    src
        Source checkpoint.
    dst
        Output path for the stripped checkpoint or copy.

    Returns
    -------
    str
        What was done: "stripped" or "copied".
    """
    model = _load(src)
    tmp = dst.with_suffix(f".tmp.{os.getpid()}")

    if hasattr(model, "distillation_heads"):
        del model.distillation_heads
        leftovers = [
            name
            for name, module in model.named_modules()
            if type(module).__name__ == "DistillationHead"
        ]
        if leftovers:
            raise ValueError(f"DistillationHead still referenced at {leftovers}")
        torch.save(model, tmp)
        action = "stripped"
    else:
        shutil.copyfile(src, tmp)
        action = "copied"

    os.replace(tmp, dst)
    # Inherit the source mtime: keeps outputs stable across reruns and lets a
    # re-uploaded (newer) source trigger a re-strip.
    stat = src.stat()
    os.utime(dst, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    return action


def main(models_dir: Path) -> None:
    """
    Strip or copy every checkpoint in a directory that needs it.

    Parameters
    ----------
    models_dir
        Directory tree containing ``*.model`` checkpoints.
    """
    now = time.time()
    for src in sorted(models_dir.rglob("*.model")):
        rel = src.relative_to(models_dir)
        if src.stem.endswith("_str"):
            continue
        if now - src.stat().st_mtime < FRESH_SECONDS:
            print(f"[strip] Skipping {rel}: modified <5 min ago (mid-copy?)")
            continue
        dst = src.with_stem(src.stem + "_str")
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            print(f"[strip] Up to date: {dst.relative_to(models_dir)}")
            continue
        try:
            action = strip_model(src, dst)
        except Exception as err:
            print(f"[strip] FAILED for {rel}: {err}")
            continue
        print(f"[strip] {action}: {rel} -> {dst.name}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
