"""
Check whether the current installation can evaluate MACE checkpoints.

For each given checkpoint this script reports which (if any) classes missing
from the installed mace the pickle needs, whether the benchmark job's
auto-strip step (strip_distillation_heads.py) can handle it, and — when it is
evaluable — runs a tiny inference (energy and forces of a water molecule) the
same way the benchmark job would load the model.

Run it on the login node to test an installation before submitting, e.g.:
    python scripts/check_model.py /ptmp/ademo/isambard/arndm/models/2k/*.model

Exit status is non-zero if any checkpoint is not evaluable here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import tempfile

import torch

# Matches e.g. "Can't get attribute 'ProductSequential' on <module 'mace...'"
MISSING_ATTR = re.compile(r"Can't get attribute '(\w+)' on <module '([\w.]+)'")


def probe_missing(path: Path, max_stubs: int = 10) -> list[str]:
    """
    Find which classes missing from this installation a checkpoint needs.

    Repeatedly attempts ``torch.load``, stubbing out each missing class the
    unpickler reports, and collects their names. All stubs are removed again
    before returning.

    Parameters
    ----------
    path
        Checkpoint file to probe.
    max_stubs
        Safety limit on the number of classes to stub.

    Returns
    -------
    list[str]
        Names of classes the checkpoint needs but the installation lacks.

    Raises
    ------
    Exception
        If loading fails for a reason other than a missing class.
    """
    from importlib import import_module

    stubbed: list[tuple[object, str]] = []
    try:
        for _ in range(max_stubs):
            try:
                torch.load(path, map_location="cpu", weights_only=False)
                break
            except AttributeError as err:
                match = MISSING_ATTR.search(str(err))
                if match is None:
                    raise
                name, module_name = match.groups()
                module = import_module(module_name)
                setattr(module, name, type(name, (torch.nn.Module,), {}))
                stubbed.append((module, name))
        return [name for _, name in stubbed]
    finally:
        for module, name in stubbed:
            delattr(module, name)


def run_inference(path: Path, head: str) -> str:
    """
    Load a checkpoint the way the benchmark job does and compute water.

    Parameters
    ----------
    path
        Checkpoint file to evaluate.
    head
        Model head to use.

    Returns
    -------
    str
        Summary of the computed energy and force shape.
    """
    from ase import Atoms
    from mace.calculators import mace_mp

    calc = mace_mp(model=str(path), head=head, device="cpu", default_dtype="float64")
    water = Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])
    water.calc = calc
    energy = water.get_potential_energy()
    return f"E(H2O) = {energy:.6f} eV, forces {water.get_forces().shape}"


def check(path: Path, head: str) -> bool:
    """
    Check one checkpoint and print a verdict.

    Parameters
    ----------
    path
        Checkpoint file to check.
    head
        Model head to use for the inference test.

    Returns
    -------
    bool
        Whether this installation (plus the job's auto-strip) can evaluate it.
    """
    print(f"=== {path.name}")
    try:
        missing = probe_missing(path)
    except Exception as err:
        print(f"  load FAILED ({type(err).__name__}: {err})")
        print("  VERDICT: NOT EVALUABLE (checkpoint unreadable)")
        return False

    if not missing:
        print("  loads natively with the installed mace")
        target = path
    elif missing == ["DistillationHead"]:
        print("  needs DistillationHead only -- the job's auto-strip handles this")
        from strip_distillation_heads import strip_model

        target = Path(tempfile.mkdtemp()) / path.with_stem(path.stem + "_str").name
        strip_model(path, target)
        print(f"  stripped to {target.name}")
    else:
        print(f"  needs classes this installation lacks: {', '.join(missing)}")
        print(
            "  VERDICT: NOT EVALUABLE -- these are part of the architecture and"
            " cannot be stripped; install the MACE fork that defines them"
        )
        return False

    try:
        print(f"  inference (head={head}): {run_inference(target, head)}")
    except Exception as err:
        print(f"  inference FAILED ({type(err).__name__}: {err})")
        print("  VERDICT: NOT EVALUABLE (loads, but inference fails)")
        return False
    print("  VERDICT: OK")
    return True


def main() -> None:
    """Check every checkpoint given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path, help="checkpoint files")
    parser.add_argument("--head", default="omat_pbe", help="model head to test")
    args = parser.parse_args()

    import mace

    print(f"mace {mace.__version__} from {Path(mace.__file__).parent}")
    results = [check(path, args.head) for path in args.models]
    print(f"\n{sum(results)}/{len(results)} model(s) evaluable")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
