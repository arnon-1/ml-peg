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


def run_inference(path: Path, head: str, device: str = "cpu", **kwargs) -> float:
    """
    Load a checkpoint the way the benchmark job does and compute water.

    Parameters
    ----------
    path
        Checkpoint file to evaluate.
    head
        Model head to use.
    device
        Device to evaluate on.
    **kwargs
        Extra keyword arguments for ``mace_mp``, e.g. ``compile_mode``.

    Returns
    -------
    float
        Potential energy of a water molecule in eV.
    """
    from ase import Atoms
    from mace.calculators import mace_mp

    calc = mace_mp(
        model=str(path), head=head, device=device, default_dtype="float64", **kwargs
    )
    water = Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])
    water.calc = calc
    energy = water.get_potential_energy()
    assert water.get_forces().shape == (3, 3)
    return energy


def check(path: Path, head: str, compile_check: bool = False) -> bool:
    """
    Check one checkpoint and print a verdict.

    Parameters
    ----------
    path
        Checkpoint file to check.
    head
        Model head to use for the inference test.
    compile_check
        Also evaluate with ``compile_mode="default"`` (on GPU if available)
        and require the energy to match the eager result. torch.compile has
        been seen to produce silently wrong energies for these models on some
        installations -- do not submit with COMPILE=1 unless this passes.

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
        energy = run_inference(target, head)
        print(f"  inference (head={head}): E(H2O) = {energy:.6f} eV, forces ok")
    except Exception as err:
        print(f"  inference FAILED ({type(err).__name__}: {err})")
        print("  VERDICT: NOT EVALUABLE (loads, but inference fails)")
        return False

    if compile_check:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            compiled = run_inference(
                target, head, device=device, compile_mode="default"
            )
        except Exception as err:
            print(f"  compiled inference FAILED ({type(err).__name__}: {err})")
            print("  VERDICT: eager OK, but do NOT submit with COMPILE=1")
            return False
        diff = abs(compiled - energy)
        print(f"  compiled ({device}): E(H2O) = {compiled:.6f} eV, |diff| = {diff:.2e}")
        if diff > 1e-5:
            print("  VERDICT: COMPILED ENERGIES WRONG -- do NOT submit with COMPILE=1")
            return False

    print("  VERDICT: OK")
    return True


def main() -> None:
    """Check every checkpoint given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path, help="checkpoint files")
    parser.add_argument("--head", default="omat_pbe", help="model head to test")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="also verify torch.compile gives the same energies as eager",
    )
    args = parser.parse_args()

    import mace

    print(f"mace {mace.__version__} from {Path(mace.__file__).parent}")
    results = [check(path, args.head, args.compile) for path in args.models]
    print(f"\n{sum(results)}/{len(results)} model(s) evaluable")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
