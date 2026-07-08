"""
Prefetch benchmark input data into ~/.cache/ml_peg without running any calcs.

Statically scans every module under ml_peg/calcs for ``download_s3_data`` /
``download_github_data`` calls (resolving literal arguments and module-level
constants via the AST) and downloads each file into the shared cache the
benchmarks read from. Run on a login node (with internet) before submitting
jobs on clusters whose batch nodes are offline:

    python scripts/prefetch_data.py           # download everything missing
    python scripts/prefetch_data.py --list    # only print what was found

NOT covered:
- Benchmarks fetching from alexandria.icams.rub.de at test time (phonons,
  high_pressure_relaxation, low_dimensional_relaxation). All are slow-marked
  and excluded from the default sweep (RUN_SLOW=0).
- CMRAds200 calls download_github_data(force=True) at TEST time, ignoring
  the cache: it needs internet on the batch node even after prefetching.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
CALCS_DIR = REPO_ROOT / "ml_peg" / "calcs"

# Calls whose arguments the AST scan cannot resolve, added by hand:
# plf547_pla15_utils.py builds key/filename with f-strings from the
# benchmark name; its two callers are PLF547 and PLA15.
EXTRA_S3 = [
    ("inputs/supramolecular/PLF547/PLF547.zip", "PLF547.zip"),
    ("inputs/supramolecular/PLA15/PLA15.zip", "PLA15.zip"),
]


def module_constants(tree: ast.Module) -> dict[str, str]:
    """
    Collect module-level string constants (NAME = "literal").

    Parameters
    ----------
    tree
        Parsed module.

    Returns
    -------
    dict[str, str]
        Constant name to string value.
    """
    consts: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            value = node.value.value
            if isinstance(value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        consts[target.id] = value
    return consts


def resolve_args(
    call: ast.Call, params: tuple[str, ...], consts: dict[str, str]
) -> dict[str, str | None]:
    """
    Resolve a call's string arguments from literals and module constants.

    Parameters
    ----------
    call
        Call node to resolve.
    params
        Parameter names, in positional order.
    consts
        Module-level string constants.

    Returns
    -------
    dict[str, str | None]
        Parameter name to resolved value (None if not statically resolvable).
    """

    def value(node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return consts.get(node.id)
        if isinstance(node, ast.JoinedStr):  # f-string of resolvable parts
            parts = [
                value(part.value)
                if isinstance(part, ast.FormattedValue)
                else value(part)
                for part in node.values
            ]
            return None if None in parts else "".join(parts)
        return None

    args: dict[str, str | None] = {}
    for param, node in zip(params, call.args, strict=False):
        args[param] = value(node)
    for keyword in call.keywords:
        if keyword.arg in params:
            args[keyword.arg] = value(keyword.value)
    return args


def scan() -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    """
    Find all statically resolvable download calls under ml_peg/calcs.

    Returns
    -------
    tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]
        S3 (key, filename) pairs, GitHub (uri, filename) pairs, and
        warnings for calls that could not be resolved.
    """
    s3: set[tuple[str, str]] = set(EXTRA_S3)
    github: set[tuple[str, str]] = set()
    warnings: list[str] = []

    for path in sorted(CALCS_DIR.rglob("*.py")):
        if path.name == "utils.py" and path.parent == CALCS_DIR / "utils":
            continue  # the helpers' own definitions
        tree = ast.parse(path.read_text(encoding="utf8"))
        consts = module_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            rel = path.relative_to(REPO_ROOT)
            if name == "download_s3_data":
                args = resolve_args(node, ("key", "filename"), consts)
                if args.get("key") and args.get("filename"):
                    s3.add((args["key"], args["filename"]))
                else:
                    warnings.append(f"{rel}:{node.lineno}: unresolved {name} call")
            elif name == "download_github_data":
                args = resolve_args(node, ("filename", "github_uri"), consts)
                if args.get("filename") and args.get("github_uri"):
                    github.add((args["github_uri"], args["filename"]))
                else:
                    warnings.append(f"{rel}:{node.lineno}: unresolved {name} call")

    return sorted(s3), sorted(github), warnings


def main() -> None:
    """Scan for benchmark downloads and prefetch them into the cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="only print what was found")
    args = parser.parse_args()

    s3, github, warnings = scan()
    print(f"Found {len(s3)} S3 file(s) and {len(github)} GitHub file(s)")
    for warning in warnings:
        print(f"WARNING: {warning} -- prefetch it manually or run its calc once")

    if args.list:
        for key, filename in s3:
            print(f"  s3: {key} -> {filename}")
        for uri, filename in github:
            print(f"  github: {uri}/{filename}")
        return

    from ml_peg.calcs.utils.utils import download_github_data, download_s3_data

    failures = []
    for key, filename in s3:
        try:
            download_s3_data(key=key, filename=filename)
        except Exception as err:
            failures.append(f"{key}: {err}")
    for uri, filename in github:
        try:
            download_github_data(filename=filename, github_uri=uri)
        except Exception as err:
            failures.append(f"{uri}/{filename}: {err}")

    for failure in failures:
        print(f"FAILED: {failure}")
    print(f"Done: {len(s3) + len(github) - len(failures)} ok, {len(failures)} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
