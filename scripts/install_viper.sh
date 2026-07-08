#!/bin/bash -l
# Set up ml-peg on MPCDF Viper-GPU (AMD MI300A, ROCm), after cloning this
# repo:
#   bash scripts/install_viper.sh
#
# Creates a BRANCH venv (.venv in the repo root) that SHARES the existing
# mace venv's site-packages ($BASE_ENV below, default ~/.venvs/mace) via a
# .pth file, so the proven ROCm torch + distillation MACE fork are reused
# without touching that env. Unlike a MACE branch clone, ml-peg's own
# dependencies (janus-core, mlipx, torch-dftd, ...) are NOT in the base env,
# so they are installed too -- into the branch venv only, with the base
# env's torch / mace-torch / e3nn / numpy pinned via a pip constraints file
# so the resolver treats them as satisfied and can never replace them (a
# fresh resolve would pull a CUDA torch, useless on MI300A).
#
# Not the README's uv route for the same reason: uv.lock resolves torch to a
# CUDA build.
#
# Also writes activate_env.sh in the repo root -- modules + branch venv --
# which run_2k_models_viper.sh sources at job start.

set -euo pipefail

BASE_ENV=${BASE_ENV:-$HOME/.venvs/mace}
MODULES=${MODULES:-rocm/7.2 python-waterboa/2025.06}

ML_PEG_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BRANCH_ENV=${BRANCH_ENV:-$ML_PEG_REPO/.venv}

echo "Repo root:  $ML_PEG_REPO"
echo "Base env:   $BASE_ENV"
echo "Branch env: $BRANCH_ENV"
echo

if [[ ! -x "$BASE_ENV/bin/python" ]]; then
    echo "ERROR: base env python not found at $BASE_ENV/bin/python" >&2
    echo "Override with: BASE_ENV=/path/to/env bash scripts/install_viper.sh" >&2
    exit 1
fi

# Avoid creating the branch venv from an already-active venv
if command -v deactivate >/dev/null 2>&1; then
    deactivate || true
fi

module purge
ml $MODULES

echo "Module python: $(which python3) ($(python3 --version))"
echo

echo "=== Creating branch venv"
python3 -m venv "$BRANCH_ENV"
source "$BRANCH_ENV/bin/activate"

BASE_SITE=$("$BASE_ENV/bin/python" -c "import site; print(site.getsitepackages()[0])")
BRANCH_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "Base site-packages:   $BASE_SITE"
echo "Branch site-packages: $BRANCH_SITE"

# Share the base env's packages, LATE in sys.path so branch installs win.
# addsitedir (not a plain path line) so the base env's own .pth files are
# processed too -- the MACE fork is installed editable there and is only
# importable through its .pth hooks.
echo "import site; site.addsitedir('$BASE_SITE')" \
    > "$BRANCH_SITE/zz-shared-base-venv.pth"

echo "=== Pinning the base env's ML stack (constraints file)"
CONSTRAINTS=$(mktemp)
python - <<'EOF' > "$CONSTRAINTS"
from importlib.metadata import version

for pkg in ("torch", "mace-torch", "e3nn", "numpy"):
    try:
        print(f"{pkg}=={version(pkg)}")
    except Exception:
        pass
EOF
cat "$CONSTRAINTS"

cd "$ML_PEG_REPO"
echo "=== Installing ml-peg (editable, with the d3 extra) into the branch venv"
python -m pip install -e ".[d3]" -c "$CONSTRAINTS"
python -m pip install pytest -c "$CONSTRAINTS"
rm -f "$CONSTRAINTS"

echo "=== Writing $ML_PEG_REPO/activate_env.sh"
cat > "$ML_PEG_REPO/activate_env.sh" <<EOF
module purge
ml $MODULES
source "$BRANCH_ENV/bin/activate"
hash -r
EOF

echo "=== Verifying"
python - <<'EOF'
import sys

print("sys.executable:", sys.executable)

import ase
import mace
import pytest
import torch

import ml_peg

print(f"ml_peg from {ml_peg.__file__}")
print(f"ase {ase.__version__} | torch {torch.__version__} | pytest {pytest.__version__}")
print(f"torch HIP (ROCm) build: {torch.version.hip or 'NO -- wrong torch?!'}")
print(f"mace {mace.__version__} from {mace.__file__}")
print(f"gpu available: {torch.cuda.is_available()} (false is fine on login nodes)")
import mace.modules.blocks as blocks
fork = [c for c in ("DistillationHead", "ProductSequential") if hasattr(blocks, c)]
print(f"fork classes present: {fork or 'NONE -- base env not shared correctly?'}")
EOF

echo
echo "Done. Next steps (login node, internet available):"
echo "  1) Check the models load:"
echo "       source activate_env.sh"
echo "       python scripts/check_model.py /ptmp/ademo/isambard/arndm/models/2k/*.model"
echo "  2) (optional) Prefetch benchmark data into ~/.cache/ml_peg:"
echo "       pytest ml_peg/calcs/*/*/calc* -s --run-mock --mock-only"
echo "  3) Submit:"
echo "       sbatch scripts/run_2k_models_viper.sh"
