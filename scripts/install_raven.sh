#!/bin/bash -l
# Set up ml-peg on MPCDF Raven, after cloning this repo branch:
#   bash scripts/install_raven.sh
#
# Creates .venv in the repo root and installs ml-peg (with the D3 extra the
# molecular benchmarks need) plus the distillation MACE fork (editable, from
# $MACE_FORK) in place of stock mace-torch, so checkpoints pickling fork
# classes (DistillationHead, rolled-dot blocks, ...) load natively.
#
# Also writes activate_env.sh in the repo root -- the same module + venv +
# cache setup as MACE_clone's -- which run_2k_models_raven.sh sources
# automatically at job start.

set -euo pipefail

ML_PEG_REPO=$(cd "$(dirname "$0")/.." && pwd)
MACE_FORK=${MACE_FORK:-$HOME/mace/MACE_clone}

if [[ ! -e "$MACE_FORK/pyproject.toml" && ! -e "$MACE_FORK/setup.py" ]]; then
    echo "ERROR: MACE fork not found at $MACE_FORK (override with MACE_FORK=...)" >&2
    exit 1
fi

module purge
ml cuda/13.2 python-waterboa/2025.06

echo "=== Creating venv at $ML_PEG_REPO/.venv (python: $(python3 --version))"
python3 -m venv "$ML_PEG_REPO/.venv"
source "$ML_PEG_REPO/.venv/bin/activate"
pip install --upgrade pip

echo "=== Installing ml-peg (with D3 support)"
pip install -e "$ML_PEG_REPO[d3]"

echo "=== Installing the MACE fork from $MACE_FORK (replaces stock mace-torch)"
pip install -e "$MACE_FORK"

echo "=== Writing $ML_PEG_REPO/activate_env.sh"
cat > "$ML_PEG_REPO/activate_env.sh" <<EOF
module purge
ml cuda/13.2 python-waterboa/2025.06
source $ML_PEG_REPO/.venv/bin/activate
export TORCHINDUCTOR_CACHE_DIR=/ptmp/\$USER/torchinductor_cache
export TRITON_CACHE_DIR=\$TORCHINDUCTOR_CACHE_DIR/triton
mkdir -p "\$TRITON_CACHE_DIR"
EOF

echo "=== Verifying"
python - <<'EOF'
import ase
import mace
import torch

import ml_peg

print(f"ml_peg ok | ase {ase.__version__} | torch {torch.__version__}")
print(f"mace {mace.__version__} from {mace.__file__}")
print(f"cuda available: {torch.cuda.is_available()} (false is fine on login nodes)")
import mace.modules.blocks as blocks
fork = [c for c in ("DistillationHead", "ProductSequential") if hasattr(blocks, c)]
print(f"fork classes present: {fork or 'NONE -- fork install failed?'}")
EOF

echo
echo "Done. Next steps (login node, internet available):"
echo "  1) Check the models load:"
echo "       source activate_env.sh"
echo "       python scripts/check_model.py /ptmp/ademo/isambard/arndm/models/2k/*.model"
echo "  2) Prefetch benchmark data into ~/.cache/ml_peg (batch nodes have no internet):"
echo "       pytest ml_peg/calcs/*/*/calc* -s --run-mock --mock-only"
echo "  3) Submit:"
echo "       sbatch scripts/run_2k_models_raven.sh"
