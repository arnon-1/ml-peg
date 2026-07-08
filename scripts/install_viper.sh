#!/bin/bash -l
# Set up ml-peg on MPCDF Viper-GPU (AMD MI300A, ROCm), after cloning this repo:
#   bash scripts/install_viper.sh
#
# Unlike install_raven.sh this does NOT create a fresh venv from uv.lock: the
# lock resolves torch to a CUDA build, which cannot drive the MI300A. Instead
# ml-peg and its dependencies are installed into the EXISTING mace venv
# ($VENV below, default ~/.venvs/mace) that already has a working ROCm torch
# and the distillation MACE fork. The installed torch / mace-torch / e3nn
# versions are pinned via a pip constraints file so nothing can replace them.
#
# Also writes activate_env.sh in the repo root -- the same module + venv setup
# as MACE_clone's -- which run_2k_models_viper.sh sources at job start.

set -euo pipefail

ML_PEG_REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=${VENV:-$HOME/.venvs/mace}
ROCM_MODULE=${ROCM_MODULE:-rocm/7.2}
PYTHON_MODULE=${PYTHON_MODULE:-python-waterboa/2025.06}

if [[ ! -e "$VENV/bin/activate" ]]; then
    echo "ERROR: venv not found at $VENV (override with VENV=...)" >&2
    exit 1
fi

module purge
ml "$ROCM_MODULE" "$PYTHON_MODULE"
source "$VENV/bin/activate"
cd "$ML_PEG_REPO"

echo "=== Pinning the already-working ML stack (constraints file)"
CONSTRAINTS=$(mktemp)
python - <<'EOF' > "$CONSTRAINTS"
from importlib.metadata import version

for pkg in ("torch", "mace-torch", "e3nn", "numpy"):
    try:
        print(f"{pkg}=={version(pkg)}")
    except Exception:  # noqa: BLE001, S110
        pass
EOF
cat "$CONSTRAINTS"

echo "=== Installing ml-peg (editable, with the d3 extra) into $VENV"
pip install -e ".[d3]" -c "$CONSTRAINTS"
pip install pytest -c "$CONSTRAINTS"
rm -f "$CONSTRAINTS"

echo "=== Writing $ML_PEG_REPO/activate_env.sh"
cat > "$ML_PEG_REPO/activate_env.sh" <<EOF
module purge
ml $ROCM_MODULE $PYTHON_MODULE
source $VENV/bin/activate
EOF

echo "=== Verifying"
python - <<'EOF'
import ase
import mace
import torch

import ml_peg

print(f"ml_peg ok | ase {ase.__version__} | torch {torch.__version__}")
print(f"torch HIP (ROCm) build: {torch.version.hip or 'NO -- torch was replaced?!'}")
print(f"mace {mace.__version__} from {mace.__file__}")
print(f"gpu available: {torch.cuda.is_available()} (false is fine on login nodes)")
import mace.modules.blocks as blocks
fork = [c for c in ("DistillationHead", "ProductSequential") if hasattr(blocks, c)]
print(f"fork classes present: {fork or 'NONE -- fork install was clobbered?'}")
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
