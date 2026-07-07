#!/bin/bash -l
# Run the full ml-peg benchmark suite on MPCDF Raven (NVIDIA A100, CUDA) for
# EVERY MACE model in the shared 2k models dir ($MODELS_DIR below).
#
# The folder is a moving target -- new models land in it at any time. So this
# job is IDEMPOTENT and SELF-REFRESHING:
#   1) At startup it rescans $MODELS_DIR/*.model and regenerates a models YAML
#      ($MODELS_YML), one omat-head entry per file (same config style as the
#      test*-omat entries in ml_peg/models/models.yml: mace_mp + head omat_pbe).
#      Files modified in the last 5 minutes are skipped -- they may still be
#      mid-copy; the next run picks them up.
#   2) It runs pytest over all calcs with --models-file pointing at that YAML.
#      ml-peg's completion markers (outputs/<model>/.completed.json) skip every
#      (model, benchmark) pair that already finished with identical inputs, so
#      only new models (or new/changed benchmarks) actually compute anything.
#      Benchmarks marked slow/very_slow are EXCLUDED (RUN_SLOW=0 default): a
#      single test outlasting the walltime never writes its completion marker,
#      so each resubmission would restart it and the loop would spin on that
#      one test forever. Completion is only tracked per (model, test), not
#      mid-test.
#   3) If RESUBMIT=1 (default) it resubmits itself with --begin=now+$RESUBMIT_DELAY.
#      --dependency=singleton (keyed on the job name) guarantees at most one
#      instance runs PER JOB NAME. Stop a stream with RESUBMIT=0 or
#      scancel -n <job name>.
#
# MULTIPLE INDEPENDENT JOBS: launch extra streams under different job names,
# e.g.  sbatch -J mlpeg_2k_b scripts/run_2k_models_raven.sh
#       sbatch -J mlpeg_2k_c scripts/run_2k_models_raven.sh
# Each stream keeps its own singleton resubmission chain (the name is
# inherited on resubmit). The streams share the work through lock files in
# $LOCK_DIR (see mlpeg_job_lock.py): a job atomically claims each test before
# running it and skips tests another live job has claimed, so no (model,
# benchmark) pair is computed twice. Locks are released when the test ends;
# locks from jobs killed mid-test go stale after MLPEG_LOCK_STALE_HOURS
# (default 25 h) and are then reclaimed.
#
# PREREQUISITES (batch jobs have NO internet):
#   - Benchmark data must be cached in ~/.cache/ml_peg first. Prefetch on a
#     login node with the mock model (cheap, but triggers every download):
#       cd $ML_PEG_REPO && pytest ml_peg/calcs/*/*/calc* -s --run-mock --mock-only
#   - Model files must be in $MODELS_DIR, e.g. from a local clone:
#       scp -J ademo@gate.mpcdf.mpg.de models/2k/*.model \
#           ademo@raven.mpcdf.mpg.de:/ptmp/ademo/isambard/arndm/models/2k/
#
#SBATCH -o /ptmp/ademo/isambard/arndm/results/logs/mlpeg_2k_%j.out
#SBATCH -e /ptmp/ademo/isambard/arndm/results/logs/mlpeg_2k_%j.err
#SBATCH -D ./
#SBATCH -J mlpeg_2k
#SBATCH --dependency=singleton
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --mail-type=none
#SBATCH --time=24:00:00

set -euo pipefail

# --- Parameters (override via env vars at submit time) ---
ML_PEG_REPO=${ML_PEG_REPO:-~/mace/ml-peg}
MODELS_DIR=${MODELS_DIR:-/ptmp/ademo/isambard/arndm/models/2k}
RESULTS_BASE=${RESULTS_BASE:-/ptmp/ademo/isambard/arndm/results}
MODELS_YML=${MODELS_YML:-$RESULTS_BASE/mlpeg/models_2k.yml}
LOCK_DIR=${LOCK_DIR:-$RESULTS_BASE/mlpeg/locks}
HEAD=${HEAD:-omat_pbe}
# Include slow-marked benchmarks (phonons, RDB7, NEBs, diatomics, ...). Leave
# at 0: a single slow test can outlast the walltime and, with no completion
# marker written, would rerun from scratch every resubmission.
RUN_SLOW=${RUN_SLOW:-0}
# Self-resubmission: keep polling $MODELS_DIR for new models. The singleton
# dependency plus the completion markers make this safe to leave on.
RESUBMIT=${RESUBMIT:-1}
RESUBMIT_DELAY=${RESUBMIT_DELAY:-1hour}
SCRIPT_PATH=${SCRIPT_PATH:-$ML_PEG_REPO/scripts/run_2k_models_raven.sh}

mkdir -p "$RESULTS_BASE/logs" "$(dirname "$MODELS_YML")" "$LOCK_DIR"

# --- Environment ---
if [[ -f "$ML_PEG_REPO/activate_env.sh" ]]; then
    source "$ML_PEG_REPO/activate_env.sh"
else
    source "$ML_PEG_REPO/.venv/bin/activate"
fi
cd "$ML_PEG_REPO"

export OMP_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Cross-job locking (see mlpeg_job_lock.py in this directory)
export MLPEG_LOCK_DIR="$LOCK_DIR"
export PYTHONPATH="$ML_PEG_REPO/scripts${PYTHONPATH:+:$PYTHONPATH}"

echo "$(date): ml-peg 2k-model sweep on $(hostname)"
echo "Models dir: $MODELS_DIR"
echo "Models YAML: $MODELS_YML"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available')"

# --- 1) Regenerate the models YAML from the current contents of $MODELS_DIR ---
# Entry style copied from the test*-omat entries in ml_peg/models/models.yml.
python - "$MODELS_DIR" "$MODELS_YML" "$HEAD" <<'EOF'
import os
import sys
import time
from pathlib import Path

models_dir, out_path, head = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
now = time.time()

entries = []
for model in sorted(models_dir.glob("*.model")):
    if now - model.stat().st_mtime < 300:
        print(f"[generate] Skipping {model.name}: modified <5 min ago (mid-copy?)")
        continue
    name = model.stem.replace("_", "-").replace(".", "-") + "-omat"
    entries.append(
        f"{name}:\n"
        "  module: mace.calculators\n"
        "  class_name: mace_mp\n"
        '  device: "auto"\n'
        "  default_dtype: float32\n"
        "  trained_on_dispersion: false\n"
        "  level_of_theory: PBE\n"
        "  kwargs:\n"
        f'    model: "{model}"\n'
        f'    head: {head}\n'
    )
    print(f"[generate] {name} -> {model}")

if not entries:
    sys.exit(f"No usable .model files found in {models_dir}")

# Write atomically: concurrent job streams regenerate the same file
tmp_path = out_path.with_suffix(f".tmp.{os.getpid()}")
tmp_path.write_text("\n".join(entries), encoding="utf8")
os.replace(tmp_path, out_path)
print(f"[generate] Wrote {len(entries)} models to {out_path}")
EOF

# --- 2) Run all benchmark calculations for models not yet completed ---
# Completion markers make this a no-op for (model, benchmark) pairs that
# already ran with identical inputs; a non-zero exit (some benchmarks failing
# for some models) must not kill the resubmission step.
SLOW_FLAG=""
if [[ "$RUN_SLOW" == "1" ]]; then SLOW_FLAG="--run-slow"; fi

pytest_status=0
srun python -m pytest -v ml_peg/calcs/*/*/calc* -s $SLOW_FLAG \
    -p mlpeg_job_lock --models-file "$MODELS_YML" || pytest_status=$?
echo "$(date): pytest finished with exit status $pytest_status"

# --- 3) Resubmit to pick up models that arrive later ---
if [[ "$RESUBMIT" == "1" ]]; then
    # Keep the job name so each stream stays its own singleton chain
    echo "Resubmitting (begin in $RESUBMIT_DELAY): $SCRIPT_PATH"
    sbatch -J "${SLURM_JOB_NAME:-mlpeg_2k}" --begin="now+$RESUBMIT_DELAY" "$SCRIPT_PATH"
else
    echo "RESUBMIT=0 -- not resubmitting."
fi

echo "$(date): done."
