#!/bin/bash -l
# Run the full ml-peg benchmark suite on MPCDF Raven (NVIDIA A100, CUDA) for
# EVERY MACE model in the shared 2k models dir ($MODELS_DIR below).
#
# The folder is a moving target -- new models land in it at any time. So this
# job is IDEMPOTENT and SELF-REFRESHING:
#   0) It prepares raw checkpoints for stock mace: models trained with the
#      distillation fork pickle a DistillationHead class stock mace lacks, so
#      strip_distillation_heads.py writes a <name>_str.model sibling for every
#      model (heads removed, or a plain copy if there were none). Idempotent
#      and concurrency-safe; upload models unstripped and forget about it.
#   1) It rescans $MODELS_DIR for *_str.model (recursively) and regenerates a
#      models YAML
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
#      so every later task would restart it from scratch. Completion is only
#      tracked per (model, test), not mid-test.
#   3) It runs as a JOB ARRAY (default 6 tasks, max 2 running at once). Each
#      array task is an independent sweep; the tasks share the work through
#      lock files in $LOCK_DIR (see mlpeg_job_lock.py): a task atomically
#      claims each test before running it and skips tests another live task
#      has claimed, so no (model, benchmark) pair is computed twice. Locks are
#      released when the test ends; locks from tasks killed mid-test go stale
#      after MLPEG_LOCK_STALE_HOURS (default 25 h) and are then reclaimed.
#      Later tasks in the array rescan $MODELS_DIR when they start, so models
#      arriving while the array works through its queue are picked up; once
#      the array is exhausted, submit it again for newer models. Override the
#      shape at submit time, e.g.:  sbatch --array=0-9%3 <this script>
#
# PREREQUISITES (batch jobs have NO internet):
#   - Benchmark data must be cached in ~/.cache/ml_peg first. Prefetch on a
#     login node with the mock model (cheap, but triggers every download):
#       cd $ML_PEG_REPO && pytest ml_peg/calcs/*/*/calc* -s --run-mock --mock-only
#   - Model files must be in $MODELS_DIR, e.g. from a local clone:
#       scp -J ademo@gate.mpcdf.mpg.de models/2k/*.model \
#           ademo@raven.mpcdf.mpg.de:/ptmp/ademo/isambard/arndm/models/2k/
#
#SBATCH -o /ptmp/ademo/isambard/arndm/results/logs/mlpeg_2k_%A_%a.out
#SBATCH -e /ptmp/ademo/isambard/arndm/results/logs/mlpeg_2k_%A_%a.err
#SBATCH -D ./
#SBATCH -J mlpeg_2k
#SBATCH --array=0-5%2
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

# --- Parameters ---
# Override by passing VAR=value as script arguments, e.g.
#   sbatch --array=0 scripts/run_2k_models_raven.sh CALCS="ml_peg/calcs/molecular_reactions/BH2O_36/calc_*.py"
# Script arguments are always forwarded by sbatch, unlike environment
# variables, which the site's Slurm policy may strip from the job.
for arg in "$@"; do
    if [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done
# Defaults to the directory sbatch was run from, i.e. submit from the repo root
ML_PEG_REPO=${ML_PEG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
MODELS_DIR=${MODELS_DIR:-/ptmp/ademo/isambard/arndm/models/2k}
RESULTS_BASE=${RESULTS_BASE:-/ptmp/ademo/isambard/arndm/results}
MODELS_YML=${MODELS_YML:-$RESULTS_BASE/mlpeg/models_2k.yml}
LOCK_DIR=${LOCK_DIR:-$RESULTS_BASE/mlpeg/locks}
HEAD=${HEAD:-omat_pbe}
# Benchmarks to run (glob(s) relative to the repo root); override to test a
# subset, e.g. CALCS="ml_peg/calcs/molecular_reactions/BH2O_36/calc_*.py"
CALCS=${CALCS:-ml_peg/calcs/*/*/calc*}
# Include slow-marked benchmarks (phonons, RDB7, NEBs, diatomics, ...). Leave
# at 0: a single slow test can outlast the walltime and, with no completion
# marker written, would rerun from scratch every resubmission.
RUN_SLOW=${RUN_SLOW:-0}
# Evaluate with torch.compile (adds compile_mode: default to every model's
# kwargs). Each test compiles its own calculator; the inductor cache on /ptmp
# amortises this across tests and tasks. CAUTION: compile has been seen to
# produce silently WRONG energies for these models on some installations --
# only enable after `check_model.py --compile` passes on this machine.
COMPILE=${COMPILE:-0}

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

if [[ "$COMPILE" == "1" ]]; then
    # Inductor needs a C++20-capable compiler; the node default g++ is too old
    module load gcc/13
    export CXX="$(command -v g++)" CC="$(command -v gcc)"
    export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/ptmp/$USER/torchinductor_cache}
    export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$TORCHINDUCTOR_CACHE_DIR/triton}
    mkdir -p "$TRITON_CACHE_DIR"
fi

# Cross-job locking (see mlpeg_job_lock.py in this directory)
export MLPEG_LOCK_DIR="$LOCK_DIR"
export PYTHONPATH="$ML_PEG_REPO/scripts${PYTHONPATH:+:$PYTHONPATH}"

echo "$(date): ml-peg 2k-model sweep, array task ${SLURM_ARRAY_TASK_ID:-?} on $(hostname)"
echo "Models dir: $MODELS_DIR"
echo "Models YAML: $MODELS_YML"
echo "Settings: RUN_SLOW=$RUN_SLOW COMPILE=$COMPILE HEAD=$HEAD CALCS=$CALCS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available')"

# --- 0) Strip distillation heads (writes <name>_str.model siblings) ---
python "$ML_PEG_REPO/scripts/strip_distillation_heads.py" "$MODELS_DIR"

# --- 1) Regenerate the models YAML from the current contents of $MODELS_DIR ---
# Entry style copied from the test*-omat entries in ml_peg/models/models.yml.
python - "$MODELS_DIR" "$MODELS_YML" "$HEAD" "$COMPILE" <<'EOF'
import os
import sys
import time
from pathlib import Path

models_dir, out_path, head = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
compile_mode = '    compile_mode: "default"\n' if sys.argv[4] == "1" else ""
now = time.time()

entries = []
# Only evaluate the stripped/verified siblings written by
# strip_distillation_heads.py, never the raw uploads. Subdirectories are
# included; their path becomes part of the model name to keep it unique
# (top-level models keep the same name as before).
for model in sorted(models_dir.rglob("*_str.model")):
    rel = model.relative_to(models_dir)
    if now - model.stat().st_mtime < 300:
        print(f"[generate] Skipping {rel}: modified <5 min ago (mid-copy?)")
        continue
    name = (
        str(rel.with_suffix("")).replace("/", "-").replace("_", "-").replace(".", "-")
        + "-omat"
    )
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
        f"    head: {head}\n" + compile_mode
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
# already ran with identical inputs. Capture pytest's exit status: some
# benchmarks failing for some models is expected and should not abort the
# task under set -e.
SLOW_FLAG=""
if [[ "$RUN_SLOW" == "1" ]]; then SLOW_FLAG="--run-slow"; fi

pytest_status=0
srun python -m pytest -v $CALCS -s $SLOW_FLAG \
    -p mlpeg_job_lock --models-file "$MODELS_YML" || pytest_status=$?
echo "$(date): pytest finished with exit status $pytest_status"
echo "$(date): array task ${SLURM_ARRAY_TASK_ID:-?} done."
