#!/bin/bash -l
# Run the ml-peg ANALYSIS stage on MPCDF Viper-GPU for the models the 2k
# sweep calculated (run_2k_models_viper.sh). Analysis aggregates ALL models
# per benchmark into the app tables/figures, so the unit of work is one
# benchmark (not one (model, benchmark) pair), and it is file reading +
# metric computation + plotting only -- no GPU work. The GPU request below
# just satisfies the partition; a single node for a few hours is plenty.
#
# Like the calc sweep, this job is IDEMPOTENT and shares work:
#   0) It first ensures mock calc outputs exist: many analyse modules read
#      outputs/mock (reference structures) at import time. Calc completion
#      markers make this step a fast no-op after its first run.
#   1) It then runs $WORKERS `ml_peg analyse` workers in parallel, all
#      loading the mlpeg_job_lock plugin, so workers claim benchmarks
#      atomically through lock files in $LOCK_DIR and no benchmark is
#      analysed twice (also across concurrent or resubmitted jobs).
#   2) Analysis completion markers (<app data>/<benchmark>/.completed.json,
#      fingerprinting the analysis sources, model configs and the CALC
#      completion markers) skip benchmarks already analysed with identical
#      inputs. Resubmitting after more calcs finish re-analyses exactly the
#      benchmarks whose calc markers changed.
#
# PREREQUISITES:
#   - $MODELS_YML written by run_2k_models_viper.sh; the analysis runs
#     against exactly that model set. Do not regenerate it here: models that
#     arrived after the calc sweep would join the set without results.
#   - Calc results under ml_peg/calcs/**/outputs in $ML_PEG_REPO.
#
# Benchmarks whose calcs are missing or incomplete fail individually without
# aborting the job, and are retried on the next submission (failures never
# write completion markers). Check the worker logs for what failed.
#
#SBATCH -o /ptmp/ademo/isambard/arndm/results/logs/mlpeg/mlpeg_analysis_%j.out
#SBATCH -e /ptmp/ademo/isambard/arndm/results/logs/mlpeg/mlpeg_analysis_%j.err
#SBATCH -D ./
#SBATCH -J mlpeg_analysis
#
#SBATCH --ntasks=1
#SBATCH --constraint="apu"
#
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=110000
#
#SBATCH --mail-type=none
#SBATCH --time=06:00:00

set -euo pipefail

# --- Parameters ---
# Override by passing VAR=value as script arguments, e.g.
#   sbatch scripts/run_2k_analysis_viper.sh CATEGORY=conformers WORKERS=4
for arg in "$@"; do
    if [[ "$arg" == *=* ]]; then export "${arg?}"; fi
done
# Defaults to the directory sbatch was run from, i.e. submit from the repo root
ML_PEG_REPO=${ML_PEG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
RESULTS_BASE=${RESULTS_BASE:-/ptmp/ademo/isambard/arndm/results}
MODELS_YML=${MODELS_YML:-$RESULTS_BASE/mlpeg/models_2k.yml}
LOCK_DIR=${LOCK_DIR:-$RESULTS_BASE/mlpeg/locks}
# Parallel ml_peg analyse workers; benchmarks are shared via lock files
WORKERS=${WORKERS:-6}
# Benchmarks to analyse (ml_peg analyse --category/--test selectors)
CATEGORY=${CATEGORY:-*}
TEST=${TEST:-*}
# Set MOCK=0 to skip the mock-output step (e.g. when it already completed)
MOCK=${MOCK:-1}

mkdir -p "$RESULTS_BASE/logs/mlpeg" "$LOCK_DIR"

if [[ ! -f "$MODELS_YML" ]]; then
    echo "Missing $MODELS_YML -- run run_2k_models_viper.sh first" >&2
    exit 1
fi

# --- Environment ---
if [[ -f "$ML_PEG_REPO/activate_env.sh" ]]; then
    source "$ML_PEG_REPO/activate_env.sh"
else
    source "$ML_PEG_REPO/.venv/bin/activate"
fi
cd "$ML_PEG_REPO"

export OMP_NUM_THREADS=4
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Cross-job locking (see mlpeg_job_lock.py in this directory); calc and
# analysis node IDs differ, so sharing the calc sweep's lock dir is safe.
export MLPEG_LOCK_DIR="$LOCK_DIR"
export PYTHONPATH="$ML_PEG_REPO/scripts${PYTHONPATH:+:$PYTHONPATH}"

echo "$(date): ml-peg analysis, job ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "Models YAML: $MODELS_YML"
echo "Settings: CATEGORY=$CATEGORY TEST=$TEST WORKERS=$WORKERS MOCK=$MOCK"

LOG_BASE="$RESULTS_BASE/logs/mlpeg/mlpeg_analysis_${SLURM_JOB_ID:-manual}"

# --- 0) Ensure mock calc outputs exist (analyse modules read them at import) ---
# --continue-on-collection-errors: an environment-broken calc module must not
# block mock outputs for everything else. test_phonons_ref scrapes
# alexandria.icams.rub.de, which batch nodes may not reach (see calc sweep).
if [[ "$MOCK" == 1 ]]; then
    echo "$(date): generating mock calc outputs (log: $LOG_BASE.mock.log)"
    status=0
    ml_peg calc --mock-only --run-slow \
        --continue-on-collection-errors \
        --deselect "ml_peg/calcs/bulk_crystal/phonons/calc_phonons.py::test_phonons_ref" \
        -p mlpeg_job_lock \
        > "$LOG_BASE.mock.log" 2>&1 || status=$?
    echo "$(date): mock calc step finished (status $status)"
fi

# --- 1) Run the analysis for benchmarks not yet analysed ---
# Worker exit statuses are captured: some benchmarks failing (e.g. calcs not
# finished for some models) is expected and must not abort the job under
# set -e. ml_peg analyse does not propagate pytest's exit code anyway.
run_worker() {
    local i=$1
    ml_peg analyse --category "$CATEGORY" --test "$TEST" \
        --models-file "$MODELS_YML" \
        -p mlpeg_job_lock \
        > "$LOG_BASE.w$i.log" 2>&1
}

echo "Worker logs: $LOG_BASE.w{1..$WORKERS}.log"
pids=()
for i in $(seq 1 "$WORKERS"); do
    run_worker "$i" &
    pids+=($!)
done
for i in "${!pids[@]}"; do
    status=0
    wait "${pids[$i]}" || status=$?
    echo "$(date): analyse worker $((i + 1)) finished (status $status)"
done

echo "$(date): analysis job done. Skipped-as-complete benchmarks are normal;"
echo "grep the worker logs for FAILED to see benchmarks needing attention."
