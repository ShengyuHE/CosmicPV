#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
PROJECT=$1
OUT=$2
STAGE=$3
export PYTHONPATH="$OUT/code/src"
export PYTHONDONTWRITEBYTECODE=1
export DIVE_EXECUTABLE="$OUT/code/bin/DIVE"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/pv-half-${SLURM_JOB_ID}"
case "$STAGE" in
  validate)
    cd "$OUT/code"
    "$PROJECT/.venv/bin/python" -m pytest -q tests/test_initial_conditions.py tests/test_geometry.py -o cache_dir="$OUT/validation/pytest_cache" --junitxml="$OUT/validation/pytest.xml"
    "$PROJECT/.venv/bin/python" -m parity_study.tests.helicity_half validate --out "$OUT"
    ;;
  measure)
    "$PROJECT/.venv/bin/python" -m parity_study.tests.helicity_half measure --out "$OUT" --task "$SLURM_ARRAY_TASK_ID"
    ;;
  summarize)
    "$PROJECT/.venv/bin/python" -m parity_study.tests.helicity_half summarize --out "$OUT"
    ;;
  *) exit 2 ;;
esac
