#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
PROJECT=$1
OUT=$2
STAGE=$3
export PYTHONPATH="$OUT/code/src"
export DIVE_EXECUTABLE="$OUT/code/bin/DIVE"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/pv-selection-geometry-${SLURM_JOB_ID}"
case "$STAGE" in
  validate)
    cd "$OUT/code"
    "$PROJECT/.venv/bin/python" -m pytest -q tests -o cache_dir="$OUT/pytest_cache" --junitxml="$OUT/pytest.xml"
    ;;
  measure)
    "$PROJECT/.venv/bin/python" -m parity_study.tests.selection_geometry measure --out "$OUT" --seed "$((1000+SLURM_ARRAY_TASK_ID))"
    ;;
  report)
    "$PROJECT/.venv/bin/python" -m parity_study.tests.selection_geometry report --out "$OUT" --project "$PROJECT"
    ;;
  bins)
    "$PROJECT/.venv/bin/python" -m parity_study.tests.selection_geometry report-bins --out "$OUT" --project "$PROJECT"
    ;;
  *) exit 2 ;;
esac
