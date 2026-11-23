#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
PROJECT=$1
OUT=$2
export PYTHONPATH="$OUT/code/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/pv-selection-comparison-${SLURM_JOB_ID}"
cd "$OUT/code"
"$PROJECT/.venv/bin/python" -m pytest -q tests/test_selection_comparison.py -o cache_dir="$OUT/pytest_cache" --junitxml="$OUT/pytest.xml"
"$PROJECT/.venv/bin/python" -m parity_study.tests.selection_comparison run --out "$OUT"
