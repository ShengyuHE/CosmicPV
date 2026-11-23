#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
PROJECT=$1
shift
export PYTHONPATH="$PROJECT/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/parity-mpl-${SLURM_JOB_ID}"
"$PROJECT/.venv/bin/python" "$PROJECT/scripts/render_field_report.py" "$@"
