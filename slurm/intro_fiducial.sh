#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
PROJECT=$1
OUT=$2
export PYTHONPATH="$OUT/code/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/pv-intro-${SLURM_JOB_ID}"
export PYDEVD_DISABLE_FILE_VALIDATION=1
cd "$OUT/code"
"$PROJECT/.venv/bin/python" -m pytest -q tests/test_amplitude_scan.py::test_physical_scales_match_generated_spectral_normalization tests/test_initial_conditions.py::test_additive_lagrangian_particle_map_and_zero_helicity_limit -o cache_dir="$OUT/validation/pytest_cache" --junitxml="$OUT/validation/pytest.xml"
"$PROJECT/.venv/bin/python" -m parity_study.tests.intro_fiducial run --out "$OUT"
