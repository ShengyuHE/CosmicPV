#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
ROOT=$1
RUN=$2
STAGE=$3
export PYTHONPATH="$RUN/code/src"
export PYTHONDONTWRITEBYTECODE=1
export DIVE_EXECUTABLE="$RUN/code/bin/DIVE"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/parity-mpl-${SLURM_JOB_ID}"
PYTHON="$ROOT/.venv/bin/python"
case "$STAGE" in
 validate)
   if "$PYTHON" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("purpose") != "best_configuration_confirmation")' "$RUN/config.json"; then
     "$PYTHON" -m parity_study.tests.best_confirmation verify --run "$RUN"
   fi
   if "$PYTHON" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("purpose") != "amplitude_scan")' "$RUN/config.json"; then
     "$PYTHON" -m parity_study.tests.amplitude_scan reuse --run "$RUN"
   else
     "$PYTHON" "$RUN/code/scripts/reuse_mesh_ensemble.py" --run "$RUN"
   fi
   "$PYTHON" -m pytest -q -o cache_dir="$RUN/validation/pytest_cache" "$RUN/code/tests" --junitxml="$RUN/validation/pytest.xml"
   "$PYTHON" -m parity_study.run validate --run "$RUN"
   ;;
 ensemble)
   "$PYTHON" -m parity_study.run seed-block --run "$RUN" --task-index "$SLURM_ARRAY_TASK_ID"
   ;;
 amplitude-report)
   "$PYTHON" "$RUN/code/scripts/render_amplitude_report.py" --run "$RUN"
   ;;
 confirmation-report)
   "$PYTHON" -m parity_study.tests.best_confirmation report --run "$RUN"
   ;;
 summary)
   if "$PYTHON" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("purpose") != "best_configuration_confirmation")' "$RUN/config.json"; then
     "$PYTHON" -m parity_study.summarize --run "$RUN" --no-notebook
     "$PYTHON" -m parity_study.tests.best_confirmation analyze --run "$RUN"
   elif "$PYTHON" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("purpose") != "amplitude_scan")' "$RUN/config.json"; then
     "$PYTHON" -m parity_study.summarize --run "$RUN" --no-notebook
     "$PYTHON" -m parity_study.tests.amplitude_scan analyze --run "$RUN"
   elif "$PYTHON" -c 'import json,sys; sys.exit("selection_protocol" not in json.load(open(sys.argv[1])))' "$RUN/config.json"; then
     "$PYTHON" -m parity_study.summarize --run "$RUN" --no-notebook
     if "$PYTHON" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("purpose") != "independent_ic_confirmation")' "$RUN/config.json"; then
       "$PYTHON" -m parity_study.tests.independent_ics confirm --run "$RUN"
     else
       "$PYTHON" "$RUN/code/scripts/check_fixed_ic_ensemble.py" --run "$RUN"
       "$PYTHON" -m parity_study.tests.selections confirm --run "$RUN"
     fi
   else
     "$PYTHON" -m parity_study.summarize --run "$RUN"
   fi
   ;;
 *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
