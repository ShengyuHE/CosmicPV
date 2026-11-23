"""Keep study entry points and saved notebook imports working after relocation."""
import importlib
from pathlib import Path
import subprocess
import sys

import pytest
from setuptools import find_packages


STUDIES = (
    'amplitude_scan', 'best_confirmation', 'independent_ics', 'selections',
    'notebooks', 'study_notebooks', 'ic_notebooks', 'selection_notebooks',
    'independent_ic_notebooks', 'amplitude_notebooks', 'best_confirmation_notebooks',
    'signal_plots', 'tutorial_examples', 'selection_geometry',
)


def test_studies_are_packaged_and_saved_notebook_imports_resolve():
    import parity_study
    package = Path(parity_study.__file__).parent
    assert 'parity_study.tests' in find_packages(str(package.parent))
    for name in STUDIES:
        current = importlib.import_module(f'parity_study.tests.{name}')
        legacy = importlib.import_module(f'parity_study.{name}')
        assert Path(current.__file__) == package/'tests'/f'{name}.py'
        assert Path(legacy.__file__) == Path(current.__file__)
        assert not (package/f'{name}.py').exists()
    for name in ('initial_conditions', 'catalogue', 'delaunay'):
        assert Path(importlib.import_module(f'parity_study.{name}').__file__).parent == package


@pytest.mark.parametrize('name', ['amplitude_scan', 'best_confirmation',
                                  'independent_ics', 'selections', 'tutorial_examples'])
def test_study_command_line_entry_points(name):
    for prefix in ('parity_study.tests', 'parity_study'):
        result = subprocess.run([sys.executable, '-m', f'{prefix}.{name}', '--help'],
                                capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stderr
        assert 'usage:' in result.stdout


def test_report_route_uses_existing_confirmation_notebook_name():
    from parity_study.tests.notebooks import report_filename
    assert report_filename('20261122_mesh_ic_L1000_N256_A16_confirmation_50') == 'field_PVdisp_B.ipynb'
