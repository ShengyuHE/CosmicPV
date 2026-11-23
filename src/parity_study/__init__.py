"""TPIVa parity violation catalogue and Delaunay experiments."""

from pathlib import Path as _Path

# Saved notebooks and older commands import study modules directly from this
# package. Resolve those imports in tests/ while new code uses
# parity_study.tests.<module>. The moved modules use absolute imports so both
# entry paths work without duplicating source files at the package root.
__path__.append(str(_Path(__file__).with_name('tests')))
del _Path
