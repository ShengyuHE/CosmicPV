"""Checked interface to the pinned, patched student DIVE executable."""
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]


def parity_stat(cat, *, L, executable=None, scratch=None, geometry=False):
    """Measure chirality; optionally append volume and all six sorted edge lengths.

    Default columns remain chi, chi_hat, l_min, l_mid, l_max (three reference
    edges). With geometry=True, columns 5:12 are geometric volume and six
    edges, including the volume of cells whose tied ordering gives zero chi.
    """
    # L is deliberately mandatory. Never silently fold a 1000 Mpc box into L=1.
    cat = np.asarray(cat, dtype=float)
    if cat.ndim != 2 or cat.shape[1] != 3 or len(cat) < 4:
        raise ValueError("Expected a catalogue with shape (N>=4, 3)")
    if not np.isfinite(cat).all() or not np.isfinite(L) or L <= 0:
        raise ValueError("Coordinates and positive box size must be finite")
    if np.any(cat < 0) or np.any(cat >= L):
        raise ValueError("Catalogue coordinates must already be in [0,L); check the box size")
    if len(np.unique(cat, axis=0)) != len(cat):
        raise ValueError("Duplicate catalogue points")
    exe = Path(executable or os.environ.get("DIVE_EXECUTABLE", PROJECT / "build/dive/DIVE"))
    if not exe.is_file():
        raise FileNotFoundError(f"Build DIVE first: {exe}")
    with tempfile.TemporaryDirectory(prefix="parity_", dir=scratch or os.environ.get("SLURM_TMPDIR")) as tmp:
        fin, fout = Path(tmp)/"catalogue.dat", Path(tmp)/"tetrahedra.dat"
        np.savetxt(fin, cat, fmt="%.17g")
        command = [str(exe), "-i", str(fin), "-o", str(fout),
                   "-l", "0", "-u", format(L, ".17g")]
        if geometry:
            command.append('-g')
        result = subprocess.run(command,
                                capture_output=True, text=True, timeout=3600)
        if result.returncode:
            raise RuntimeError(f"DIVE failed ({result.returncode}):\n{result.stderr}\n{result.stdout}")
        data = np.loadtxt(fout, ndmin=2)
    expected_columns = 12 if geometry else 5
    if data.shape[1] != expected_columns or not len(data) or not np.isfinite(data).all():
        raise ValueError(f"Expected {expected_columns} finite DIVE columns; got {data.shape}")
    chi, shape, l1, l2, l3 = data[:, :5].T
    if np.any(l1 <= 0) or np.any(l1 > l2) or np.any(l2 > l3):
        raise ValueError("Invalid DIVE lengths")
    if np.any(abs(shape) > 1 + 1e-10) or not np.allclose(shape, chi/(l1*l2*l3), rtol=1e-12, atol=1e-14):
        raise ValueError("Inconsistent DIVE chirality columns")
    if geometry:
        from .geometry import tetrahedron_geometry
        tetrahedron_geometry(data)
    return data


def statistics(data):
    return np.array([data[:, 0].mean(), data[:, 1].mean(), np.sign(data[:, 0]).mean()])
