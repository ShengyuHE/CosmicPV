"""Parity-even tetrahedron shape and size, independent of vertex ordering."""
import numpy as np


def edge_geometry(volume, edges):
    """Six-edge mean ratio Q=12(3V)^(2/3)/sum(l_ij^2), in [0,1].

    Q=1 for a regular tetrahedron and tends to zero for a flattened one.
    Both Q and the all-edge ratio are invariant under uniform scaling.
    edge_rms retains the physical length unit; volume must be nonnegative.
    This routine validates numerical bounds, not full edge realizability.
    """
    volume, edges = np.asarray(volume, float), np.asarray(edges, float)
    if (edges.shape != volume.shape+(6,) or not np.isfinite(edges).all()
            or not np.isfinite(volume).all() or np.any(edges <= 0) or np.any(volume < 0)):
        raise ValueError('Require nonnegative volume and six positive finite edge lengths')
    scale = np.max(edges, axis=-1)
    normalized = edges/scale[..., None]
    quality = 12*np.cbrt(3*(volume/scale/scale/scale))**2/np.sum(normalized**2, axis=-1)
    if np.any(quality > 1+1e-10):
        raise ValueError('Volume and edges imply an impossible mean-ratio quality > 1')
    return dict(quality=np.clip(quality, 0, 1), edge_ratio=np.min(normalized, axis=-1),
                edge_rms=scale*np.sqrt(np.mean(normalized**2, axis=-1)), volume=volume)


def tetrahedron_geometry(tetrahedra):
    """Read optional 12-column DIVE output without confusing tied chi with volume."""
    data = np.asarray(tetrahedra, float)
    if data.ndim != 2 or data.shape[1] != 12:
        raise ValueError('Six-edge geometry requires 12-column DIVE output (geometry=True)')
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite tetrahedron measurements')
    volume, edges = data[:, 5], data[:, 6:12]
    if np.any(np.diff(edges, axis=1) < 0):
        raise ValueError('DIVE six-edge lengths must be sorted')
    nonzero = data[:, 0] != 0
    if not np.allclose(6*volume[nonzero], abs(data[nonzero, 0]), rtol=1e-12, atol=0):
        raise ValueError('Geometric volume disagrees with nonzero chirality')
    return edge_geometry(volume, edges)
