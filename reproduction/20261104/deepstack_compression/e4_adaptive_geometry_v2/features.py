"""Pure geometric descriptors using stored EADP spatial IDs and P3 coordinates."""
import math
import numpy as np


def descriptors(pos, start, ids):
    if type(start) is not int or start < 0 or not isinstance(ids, list) or len(ids) != 128 or len(set(ids)) != 128 or any(type(i) is not int or not 0 <= i < 1024 for i in ids):
        raise ValueError("invalid selection")
    if not isinstance(pos, list) or len(pos) != 3 or any(not isinstance(axis, list) or len(axis) != 1 or len(axis[0]) < start + 1024 for axis in pos):
        raise ValueError("invalid P3 shape")
    coords = np.asarray([[pos[1][0][start + i], pos[2][0][start + i]] for i in range(1024)])
    if not np.issubdtype(coords.dtype, np.integer) or len(set(map(tuple, coords))) != 1024: raise ValueError("noninteger or duplicate original coordinates")
    low, high = coords.min(axis=0), coords.max(axis=0)
    extent = high - low + 1
    if np.any(extent < 2) or int(extent[0] * extent[1]) != 1024: raise ValueError("unexpected image grid")
    selected = coords[ids]
    norm = (selected - low) / (extent - 1)
    diff = norm[:, None, :] - norm[None, :, :]
    distances = np.sqrt((diff * diff).sum(axis=2)); np.fill_diagonal(distances, np.inf)
    nearest = float(distances.min(axis=1).mean())
    def bins(n): return ((selected - low) * n // extent).astype(int)
    cells8 = bins(8); occupancy = len(set(map(tuple, cells8)))
    cells4 = bins(4); _, counts = np.unique(cells4, axis=0, return_counts=True)
    probability = counts / len(selected); entropy = float(-(probability * np.log2(probability)).sum())
    if not math.isfinite(nearest) or not math.isfinite(entropy): raise ValueError("nonfinite geometry")
    return {"nearest_mean": nearest, "occupied8": occupancy, "entropy4": entropy}
