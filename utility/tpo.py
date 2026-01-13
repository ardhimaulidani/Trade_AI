import numpy as np

# Parameters
TICK_SIZE = 0.25       # instrument tick size (set to your instrument tick)
VA_COVER = 0.70       # value area coverage (70%)
COMPOSITE_OVERLAP_THRESH = 0.7  # overlap threshold (70%)

# -------------------------
# Helpers: bins & rebinning
# -------------------------
def make_bins(low, high, tick=TICK_SIZE, min_bins=10):
    """Create price bin edges and centers for one day with a small buffer."""
    if np.isnan(low) or np.isnan(high) or low == high:
        # fallback small range
        low = low - tick if not np.isnan(low) else 0.0
        high = high + tick if not np.isnan(high) else tick * (min_bins + 1)
    # add a small buffer to avoid zero-width
    buf = max(tick, (high - low) * 0.01)
    pmin = np.floor((low - buf) / tick) * tick
    pmax = np.ceil((high + buf) / tick) * tick
    # ensure at least min_bins
    if pmax <= pmin:
        pmax = pmin + tick * min_bins
    edges = np.arange(pmin, pmax + tick, tick)
    if len(edges) < 2:
        edges = np.arange(pmin, pmin + tick * (min_bins + 1), tick)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers

def rebin_counts(old_counts, old_edges, new_edges):
    """
    Map old_counts (defined on old_edges) into new_counts (defined on new_edges).
    Simple approach: map each old bin center to a new bin index and add counts there.
    """
    if old_counts is None or old_edges is None:
        return np.zeros(len(new_edges)-1, dtype=np.int32)
    old_centers = (old_edges[:-1] + old_edges[1:]) / 2.0
    new_counts = np.zeros(len(new_edges)-1, dtype=old_counts.dtype if old_counts is not None else np.int32)
    if len(old_centers) == 0:
        return new_counts
    idxs = np.searchsorted(new_edges, old_centers) - 1
    idxs = np.clip(idxs, 0, len(new_counts)-1)
    for oi, ni in enumerate(idxs):
        new_counts[ni] += int(old_counts[oi])
    return new_counts

def update_counts(counts, row, bin_edges):
    """Incrementally update TPO counts with one bar on bin_edges grid (edges = price edges)."""
    low, high = float(row["low"]), float(row["high"])
    n_centers = len(counts)

    # If degenerate bar or equal low/high, treat as single center
    if high <= low:
        idx = np.searchsorted(bin_edges, low, side="right") - 1
        idx = np.clip(idx, 0, n_centers - 1)
        counts[idx] += 1
        return counts

    # For a bar range, compute left/right indices inclusive of covered centers
    # left: smallest center whose edge interval includes low
    # right: largest center whose edge interval includes high
    left = np.searchsorted(bin_edges, low, side="right") - 1
    right = np.searchsorted(bin_edges, high, side="right") - 1

    left = np.clip(left, 0, n_centers - 1)
    right = np.clip(right, 0, n_centers - 1)

    if right < left:
        # degenerate due to rounding — map to nearest center
        mid = 0.5 * (low + high)
        idx = np.searchsorted(bin_edges, mid, side="right") - 1
        idx = np.clip(idx, 0, n_centers - 1)
        counts[idx] += 1
    else:
        counts[left:right + 1] += 1

    return counts

def poc_val_vah(counts, centers, cover=0.70):
    """Compute POC, VAL, VAH by expanding around POC until cover fraction included."""
    total = counts.sum()
    if total == 0:
        return np.nan, np.nan, np.nan

    poc_idx = int(np.argmax(counts))
    poc = float(centers[poc_idx])

    # order indices by distance from POC (tie-breaker: higher count then lower price)
    idx_order = np.argsort(np.abs(centers - poc))
    cum = 0
    included = np.zeros_like(counts, dtype=bool)
    for idx in idx_order:
        included[idx] = True
        cum += counts[idx]
        if cum / total >= cover:
            break

    # compute min/max center included
    included_idxs = np.where(included)[0]
    if included_idxs.size == 0:
        return poc, poc, poc

    val = float(centers[included_idxs.min()])
    vah = float(centers[included_idxs.max()])
    return poc, val, vah
