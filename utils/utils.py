import numpy as np
from typing import Tuple, Dict
import json
import os
import pandas as pd

# ----------------------------
# Core adaptive update helpers
# ----------------------------

def _linear_adaptive(prev: Tuple[float, float, float],
                     best: float,
                     bounds: Tuple[float, float],
                     *,
                     iter_idx: int,
                     interior_frac: float = 0.2,
                     shrink_interior: float = 0.6,
                     shrink_edge: float = 0.85,
                     expand_if_at_global_edge: float = 1.25,
                     min_step: float = 0.25,
                     target_points_base: int = 41,
                     target_points_growth: float = 1.0) -> Tuple[float, float, float]:
    """
    Adaptive (start, stop, step) for td or Tc on a linear axis.

    prev:  (start, stop, step) from last round
    best:  latest best estimate (centre)
    bounds: hard (lo, hi)
    iter_idx: non-negative iteration index
    """
    lo, hi = map(float, bounds)
    start0, stop0, step0 = map(float, prev)
    start0, stop0 = min(start0, stop0), max(start0, stop0)
    width0 = max(stop0 - start0, min_step)

    # Position of best inside previous window (0 at start, 1 at stop)
    if width0 <= 0:
        pos = 0.5
    else:
        pos = (float(best) - start0) / width0

    # Choose shrink/expand factor
    if interior_frac <= pos <= 1.0 - interior_frac:
        factor = shrink_interior                      # aggressive shrink
    else:
        factor = shrink_edge                          # mild shrink near edge

    # If we are pressed against global bounds, allow slight expansion
    near_global_edge = (np.isclose(start0, lo) or np.isclose(stop0, hi))
    if near_global_edge and (pos <= 0.02 or pos >= 0.98):
        factor = max(factor, 1.0 / expand_if_at_global_edge)  # expand window

    # New target width
    width_new = max(min_step, width0 * factor)

    # Centre at best and clamp to bounds
    c = float(best)
    start = max(lo, c - 0.5 * width_new)
    stop  = min(hi, c + 0.5 * width_new)

    # Guard: ensure at least min_step span
    if stop - start < min_step:
        mid = np.clip(c, lo, hi)
        start = max(lo, mid - 0.5 * min_step)
        stop  = min(hi, mid + 0.5 * min_step)
        if stop - start < min_step:
            stop = min(hi, start + min_step)

    # Step: gradually refine with iteration (optional growth in points)
    tgt_pts = int(max(2, round(target_points_base * (target_points_growth ** iter_idx))))
    step = max(min_step, (stop - start) / (tgt_pts - 1))

    # Snap step to a multiple of min_step (keeps nice frame-aligned grids)
    m = max(1, int(round(step / min_step)))
    step = m * min_step

    # Recompute stop so arange(...) has >= 2 samples
    n = max(2, int(np.floor((stop - start) / step)) + 1)
    stop = start + (n - 1) * step

    # Final clamp if rounding pushed us out
    if stop > hi:
        shift = stop - hi
        start = max(lo, start - shift)
        stop  = start + (n - 1) * step

    return float(start), float(stop), float(step)


def _log_adaptive(prev: Tuple[float, float, int],
                  best: float,
                  bounds: Tuple[float, float],
                  *,
                  iter_idx: int,
                  interior_frac: float = 0.2,
                  shrink_interior: float = 0.6,
                  shrink_edge: float = 0.85,
                  expand_if_at_global_edge: float = 1.25,
                  min_ratio: float = 1.05,
                  num_base: int = 80,
                  num_growth: float = 1.1) -> Tuple[float, float, int]:
    """
    Adaptive (min_k, max_k, num_k) for k2 on a log axis.

    prev:  (min_k, max_k, num_k) from last round
    best:  latest best estimate (>0); if <=0 or nonfinite, defaults applied
    bounds: hard (lo, hi)
    """
    lo, hi = map(float, bounds)
    min0, step0, num0 = float(prev[0]), float(prev[1]), int(prev[2])
    min0 = max(lo, min0)
    max0 = min(hi, step0*num0)
    if not np.isfinite(best) or best <= 0:
        best = np.sqrt(min0 * max0) if min0 > 0 and max0 > min0 else 1e-2

    # Work in log space
    lmin0, lmax0 = np.log(max(min0, 1e-12)), np.log(max(max0, min0 * (1.0 + 1e-6)))
    lbest = np.log(best)

    width0 = max(lmax0 - lmin0, np.log(min_ratio))   # at least minimal width
    pos = (lbest - lmin0) / width0                   # 0..1 within previous log window

    # Shrink/expand choice
    if interior_frac <= pos <= 1.0 - interior_frac:
        factor = shrink_interior
    else:
        factor = shrink_edge

    # If pressed against global bounds in log-space, allow expansion
    near_global_edge = (np.isclose(min0, lo) or np.isclose(max0, hi))
    if near_global_edge and (pos <= 0.02 or pos >= 0.98):
        factor = max(factor, 1.0 / expand_if_at_global_edge)

    # New half-width in log space
    half_new = 0.5 * width0 * factor
    half_new = max(0.5 * np.log(min_ratio), half_new)

    lstart = lbest - half_new
    lstop  = lbest + half_new

    # Clamp to bounds
    lstart = max(np.log(max(lo, 1e-12)), lstart)
    lstop  = min(np.log(max(hi, lo * (1.0 + 1e-6))), lstop)

    # Ensure minimal dynamic range
    if lstop - lstart < np.log(min_ratio):
        mid = np.clip(lbest, lstart, lstop)
        lstart = mid - 0.5 * np.log(min_ratio)
        lstop  = mid + 0.5 * np.log(min_ratio)
        # Snap to bounds if necessary
        lstart = max(np.log(max(lo, 1e-12)), lstart)
        lstop  = min(np.log(max(hi, lo * (1.0 + 1e-6))), lstop)

    min_k = float(np.exp(lstart))
    max_k = float(np.exp(lstop))

    # Increase resolution modestly with iteration
    num_k = int(max(2, round(num_base * (num_growth ** iter_idx))))

    return min_k, max_k, num_k

# ---------------------------------
# Public API: one-step range update
# ---------------------------------

def update_search_ranges(prev_td: Tuple[float, float, float],
                         prev_Tc: Tuple[float, float, float],
                         prev_k2: Tuple[float, float, int],
                         best_td: float,
                         best_Tc: float,
                         best_k2: float,
                         *,
                         iter_idx: int,
                         td_bounds: Tuple[float, float] = (-20.0, 20.0),
                         Tc_bounds: Tuple[float, float] = (0.0, 30.0),
                         k2_bounds: Tuple[float, float] = (1e-6, 1.0),
                         min_step: float = 0.25) -> Dict[str, Tuple]:
    """
    Adaptive, iteration-aware update of (td, Tc, k2) search ranges.

    Returns a dict:
      {
        'td_params': (start, stop, step),   # seconds (td may be negative)
        'Tc_params': (start, stop, step),   # seconds (Tc >= 0)
        'k2_params': (min_k, max_k, num_k)  # s^-1   (k2 > 0; 0 is added later by your logspace routine)
      }
    """
    # Enforce Tc >= 0 in the centre used for recentering
    best_Tc = max(1, float(best_Tc))

    td_params = _linear_adaptive(prev_td, best_td, td_bounds,
                                 iter_idx=iter_idx,
                                 min_step=min_step)

    Tc_params = _linear_adaptive(prev_Tc, best_Tc, Tc_bounds,
                                 iter_idx=iter_idx,
                                 min_step=min_step)

    k2_params = _log_adaptive(prev_k2, best_k2, k2_bounds,
                              iter_idx=iter_idx)

    return {'td_params': td_params, 'Tc_params': Tc_params, 'k2_params': k2_params}




def parsePetMetricsJson(jsonPath, PET=None):
    """
    Load a PET metrics JSON file and return a pandas DataFrame with selected metadata.

    Parameters:
    - jsonPath (str): Path to the JSON file.

    Returns:
    - pd.DataFrame: DataFrame with columns: patient, organ, mean, path, filename
    """
    with open(jsonPath, 'r') as f:
        jsonData = json.load(f)

    patient = jsonData.get("Patient")
    maskLabels = jsonData.get("Mask Labels", {})
    metrics = jsonData.get("Metrics", {})

    rows = []
    # Assume only one PET dataset key

    for petKey in jsonData.get("Procedure")["PET list"]:

        organs = metrics[petKey].get("organs", {})

        for organId, organData in organs.items():
            organName = maskLabels.get(organId, f"Organ_{organId}")
            row = {
                "subject": patient,
                "scanKey":petKey,
                "region": organName,
                "mean": np.array(organData.get("avg")),
                "size":organData.get("size"),
                "framing":jsonData.get('framing [s]')[petKey],
                "path": jsonPath,
                "filename": os.path.basename(jsonPath)
            }
            rows.append(row)

    return pd.DataFrame(rows)