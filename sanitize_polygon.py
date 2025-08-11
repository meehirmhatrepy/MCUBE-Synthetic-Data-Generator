import numpy as np

def sanitize_polygon(poly, img_w, img_h):
    """Ensure polygon is Nx2 float, clip to image bounds, fix NaN/inf and (0,0) points.

    Repairs strategy:
      - convert to float32 Nx2
      - replace NaN/inf with centroid
      - clip coords to [0, img_w-1] / [0, img_h-1]
      - if any point is exactly (0,0) (or extremely close), replace it with avg of neighbors.
    Returns sanitized polygon as float32 Nx2.
    """
    poly = np.asarray(poly, dtype=np.float32)
    if poly.size == 0:
        return poly.reshape((-1,2)).astype(np.float32)
    # reshape if flat
    if poly.ndim == 1:
        poly = poly.reshape((-1,2))
    elif poly.ndim == 2 and poly.shape[1] != 2:
        # try reshape
        poly = poly.reshape((-1,2))
    # replace NaN/inf with centroid (compute safe centroid)
    if np.isnan(poly).any() or np.isinf(poly).any():
        valid = poly[~(np.isnan(poly).any(axis=1) | np.isinf(poly).any(axis=1))]
        if valid.shape[0] > 0:
            centroid = valid.mean(axis=0)
        else:
            centroid = np.array([img_w/2.0, img_h/2.0], dtype=np.float32)
        mask_bad = np.isnan(poly).any(axis=1) | np.isinf(poly).any(axis=1)
        poly[mask_bad] = centroid
        print(f"[sanitize] replaced NaN/inf in polygon with centroid {centroid}")
    # clip to image bounds
    poly[:,0] = np.clip(poly[:,0], 0.0, float(max(0, img_w-1)))
    poly[:,1] = np.clip(poly[:,1], 0.0, float(max(0, img_h-1)))
    # detect (0,0) points (or very close to 0) which are often sentinel bugs
    zero_mask = (np.isclose(poly[:,0], 0.0, atol=1e-3) & np.isclose(poly[:,1], 0.0, atol=1e-3))
    if zero_mask.any():
        N = poly.shape[0]
        if N >= 3:
            for idx in np.where(zero_mask)[0]:
                # neighbors
                prev_idx = (idx - 1) % N
                next_idx = (idx + 1) % N
                cand = (poly[prev_idx] + poly[next_idx]) / 2.0
                # if candidate is also zero, fall back to polygon centroid
                if np.isclose(cand[0], 0.0, atol=1e-3) and np.isclose(cand[1], 0.0, atol=1e-3):
                    cand = poly.mean(axis=0)
                poly[idx] = cand
            print(f"[sanitize] repaired {zero_mask.sum()} (0,0) vertices by neighbor interpolation")
        else:
            # degenerate polygon - set to small rectangle center
            poly = poly.copy()
            cx = img_w/2.0; cy = img_h/2.0
            poly = np.array([[cx-1,cy-1],[cx+1,cy-1],[cx+1,cy+1],[cx-1,cy+1]], dtype=np.float32)
            print("[sanitize] degenerate polygon fixed to small center rectangle")
    return poly.astype(np.float32)