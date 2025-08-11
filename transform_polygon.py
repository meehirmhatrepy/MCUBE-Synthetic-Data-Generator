import math
import numpy as np  
from sanitize_polygon import sanitize_polygon

def transform_polygon(poly, translate=(0,0), angle_deg=0.0, scale=1.0, origin=None, img_w=None, img_h=None):
    """Apply similarity transform to every vertex in poly (keeps same number of points).

    If img_w and img_h are provided, the result is sanitized (clipped/fixed) to image bounds.
    """
    # Default origin is (0,0) if not provided
    if origin is None:
        origin = np.array([0.0, 0.0])
    pts = poly - origin
    theta = math.radians(angle_deg)
    c = math.cos(theta); s = math.sin(theta)
    R = np.array([[c, -s],[s, c]])
    pts = pts * scale
    pts = pts.dot(R.T)
    pts = pts + origin + np.array(translate)
    if img_w is not None and img_h is not None:
        pts = sanitize_polygon(pts, img_w, img_h)
    return pts.astype(np.float32)