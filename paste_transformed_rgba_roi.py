import math
import numpy as np
import cv2
def paste_transformed_rgba_roi(background, rgba, src_origin, centroid_src, centroid_dst, angle_deg, scale):
    H, W = background.shape[:2]
    h, w = rgba.shape[:2]
    # similarity parameters
    theta = math.radians(angle_deg)
    c = math.cos(theta); s = math.sin(theta)
    a = scale * c; b = -scale * s; d = scale * s; e = scale * c
    src_origin = np.array(src_origin, dtype=np.float32)
    centroid_src = np.array(centroid_src, dtype=np.float32)
    centroid_dst = np.array(centroid_dst, dtype=np.float32)
    tx = centroid_dst[0] - (a * centroid_src[0] + b * centroid_src[1])
    ty = centroid_dst[1] - (d * centroid_src[0] + e * centroid_src[1])
    A = np.array([[a, b], [d, e]], dtype=np.float32)
    t = np.array([tx, ty], dtype=np.float32)

    corners_local = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    src_glob_pts = corners_local + src_origin
    dst_pts = (src_glob_pts.dot(A.T) + t)
    min_xy = np.floor(dst_pts.min(axis=0)).astype(int)
    max_xy = np.ceil(dst_pts.max(axis=0)).astype(int)
    x1 = max(0, min_xy[0]); y1 = max(0, min_xy[1])
    x2 = min(W, max_xy[0]); y2 = min(H, max_xy[1])
    if x2 <= x1 or y2 <= y1:
        return background
    roi_w = x2 - x1; roi_h = y2 - y1
    offset = (A.dot(src_origin) + t - np.array([x1, y1], dtype=np.float32))
    Mprime = np.array([[a, b, offset[0]],[d, e, offset[1]]], dtype=np.float32)
    warped = cv2.warpAffine(rgba, Mprime, (roi_w, roi_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    alpha = warped[:,:,3:4].astype(np.float32)/255.0
    rgb = warped[:,:,:3].astype(np.float32)
    roi = background[y1:y2, x1:x2].astype(np.float32)
    comp = (alpha * rgb + (1-alpha) * roi).astype(np.uint8)
    out = background.copy()
    out[y1:y2, x1:x2] = comp
    return out