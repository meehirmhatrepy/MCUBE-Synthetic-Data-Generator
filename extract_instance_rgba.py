import numpy as np
from poly_to_mask import poly_to_mask
def extract_instance_rgba(img, poly):
    h, w = img.shape[:2]
    xs = poly[:,0]; ys = poly[:,1]
    minx = max(0, int(xs.min())); miny = max(0, int(ys.min()))
    maxx = min(w, int(xs.max())+1); maxy = min(h, int(ys.max())+1)
    if maxx<=minx or maxy<=miny:
        return None, (0,0)
    crop = img[miny:maxy, minx:maxx].copy()
    mask_full = poly_to_mask(poly, h, w)
    mask_crop = mask_full[miny:maxy, minx:maxx].copy()
    rgba = np.zeros((crop.shape[0], crop.shape[1], 4), dtype=np.uint8)
    rgba[:,:,:3] = crop
    rgba[:,:,3] = mask_crop
    return rgba, (minx, miny)