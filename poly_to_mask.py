import numpy as np
import cv2
def poly_to_mask(poly, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = poly.reshape((-1,1,2)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask