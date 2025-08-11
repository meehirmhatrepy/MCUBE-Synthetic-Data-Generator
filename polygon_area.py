import numpy as np
import cv2
def polygon_area(poly):
    return abs(cv2.contourArea(poly.astype(np.float32)))