import numpy as np
def polygon_centroid(poly):
    x = poly[:,0]; y = poly[:,1]
    a = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    A = a / 2.0
    if abs(A) < 1e-6:
        return np.array([x.mean(), y.mean()])
    cx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * A)
    cy = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6 * A)
    return np.array([cx, cy])