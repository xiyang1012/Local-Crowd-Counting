import os
import sys
import cv2
import numpy as np
from math import floor
 
def get_density_map_gaussian(img, points, k_size, sigma):

    # points (w, h)
    [w, h, c] = img.shape
    im_density = np.zeros((h, w), dtype=float)

    if len(points) == 0:
        return im_density

    for j in range(len(points[:, 1])):
        f_sz = k_size
        H = np.multiply(cv2.getGaussianKernel(f_sz, sigma),
                        (cv2.getGaussianKernel(f_sz, sigma)).T)  # H.shape == (r, c)

        x = min(w, max(1, abs(int( floor(points[j, 0] )))))
        y = min(h, max(1, abs(int( floor(points[j, 1] )))))
        if x > w or y > h:
            continue

        x1 = x - int(floor(f_sz / 2))
        y1 = y - int(floor(f_sz / 2))
        x2 = x + int(floor(f_sz / 2))
        y2 = y + int(floor(f_sz / 2))
        dfx1 = 0
        dfy1 = 0
        dfx2 = 0
        dfy2 = 0
        change_H = False

        if x1 < 0:
            dfx1 = abs(x1)
            x1 = 0
            change_H = True
        if y1 < 0:
            dfy1 = abs(y1)
            y1 = 0
            change_H = True
        if x2 > w-1:
            dfx2 = x2 - (w-1)
            x2 = w-1
            change_H = True
        if y2 > h-1:
            dfy2 = y2 - (h-1)
            y2 = h-1
            change_H = True

        x1h = 1 + dfx1
        y1h = 1 + dfy1
        x2h = f_sz - dfx2
        y2h = f_sz - dfy2
        if change_H is True:
            # H = fspecial('Gaussian', [float(y2h - y1h + 1), float(x2h - x1h + 1)], sigma);
            H = np.multiply(cv2.getGaussianKernel(int(y2h - y1h + 1), sigma),
                            (cv2.getGaussianKernel(int(x2h - x1h + 1), sigma)).T)  # H.shape == (r, c)

        im_density[y1: y2+1, x1: x2+1] = im_density[y1: y2+1, x1: x2+1] +  H

    return im_density
