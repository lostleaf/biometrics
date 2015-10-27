import cv2
import numpy as np
from find_center_range import find_center_range
from fit_circle import fit_circle
import matplotlib.pyplot as plt

def test():
    im_path = "04252d427.tiff"
    # im_path = "/Volumes/Untitled/2008-03-11_13/05460/05460d96.tiff"
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (160, 120))
    x, y, l = find_center_range(im)
    im_canny = cv2.Canny(im, 50, 200)
    x, y, r = fit_circle(im_canny, x, y, l, True)
    cv2.circle(im, (y, x), r, 255, thickness=1)
    x, y, r = fit_circle(im_canny, x, y, l, False)
    cv2.circle(im, (y, x), r, 255, thickness=1)
    plt.imshow(im, cmap='gray')
    plt.show()

if __name__ == "__main__":
    test()
