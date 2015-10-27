import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_center_range(im, length_low=20, length_high=38):
    im_h, im_l = im.shape
    avg_min, ans_x, ans_y, ans_l = None, None, None, None
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, None)
    # plt.imshow(im, cmap='gray')
    # plt.show()
    for l in xrange(length_low, length_high):
        for x in xrange(0, im_h - l):
            for y in xrange(0, im_l - l):
                avg = np.sum(im[x : x + l, y : y + l]) / float(l * l)
                if avg_min == None or avg < avg_min:
                    avg_min = avg
                    ans_x, ans_y, ans_l = x, y, l
    return ans_x, ans_y, ans_l

def test():
    im_path = "04252d427.tiff"
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (160, 120))
    x, y, l = find_center_range(im)
    cv2.rectangle(im, (y, x), (y + l, x + l), 255)
    plt.imshow(im, cmap='gray')
    plt.show()

if __name__ == "__main__":
    test()
