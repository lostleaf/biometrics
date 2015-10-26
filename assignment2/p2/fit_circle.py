import cv2
import numpy as np
from find_center_range import find_center_range
import matplotlib.pyplot as plt

def fit_circle(im, cx, cy, cl, is_pupil):
    ans_x, ans_y, ans_r = None, None, None
    ans = None
    r_low, r_high = int(cl * 1.414), int(cl * 2)
    if is_pupil:
        r_low, r_high = int(cl / 1.414), int(cl * 1.414)
    for x in xrange(cx, cx + cl):
        for y in xrange(cy, cy + cl):
            prev = None
            for r in xrange(r_low, r_high):
                # print x, y, r
                cur = []
                for theta in np.linspace(0, 2*np.pi, 600):
                    xx, yy = int(round(x + r * np.cos(theta))), int(round(y + r * np.sin(theta)))
                    cur.append(im[xx, yy])  
                cur = np.array(cur)    
                # print cur.shape
                if prev is not None:
                    int_val = np.sum(np.abs(cur - prev))
                    # print r, int_val
                    if ans is None or int_val > ans:
                        ans = int_val
                        ans_x, ans_y, ans_r = x, y, r
                prev = cur
    return ans_x, ans_y, ans_r

def test():
    im_path = "/Volumes/Untitled/2008-03-11_13/05176/05176d354.tiff"
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (160, 120))
    x, y, l = find_center_range(im)
    # cv2.rectangle(im, (y, x), (y + l, x + l), 255)
    x, y, r = fit_circle(im, x, y, l, True)
    print type(x), type(y), type(r)
    cv2.circle(im, (y, x), r, 255, thickness=1) 
    plt.imshow(im, cmap='gray')
    plt.show()

if __name__ == "__main__":
    test()
