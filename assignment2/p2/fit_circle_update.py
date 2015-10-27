import cv2
import numpy as np
from find_center_range import find_center_range
import matplotlib.pyplot as plt
import os


def find_location_to_process(rootDir):
    list_dirs = os.walk(rootDir)
    count = 0
    for root, dirs, files in list_dirs:
        for f in files:
            if f[-4:]=='tiff':
                print os.path.join(root, f)
                image_process(os.path.join(root, f),count)
                count+=1

def fit_circle(im, cx, cy, cl, is_pupil):
    ans_x, ans_y, ans_r = None, None, None
    ans = None
    r_low, r_high = int(cl*2), int(cl * 4)
    if is_pupil:
        r_low, r_high = int(cl / 2), int(cl+1)
    for x in xrange(cx, cx + cl):
        for y in xrange(cy, cy + cl):
            prev = None
            for r in xrange(r_low, r_high):
                # print x, y, r
                cur = []
                prex, prey = None, None
                for theta in np.linspace(0, 2*np.pi, 600):
                    if np.cos(theta) > -0.2 and np.cos(theta) < 0.2:
                        continue
                    if not is_pupil and np.cos(theta) > -0.7 and np.cos(theta) < 0.7:
                        continue
                    xx, yy = int(round(x + r * np.sin(theta))), int(round(y + r * np.cos(theta)))
                    if (xx!=prex or yy!=prey) and yy<160 and xx<120:
                        # if is_pupil==False:
                            # print "haha", xx,yy,"test",im[xx,yy]
                        cur.append(im[xx, yy])
                cur = np.array(cur)
                # print cur.shape
                if prev is not None:
                    int_val = np.mean(cur) - np.mean(prev)
                    # print r, int_val
                    if ans is None or int_val > ans:
                        ans = int_val
                        ans_x, ans_y, ans_r = x, y, r
                        # print x, y, r
                prev = cur
    return ans_x, ans_y, ans_r


def image_process(im_path,count):
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (160, 120))
    rx, ry, rl = find_center_range(im)
    src = cv2.medianBlur(im,3)
    px, py, pr = fit_circle(src, rx, ry, rl, True)
    cv2.circle(im, (py, px), pr, 255, thickness=1)
    ix, iy, ir = fit_circle(src, int(px - pr/2), int(py - pr/2), pr, False)
    cv2.circle(im, (iy, ix), ir, 255, thickness=1)
    plt.imshow(im, cmap='gray')
    # plt.show()
    print count
    plt.savefig(str(count)+'_hw.tiff')

def test():
    #location of the image file location
    find_location_to_process('/Users/Xiangyu/Documents/bio/hw2/biometrics/assignment2/2008-03-11_13-2')


if __name__ == "__main__":
    test()
