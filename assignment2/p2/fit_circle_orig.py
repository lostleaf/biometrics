import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def find_center_range(im, length_low=20, length_high=38):
    im_h, im_l = im.shape
    avg_min, ans_x, ans_y, ans_l = None, None, None, None
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, None)
    for l in xrange(length_low, length_high):
        for x in xrange(0, im_h - l):
            for y in xrange(0, im_l - l):
                avg = np.sum(im[x : x + l, y : y + l]) / float(l * l)
                if avg_min == None or avg < avg_min:
                    avg_min = avg
                    ans_x, ans_y, ans_l = x, y, l
    return ans_x, ans_y, ans_l

def find_location_to_process(rootDir, dstDir, start):
    list_dirs = os.walk(rootDir)
    count = 0

    dstDir = os.path.join(os.path.dirname("__dir__"),dstDir)
    if not os.path.exists(dstDir):
        os.mkdir(dstDir)
    for root, dirs, files in list_dirs:
        for f in files:
            if f[-4:]=='tiff':
                if count >= start:
                    print os.path.join(root, f)
                    image_process(os.path.join(root, f),count,dstDir)
                count+=1

def fit_circle(im, cx, cy, cl, is_pupil):
    ans_x, ans_y, ans_r = None, None, None
    ans = None
    r_low, r_high = int(cl * 2), int(cl * 4)
    if is_pupil:
        r_low, r_high = int(cl / 3), int(cl)
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
                    if not is_pupil and np.cos(theta) > -0.75 and np.cos(theta) < 0.75:
                        continue
                    xx, yy = int(round(x + r * np.sin(theta))), int(round(y + r * np.cos(theta)))
                    if (xx!=prex or yy!=prey) and yy<160 and xx<120:
                        # if is_pupil==False:
                            # print "haha", xx,yy,"test",im[xx,yy]
                        cur.append(im[xx, yy])
                    prex, prey = xx, yy

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


def image_process(im_path,count,dst_path):
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    #print os.path.exists(im_path)
    src = cv2.resize(im, (160,120))

    rx, ry, rl = find_center_range(src)
    src = cv2.medianBlur(src,3)
    px, py, pr = fit_circle(src, rx, ry, rl, True)
    cv2.circle(im, (4*py, 4*px), 4*pr, 255, thickness=1)
    ix, iy, ir = fit_circle(src, int(px - pr/2), int(py - pr/2), pr, False)
    cv2.circle(im, (4*iy, 4*ix), 4*ir, 255, thickness=1)
    #plt.imshow(im, cmap='gray')
    #cv2.imshow('0',im)
    #cv2.waitKey(0)
    print count
    dst = os.path.join(dst_path, str(count)+'_hw.tiff')
    cv2.imwrite(dst, im)
    #print dst

def test():
    #location of the image file location
    find_location_to_process('2008-03-11_13', 'result_ori'); 


if __name__ == "__main__":
    srcdir = 'D:\\Files\\Learning\\Northwestern\\EECS 495 Biometric\\hw\\2008-03-11_13'
    dstdir = 'result_ori'
    start = 0
    if len(sys.argv) > 0:
        print sys.argv[0]
    if len(sys.argv) > 1:
        start = int(sys.argv[1])
    if len(sys.argv) > 2:
        srcdir = sys.argv[2]
    if len(sys.argv) > 3:
        dstdir = sys.argv[3]

    find_location_to_process(srcdir, dstdir, start); 
