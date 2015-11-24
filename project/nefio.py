import cv2
import numpy as np
import rawpy
import matplotlib.pyplot as plt
import imageio

def nefread(path, resize_scale = 1, gray = False):
    raw = rawpy.imread(path)
    rgb = raw.postprocess()
    (r, g, b)=cv2.split(rgb)
    img = cv2.merge([b,g,r])
    if resize_scale != 1:
        img = cv2.resize(img, None,fx=resize_scale, fy=resize_scale)

    if gray:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = gray_image

    return img
