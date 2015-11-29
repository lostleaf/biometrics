import cv2
import numpy as np
import matplotlib as plt
import nefio
import dlib


landmarkmodel = './model/shape_predictor_68_face_landmarks.dat'

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))


dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor(landmarkmodel)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = dlib_detector(im, 1)
    
    if len(rects) == 0:
        return np.matirx([])
    if len(rects) > 1:
        cv2.imsave("manyFaces.tiff", im)

    return np.matrix([[p.x, p.y] for p in dlib_predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def warp_trans_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpPerspective(im,
                        M[0],
                       (dshape[1], dshape[0]),
                       dst=output_im,
                       flags=cv2.WARP_INVERSE_MAP)
    return output_im


def align_face(im1, im2):
    landmarks1 = get_landmarks(im1)
    landmarks2 = get_landmarks(im2)

    # Affine Transform
    M = transformation_from_points(landmarks1[JAW_POINTS + NOSE_POINTS],
                               landmarks2[JAW_POINTS + NOSE_POINTS])
    aligned_im2 = warp_im(im2, M, im1.shape)

    """
    # Prespective Transform
    M = cv2.findHomography(landmarks1.astype(float), landmarks2.astype(float), cv2.RANSAC)
    aligned_im2 = warp_trans_im(im2, M, im1.shape)

    """
    return aligned_im2


def KeyPoint_convert(points):
    keypoints = []
    for pt in points:
        keypoints.append(cv2.KeyPoint(pt[0,0],pt[0,1],1))
    return np.array(keypoints)


def zscore(v):
    return (v - np.mean(v)) / np.std(v)

def znorm(m, means, stds):
    ncols = means.shape[1]
    for i in range(ncols):
        m[:,i] = (m[:,i] - means[i]) / stds[i]
    return m

def cosineDistance(vec1, vec2):
    if vec1.ndim == 1:
        return float(np.dot(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    else:
        nrows = vec1.shape[0]
        dis = np.zeros(nrows)
        for i in range(nrows):
            dis[i] = float(np.dot(vec1[i], vec2[i])) / (np.linalg.norm(vec1[i]) * np.linalg.norm(vec2[i]))
        return dis.mean()

def detectKeyPoints(im, mask=None, featureType="SIFT"):
    return

def computeFeature(im, points, mask=None, featureType="SIFT"):
    descriptor = cv2.DescriptorExtractor_create(featureType)
    return descriptor.compute(im, points)
    
