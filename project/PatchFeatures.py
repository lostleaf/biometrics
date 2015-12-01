import facelib as flib
import nefio
import cv2
import numpy as np
import csvio
import os
import sys

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))


def pack_keypoint(keypoints, descriptors):
    kpts = np.array([[kp.pt[0], kp.pt[1], kp.size,
                  kp.angle, kp.response, kp.octave,
                  kp.class_id]
                 for kp in keypoints])
    desc = np.array(descriptors)
    return kpts, desc

def unpack_keypoint(array):
    try:
        kpts = array[:,:7]
        desc = array[:,7:]
        keypoints = [cv2.KeyPoint(x, y, _size, _angle, _response, int(_octave), int(_class_id))
                 for x, y, _size, _angle, _response, _octave, _class_id in list(kpts)]
        return keypoints, np.array(desc)
    except(IndexError):
        return np.array([]), np.array([])


def rectHull(box):
    print box
    x = box.left()
    y = box.right()
    w = box.width()
    h = box.height()

    return np.array([(x,y),(x+w,y),(x+w,y+h),(x, y+h)])

def test():
    im = cv2.imread("face2.tiff")
    im = cv2.resize(im, (400,600))
    rects = flib.detect_face(im)


    x = rects[0].left()
    y = rects[0].top()
    w = rects[0].width()
    h = rects[0].height()

    landmarks = flib.get_landmarks(im)

    cv2.imwrite("face_detection.tiff", im)
    
    mask = np.zeros(im.shape[:2], np.uint8)
    contour = np.array(list(landmarks[JAW_POINTS])+[np.matrix((x,y)),np.matrix((x+w,y))])
    #print contour
    region = cv2.convexHull(contour)
    cv2.fillPoly(mask,[region],1)

    eyes = landmarks[LEFT_EYE_POINTS+RIGHT_EYE_POINTS+RIGHT_BROW_POINTS+LEFT_BROW_POINTS]
    lefteye = cv2.convexHull(landmarks[LEFT_BROW_POINTS+LEFT_EYE_POINTS])
    lefteye = np.int0(cv2.cv.BoxPoints(cv2.minAreaRect(lefteye)))
    
    righteye = cv2.convexHull(landmarks[RIGHT_BROW_POINTS+RIGHT_EYE_POINTS])
    righteye = np.int0(cv2.cv.BoxPoints(cv2.minAreaRect(righteye)))
    nose = cv2.convexHull(landmarks[NOSE_POINTS])
    mouth = cv2.convexHull(landmarks[MOUTH_POINTS])

    cv2.fillPoly(mask, [lefteye, righteye, nose, mouth], 0)
    #cv2.fillPoly(mask, [eyes, nose, mouth], 0)
    """
    cv2.fillConvexPoly(mask, lefteye, 0)
    cv2.fillConvexPoly(mask, righteye, 0)
    cv2.fillConvexPoly(mask, nose, 0)
    cv2.fillConvexPoly(mask, mouth, 0)
    """
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel)
    
    cv2.imwrite("mask.tiff", cv2.bitwise_and(im,im, mask=mask))
    #cv2.imwrite("mask.tiff", im*mask)
    #cv2.waitKey(0)

    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(im, mask)
    #print kp, des
    res = cv2.drawKeypoints(im,kp)
    cv2.imwrite("keypoints.tiff", res)
    #cv2.fillConvexPoly(mask, 

    

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)
    

def readfeatures(feaType, rootdir = None, start = 0, end =5000):
    listpath = "./list/frontface_list.csv";
    rootDir = "J:/Biometrics/images/"
    piclist = csvio.readList(listpath)

    templatepath = "./model/template_landmark.npy"
    

    dstDir = os.path.join("./features","nonlandmark", feaType)
    if rootdir!=None:
        dstDir = os.path.join(rootdir, feaType)
        
    if not os.path.exists(dstDir):
        os.mkdir(dstDir)

    i = 0
    train_feature, test_feature = [], []
    train_id, test_id = [],[]

    preperson = ""
    for row in piclist:
        i += 1
        if i < start:
            continue
        if i > end:
            break

        row = row[0].split(' ')
        picname = row[0].strip()
        person = picname.split("d")[0]
        
        filename = "frontface_landmarks_{0}.npz".format(picname.split(".")[0])
        filepath = os.path.join(dstDir,filename)
        
        print(i, row, filepath)
        
        if not os.path.exists(filepath):
            continue
    
        r = np.load(filepath)
        features = r["feature_array"]

        if preperson == person:
            test_feature.append(features)
            test_id += [person]
        else:
            train_feature.append(features)
            train_id += [person]
        preperson = person

    print "Finished"
    return train_feature, train_id, test_feature, test_id


if __name__ == "__main__":

    feaType = "SURF"
    start = 0
    end = 5000
    if len(sys.argv) > 0:
        print sys.argv[0]
        
    if len(sys.argv) > 1:
        feaType = sys.argv[1]
    
    if len(sys.argv) > 2:
        start = int(sys.argv[2])

    if len(sys.argv) > 3:
        end = int(sys.argv[3])
        
    a,b,c,d = readfeatures(feaType, None)
    #print a
    #print b
    #print c
    #print d

    np.savez("features/nonlandmark/train_nonlandmark_surf.npz", feature = a, label = b)
    np.savez("features/nonlandmark/test_nonlandmark_surf.npz", feature = c, label = d)
