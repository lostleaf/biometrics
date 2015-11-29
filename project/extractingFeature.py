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


def process(feaType, start = 0, end =5000):
    listpath = "./list/frontface_list.csv";
    rootDir = "J:/Biometrics/images/"
    piclist = csvio.readList(listpath)

    templatepath = "./model/template_landmark.npy"

    dstDir = os.path.join("./data", feaType)
    if not os.path.exists(dstDir):
        os.mkdir(dstDir)

    template_marks = np.matrix(np.load(templatepath))
    #print(template_marks[NOSE_POINTS])
    i = 0
    keypoints = []
    features = []
    for row in piclist:
        i += 1
        if i < start:
            continue
        if i > end:
            break

        row = row[0].split(' ')
        picname = row[0].strip()
        picdir = picname.split("d")[0]
        picpath = os.path.join(rootDir, picdir, picname)

        print(i, row, picpath)
        
        im = nefio.nefread(picpath, 0.125)
        
        landmarks = flib.get_landmarks(im)
        #print(landmarks[NOSE_POINTS])

        
        # Do alignment
        M = flib.transformation_from_points(template_marks[NOSE_POINTS], landmarks[NOSE_POINTS])
        im = flib.warp_im(im, M, im.shape)

        landmarks = flib.get_landmarks(im)
        
        kps = flib.KeyPoint_convert(landmarks[FACE_POINTS])
        kps, fea = flib.computeFeature(im, kps, None, feaType)

        # save
        dstname = "frontface_landmarks_{0}.npz".format(picname.split(".")[0])
        dstpath = os.path.join(dstDir,dstname)

        pts, des = pack_keypoint(kps, fea)
        np.savez(dstpath, points_array = pts, feature_array = des)
        """
        keypoints.append(kps)
        features.append(fea)
        if i%10:
            temppath = os.path.join(dstDir,"temp_{0}.npy".format(int(i//10)))
            pts, des = pack_keypoint(keypoints[i-10:i], features[i-10:i])
            np.savez(temppath, points_array = pts, feature_array = des)
            print("Saved:", temppath)
        """

    #pts, des = pack_keypoint(keypoints, features)
    #np.savez(dstpath, points_array = pts, feature_array = des)
    print "Finished"


if __name__ == "__main__":

    feaType = "ORB"
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
        
    process(feaType,start,end)
        
