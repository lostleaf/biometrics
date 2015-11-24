import cv2
import numpy as np
import matplotlib as plt
import nefio
import dlib

imagePath = 'J:\Biometrics\images\90292\90292d11.nef'
detectorPath = './model/haarcascade_frontalface_alt_tree.xml'
landmarkmodel = './model/shape_predictor_68_face_landmarks.dat'


img = nefio.nefread(imagePath, 0.2)
detector = cv2.CascadeClassifier(detectorPath)

faces = detector.detectMultiScale(
    img,
    scaleFactor = 1.2,
    minNeighbors = 5,
    minSize = (50, 50),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

#for (x, y, w, h) in faces:
#    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#cv2.imshow("face", img)
#cv2.waitKey(0)



dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor(landmarkmodel)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = dlib_detector(im, 1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

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

def detect_face(img):
    return

landmarks = get_landmarks(img)
marked_face = annotate_landmarks(img, landmarks)
cv2.imshow("face", marked_face)
cv2.waitKey(0)
