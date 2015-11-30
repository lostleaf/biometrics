from numpy import *
from cv2 import *
with load('frontface_landmarks_90004d17.npz') as data:
    points = data['points_array']
    features = data['feature_array']

with load('frontface_landmarks_90004d18.npz') as data:
    _points = data['points_array']
    _features = data['feature_array']

print features[0].size
print _features[0].size

bf = BFMatcher(NORM_HAMMING,crossCheck=True)
matches = bf.match(points,_points)
matches = sorted(matches, key = lambda x:x.distance)
print "test"
