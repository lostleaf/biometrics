import numpy as np
import sys
import os


r = np.load("./result/eyes_landmark_brisk_scores.npz")
scores1 = r['score']
pairs = r['pair']



r = np.load("./result/eyes_landmark_orb_scores.npz")
scores2 = r['score']

r = np.load("./result/eyes_landmark_sift_scores.npz")
scores3 = r['score']



r = np.load("./result/eyes_landmark_surf_scores.npz")
scores4 = r['score']


scores = scores1 * scores2 * scores3 * scores4 

print scores
np.savez("./result/eyes_landmark_4fea_scores.npz", score = scores, pair = pairs)
