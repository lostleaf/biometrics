import numpy as np
import sys
import os

"""
This file is used to evaluation the algrithm
by caculating TP and FN
"""
def evaluate(scorefile):
    r = np.load(scorefile)
    scores = r['score']
    pairs = r['pair']

    total = len(scores)
    print total
    
    metrics = []
    for thres in np.arange(0.6, 0.9, 0.005, dtype = np.float16):
        TP = 0
        TClaim = 0
        FP = 0
        FClaim = 0

        for i in range(total):
            #print scores[i], pairs[i]
            score = scores[i]
            if pairs[i][0] == pairs[i][1]:
                TClaim += 1
                if  np.float16(score) > thres:
                    TP += 1
            else:
                FClaim += 1
                if np.float16(score)  > thres:
                    FP += 1

        Precision = float( TP + FClaim - FP) / total

        metrics.append(tuple([thres, TP, TClaim, FP, FClaim, Precision]))
        #print thres, TP, TClaim, FN, FClaim, Precision

    return metrics

met = evaluate('./result/eyes_landmark_orb_scores.npz')
for m in met:
    print m

np.save("./evaluation/eyes_landmark_orb_eval.npy", met)
np.savetxt("./evaluation/eyes_landmark_orb_eval.txt", met)
