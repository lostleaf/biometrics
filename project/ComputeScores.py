import numpy as np
import csvio
import os
import sys


RIGHT_BROW_POINTS = list(range(0, 5))
LEFT_BROW_POINTS = list(range(5, 10))
RIGHT_EYE_POINTS = list(range(19, 25))
LEFT_EYE_POINTS = list(range(25, 31))
NOSE_POINTS = list(range(10, 18))
MOUTH_POINTS = list(range(31, 51))

BROWS_POINTS =  RIGHT_BROW_POINTS + LEFT_BROW_POINTS
EYES_POINTS = RIGHT_EYE_POINTS + LEFT_EYE_POINTS


def avgCosineDistance(fea1, fea2, indices = NOSE_POINTS):
    if fea1.ndim == 1:
        return float(np.dot(fea1, fea2)) / (np.linalg.norm(fea1) * np.linalg.norm(fea2))
    elif fea2.ndim == 2:
        if indices != None:
            fea1 = fea1[indices, :]
            fea2 = fea2[indices, :]


        nrows = fea1.shape[0] 
        dis = np.zeros(nrows).astype(float)
        
        for i in range(nrows):
            dis[i] = float(np.dot(fea1[i], fea2[i])) / (np.linalg.norm(fea1[i]) * np.linalg.norm(fea2[i]))
        return dis.mean()


if __name__ == "__main__":

    rootdir = "./result"
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    output_name = "nose_landmark_sift_scores.npz"
    
    template_file = "./features/train_landmark_sift_norm.npz"
    query_file = "./features/test_landmark_sift_norm.npz"
    
    if len(sys.argv) > 0:
        print sys.argv[0]
        
    if len(sys.argv) > 1:
        output_name = int(sys.argv[1])
        
    if len(sys.argv) > 2:
        template_file = sys.argv[1]
    
    if len(sys.argv) > 3:
        query_file = int(sys.argv[2])

    output_file = os.path.join(rootdir, output_name)

    template = np.load(template_file)
    query = np.load(query_file)

    template_fea, template_label = template['feature'], template['label']
    query_fea, query_label = query['feature'], query['label']

    #print(len(query_fea), len(query_label))
    twin_map = csvio.readTwinPair("./list/twin.csv")

    
    scores = []
    pairs = []
    qstart = 0
    qend = 0
    for i in range(len(template_label)):
        
        tid = template_label[i]
        tfea = template_fea[i]

        if not tid in twin_map.keys():
            continue

        trueSim = []
        falseSim = []

        # self matching        
        qstart = list(query_label).index(tid)
        qend = qstart
        while qend < len(query_label):
            if query_label[qstart] != query_label[qend]:
                break
            qend += 1

        print i, (qstart, qend),
        
        for j in range(qstart,qend):
            qid = query_label[j]
            qfea = query_fea[j]

            distance = avgCosineDistance(tfea, qfea)
            score = 0.5 + distance / 2
            pair = (tid, qid)
            scores.append(score)
            pairs.append(pair)

            trueSim.append(score)


        qstart = list(query_label).index(twin_map[tid])
        qend = qstart
        while qend < len(query_label):
            if query_label[qstart] != query_label[qend]:
                break
            qend += 1

        print (qstart, qend)
        
        for j in range(qstart,qend):
            qid = query_label[j]
            qfea = query_fea[j]

            distance = avgCosineDistance(tfea, qfea)
            score = 0.5 + distance / 2
            pair = (tid, qid)
            scores.append(score)
            pairs.append(pair)

            falseSim.append(score)
        
        avgTrueSim = np.array(trueSim).mean()
        avgFalseSim = np.array(falseSim).mean()
        
        print(i,tid, "T:", avgTrueSim, "F:", avgFalseSim)
        
    
    np.savez(output_file, score = scores, pair = pairs)
