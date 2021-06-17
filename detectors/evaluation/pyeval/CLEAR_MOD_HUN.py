import numpy as np
from detectors.evaluation.pyeval.getDistance import getDistance
from sklearn.utils.linear_assignment_ import linear_assignment

def CLEAR_MOD_HUN(gt,det):
    """

    @param gt: the ground truth result matrix
    @param det: the detection result matrix
    @return: MODA, MODP, recall, precision

    compute CLEAR Detection metrics according to
    PERFORMANCE EVALUATION PROTOCOL FOR FACE, PERSON AND
    VEHICLE DETECTION & TRACKING IN VIDEO ANALYSIS AND
    CONTENT EXTRACTION (VACE-II)
    CLEAR – CLASSIFICATION OF EVENTS, ACTIVITIES AND RELATIONSHIPS
    Submitted to Advanced Research and Development Activity

    metrics contains the following
    [1]   recall	- recall = percentage of detected targets
    [2]   precision	- precision = percentage of correctly detected targets
    [3]	MODA          - N-MODA
    [4]	MODP          - N-MODP
    """
    td = 50/2.5

    F = int(max(gt[:, 0])) + 1
    N = int(max(det[:, 1])) + 1
    Fgt = int(max(gt[:, 0])) + 1
    Ngt = int(max(gt[:, 1])) + 1

    M = np.zeros((F, Ngt))

    c = np.zeros((1, F))
    fp = np.zeros((1, F))
    m = np.zeros((1, F))
    g = np.zeros((1, F))


    d = np.zeros((F, Ngt))
    distances = np.inf * np.ones((F, Ngt))

    for t in range(1, F + 1):
        GTsInFrames = np.where(gt[:, 0] == t - 1)
        DetsInFrames = np.where(det[:, 0] == t - 1)
        GTsInFrame = GTsInFrames[0]
        DetsInFrame = DetsInFrames[0]
        GTsInFrame = np.reshape(GTsInFrame,(1, GTsInFrame.shape[0]))
        DetsInFrame = np.reshape(DetsInFrame,(1, DetsInFrame.shape[0]))

        Ngtt = GTsInFrame.shape[1]
        Nt = DetsInFrame.shape[1]
        g[0,t - 1] = Ngtt

        if GTsInFrame is not None and DetsInFrame is not None:
            dist = np.inf * np.ones((Ngtt, Nt))
            for o in range(1, Ngtt + 1):
                GT = gt[GTsInFrame[0][o - 1]][2:4]
                for e in range(1, Nt + 1):
                    E = det[DetsInFrame[0][e - 1]][2:4]
                    dist[o-1, e-1] = getDistance(GT[0], GT[1], E[0], E[1])
            tmpai=dist
            tmpai = np.array(tmpai)

            # Please notice that the price/distance of are set to 100000 instead of np.inf, since the Hungarian Algorithm implemented in
            # sklearn will suffer from long calculation time if we use np.inf.
            tmpai[tmpai>td] = 100000
            if not tmpai.all() == 100000:
                HUN_res = linear_assignment(tmpai)
                tmp_zeros = np.zeros(tmpai.shape)
                for entry in HUN_res:
                    if entry[0] < tmp_zeros.shape[0] and entry[1] < tmp_zeros.shape[1]:
                        tmp_zeros[entry[0], entry[1]] = 1
                matches = tmp_zeros
                u,v = np.where(matches > 0)
                for mmm in range(1, len(u) + 1):
                    M[t - 1, u[mmm - 1]] = v[mmm - 1]
        curdetected = np.where(M[t - 1, :])
        curdetected = curdetected[0]

        c[0][t - 1] = curdetected.shape[0]
        for ct in curdetected:
            eid = M[t-1, ct]
            gtX = gt[GTsInFrame[0][ct], 2]

            gtY = gt[GTsInFrame[0][ct], 3]

            stX = det[DetsInFrame[0][int(eid)], 2]
            stY = det[DetsInFrame[0][int(eid)], 3]

            distances[t - 1, ct] = getDistance(gtX, gtY, stX, stY);
        fp[0][t-1] = Nt - c[0][t-1]
        m[0][t-1] = g[0][t-1] - c[0][t-1]

    MODP = sum(1 - distances[distances < td] / td) / np.sum(c) * 100 if sum(1 - distances[distances < td] / td) / np.sum(c) * 100 > 0 else 0
    MODA = (1 - ((np.sum(m) + np.sum(fp)) / np.sum(g))) * 100 if (1 - ((np.sum(m) + np.sum(fp)) / np.sum(g))) * 100 > 0 else 0
    recall = np.sum(c) / np.sum(g) * 100 if np.sum(c) / np.sum(g) * 100 > 0 else 0
    precision = np.sum(c) / (np.sum(fp) + np.sum(c)) * 100 if np.sum(c) / (np.sum(fp) + np.sum(c)) * 100 > 0 else 0

    return MODP, MODA, recall, precision