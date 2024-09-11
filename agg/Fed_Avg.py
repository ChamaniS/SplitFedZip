#federated averaging
#Averaging local SGD updates for the primal problem

import copy
import torch


def fedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
    return w_avg

