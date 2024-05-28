"""
    Version: 1.0 (2024-5)

    Written by: Wenjun Luo (Luowenjunn@outlook.com)
"""

import torch
import numpy as np
from types import SimpleNamespace
from sklearn import metrics




def semi_supervised(y, rate, random_state, type = 1):
    nSmp = y.shape[0]
    Clabel = torch.unique(y)
    Cnum = len(Clabel)

    kk = []
    LabelInd = []
    rangeddd = np.arange(nSmp)
    for clas in Clabel:
        ind = y == clas
        number = (ind).sum()
        if rate < 1:
            nn = int(number*rate)
            if nn == 0:
                nn = 1
            LabelInd += [random_state.choice(rangeddd[ind], nn, replace = False)]
            kk += [int(clas)]*nn
        else:
            LabelInd += [random_state.choice(rangeddd[ind], rate, replace = False)]
            kk += [int(clas)]*rate
    LabelInd = np.hstack(LabelInd)
    UnLabelInd = np.setdiff1d(np.arange(nSmp), LabelInd)

    nl = len(LabelInd)
    if type == 1:
        ind = torch.vstack([torch.as_tensor(kk), torch.as_tensor(LabelInd)])
        Y = torch.sparse_coo_tensor(ind, torch.ones(nl, dtype=torch.float64), (Cnum, nSmp))
    elif type == 2:
        
        ind = torch.vstack([torch.as_tensor(kk), torch.arange(nl)])
        Y = torch.sparse_coo_tensor(ind, torch.ones(len(LabelInd), dtype=torch.float64), (Cnum, nSmp))
    return Y.coalesce(), LabelInd, UnLabelInd

def gen_data(X_raw, y_raw, rate, SampleInd = None, shuffle = True, random_state = None):
    if random_state is None:
        random_state = np.random.RandomState()
    nSmp = y_raw.shape[0]
    if SampleInd is None:
        SampleInd = np.arange(nSmp)
    if shuffle:
        random_state.shuffle(SampleInd)

    X = X_raw[SampleInd]
    y = y_raw[SampleInd]
    Y, LabelInd, UnLabelInd = semi_supervised(y, rate, random_state, type=1)

    return X, y, Y, LabelInd, UnLabelInd



def eval_metrics(testlabel, res, **Parm):
    metric = SimpleNamespace(**Parm)
    return_parm = []
    if hasattr(metric, 'ACC'):
        ACC = (testlabel==res).sum()/len(testlabel)
        return_parm += [ACC]
        metric.ACC += [ACC]
    if hasattr(metric, 'NMI'):
        NMI = metrics.normalized_mutual_info_score(testlabel, res)
        return_parm += [NMI]
        metric.NMI += [NMI]
    if hasattr(metric, 'ARI'):
        ARI = metrics.adjusted_rand_score(testlabel, res)
        return_parm += [ARI]
        metric.ARI += [ARI]
    if hasattr(metric, 'FSC'):
        FSC = metrics.f1_score(testlabel, res, average='macro')
        return_parm += [FSC]
        metric.FSC += [FSC]
    return return_parm
