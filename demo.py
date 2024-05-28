import numpy as np
import torch
from scipy import io
from itertools import product
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import warnings
warnings.filterwarnings('ignore')

from Utils.util import *
from Utils.Graphs import *

from Methods.HCSSMF_conj_fast import HCSSMF_conj


def run(DataName, rate, **kwargs):
    settings = SimpleNamespace(**kwargs)

    rowData = io.loadmat('mat_data/'+DataName+'.mat')
    X_raw = torch.as_tensor(rowData['fea'].astype(np.float64))
    y_raw = torch.as_tensor(rowData['gnd'].astype(np.int64)-1).view(-1)

    # if X.min() >= 0:
    #     X /=torch.maximum(torch.norm(X, dim=1), torch.tensor(1e-12))[:,None]
    
    Mtime = settings.Mtime if hasattr(settings, 'Mtime') else 1
    CG_method = settings.CG_method if hasattr(settings, 'CG_method') else 'SDMFRPRP'
    random_init = settings.random_init if hasattr(settings, 'random_init') else False


    ALLACC, ALLNMI, ALLARI, ALLFSC, ALLTime, ALLGraphTIME = [], [], [], [], [], []
    for IterTime in range(Mtime):
        random_state = np.random.RandomState(settings.random_seed) if hasattr(settings, 'random_seed') else None
        X, y, Y, _, UnLabelInd = gen_data(X_raw, y_raw, rate, random_state = random_state)


        H, time, graphtime = HCSSMF_conj(X.T, Y, UnLabelInd, CG_method = CG_method, random_init = random_init)

        Label = H.argmax(0).numpy()
        res = Label[UnLabelInd]
        testlabel = np.asarray(y[UnLabelInd])
        ACC, NMI, ARI, FSC = eval_metrics(testlabel, res, ACC=ALLACC, NMI=ALLNMI, ARI=ALLARI, FSC=ALLFSC)
        print('Mtime%2d'%(IterTime+1),'ACC=%.4f'%ACC, 'NMI=%.4f'%NMI, 'ARI=%.4f'%ARI, 'FSC=%.4f'%FSC)
        ALLTime += [time]
        ALLGraphTIME += [graphtime]

    AvgACC = np.mean(ALLACC)
    stdACC = np.std(ALLACC)
    AvgNMI = np.mean(ALLNMI)
    stdNMI = np.std(ALLNMI)
    AvgARI = np.mean(ALLARI)
    stdARI = np.std(ALLARI)
    AvgFSC = np.mean(ALLFSC)
    stdFSC = np.std(ALLFSC)
    AvgTime = np.mean(ALLTime)
    stdTime = np.std(ALLTime)
    AvgGraph = np.mean(ALLGraphTIME)
    stdGraph = np.std(ALLGraphTIME)
    print('ACC: %05.2f±%05.2f'%(AvgACC*100, stdACC*100))
    print('NMI: %05.2f±%05.2f'%(AvgNMI*100, stdNMI*100))
    print('ARI: %05.2f±%05.2f'%(AvgARI*100, stdARI*100))
    print('FSC: %05.2f±%05.2f'%(AvgFSC*100, stdFSC*100))
    print('Time: %05.4f±%05.4f'%(AvgTime, stdTime))
    print('GraphTime: %05.4f±%05.4f'%(AvgGraph, stdGraph))

                    


if __name__ == '__main__':
    ALLGraphTIME = []
    ALLNAME = []
    methods = [
        'HCSSMF', 
        ]
    datasets =[
        ['COIL20', 20],
        ['COIL100_obj',  100],
    ]

    for (method, dataset) in product(methods, datasets):
        name = dataset[1]
        DataName = dataset[0]
        
        run(DataName, 0.3, Mtime=10)



