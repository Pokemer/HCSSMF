"""
    Version: 1.0 (2024-5)

    Written by: Wenjun Luo (Luowenjunn@outlook.com)
"""


import numpy as np
import torch
from Utils.Graphs import generate_graph
from CG_utils.betas import betas
from CG_utils.CG import CG
import time as t

class P():
    def __init__(self, Y, S, UnLabelInd, alpha, random_init = False) -> None:
        

        self.c, n = Y.shape
        self.C = Y.sum(1).values()
        self.C = torch.sparse_coo_tensor(torch.arange(self.c).repeat(2,1), self.C)
        self.Ic = torch.sparse_coo_tensor(torch.arange(self.c).repeat(2,1), torch.ones(self.c, dtype=S.dtype))
        lun = UnLabelInd.shape[0]
        self.A2 = torch.sparse_coo_tensor(torch.vstack((torch.arange(lun), torch.tensor(UnLabelInd))),
                                     torch.ones(lun, dtype=S.dtype), (lun, n))
        self.AS = self.A2@S
        self.YS = Y@S
        self.alpha = alpha
        if random_init:
            print('Random init Z, and the first 5 elements are:', end = '')
            Z = 2*torch.rand((self.c, lun), dtype=S.dtype)-1
            print(Z[0,:5].numpy())
        else:
            Z = torch.zeros((self.c, lun), dtype=S.dtype)
        self.x0 = Z.reshape(-1)
        
    def obj(self, x, gradient = False):
        Z = x.reshape(self.c, -1)
        ZZ = Z@Z.T
        AS = self.AS
        ZAS = Z@AS
        objf = self.alpha*torch.trace(ZZ)-(2+self.alpha)*torch.square(ZAS+self.YS).sum()+torch.square(ZZ+self.C).sum()
        
        if gradient == False:
            return objf
        grad = (4*ZZ+2*self.alpha*self.Ic+4*self.C)@Z - 2*(2+self.alpha)*(ZAS+self.YS)@AS.T
        grad = grad.reshape(-1)
        return objf, grad

def HCSSMF_conj(X, Y, UnLabelInd, *, p=5, alpha=100, CG_method = 'SDMFRPRP', random_init = False):
    Y = torch.tensor(Y).to_sparse()

    start = t.time()
    Sh = generate_graph(X.T, p=p, graph_type='nie', symmetric = False, self_connect=True, normalize = 'left', half=True)
    graphtime = t.time()-start
    p = P(Y, Sh, UnLabelInd, alpha, random_init=random_init)
    cg = CG(p)
    state = cg.step(betas[CG_method])
    Z = state['x'].reshape(p.c, -1)
    time = state['time']
    H = Z@p.A2+Y
    return H, time, graphtime
    