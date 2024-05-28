"""
    Version: 1.0 (2024-5)

    Written by: Wenjun Luo (Luowenjunn@outlook.com)
"""

import torch
import numpy as np
from .util import *
import hnswlib
from types import SimpleNamespace


def get_sparse_sum(X, axis=0, default=1):
    X_sum = X.sum(axis)
    s = default*torch.ones(X_sum.shape, dtype=X.dtype)
    s[X_sum.indices()[0]] = X_sum.values()
    return s


def sparse_eye(n, dtype=torch.float64):
    return torch.sparse_coo_tensor(torch.arange(n).repeat(2,1), torch.ones(n, dtype=dtype))

def drop_zero(X):
    X = X.coalesce()
    indices, values = X.indices(), X.values()
    nzind = values != 0
    return torch.sparse_coo_tensor(
        indices=indices[:, nzind],
        values=values[nzind],
        size = X.shape,
    )


def knn(X, U, **kwargs):
    pnn = kwargs['p'] if 'p' in kwargs else 5
    kernel = kwargs['knn_kernel'] if 'knn_kernel' in kwargs else 'torch'
    if 'get_distance' in kwargs and kwargs['get_distance']:
        get_distance = True
        kernel = 'torch'
    else:
        get_distance = False
    match kernel:
        case 'hnswlib':
            num_elements, dim = X.shape
            p = hnswlib.Index(space='l2', dim=dim)
            p.init_index(max_elements=num_elements, ef_construction=100, M=1000)
            p.set_ef(1000)
            p.add_items(U)
            ind, v = p.knn_query(X.float().numpy(), k=pnn)
            ind = torch.from_numpy(ind).long()
            v = torch.from_numpy(v).double()
        case 'torch':
            dist = torch.cdist(X, U)
            v, ind = torch.topk(dist, pnn, largest=False)
    if get_distance:
        return ind, v, dist
    return ind, v

def normalize(S, axis = 1, **kwargs):
    # symmetric normalization
    normalize = kwargs['normalize'] if 'normalize' in kwargs else 'symmetric'
    match normalize:
        case 'symmetric' | 'sym':
            if S.is_sparse:
                gd = get_sparse_sum(S, axis=axis)**(-.5)
            else: 
                gd = S.sum(axis)**(-.5)
            S = gd.view(-1,1)*S*gd
        case 'left':
            if S.is_sparse:
                gd = get_sparse_sum(S, axis=axis)**(-1)
            else:
                gd = S.sum(axis)**(-1)
            S = gd.view(-1,1)*S
        case 'right':
            if S.is_sparse:
                gd = get_sparse_sum(S, axis=axis)**(-1)
            else:
                gd = S.sum(axis)**(-1)
            S = S*gd
    return S 

def laplacian(S, **kwargs):
    options = SimpleNamespace(**kwargs)
    if hasattr(options, 'normalize') and options.normalize:
        if S.is_sparse:
            D = sparse_eye(S.shape[0])
        else:
            D = torch.eye(S.shape[0], dtype=S.dtype)
    else:
        if S.is_sparse:
            D = get_sparse_sum(S, axis=1)
        else:
            D = S.sum(1)
    L = D-S
    return S, D, L


def generate_graph(X, U=None, **kwargs):
    options = SimpleNamespace(**kwargs)
    options.p = options.p if hasattr(options, 'p') else 5
    options.self_connect = options.self_connect if hasattr(options, 'self_connect') else True if U is not None else False
    options.symmetric = options.symmetric if hasattr(options, 'symmetric') else False if U is not None else 'max'
    options.normalize = options.normalize if hasattr(options, 'normalize') else 'left' if U is not None else 'symmetric'
    options.laplace = options.laplace if hasattr(options, 'laplace') else False

    if U is None and options.self_connect:
        options.p += 1
    
    options.graph_type = options.graph_type if hasattr(options, 'graph_type') else 'Cai'
    
    match options.graph_type.lower():
        case 'hyper' | 'hypergraph':
            S = HyperGraph(X, U, **vars(options))
        case 'nie':
            S = Nie_graph(X, U, **vars(options))
        case _:
            raise('Graph type does not exist!')
    
    if hasattr(options, 'symmetric') and options.symmetric:
        match options.symmetric.lower():
            case('max'):
                # max(W, W.T)
                cha = (S.T-S).coalesce()
                S = S+(torch.abs(cha)+cha)/2
            case('sum'):
                S = (S+S.T)/2

    if hasattr(options, 'normalize') and options.normalize:
        S = normalize(S, **vars(options))

    S = drop_zero(S)
    if hasattr(options, 'laplace') and options.laplace:
        S, D, L = laplacian(S, **vars(options))
        return S.coalesce(), D.coalesce(), L.coalesce()
    else:
        return S.coalesce()


def HyperGraph(X, U=None, **kwargs):
    '''
    
    '''
    options = SimpleNamespace(**kwargs)
    options.p = options.p if hasattr(options, 'p') else 5

    nSmp = X.shape[0]
    if U is not None:
        mArch = U.shape[0]
        options.self_connect = True
    else:
        mArch = nSmp
        U = X
    self_connect = options.self_connect if hasattr(options, 'self_connect') else False
    sigma_mode = options.sigma_mode if hasattr(options, 'sigma_mode') else 2
    temp = 0 if self_connect else 1
    options.p += temp
    
    if hasattr(options, 'sigma'):
        ind, value = knn(X, U, **vars(options))
        sigma = options.sigma
    elif sigma_mode == 1:
        ind, value = knn(X, U, **vars(options))
        sigma = value[:, 1:].mean()
    else:
        options.get_distance = True
        ind, value, dist = knn(X, U, **vars(options))
        sigma = torch.sqrt(dist).mean()
    options.p -= temp

    if not self_connect:
        ind, value = ind[:, 1:], value[:, 1:]
    
    A = torch.exp(-(dist/sigma)**2)

    H = torch.sparse_coo_tensor(
        torch.vstack((torch.arange(ind.shape[0]).repeat(options.p,1).T.reshape(-1), ind.view(-1))),
        torch.ones(ind.shape[0]*(options.p)),
        (nSmp, mArch),
    )

    if U is None:
        W = get_sparse_sum(H.T*A, axis=0)**(1/2)
        invDe = 1/get_sparse_sum(H, 1)**(1/2)
        S = H.T*(W*invDe)
    else:
        W = get_sparse_sum(H*A, axis=0)**(1/2)
        invDe = 1/get_sparse_sum(H, 0)**(1/2)
        S = H*(W*invDe)
    if (hasattr(options, 'half') and options.half) or U is not None:
        return S
    S = S@S.T
    # S=(S+S.T)/2
    return S
    
        



def Nie_graph(X, U=None, **kwargs):
    '''
    X: data
    U: anchor if possible
    kwargs:
        p: p nearest neighbors
        self_connect: self connect or not
    '''
    options = SimpleNamespace(**kwargs)
    options.p = options.p if hasattr(options, 'p') else 5

    nSmp = X.shape[0]
    if U is not None:
        mArch = U.shape[0]
        options.self_connect = True
    else:
        mArch = nSmp
        U = X
    self_connect = options.self_connect if hasattr(options, 'self_connect') else False
    temp = 1 if self_connect else 2
    options.p += temp
    ind, v = knn(X, U, **vars(options))
    ind, v = (ind[:, :-1], v) if self_connect else (ind[:, 1:-1], v[:, 1:])
    v[:, :-1] = (v[:, :-1]-v[:, -1].view(-1,1))
    v[:, :-1] /= v[:, :-1].sum(1, keepdim=True)
    v[v.isnan()] = 1/(options.p-temp)

    ind = torch.vstack((torch.arange(nSmp).repeat(1, options.p-temp), ind.T.reshape(-1)))
    return torch.sparse_coo_tensor(ind, v[:,:-1].T.reshape(-1), (nSmp, mArch))

