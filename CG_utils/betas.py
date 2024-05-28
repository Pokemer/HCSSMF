"""
    Version: 1.0 (2024-5)

    Written by: Wenjun Luo (Luowenjunn@outlook.com)
"""


import numpy as np
import torch
from .CG import HZ_tol, HZ_return
from functools import partial

def kuangjia(state, restart, theta, beta, tri = False):
    state['gkg'] = torch.dot(state['gtemp'], state['g'])
    if restart and torch.abs(state['gkg'])>=0.1*state['gnorm2']:
        state['Restart'] = False
        state['IterRestart'] = 0
        state['IterQuad'] = 0

        state['x'] = state['xtemp'].clone()

        state['g'] = state['gtemp'].clone()
        state['d'] = -state['g']
        state['gnorm'] = torch.abs(state['g']).max()
        state['gnorm2'] = torch.dot(state['g'], state['g'])

        if HZ_tol(state['gnorm'], state):
            status = 0
            return HZ_return(status, state['iteration'], state)
        state['dphi0'] = -state['gnorm2']
        state['dnorm2'] = state['gnorm2']
        state['beta'] = 0
    elif tri:
        state['x'] = state['xtemp'].clone()
        beta(state)
        state['d'] = -state['gtemp'] + state['beta'] * state['d'] + state['tri_t'] * (state['gtemp'] - state['g'])
        state['g'] = state['gtemp'].clone()
    else:
        state['x'] = state['xtemp'].clone()
        if theta:
            beta(state)
            state['g'] = state['gtemp'].clone()
            state['d'] = -(1 +  state['alpha'] * state['beta']*state['df']/state['gnorm2'])*state['g'] + state['beta'] * state['d'] * state['alpha']
        else:
            beta(state)
            state['g'] = state['gtemp'].clone()
            state['d'] = -state['g'] + state['beta'] * state['d']




def SDMDYHS(state):
    dkyk = state['dphi'] - state['dphi0']
    gkyk = state['gnorm2'] - state['gkg']
    if (gkyk > 0) and (dkyk > 0):
        if state['gkg'] < 0:
            state['beta'] = state['gnorm2']/dkyk/ state['alpha']
            return
        n = state['n']
        sy = state['alpha'] * dkyk
        sg = state['alpha'] * state['dphi0']
        sgn = state['alpha'] * state['dphi']
        yy = state['gnorm2'] - 2*state['gkg'] + state['gnorm2old']
        ss = state['alpha'] * state['alpha'] * state['dnorm2']
        gg = state['gnorm2old']
        gy = state['gkg']-state['gnorm2old']

        gama = min(sy/(yy+sgn),1)
        delta = (n-gama*yy/sy)/(n-1)
        O = dkyk*state['alpha']
        la = (O*sg)/(ss*gg) - gy/gg+ (1/delta)*(O*gy)/(sy*gg) -(1/gama + (1/delta)*yy/sy)*(O*sg)/(sy*gg)

        if torch.isfinite(la):
            if la < 0:
                state['beta'] = state['gnorm2']/dkyk / state['alpha']
                return 
            elif la > 1:
                state['beta'] = gkyk/dkyk / state['alpha']
                return 
            else:
                state['beta'] = (state['gnorm2']-la*state['gkg'])/dkyk / state['alpha']
                return
        else:
            state['beta'] = gkyk/dkyk / state['alpha']
            return
    else:
        state['beta'] = 0
        state['Restart'] = False
        state['IterRestart'] = 0
        state['IterQuad'] = 0
        return

def SDMDYHS2(state):
    dkyk = state['dphi'] - state['dphi0']
    gkyk = state['gnorm2'] - state['gkg']
    if (gkyk > 0) and (dkyk > 0):
        if state['gkg'] < 0:
            state['beta'] = state['gnorm2']/dkyk/ state['alpha']
            return
        n = state['n']
        sy = state['alpha'] * dkyk
        sg = state['alpha'] * state['dphi0']
        sgn = state['alpha'] * state['dphi']
        yy = state['gnorm2'] - 2*state['gkg'] + state['gnorm2old']
        ss = state['alpha'] * state['alpha'] * state['dnorm2']
        gg = state['gnorm2old']
        gy = state['gkg']-state['gnorm2old']

        gama = min(sy/(yy+sgn),1)
        delta = (n-gama*yy/sy)/(n-1)
        O = dkyk*state['alpha']
        la = (O*sg)/(ss*gg) - gy/gg+ (1/delta)*(O*gy)/(sy*gg) -(1/gama + (1/delta)*yy/sy)*(O*sg)/(sy*gg)
        la = -la

        if torch.isfinite(la):
            if la < 0:
                state['beta'] = state['gnorm2']/dkyk / state['alpha']
                return 
            elif la > 1:
                state['beta'] = gkyk/dkyk / state['alpha']
                return 
            else:
                state['beta'] = (state['gnorm2']-la*state['gkg'])/dkyk / state['alpha']
                return
        else:
            state['beta'] = gkyk/dkyk / state['alpha']
            return
    else:
        state['beta'] = 0
        state['Restart'] = False
        state['IterRestart'] = 0
        state['IterQuad'] = 0
        return

def MCDLS(state):
    dkgk = state['dphi0']
    gkyk = state['gnorm2'] - state['gkg']
    if -dkgk >0 and gkyk > 0:
        if state['gkg'] < 0:
            state['beta'] = state['gnorm2'] / (-dkgk)/ state['alpha']
            return 
        n = state['n']
        dkyk = state['dphi'] - state['dphi0']
        sy = state['alpha'] * dkyk
        sg = state['alpha'] * state['dphi0']
        sgn = state['alpha'] * state['dphi']
        yy = state['gnorm2'] - 2*state['gkg'] + state['gnorm2old']
        ss = state['alpha'] * state['alpha'] * state['dnorm2']
        gg = state['gnorm2old']
        gy = state['gkg']-state['gnorm2old']
        
        gama = min(sy/(yy+sgn),1)
        delta = (n-gama*yy/sy)/(n-1)
        O = -dkgk*state['alpha']
        la = (O*sg)/(ss*gg) - gy/gg+ (1/delta)*(O*gy)/(sy*gg) -(1/gama + (1/delta)*yy/sy)*(O*sg)/(sy*gg)

        if torch.isfinite(la):
            if la < 0:
                state['beta'] = state['gnorm2']/(-dkgk) / state['alpha']
                return
            elif la > 1:
                state['beta'] = gkyk / (-dkgk) / state['alpha']
                return
            else:
                state['beta'] = (state['gnorm2'] - la*state['gkg'])/(-dkgk) / state['alpha']
                return
        else:
            state['beta'] = gkyk/(-dkgk) / state['alpha']
            return
    else:
        state['beta'] = 0
        state['Restart'] = False
        state['IterRestart'] = 0
        state['IterQuad'] = 0
        return

def MFRPRP(state):
    gkyk = state['gnorm2'] - state['gkg']
    if gkyk > 0:
        if state['gkg'] < 0:
            state['beta'] = state['gnorm2'] / state['gnorm2old'] / state['alpha']
            return 
        n = state['n']
        dkyk = state['dphi'] - state['dphi0']
        sy = state['alpha'] * dkyk
        sg = state['alpha'] * state['dphi0']
        sgn = state['alpha'] * state['dphi']
        yy = state['gnorm2'] - 2*state['gkg'] + state['gnorm2old']
        ss = state['alpha'] * state['alpha'] * state['dnorm2']
        gg = state['gnorm2old']
        gy = state['gkg']-state['gnorm2old']
        
        gama = min(sy/(yy+sgn),1)
        delta = (n-gama*yy/sy)/(n-1)
        O = state['alpha']*gg
        la = (O*sg)/(ss*gg) - gy/gg+ (1/delta)*(O*gy)/(sy*gg) -(1/gama + (1/delta)*yy/sy)*(O*sg)/(sy*gg)

        if torch.isfinite(la):
            if la < 0:
                state['beta'] = state['gnorm2']/state['gnorm2old']/ state['alpha']
                return
            elif la > 1:
                state['beta'] = gkyk / state['gnorm2old']/ state['alpha']
                return
            else:
                state['beta'] = (state['gnorm2'] - la*state['gkg'])/state['gnorm2old']/ state['alpha']
                return
        else:
            state['beta'] = gkyk/state['gnorm2old']/ state['alpha']
            return
    else:
        state['beta'] = 0
        state['Restart'] = False
        state['IterRestart'] = 0
        state['IterQuad'] = 0
        return

def CSDCG(state, O):
    gkyk = state['gnorm2'] - state['gkg']
    if gkyk > 0 :
        if state['gkg'] < 0:
            state['beta'] = state['gnorm2'] / O
            return 
        n = state['n']
        dkyk = state['dphi'] - state['dphi0']
        sy = state['alpha'] * dkyk
        sg = state['alpha'] * state['dphi0']
        sgn = state['alpha'] * state['dphi']
        yy = state['gnorm2'] - 2*state['gkg'] + state['gnorm2old']
        ss = state['alpha'] * state['alpha'] * state['dnorm2']
        gg = state['gnorm2old']
        gy = state['gkg']-state['gnorm2old']

        gama = min(sy/(yy+sgn),1)
        delta = (n-gama*yy/sy)/(n-1)
        la = (O*sg)/(ss*gg) - gy/gg+ (1/delta)*(O*gy)/(sy*gg) -(1/gama + (1/delta)*yy/sy)*(O*sg)/(sy*gg)

        if torch.isfinite(la):
            if la < 0:
                state['beta'] = state['gnorm2'] / O
                return
            elif la > 1:
                state['beta'] = gkyk / O
                return
            else:
                state['beta'] = (state['gnorm2'] - la*state['gkg']) / O
                return
        else:
            state['beta'] = gkyk / O
            return
    else:
        state['beta'] = 0
        state['Restart'] = False
        state['IterRestart'] = 0
        state['IterQuad'] = 0
        return




def GD(state):
    state['beta'] = 0
    return



    




betas = {
    'SDMDYHS' : partial(kuangjia, beta = SDMDYHS, restart = True, theta = True),
    'SDMDYHS2' : partial(kuangjia, beta = SDMDYHS2, restart = True, theta = True),
    'SDMCDLS' : partial(kuangjia, beta = MCDLS, restart = True, theta = True),
    'SDMFRPRP' : partial(kuangjia, beta = MFRPRP, restart = True, theta = True),
    'GD' : partial(kuangjia, beta = GD, restart = False, theta = False),
}