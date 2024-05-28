"""
    Version: 1.0 (2024-5)

    Written by: Wenjun Luo (Luowenjunn@outlook.com)
"""

import numpy as np
from functools import reduce
from collections import defaultdict
from functools import partial
import time
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
# import tensorflow as tf

INT_INF = 9223372036854775807



def HZ_tol(gnorm, Com):
    if Com['Parm']['StopRule']:
        if gnorm <= Com['tol']:
            return 1
    elif gnorm <= Com['tol']*(1 + abs(Com['f'])):
        return 1
    return 0
    
def cg_tol(gnorm, Com):
    """
    check the algorithms' convergence
    """

    #check the stop rule
    if Com['Parm']['StopRule']:
        if gnorm < Com['tol']:
            return True
    elif gnorm <= Com['tol']*(1 + abs(Com['f'])):
        return True
    return False

def HZ_evaluate(what, nan, Com):
    
    Parm = Com['Parm']
    x = Com['x']
    d = Com['d']
    xtemp = Com['xtemp']
    gtemp = Com['gtemp']
    alpha = Com['alpha'].clone()

    # reduce setpsiz if function value is nan
    if nan == 'y' or nan == 'p':
        if what == 'f':
            xtemp = x+alpha*d
            Com['xtemp'] = xtemp
            Com['f'] = Com['fun'](xtemp)
            Com['nf']+=1
            if not torch.isfinite(Com['f']):
                flag = True
                for i in range(Parm['ntries']):
                    if nan == 'p':
                        alpha = Com['alphaold'] + .8*(alpha - Com['alphaold'])
                    else:
                        alpha *= Parm['nan_decay']
                    xtemp = x + alpha * d
                    Com['xtemp'] = xtemp
                    Com['f'] = Com['fun'](xtemp)
                    Com['nf']+=1
                    if torch.isfinite(Com['f']):
                        flag = False
                        break
                if flag:
                    return 11
            Com['alpha'] = alpha
        elif what == 'g':
            xtemp = x+alpha*d
            Com['xtemp'] = xtemp
            gtemp = Com['grad'](xtemp)
            Com['gtemp'] = gtemp
            Com['ng']+=1
            Com['df'] = torch.dot(gtemp, d)
            if not torch.isfinite(Com['df']):
                flag = True
                for i in range(Parm['ntries']):
                    if nan == 'p':
                        alpha = Com['alphaold'] + .8*(alpha - Com['alphaold'])
                    else:
                        alpha *= Parm['nan_decay']
                    xtemp = x+alpha*d
                    Com['xtemp'] = xtemp
                    gtemp = Com['grad'](xtemp)
                    Com['gtemp'] = gtemp
                    Com['ng']+=1
                    Com['df'] = torch.dot(gtemp, d)
                    if torch.isfinite(Com['df']):
                        flag = False
                        break
                if flag:
                    return 11
                Com['rho'] = Parm['nan_rho']
            else:
                Com['rho'] = Parm['rho']
            Com['alpha'] = alpha
        else:
            xtemp = x+alpha*d
            Com['xtemp'] = xtemp
            Com['f'], gtemp = Com['fandgrad'](xtemp)
            Com['gtemp'] = gtemp
            Com['df'] = torch.dot(gtemp, d)
            Com['nf']+=1
            Com['ng']+=1

            if not torch.isfinite(Com['f']) or not torch.isfinite(Com['df']):
                flag = True
                for i in range(Parm['ntries']):
                    if nan == 'p':
                        alpha = Com['alphaold'] + .8*(alpha - Com['alphaold'])
                    else:
                        alpha *= Parm['nan_decay']
                    xtemp = x+alpha*d
                    Com['xtemp'] = xtemp
                    Com['f'], gtemp = Com['fandgrad'](xtemp)
                    Com['gtemp'] = gtemp
                    Com['df'] = torch.dot(gtemp, d)
                    Com['nf']+=1
                    Com['ng']+=1
                    if torch.isfinite(Com['f']) and torch.isfinite(Com['df']):
                        flag = False
                        break
                if flag:
                    return 11
                Com['rho'] = Parm['nan_rho']
            else:
                Com['rho'] = Parm['rho']
            Com['alpha'] = alpha
    else:
        if what == 'fg':
            xtemp = x+alpha*d
            Com['xtemp'] = xtemp
            Com['f'], gtemp = Com['fandgrad'](xtemp)
            Com['gtemp'] = gtemp
            Com['df'] = torch.dot(gtemp, d)
            Com['nf']+=1
            Com['ng']+=1
            if not torch.isfinite(Com['f']).all() or not torch.isfinite(Com['df']).all():
                return 11
        elif what == 'f':
            xtemp = x+alpha*d
            Com['xtemp'] = xtemp
            Com['f'] = Com['fun'](xtemp)
            Com['nf']+=1
            if not torch.isfinite(Com['f']) :
                return 11
        else:
            xtemp = x+alpha*d
            Com['xtemp'] = xtemp
            gtemp = Com['grad'](xtemp)
            Com['gtemp'] = gtemp
            Com['ng']+=1
            Com['df'] = torch.dot(gtemp, d)
            if not torch.isfinite(Com['df']):
                return 11
    return 0


    
            


def HZ_Wolfe(alpha, f, dphi, Com):
    """
    Check whether the Wolfe or the approximate Wolfe conditions are satisfied
    """
    if dphi >= Com['wolfe_lo']:

        #test original Wolfe conditions
        if f - Com['f0'] <= alpha * Com['wolfe_hi']:
            #Wolfe conditions hold
            return True

        #test approximate Wolfe conditions
        elif Com['AWolfe']:
            if ((f <= Com['fpert']) and (dphi <= Com['awolfe_hi'])):
                #Approximate Wolfe conditions hold
                return True
    
    return False



def HZ_cubic(a, fa, da, b, fb, db):
    """
    Compute the minimizer of a Hermite cubic. If the computed minimizer
    outside [a, b], return -1 (it is assumed that a >= 0).
    """

    delta = b - a
    if delta == 0:
        return a
    v = da + db - 3. * (fb - fa) / delta
    t = v*v - da*db
    if t < 0:
        if abs(da) < abs(db):
            c = a - (a-b)*(da/(da-db))
        elif (da != db):
            c = b - (a-b)*(db/(da-db))
        else:
            c = -1
        return c
    
    if delta > 0:
        w = torch.sqrt(t)
    else:
        w = -torch.sqrt(t)
    d1 = da + v - w
    d2 = db + v + w
    if (d1 == 0) and (d2 == 0):
        return -1.
    if abs(d1) >= abs(d2):
        c = a + delta*da/d1
    else:
        c = b - delta*db/d2
    return c
    

def HZ_contract(A, fA, dA, B, fB, dB, Com):
    AWolfe = Com['AWolfe']
    Parm = Com['Parm']

    a = A
    fa = fA
    da = dA
    b = B
    fb = fB
    db = dB
    f1 = fb
    d1 = db
    toggle = 0
    width = 0
    # old = 0.
    # dold = 0.
    # fold = 0.

    for iter in range(Parm['nshrink']):
        if (toggle == 0) or ((toggle == 2) and ((b-a) <= width)):

            #cubic based on bracketing interval
            alpha = HZ_cubic(a, fa, da, b, fb, db)
            toggle = 0
            width = Parm['gamma'] * (b-a)
            if iter:
                Com['QuadOK'] = True
        elif toggle == 1:
            Com['QuadOK'] = True
            if old < a:
                alpha = HZ_cubic(a, fa, da, old, fold, dold)
            else:
                alpha = HZ_cubic(a, fa, da, b, fb, db)
        else:
            alpha = .5*(a+b)
            Com['QuadOK'] = False
        
        if (alpha <= a) or (alpha >=b):
            alpha = .5*(a+b)
            Com['QuadOK'] = False

        toggle+=1
        if toggle>2:
            toggle = 0
        
        Com['alpha'] = alpha
        # evaluate function and gradient
        status = HZ_evaluate ("fg", "n", Com)
        if status:
            return a, fa, da, b, fb, db, status
        

        f = Com['f'].clone()
        df = Com['df'].clone()

        if Com['QuadOK']:
            if HZ_Wolfe(alpha, f, df, Com):
                return a, fa, da, b, fb, db, 0
        if not AWolfe:
            f -= alpha*Com['wolfe_hi']
            df -= Com['wolfe_hi']
        if df >= 0:
            return a, fa, da, alpha, f, df, -2
        if f <= Com['fpert']:
            old = a
            a = alpha
            fold = fa
            fa = f
            dold = da
            da = df
        else:
            fold = fb
            dold = db

            old = b
            b = alpha
            fb = f
            db = df

    # see if the cost is small enough to change the PertRule  
    if abs(fb) <= Com['SmallCost']:
        Com['PertRul'] = False
    
    # increase eps if slope is negative after Parm->nshrink iterations
    t = Com['f0']
    if Com['PertRule']:
        if t != 0:
            Com['eps'] = Parm['egrow']*(f1-t) / abs(t)
            Com['fpert'] = t + abs(t) * Com['eps']
        else:
            Com['fpert'] = 2.*f1
    else:
        Com['eps'] = Parm['egrow']*(f1-t)
        Com['fpert'] = t + Com['eps']
    
    Com['neps'] += 1
    return a, fa, da, b, fb, db, -1



def HZ_Linesearch(Com):
    """
    Approximate Wolfe line search routine

    Return:
       -2 (function nan)
        0 (Wolfe or approximate Wolfe conditions satisfied)
        3 (slope always negative in line search)
        4 (number line search iterations exceed nline)
        6 (excessive updating of eps)
        7 (Wolfe conditions never satisfied)
    """

    AWolfe = Com['AWolfe']
    Parm = Com['Parm']


    if Com['QuadOK']:
        # evaluate function and gradient at alpha (initial guess)
        status = HZ_evaluate('fg', 'y', Com)
        fb = Com['f'].clone()
        if not AWolfe:
            fb -= Com['alpha'] * Com['wolfe_hi']
        qb = True
    else:
        # evaluate gradient
        status = HZ_evaluate ("g", "y", Com)
        qb = False

    if status:
        return status
    b = Com['alpha']
    
    if AWolfe:
        db = Com['df']
        d0 = da = Com['df0']
    else:
        db = Com['df'] - Com['wolfe_hi']
        d0 = da = Com['df0'] - Com['wolfe_hi']
    
    a = 0
    a1 = 0
    d1 = d0
    fa = Com['f0'].clone()
    # a0 = 0.
    # fa0 = 0.
    # da0 = 0.
    # b0 = 0.
    # fb0 = 0.
    # db0 = 0.
    
    # if a quadratic interpolation step performed, check Wolfe conditions
    if Com['QuadOK'] and (Com['f'] <= Com['f0']):
        if HZ_Wolfe(b, Com['f'], Com['df'], Com):
            return 0
    
    # if a Wolfe line search and the Wolfe conditions have not been satisfied
    if not AWolfe:
        Com['Wolfe'] = True
    
    # Find initial interval [a,b] such that
    # da <= 0, db >= 0, fa <= fpert = [(f0 + eps*fabs (f0)) or (f0 + eps)]
    rho = Com['rho']
    ngrow = 1
    while db < 0:
        if not qb:
            # evaluate function
            status = HZ_evaluate ("f", "n", Com)
            if status:
                return status
            if AWolfe:
                fb = Com['f']
            else:
                fb = Com['f'] - b * Com['wolfe_hi']
            qb = True
        if fb > Com['fpert']:
            a, fa, da, b, fb, db, status = HZ_contract(a, fa, da, b, fb, db, Com)
            if (status == 0):
                return 0
            if status == -2:
                break
            elif Com['neps'] > Parm['neps']:
                return 6
        

        # expansion phase
        ngrow+=1
        if ngrow > Parm['ntries']:
            return 3
        # update interval (a replaced by b)
        a = b
        fa = fb
        da = db
        # store old values of a and corresponding derivative
        d2 = d1
        d1 = da
        a2 = a1
        a1 = a

        bmin = rho*b
        if (ngrow == 2) or (ngrow == 3) or (ngrow == 6):
            if d1 > d2:
                if ngrow == 2:
                    b = a1 - (a1-a2)*(d1/(d1-d2))
                else:
                    if (d1-d2)/(a1-a2) >= (d2-d0)/a2:
                        b = a1 - (a1-a2)*(d1/(d1-d2))
                    else:
                        b = a1 - Parm['SecantAmp']*(a1-a2)*(d1/(d1-d2))
                # safeguard growth
                b = min(b, Parm['ExpandSafe']*a1)
            else:
                rho *= Parm['RhoGrow']
        else:
            rho *= Parm['RhoGrow']
        b = max(bmin, b)
        Com['alphaold'] = Com['alpha']
        Com['alpha'] = b
        #evaluate gradient
        status = HZ_evaluate ("g", "p", Com)
        if status:
            return status
        b = Com['alpha']
        qb = False
        if AWolfe:
            db = Com['df']
        else:
            db = Com['df'] - Com['wolfe_hi']
    



    # we now have fa <= fpert, da >= 0, db <= 0
    toggle = 0
    width = b - a
    qb0 = False
    for iteration in range(Parm['nline']):
        
        # determine the next iterate
        if (toggle == 0) or ((toggle == 2) and ((b-a) <= width)):
            Com['QuadOK'] = True
            if Com['UseCubic'] and qb:
                alpha = HZ_cubic(a, fa, da, b, fb, db)
                if alpha < 0:
                    if -da < db :
                        alpha = a - (a-b)*(da/(da-db))
                    elif da != db:
                        alpha = b - (a-b)*(db/(da-db))
                    else:
                        alpha = torch.tensor(-1.)
            else:
                if -da < db :
                    alpha = a - (a-b)*(da/(da-db)) 
                elif da != db:
                    alpha = b - (a-b)*(db/(da-db))
                else:
                    alpha = torch.tensor(-1.)
            width = Parm['gamma']*(b-a)
        elif toggle == 1:
            Com['QuadOK'] = True
            if Com['UseCubic']:
                if Com['alpha'] == a:
                    alpha = HZ_cubic(a0, fa0, da0, a, fa, da)
                elif qb0:
                    alpha = HZ_cubic(b, fb, db, b0, fb0, db0)
                else:
                    alpha = torch.tensor(-1)
                
                # if alpha no good, use cubic between a and b
                if alpha <= a or alpha >= b:
                    if qb:
                        alpha = HZ_cubic(a, fa, da, b, fb, db)
                    else:
                        alpha = torch.tensor(-1.)
                
                # if alpha still no good, use secant method
                if alpha < 0:
                    if (-da < db):
                        alpha = a - (a-b)*(da/(da-db))
                    elif da != db:
                        alpha = b - (a-b)*(db/(da-db))
                    else:
                        alpha = torch.tensor(-1.)
            else: #use secant
                if (Com['alpha'] == a) and (da > da0):
                    alpha = a - (a-a0)*(da/(da-da0))
                elif db < db0:
                    alpha = b - (b-b0)*(db/(db-db0))
                else:
                    if -da < db:
                        alpha = a - (a-b)*(da/(da-db))
                    elif da != db:
                        alpha = b - (a-b)*(db/(da-db))
                    else:
                        alpha = torch.tensor(-1.)
                
                if (alpha <= a) or (alpha >= b):
                    if -da < db:
                        alpha = a - (a-b)*(da/(da-db))
                    elif da != db:
                        alpha = b - (a-b)*(db/(da-db))
                    else:
                        alpha = torch.tensor(-1.)
        
        else:
            alpha = .5*(a+b)
            Com['QuadOK'] = False
        
        if (alpha<=a) or (alpha >=b):
            alpha = .5*(a+b)
            if (alpha == a) or (alpha == b):
                return 7
            Com['QuadOK'] = False
        
        if toggle == 0:
            a0 = a 
            b0 = b 
            da0 = da 
            db0 = db 
            fa0 = fa 
            if qb:
                fb0 = fb
                qb0 = True
        
        toggle+=1
        if toggle > 2:
            toggle = 0
        
        Com['alpha'] = alpha
        # evaluate function and gradient
        status = HZ_evaluate ("fg", "n", Com)
        if status:
            return status


        Com['alpha'] = alpha
        f = Com['f'].clone()
        df = Com['df'].clone()
        if Com['QuadOK']:
            if HZ_Wolfe(alpha, f, df, Com):
                return 0
            
        if not AWolfe:
            f -= alpha*Com['wolfe_hi']
            df -= Com['wolfe_hi']
        if df >= 0:
            b = alpha
            fb = f
            db = df
            qb = True
        elif f <= Com['fpert']:
            a = alpha
            da = df
            fa = f
        else:
            B = b
            if qb:
                fB = fb
            dB = db
            b = alpha
            fb = f
            db = df

            # contract interval [a, alpha] 
            a, fa, da, b, fb, db, status = HZ_contract(a, fa, da, b, fb, db, Com)
            if status == 0:
                return 0
            if status == -1: # eps reduced, use [a, b] = [alpha, b]
                if Com['neps'] > Parm['neps']:
                    return 6
                a = b
                fa = fb
                da = db
                b = B
                if qb:
                    fb = fB
                db = dB
            else:
                qb = True
    return 4



def HZ_default():
    return dict(
        AWolfe = False,
        # AWolfe = True, 
        AWolfeFac = 1.e-3,
        Qdecay = .7,
        nslow = 1000,
        StopRule = True,
        # StopRule = False,
        StopFac = 0.e-12,
        PertRule = True,
        eps = 1.e-6,
        egrow = 10.,
        QuadStep = True,
        QuadCutOff = 1.e-12,
        QuadSafe = 1.e-10,
        UseCubic = True,
        CubicCutOff = 1.e-12,
        SmallCost = 1.e-30,
        debug = False,
        debugtol = 1.e-10,
        step = 0,
        # step = 1,
        maxit = 10000,
        ntries = 50,
        ExpandSafe = 200.,
        SecantAmp = 1.05,
        RhoGrow = 2.0,
        neps = 5, 
        nshrink = 10,
        nline = 50,
        restart_fac = 6.0,
        feps = 0,
        nan_rho = 1.3,
        nan_decay = 0.1,
        delta = .1,
        sigma = .9,
        gamma = .66,
        rho = 5.,
        psi0 = .01,
        psi_lo = 0.1,
        psi_hi = 10.,
        psi1 = 1.0,
        psi2 = 2.,
        AdaptiveBeta = False,
        BetaLower = 0.4,
        theta = 1.0,
        qeps = 1.e-12,
        qrestart = 6,
        qrule = 1.e-8,
    )



def evalfun(x, p):
    return torch.as_tensor(p.obj(x))

def evalgradient(x, p):
    return p.obj(x, gradient = True)[1]

def evalfandg(x, p):
    a, b = p.obj(x, gradient = True)
    return torch.as_tensor(a), b


class CG():
    """Implements L-BFGS algorithm, heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """
        
    def __init__(self,
                p,
                lr=1,
                max_iter=10000,
                max_eval=None,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                # history_size=3,
                line_search_fn="strongbacktracing"):
        
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        
        defaults = dict(
            lr=lr,
            max_iter = max_iter,
            max_eval = max_eval,
            tolerance_grad = tolerance_grad,
            tolerance_change = tolerance_change,
            # history_size = history_size,
            line_search_fn = line_search_fn,
        )

        self.state = defaultdict(dict)
        self.param_groups = [defaults]

        self.x = p.x0.clone()
        self.cutest = p
        
        
        
    def evalfunc(self, x):
        return self.cutest.obj(x)

    def evalgrad(self, x):
        return self.cutest.obj(x, gradient=True)[1]
    
    def evalfandg(self, x):
        return self.cutest.obj(x, gradient=True)


    def step(self, direction, grad_tol=1.e-6, Parm = None, update_dphi0 = False):
        
        if Parm is None:
            Parm = HZ_default()
        
        
        state = {}
        
        state['objhist'] = []
        state['NACGM_e'] = 1

        # state['fun'] = self.cutest.obj
        state['fun'] = partial(evalfun, p = self.cutest)
        state['grad'] = partial(evalgradient, p = self.cutest)
        state['fandgrad'] = partial(evalfandg, p = self.cutest)
        # state['fandgrad'] = partial(self.cutest.obj, gradient = True)


        n = self.x.shape[0]
        qrestart = min(n, Parm['qrestart'])
        
        state['Parm'] = Parm
        state['eps'] = Parm['eps']
        state['PertRule'] = Parm['PertRule']
        state['Wolfe'] = False
        state['nf'] = 0
        state['ng'] = 0
        iter = 0
        QuadF = False

        state['xtemp'] = torch.zeros(self.x.shape)
        state['gtemp'] = torch.zeros(self.x.shape)
        state['d'] = torch.zeros(self.x.shape)
        state['x'] = self.x
        state['n'] = n
        state['neps'] = 0
        state['AWolfe'] = Parm['AWolfe']

        # StopRule = Parm['StopRule']

        nrestart = int(n*Parm['restart_fac'])

        maxit = Parm['maxit']
        f = torch.tensor(0)
        fbest = np.inf
        gbest = np.inf
        nslow = 0
        slowlimit = 2*n + Parm['nslow']

        Ck = 0
        Qk = 0

        state['alpha'] = torch.tensor(0)
        state['starttime'] = time.time()
        status = HZ_evaluate ("fg", "n", state)
        f = state['f'].clone()
        state['objhist']+=[f]
        if status:
            return HZ_return(status, 0, state)
        state['g'] = state['gtemp'].clone()
        state['f0'] = f + f
        state['SmallCost'] = abs(f) * Parm['SmallCost']
        xnorm = torch.abs(state['x']).max()
        state['d'] = -state['g'].clone()
        state['gnorm'] = torch.abs(state['g']).max()
        state['gnorm2'] = torch.dot(state['g'], state['g'])
        state['dnorm2'] = state['gnorm2'].clone()

        if Parm['StopRule']:
            tol = max(state['gnorm']*Parm['StopFac'], grad_tol)
        else:
            tol = grad_tol
        state['tol'] = tol
        
        if HZ_tol(state['gnorm'], state):
            iteration = 0
            status = 0
            return HZ_return(status, iteration, state)
        
        state['dphi0'] = -state['gnorm2']
        delta2 = 2*Parm['delta'] - 1
        alpha = torch.tensor(Parm['step'])
        if alpha == 0:
            if xnorm == 0:
                if f != 0:
                    alpha = 2.*abs (f)/state['gnorm2']
                else:
                    alpha = torch.tensor(1.)
            else:
                alpha = Parm['psi0']*xnorm/state['gnorm']
        
        state['df0'] = -2.0*abs(f)/alpha

        state['Restart'] = False
        state['IterRestart'] = 0
        # IterSub = 0
        # NumSub = 0
        state['IterQuad'] = 0
        
        for iteration in range(1, maxit+1):
            # alphaold = alpha
            state['QuadOK'] = False
            alpha = Parm['psi2'] * alpha
            if f != 0:
                t = abs((f-state['f0'])/f)
            else:
                t = 1
            state['UseCubic'] = True
            if ((t < Parm['CubicCutOff'] or not Parm['UseCubic'])):
                state['UseCubic'] = False
            if Parm['QuadStep']:

                # test if quadratic interpolation step should be tried 
                if ((t > Parm['QuadCutOff']) and (abs(f) >= state['SmallCost'])) or QuadF:
                    if QuadF:
                        state['alpha'] = Parm['psi1'] * alpha
                        status = HZ_evaluate ("g", "y", state)
                        if status:
                            return HZ_return(status, iteration, state)
                        if state['df'] > state['dphi0']:
                            alpha = -state['dphi0']/((state['df']-state['dphi0'])/state['alpha'])
                            state['QuadOK'] = True
                    else:
                        t = max(Parm['psi_lo'], state['df0']/(state['dphi0']*Parm['psi2']))
                        state['alpha'] = min(t, Parm['psi_hi']) * alpha
                        status = HZ_evaluate ("f", "y", state)
                        if status:
                            return HZ_return(status, iteration, state)
                        ftemp = state['f']
                        denom = 2.*(((ftemp-f)/state['alpha'])-state['dphi0'])
                        if denom > 0:
                            t = torch.tensor(-float(state['dphi0'])*float(state['alpha'])/float(denom))

                            if ftemp >= f:
                                alpha = max(t, state['alpha']*Parm['QuadSafe'])
                            else:
                                alpha = t
                            state['QuadOK'] = True
            
            state['f0'] = f.clone()
            state['df0'] = state['dphi0'].clone()

            # parameters in Wolfe and approximate Wolfe conditions, and in update

            Qk = Parm['Qdecay'] * Qk + 1
            Ck = Ck + (abs (f) - Ck)/Qk

            if state['PertRule']:
                state['fpert'] = f + state['eps']*abs(f)
            else:
                state['fpert'] = f + state['eps']

            state['wolfe_hi'] = Parm['delta'] * state['dphi0']
            state['wolfe_lo'] = Parm['sigma'] * state['dphi0']
            state['awolfe_hi'] = delta2 * state['dphi0']
            state['alpha'] = alpha


            # perform line search
            status = HZ_Linesearch(state)
            # status = _strong_wolfe(state)

            if ((status > 0) and not state['AWolfe']):
                if status != 3:
                    state['AWolfe'] = True
                    status = HZ_Linesearch(state)
            

            alpha = state['alpha'].clone()
            f = state['f'].clone()
            state['dphi'] = state['df'].clone()

            if status:
                return HZ_return(status, iteration, state)
            
            # test for convergence to within machine epsilon
            # [set feps to zero to remove this test]
            if -alpha * state['dphi0'] <= Parm['feps'] * abs(f):
                status = 1
                return HZ_return(status, iteration, state)
            
            # test how close the cost function changes are to that of a quadratic
            # QuadTrust = 0 means the function change matches that of a quadratic
            t = alpha * (state['dphi']+state['dphi0'])
            if abs(t) <= Parm['qeps'] * min(Ck, 1):
                QuadTrust = 0
            else:
                QuadTrust = abs((2.0*(f-state['f0'])/t) - 1)
            if QuadTrust <= Parm['qrule']:
                state['IterQuad'] += 1
            else:
                state['IterQuad'] = 0

            if state['IterQuad'] == qrestart:
                QuadF = True
            state['IterRestart'] += 1
            if not state['AWolfe']:
                if abs(f-state['f0']) < Parm['AWolfeFac'] * Ck:
                    state['AWolfe'] = True
                    if state['Wolfe']:
                        state['Restart'] = True
            
            if (state['IterRestart'] >= nrestart) or ((state['IterQuad'] == qrestart) and (state['IterQuad'] != state['IterRestart'])):
                state['Restart'] = True
            
            state['iteration'] = iteration
            if state['Restart']:
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
                    return HZ_return(status, iteration, state)
                state['dphi0'] = -state['gnorm2']
                state['dnorm2'] = state['gnorm2']
                state['beta'] = 0
            
            else:
                state['gnorm2old'] = state['gnorm2']
                state['gnorm2'] = torch.dot(state['gtemp'], state['gtemp'])
                direction(state)
                state['gnorm'] = torch.abs(state['g']).max()
                
                state['dnorm2'] = torch.dot(state['d'], state['d'])
                if update_dphi0:
                    state['dphi0'] = -state['gnorm2'] + state['beta']*state['dphi']
                else:
                    state['dphi0'] = torch.dot(state['g'], state['d'])
                if HZ_tol(state['gnorm'], state):
                    status = 0
                    return HZ_return(status, iteration, state)
                
                if Parm['debug']:
                    t = torch.dot(state['d'], state['g'])
                    if abs(t-state['dphi0']) > Parm['debugtol']*abs(state['dphi0']):
                        print("Warning, dphi0 != d'g!\n")
                        print("dphi0:%13.6e, d'g:%13.6e\n" % (state['dphi0'], t))
            
            # test for slow convergence
            if (f < fbest) or (state['gnorm2'] < gbest):
                nslow = 0
                if f < fbest:
                    fbest = f
                if state['gnorm2'] < gbest:
                    gbest = state['gnorm2']
            else:
                nslow += 1
            
            if nslow > slowlimit:
                status = 9
                return HZ_return(status, iteration, state)
            
            if Parm['debug']:
                if f > (state['f0'] + Parm['debugtol'] * Ck):
                    status = 8
                    return HZ_return(status, iteration, state)
            
            if state['dphi0'] > 0:
                status = 5
                return HZ_return(status, iteration, state)
            
            state['objhist']+=[f]

        status = 2
        return HZ_return(status, iteration, state)

    
    # EXIT:


def HZ_return(status, iteration, state):
    end = time.time() - state['starttime']
    if status == 11:
        state['gnorm'] = torch.inf
    Stat = {}
    Stat['x'] = state['xtemp']
    Stat['status'] = status
    Stat['nfunc'] = state['nf']
    Stat['ngrad'] = state['ng']
    Stat['iter'] = iteration
    Stat['objhist'] = state['objhist']
    Stat['time'] = end
    if status < 10:
        Stat['f'] = state['f']
        Stat['gnorm'] = state['gnorm']
    
    if status > 0 and status < 10:
        # x = state['xtemp'].copy()
        gnorm = torch.abs(state['g']).max()
        Stat['gnorm'] = state['gnorm']
    if status >= 10:
        Stat['f'] = np.nan
        Stat['gnorm'] = np.nan

    return Stat
        


            







        



