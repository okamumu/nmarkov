import numpy as np
import scipy.sparse as sp
import nmarkov._nmarkov as _nm
import nmarkov.smatrix as sm

def sprob(Q, x0 = None):
    return nctmc(Q).sprob(x0)

def ssen(Q, b, pis, x0 = None):
    return nctmc(Q).ssen(b, pis, x0)

def tprob(Q, t, x0, trans = True, cx0 = None):
    return nctmc(Q).tprob(t, x0, trans, cx0)

def trwd(Q, t, x0, rwd, trans = True, cx0 = None):
    return nctmc(Q).trwd(t, x0, rwd, trans, cx0)

def mexpAx(Q, t = 1.0, x = None, trans = False):
    return nctmc(Q).mexpAx(t, x, trans)

def cmexpAx(Q, t = 1.0, x = None, trans = False, cx = None):
    return nctmc(Q).cmexpAx(t, x, trans, cx)

class deformula:
    def __init__(self):
        self.zero = 1.0e-12
        self.reltol = 1.0e-8
        self.startd = 8
        self.maxiter = 12
    
    def integrate(self, f, domain, *args, **kwargs):
        if (0, np.inf) == domain:
            return self._zerotoinf(f, *args, **kwargs)
        else:
            return self._monetoone(f, domain[0], domain[1], *args, **kwargs)

    def _zerotoinf(self, f, *args, **kwargs):
        return _nm.deformula_zerotoinf(f=lambda x:f(x, *args, **kwargs), zero=self.zero, reltol=self.reltol, startd=self.startd, maxiter=self.maxiter)
    
    def _monetoone(self, f, lower, upper, *args, **kwargs):
        dx1 = upper - lower
        dx2 = upper + lower
        res = _nm.deformula_monetoone(f=lambda x: dx1 * f((dx1 * x + dx2)/2.0, *args, **kwargs)/2.0, zero=self.zero, reltol=self.reltol, startd=self.startd, maxiter=self.maxiter)
        return (res[0], (res[1] * dx1 + dx2) / 2.0, res[2], res[3], res[4], res[5])

class _SparseCTMC:
    def __init__(self, Q, params):
        self.Q = Q
        self.params = params

    def _sprob_gth(self):
        return _nm.ctmc_st_gth_sparse(self.Q)
    
    def _sprob_gs(self, x0):
        return _nm.ctmc_st_gs_sparse(self.Q, x0, self.params)

    def _ssen_gs(self, x0, b, pis):
        return _nm.ctmc_stsen_gs_sparse(self.Q, x0, b, pis, self.params)

    def _tprob(self, dt, x0, trans, cx0):
        return _nm.ctmc_tran_sparse(self.Q, x0, cx0, dt, trans, self.params)

    def _trwd(self, dt, x0, rwd, trans, cx0):
        return _nm.ctmc_tran_rwd_sparse(self.Q, x0, cx0, rwd, dt, trans, self.params)
    
    def _mexpAx(self, x, t, trans):
        return _nm.ctmc_mexp_sparse(self.Q, x, t, trans, self.params)

    def _cmexpAx(self, x, cx, t, trans):
        return _nm.ctmc_mexpint_sparse(self.Q, x, cx, t, trans, self.params)

    def _mexpAx_mix(self, x, w, t, trans):
        return _nm.ctmc_mexp_mix_sparse(self.Q, x, w, t, trans, self.params)

class _DenseCTMC:
    def __init__(self, Q, params):
        self.Q = Q
        self.params = params

    def _sprob_gth(self):
        return _nm.ctmc_st_gth_dense(self.Q)
    
    def _sprob_gs(self, x0):
        try:
            res = _nm.ctmc_st_gs_dense(self.Q, x0, self.params)
            if self.params.info == -1:
                raise RuntimeError("Do not convergence")
        except RuntimeError:
            pass
        return _nm.ctmc_st_gs_dense(self.Q, x0, self.params)

    def _ssen_gs(self, x0, b, pis):
        return _nm.ctmc_stsen_gs_dense(self.Q, x0, b, pis, self.params)

    def _tprob(self, dt, x0, trans, cx0):
        return _nm.ctmc_tran_dense(self.Q, x0, cx0, dt, trans, self.params)

    def _trwd(self, dt, x0, rwd, trans, cx0):
        return _nm.ctmc_tran_rwd_dense(self.Q, x0, cx0, rwd, dt, trans, self.params)

    def _mexpAx(self, x, t, trans):
        return _nm.ctmc_mexp_dense(self.Q, x, t, trans, self.params)

    def _cmexpAx(self, x, cx, t, trans):
        return _nm.ctmc_mexpint_dense(self.Q, x, cx, t, trans, self.params)

    def _mexpAx_mix(self, x, w, t, trans):
        return _nm.ctmc_mexp_mix_dense(self.Q, x, w, t, trans, self.params)

class nctmc:
    def __init__(self, Q):
        self.params = _nm.Params()
        self.params.eps = np.sqrt(np.finfo(float).eps)
        if isinstance(Q, sm._smatrix):
            self.n = Q.shape[0]
            self.sprob_method = 'GS'
            self.Q = _SparseCTMC(Q.tocsc(padding_diag = True), self.params)
        elif isinstance(Q, sp.spmatrix):
            self.n = Q.shape[0]
            self.sprob_method = 'GS'
            self.Q = _SparseCTMC(Q, self.params)
        else:
            self.n = Q.shape[0]
            self.sprob_method = 'GTH'
            self.Q = _DenseCTMC(Q, self.params)

    def sprob(self, x0 = None):
        if self.sprob_method == 'GTH':
            return self.Q._sprob_gth()
        elif self.sprob_method == 'GS':
            if x0 == None:
                x0  = [1.0/self.n for i in range(self.n)]
            return self.Q._sprob_gs(x0)

    def ssen(self, b, pis, x0 = None):
        if x0 == None:
            x0  = [1.0/self.n for i in range(self.n)]
        return self.Q._ssen_gs(x0, b, pis)

    def tprob(self, t, x0, trans = True, cx0 = None):
        if cx0 == None:
            cx0 = np.zeros_like(x0)
        dt = np.insert(np.diff(t), 0, t[0])
        res = self.Q._tprob(dt, x0, trans, cx0)
        return {'t':t, 'prob':res[0], 'cprob':res[1]}

    def trwd(self, t, x0, rwd, trans = True, cx0 = None):
        if cx0 == None:
            cx0 = np.zeros_like(x0)
        dt = np.insert(np.diff(t), 0, t[0])
        res = self.Q._trwd(dt, x0, rwd, trans, cx0)
        if rwd.ndim == 1:
            return {'t':t, 'prob':res[0], 'cprob':res[1], 'irwd':res[2], 'crwd':res[3]}
        else:
            n = len(t)
            m = rwd.shape[1]
            return {'t':t, 'prob':res[0], 'cprob':res[1], 'irwd':res[2].reshape(n,m), 'crwd':res[3].reshape(n,m)}

    def mexpAx(self, t = 1.0, x = None, trans = False):
        if x == None:
            x = np.eye(self.n)
        return self.Q._mexpAx(x, t, trans)

    def cmexpAx(self, t = 1.0, x = None, trans = False, cx = None):
        if x == None:
            x = np.eye(self.n)
        if cx == None:
            cx = np.zeros_like(x)
        res = self.Q._cmexpAx(x, cx, t, trans)
        return {'x':res[0], 'cx':res[1]}

    def mexpAx_mix(self, f, x = None, domain = (0, np.inf), trans = False, *args, **kwargs):
        if x == None:
            x = np.eye(self.n)
        res = deformula().integrate(f, domain, *args, **kwargs)
        return self.Q._mexpAx_mix(x, res[2], res[1], trans) * res[4]

