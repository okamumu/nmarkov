import numpy as np
import scipy.sparse as sp
import nmarkov._nmarkov as _nm

def sprob(Q, x0 = None):
    return nctmc(Q).sprob(x0)

def ssen(Q, b, pis, x0 = None):
    return nctmc(Q).ssen(b, pis, x0)

def tprob(Q, t, x0, trans = True, cx0 = None):
    return nctmc(Q).tprob(t, x0, trans, cx0)

def trwd(Q, t, x0, rwd, trans = True, cx0 = None):
    return nctmc(Q).trwd(t, x0, rwd, trans, cx0)

def mexpAx(Q, x = None, t = 1.0, trans = False):
    return nctmc(Q).mexpAx(x, t, trans)

def cmexpAx(Q, x = None, t = 1.0, trans = False, cx = None):
    return nctmc(Q).cmexpAx(x, t, trans, cx)

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

class _DenseCTMC:
    def __init__(self, Q, params):
        self.Q = Q
        self.params = params

    def _sprob_gth(self):
        return _nm.ctmc_st_gth_dense(self.Q)
    
    def _sprob_gs(self, x0):
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

class nctmc:
    def __init__(self, Q):
        self.params = _nm.Params()
        self.params.eps = np.sqrt(np.finfo(float).eps)
        if isinstance(Q, smatrix):
            self.n = Q._get_shape()[0]
            self.sprob_method = 'GS'
            self.Q = _SparseCTMC(Q.tocsc(), self.params)
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
        return self.Q._tprob(dt, x0, trans, cx0)

    def trwd(self, t, x0, rwd, trans = True, cx0 = None):
        if cx0 == None:
            cx0 = np.zeros_like(x0)
        dt = np.insert(np.diff(t), 0, t[0])
        return self.Q._trwd(dt, x0, rwd, trans, cx0)

    def mexpAx(self, x = None, t = 1.0, trans = False):
        if x == None:
            x = np.eye(self.n)
        return self.Q._mexpAx(x, t, trans)

    def cmexpAx(self, x = None, t = 1.0, trans = False, cx = None):
        if x == None:
            x = np.eye(self.n)
        if cx == None:
            cx = np.zeros_like(x)
        return self.Q._cmexpAx(x, cx, t, trans)

class smatrix:
    def _get_shape(self):
        pass

    def _get_elem(self, xstart = 0, ystart = 0):
        pass

    def tocoo(self, xstart = 0, ystart = 0, diag = True):
        x = self._get_elem(xstart, ystart)
        shape = self._get_shape()
        if diag == True:
            i = np.array(range(min(shape)), dtype=np.int32)
            v = np.zeros(min(shape))
            x = (np.hstack((x[0], i)), np.hstack((x[1], i)), np.hstack((x[2], v)))
        Q = sp.coo_matrix((x[2], (x[0], x[1])), shape=self._get_shape())
        Q.sum_duplicates()
        return Q

    def tocsc(self, xstart = 0, ystart = 0, diag = True):
        Q = self.tocoo(xstart, ystart, diag)
        return Q.tocsc()

    def tocsr(self, xstart = 0, ystart = 0, diag = True):
        Q = self.tocoo(xstart, ystart, diag)
        return Q.tocsr()

class bmatrix(smatrix):
    def __init__(self):
        self.rows = {}
        self.cols = {}
        self.elem = {}

    def __str__(self):
        return 'smatrix({})'.format(self.elem)

    def _get_shape(self):
        x = sum(self.rows.values())
        y = sum(self.cols.values())
        return (x,y)

    def set(self, i, j, elem):
        shape = elem._get_shape()
        self.rows[i] = shape[0]
        self.cols[j] = shape[1]
        self.elem[(i,j)] = elem

    def _get_elem(self, xstart = 0, ystart = 0):
        xindex = {}
        xlabels = sorted(self.rows.items(), key=lambda x:x[0])
        xindex[xlabels[0][0]] = xstart
        for i in range(1,len(self.rows)):
            xindex[xlabels[i][0]] = xindex[xlabels[i-1][0]] + xlabels[i][1]
        yindex = {}
        ylabels = sorted(self.cols.items(), key=lambda x:x[0])
        yindex[ylabels[0][0]] = ystart
        for i in range(1,len(self.cols)):
            yindex[ylabels[i][0]] = yindex[ylabels[i-1][0]] + ylabels[i][1]
        
        i = np.array([], dtype=np.int32)
        j = np.array([], dtype=np.int32)
        v = np.array([], dtype=np.float)
        for x in self.elem.items():
            res = x[1]._get_elem(xindex[x[0][0]], yindex[x[0][1]])
            i = np.hstack((i, res[0]))
            j = np.hstack((j, res[1]))
            v = np.hstack((v, res[2]))
        return (i,j,v)

class nmatrix(smatrix):
    def __init__(self, A):
        self.A = A

    def _get_shape(self):
        return self.A.shape

    def _get_elem(self, xstart = 0, ystart = 0):
        A = sp.coo_matrix(self.A)
        return (xstart+A.row, ystart+A.col, A.data)

    def __repr__(self):
        return self.A.__repr__()

    def __str__(self):
        return self.A.__str__()

class zeros(nmatrix):
    def __init__(self, shape):
        super().__init__(sp.coo_matrix(shape, dtype=np.float))

class eye(nmatrix):
    def __init__(self, m, n=None, k=0):
        super().__init__(sp.eye(m=m, n=n, k=k, dtype=np.float, format='coo'))


## methods

def cblock(*args):
    A = bmatrix()
    for i in range(len(args)):
        if isinstance(args[i], smatrix):
            A.set(0, i, args[i])
        else:
            A.set(0, i, nmatrix(args[i]))
    return A

def rblock(*args):
    A = bmatrix()
    for i in range(len(args)):
        if isinstance(args[i], smatrix):
            A.set(i, 0, args[i])
        else:
            A.set(i, 0, nmatrix(args[i]))
    return A

def block(list):
    A = bmatrix()
    for i in range(len(list)):
        for j in range(len(list[i])):
            if isinstance(list[i][j], smatrix):
                A.set(i, j, list[i][j])
            else:
                A.set(i, j, nmatrix(list[i][j]))
    return A
