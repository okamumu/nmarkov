import numpy as np
import scipy.sparse as sp
import functools

class _smatrix:
    def _get_elem(self):
        pass

    def __add__(self, other):
        return _binplus(self, other)

    def __sub__(self, other):
        return _binminus(self, other)

    def __pos__(self):
        return self

    def __neg__(self):
        return _unaryminus(self)

    def __and__(self, other):
        return _cblock(self, other)

    def __or__(self, other):
        return _rblock(self, other)

    def tocoo(self, padding_diag = False):
        x = self._get_elem()
        if padding_diag == True:
            i = np.array(range(min(self.shape)), dtype=np.int32)
            v = np.zeros(min(self.shape))
            x = (np.hstack((x[0], i)), np.hstack((x[1], i)), np.hstack((x[2], v)))
        Q = sp.coo_matrix((x[2], (x[0], x[1])), shape=self.shape)
        Q.sum_duplicates()
        return Q

    def tocsc(self, padding_diag = False):
        Q = self.tocoo(padding_diag)
        return Q.tocsc()

    def tocsr(self, padding_diag = False):
        Q = self.tocoo(padding_diag)
        return Q.tocsr()

class _cblock(_smatrix):
    def __init__(self, A, B):
        self.A = smatrix(A)
        self.B = smatrix(B)
        self.shape = (max(self.A.shape[0], self.B.shape[0]), self.A.shape[1] + self.B.shape[1])

    def __str__(self):
        return 'cblock({}, {})'.format(self.A, self.B)

    def _get_elem(self):
        a = self.A._get_elem()
        b = self.B._get_elem()
        i = np.hstack((a[0], b[0]))
        j = np.hstack((a[1], self.A.shape[1] + b[1]))
        v = np.hstack((a[2], b[2]))
        return (i,j,v)

class _rblock(_smatrix):
    def __init__(self, A, B):
        self.A = smatrix(A)
        self.B = smatrix(B)
        self.shape = (self.A.shape[0] + self.B.shape[0], max(self.A.shape[1], self.B.shape[1]))

    def __str__(self):
        return 'rblock({}, {})'.format(self.A, self.B)

    def _get_elem(self):
        a = self.A._get_elem()
        b = self.B._get_elem()
        i = np.hstack((a[0], self.A.shape[0] + b[0]))
        j = np.hstack((a[1], b[1]))
        v = np.hstack((a[2], b[2]))
        return (i,j,v)

class _block(_smatrix):
    def __init__(self, data):
        self.rows = {}
        self.cols = {}
        self.elem = {}
        for k in range(len(data[0])):
            i = data[1][0][k]
            j = data[1][1][k]
            self._set(i, j, data[0][k])
        self.shape = (sum(self.rows.values()), sum(self.cols.values()))

    def __str__(self):
        return '_smatrix({})'.format(self.elem)

    def _set(self, i, j, elem):
        elem = smatrix(elem)
        shape = elem.shape
        self.rows[i] = shape[0]
        self.cols[j] = shape[1]
        self.elem[(i,j)] = elem

    def _get_elem(self):
        xindex = {}
        xlabels = sorted(self.rows.items(), key=lambda x:x[0])
        xindex[xlabels[0][0]] = 0
        for i in range(1,len(self.rows)):
            xindex[xlabels[i][0]] = xindex[xlabels[i-1][0]] + xlabels[i][1]
        yindex = {}
        ylabels = sorted(self.cols.items(), key=lambda x:x[0])
        yindex[ylabels[0][0]] = 0
        for i in range(1,len(self.cols)):
            yindex[ylabels[i][0]] = yindex[ylabels[i-1][0]] + ylabels[i][1]
        i = np.array([], dtype=np.int32)
        j = np.array([], dtype=np.int32)
        v = np.array([], dtype=np.float)
        for x in self.elem.items():
            res = x[1]._get_elem()
            i = np.hstack((i, xindex[x[0][0]] + res[0]))
            j = np.hstack((j, yindex[x[0][1]] + res[1]))
            v = np.hstack((v, res[2]))
        return (i,j,v)

class _kron(_smatrix):
    def __init__(self, A, B):
        self.A = smatrix(A)
        self.B = smatrix(B)
        self.shape = (self.A.shape[0] * self.B.shape[0], self.A.shape[1] * self.B.shape[1])

    def __str__(self):
        return 'kron({}, {})'.format(self.A, self.B)

    def _get_elem(self):
        a = self.A._get_elem()
        b = self.B._get_elem()
        A = sp.coo_matrix((a[2], (a[0], a[1])))
        B = sp.coo_matrix((b[2], (b[0], b[1])))
        C = sp.kron(A,B, format='coo')
        return (C.row, C.col, C.data)

class _kronsum(_smatrix):
    def __init__(self, A, B):
        self.A = smatrix(A)
        self.B = smatrix(B)
        self.shape = (self.A.shape[0] * self.B.shape[0], self.A.shape[1] * self.B.shape[1])

    def __str__(self):
        return 'kronsum({}, {})'.format(self.A, self.B)

    def _get_elem(self):
        a = self.A._get_elem()
        b = self.B._get_elem()
        A = sp.coo_matrix((a[2], (a[0], a[1])))
        B = sp.coo_matrix((b[2], (b[0], b[1])))
        C = sp.kronsum(A,B, format='coo')
        return (C.row, C.col, C.data)

class _binplus(_smatrix):
    def __init__(self, A, B):
        self.A = smatrix(A)
        self.B = smatrix(B)
        self.shape = (max(self.A.shape[0], self.B.shape[0]), max(self.A.shape[1], self.B.shape[1]))

    def __str__(self):
        return '({} + {})'.format(self.A, self.B)

    def _get_elem(self):
        a = self.A._get_elem()
        b = self.B._get_elem()
        i = np.hstack((a[0], b[0]))
        j = np.hstack((a[1], b[1]))
        v = np.hstack((a[2], b[2]))
        return (i,j,v)

class _binminus(_smatrix):
    def __init__(self, A, B):
        self.A = smatrix(A)
        self.B = smatrix(B)
        self.shape = (max(self.A.shape[0], self.B.shape[0]), max(self.A.shape[1], self.B.shape[1]))

    def __str__(self):
        return '({} - {})'.format(self.A, self.B)

    def _get_elem(self):
        a = self.A._get_elem()
        b = self.B._get_elem()
        i = np.hstack((a[0], b[0]))
        j = np.hstack((a[1], b[1]))
        v = np.hstack((a[2], -b[2]))
        return (i,j,v)

class _unaryminus(_smatrix):
    def __init__(self, A):
        self.A = smatrix(A)
        self.shape = self.A.shape

    def __str__(self):
        return '(-{})'.format(self.A)

    def _get_elem(self):
        a = self.A._get_elem()
        return (a[0],a[1],-a[2])

class _nmatrix(_smatrix):
    def __init__(self, A):
        self.A = A
        self.shape = A.shape

    def _get_elem(self):
        A = sp.coo_matrix(self.A)
        return (A.row, A.col, A.data)

    def __repr__(self):
        return self.A.__repr__()

    def __str__(self):
        return self.A.__str__()

def zeros(shape):
    return _nmatrix(sp.coo_matrix(shape, dtype=np.float))

def eye(m, n=None, k=0):
    return _nmatrix(sp.eye(m=m, n=n, k=k, dtype=np.float, format='coo'))

def diags(diagonals, offsets=0, shape=None):
    return _nmatrix(sp.diags(diagonals, offsets, shape, format='coo', dtype=np.float))

def array(x):
    return _nmatrix(np.array(x, dtype=np.float))

def smatrix(A):
    if isinstance(A, _smatrix):
        return A
    else:
        return _nmatrix(A)

def cblock(*args):
    return functools.reduce(lambda x,y: _cblock(x,y), [smatrix(x) for x in args])

def rblock(*args):
    return functools.reduce(lambda x,y: _rblock(x,y), [smatrix(x) for x in args])

def kron(*args):
    return functools.reduce(lambda x,y: _kron(x,y), [smatrix(x) for x in args])

def kronsum(*args):
    return functools.reduce(lambda x,y: _kronsum(x,y), [smatrix(x) for x in args])

def block(list):
    ilist = []
    jlist = []
    vlist = []
    for i in range(len(list)):
        for j in range(len(list[i])):
            ilist.append(i)
            jlist.append(j)
            vlist.append(list[i][j])
    return _block((vlist, (ilist, jlist)))

