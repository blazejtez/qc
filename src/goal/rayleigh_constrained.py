import cupy as cp

def gram_schmidt(vectors):
    orthogonal_vectors = []
    for v in vectors:
        w = v.copy()
        for u in orthogonal_vectors:
            w -= u.dot(u.T.dot(v))
        w /= cp.linalg.norm(w)
        orthogonal_vectors.append(w)
    return orthogonal_vectors

def orthogonalize(x0, eigenvectors):
    for vec in eigenvectors:
        x0 -= vec.dot(vec.T.dot(x0))
    x0 /= cp.linalg.norm(x0)
    return x0


class GoalGradient():
    def __init__(self, A, x, Y=None):
        self.A = A
        self.x = x
        self.Y = Y
        self.xtAx_cached = None
        self.xtx_cached = None
        self._lambd_t_Y_t_x_cached = None


    def xtAx(self, x):
        if x is self.x and self.xtAx_cached is not None:
            return self.xtAx_cached
        self.xtAx_cached = x.T.dot(self.A.matvec(x))
        self.x = x
        return self.xtAx_cached

    def xtx(self, x):
        if x is self.x and self.xtx_cached is not None:
            return self.xtx_cached
        self.xtx_cached = x.T.dot(x)
        self.x = x
        return self.xtx_cached

    def objective_function(self, x, lambd):
        if self.Y is not None and lambd is not None:
            return self.xtAx(x) / self.xtx(x) + lambd.T.dot(self.Y.T.dot(x))
        else:
            return self.xtAx(x) / self.xtx(x)

    def gradient_x(self, x, lambd):
        num = 2 * self.A.matvec(x)
        denom = self.xtx(x)
        xtAx_value = self.xtAx(x)
        term1 = (num / denom) - (2 * xtAx_value * x / denom ** 2)
        if self.Y is not None:
            term2 = self.Y.dot(lambd)
            gradient = term1 + term2
        else:
            gradient = term1
        return gradient

    def gradient_lambda(self, x):
        if self.Y is not None:
            return self.Y.T.dot(x)
