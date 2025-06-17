import cupy as cp

class GoalGradient():
    def __init__(self, A, Y=None):
        self.A = A
        self.Y = Y

    def xtAx(self, x):
        return x.T.dot(self.A.matvec(x))

    def xtx(self, x):
        return x.T.dot(x)

    def objective_function(self, x, lambd):
        xtAx = self.xtAx(x)
        xtx = self.xtx(x)
        if lambd is not None:
            lambYT = lambd.T.dot(self.Y.T.dot(x))
            return xtAx/xtx + lambYT, xtAx/xtx, lambYT
        else:
            return xtAx/xtx, xtAx/xtx, 0

    def gradient_x(self, x, lambd):
        Ax = self.A.matvec(x)
        xtAx = self.xtAx(x)
        xtx = self.xtx(x)
        a = 2/xtx
        b = xtAx/xtx * x
        term1 = 2/xtx * (Ax - (xtAx/xtx * x))
        if self.Y is not None:
            Ylamb = self.Y.dot(lambd)
            term2 = 2 * lambd.T.dot(self.Y.T.dot(x)) * Ylamb
            gradient = term1 + Ylamb
        else:
            gradient = term1
        return gradient

    def gradient_lambda(self, x):
        if self.Y is not None:
            return self.Y.T.dot(x)

    def goal_gradient(self, x, lambd):
        return self.gradient_x(x, lambd), self.gradient_lambda(x)

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
