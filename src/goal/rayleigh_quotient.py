class GoalGradient:
    def __init__(self, hamiltonian, x):
        self.hamiltonian = hamiltonian
        self.x = x
        self.xtAx_cached = None
        self.xtx_cached = None

    def xtAx(self, x, A):
        # Use cached value if applicable
        if x is self.x and A is self.hamiltonian and self.xtAx_cached is not None:
            return self.xtAx_cached
        # Compute and cache otherwise
        self.xtAx_cached = x.T.dot(A.matvec(x))
        return self.xtAx_cached

    def xtx(self, x):
        # Use cached value if applicable
        if x is self.x and self.xtx_cached is not None:
            return self.xtx_cached
        # Compute and cache otherwise
        self.xtx_cached = x.T.dot(x)
        return self.xtx_cached

    def objective_function(self, x, A):
        return self.xtAx(x, A) / self.xtx(x)

    def gradient(self, x, A):
        # Calculate components of the gradient
        num = 2 * A.matvec(x)
        denom = self.xtx(x)
        xtAx_value = self.xtAx(x, A)
        return (num / denom) - (2 * xtAx_value * x / denom ** 2), xtAx_value/denom