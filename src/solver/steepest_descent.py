import cupy as cp
from hamiltonian.hamiltonian import HamiltonianOperatorCuPy


# Definiujemy funkcję celu: (x^T A x) / (x^T x)
def xtx(x: cp.ndarray, A) -> cp.float32:
    return x.T.dot(A.matvec(x)) / (x.T.dot(x))


def objective_function(x: cp.ndarray, A) -> cp.float32:
    return xtx(x, A) / (x.T.dot(x))


# Definiujemy gradient funkcji celu
def gradient(x: cp.ndarray, A):
    num = 2 * (A.matvec(x))
    denom = (x.T @ x)
    return (num / denom) - (2 * x.T.dot(A.matvec(x)) * x / (denom ** 2).T)


# Znalezienie optymalnego kroku alpha
def line_search(x, A, grad):
    # Funkcja celu dla alpha
    def f(alpha):
        return objective_function(x - alpha * grad, A)

    # Szukanie optymalnego alpha (możemy użyć np. metody złotego podziału lub prostego przeszukiwania)
    alpha = 1.0
    tol = 1e-5
    step = 0.1
    while True:
        if f(alpha - step) < f(alpha):
            alpha -= step
        elif f(alpha + step) < f(alpha):
            alpha += step
        else:
            break
        if step < tol:
            break
        step *= 0.9  # Zmniejszamy krok, aby znaleźć dokładniejsze alpha

    return alpha


# Algorytm Steepest Descent
def steepest_descent(A, x0, tol=1e-5, max_iter=1000):
    x = x0
    for i in range(max_iter):
        grad = gradient(x, A)

        # Szukanie optymalnego kroku alpha
        alpha = line_search(x, A, grad)

        # Aktualizacja wektora x
        x_new = x - alpha * grad

        # Sprawdzanie zbieżności
        print(f"Iteracja {i}: Eigenvalue: {xtx(x_new, A)}, Residual norm: {cp.linalg.norm(x_new - x)}")

        if cp.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x


# Define the grid
N = 300
L = 10.0  # Define the size of the grid (in atomic units, for example)

x = cp.linspace(-L, L, N, dtype=cp.float32)
y = cp.linspace(-L, L, N, dtype=cp.float32)
z = cp.linspace(-L, L, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

# Compute the Gaussian-type orbital values
gaussian_orbital = cp.exp(-0.5 * (X ** 2 + Y ** 2 + Z ** 2), dtype=cp.float32)
x0 = cp.reshape(gaussian_orbital, (len(x) * len(y) * len(z)))

# Inicjalizacja macierzy A i wektora początkowego x0
xl = cp.linspace(-5, 5, N, dtype=cp.float32)
yl = xl
zl = xl
A = HamiltonianOperatorCuPy(xl, yl, zl)

# Minimalizacja funkcji metodą Steepest Descent
result = steepest_descent(A, x0)
min_value = objective_function(result, A)

print("Zminimalizowane x:", result)
print("Minimalna wartość funkcji:", min_value)
