import time

import cupy as cp
import portion as P

import Praktyki.cut_box_3D as box
import hamiltonian.hamiltonian as H
import data_structure.raster as raster
from data_structure.util_cub import save_basic, load_basic
from goal.rayleigh_constrained import GoalGradient, orthogonalize, gram_schmidt
ALPHA = 0.28294212105225837470023780155114

from goal.wavefunction import hydrogen_2px, hydrogen_1s, hydrogen_2s, hydrogen_2py, hydrogen_2pz


def gradient_descent_constrained(goal_gradient, x0, lambd0, lr_x=0.001, lr_lambda=0.1, tol=0.005, max_iter=1000):
    x = x0
    lambd = lambd0
    A = goal_gradient.A

    # Adam optimizer parameters
    beta1 = 0.9
    beta2 = 0.99
    epsilon = 1e-8

    # Initialize first and second moments
    m_x = cp.zeros_like(x)
    v_x = cp.zeros_like(x)
    if lambd is not None and lambd.size > 0:
        m_lambda = cp.zeros_like(lambd)
        v_lambda = cp.zeros_like(lambd)
    else:
        m_lambda = None
        v_lambda = None

    t = 0  # Timestep for bias correction

    for i in range(max_iter):
        t += 1  # Increment timestep

        # Compute gradients
        grad_x = goal_gradient.gradient_x(x, lambd)
        grad_lambda = goal_gradient.gradient_lambda(x)
        prev_eigenvalue = goal_gradient.objective_function(x, lambd)
        constraint_violation = 0
        if grad_lambda is not None:
            constraint_violation = cp.linalg.norm(goal_gradient.gradient_lambda(x))

        # Update biased first moment estimate for x
        m_x = beta1 * m_x + (1 - beta1) * grad_x
        # Update biased second raw moment estimate for x
        v_x = beta2 * v_x + (1 - beta2) * cp.square(grad_x)
        # Compute bias-corrected first moment estimate for x
        m_hat_x = m_x / (1 - beta1 ** t)
        # Compute bias-corrected second raw moment estimate for x
        v_hat_x = v_x / (1 - beta2 ** t)
        # Update x
        x_new = x - lr_x * m_hat_x / (cp.sqrt(v_hat_x) + epsilon)

        if grad_lambda is not None:
            # Update biased first moment estimate for lambda
            m_lambda = beta1 * m_lambda + (1 - beta1) * grad_lambda
            # Update biased second raw moment estimate for lambda
            v_lambda = beta2 * v_lambda + (1 - beta2) * cp.square(grad_lambda)
            # Compute bias-corrected first moment estimate for lambda
            m_hat_lambda = m_lambda / (1 - beta1 ** t)
            # Compute bias-corrected second raw moment estimate for lambda
            v_hat_lambda = v_lambda / (1 - beta2 ** t)
            # Update lambda
            lambd_new = lambd + lr_lambda * m_hat_lambda / (cp.sqrt(v_hat_lambda) + epsilon)
        else:
            lambd_new = lambd

        # Normalize x_new to prevent it from becoming too small or large
        x_new = x_new / cp.linalg.norm(x_new)

        # Compute the eigenvalue (objective function value)
        eigenvalue, x_factor, lambda_factor = goal_gradient.objective_function(x_new, lambd_new)
        '''
        if abs(prev_eigenvalue - eigenvalue) < tol:
            print("Eigenvalue converged:",abs(prev_eigenvalue - eigenvalue) )
            time.sleep(5)
            break
        '''

        # Check convergence
        step_x = cp.linalg.norm(x_new - x)
        if lambd is not None:
            step_lambda = cp.linalg.norm(lambd_new - lambd)
        if step_x < tol:
            print(f"Converged at iteration {i}.")
            break
        if i % 100 == 1 or i == max_iter - 1:
            # Iteration number, eigenvalue, norm change, constraint violation
            if lambd is not None:
                print(f"{i:6d}, {eigenvalue[0, 0]:.7f}, {x_factor[0,0]:.7f}, {lambda_factor[0,0]:.7f}, {step_x:.7f}, {constraint_violation:.7f}")
            else:
                print(f"{i:6d}, {eigenvalue[0, 0]:.7f}, {step_x:.7f}")
        # Update variables for next iteration
        x = x_new
        lambd = lambd_new

    return x, lambd

def find_lowest_eigenvalues(A, initial_x, num_eigenvalues=5, lr_x=1e-5, lr_lambda=0.1, tol=1e-5, max_iter=1000, initial_Y=None, initial_lambd=None):
    if initial_Y is not None:
        eigenvectors = initial_Y
        eigenvalues = initial_lambd
    else:
        eigenvectors = []
        eigenvalues = []
    x0 = initial_x

    for i in range(num_eigenvalues):
        if eigenvalues:
            lambd0 = cp.array(eigenvalues, dtype=cp.float32).reshape((len(eigenvalues), 1))
        else:
            lambd0 = None
        Y = cp.hstack(eigenvectors) if eigenvectors else None

        # Create GoalGradient instance with the current Y
        goal_gradient = GoalGradient(A, Y=Y)

        # Perform constrained gradient descent
        x, lambd = gradient_descent_constrained(goal_gradient, x0, lambd0, lr_x=lr_x, lr_lambda=lr_lambda, tol=tol, max_iter=max_iter)

        # Compute the eigenvalue
        eigenvalue = goal_gradient.objective_function(x, lambd)[1]

        # Store the eigenvector and eigenvalue
        eigenvectors.append(x)
        eigenvalues.append(eigenvalue)

        eigenvectors_path_template = "eigenvector_{i}.cub"
        eigenvalues_path = "eigenvalues.txt"
        file_path = eigenvectors_path_template.format(i=len(eigenvectors))

        save_basic(file_path, cp.asnumpy(eigenvectors[-1].reshape((len(xl), len(yl), len(zl)))))
        with open(eigenvalues_path, "w") as f:
            for idx, eigenvalue in enumerate(eigenvalues):
                f.write(f"Eigenvalue {idx + 1}: {eigenvalue}\n")
        
        # Prepare for next iteration
        x0 = cp.random.rand(*x.shape).astype(cp.float32)
        eigenvectors = gram_schmidt(eigenvectors)
        x0 = orthogonalize(x0, eigenvectors)

    return eigenvalues, eigenvectors


def numerical_gradient_x(goal_gradient, x, A, lambd, epsilon=1e-8):
    grad = cp.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        f_plus = goal_gradient.objective_function(x_plus, A, lambd)
        f_minus = goal_gradient.objective_function(x_minus, A, lambd)
        grad[i] = (f_plus - f_minus) / (2 * epsilon)
    return grad


print("Iteration number, eigenvalue, norm change, constraint violation")
HYDROGEN_RADIUS = 20.
ALPHA = 0.282942121052

interval = P.closed(-HYDROGEN_RADIUS, HYDROGEN_RADIUS)
box_ = box.box3D(interval, interval, interval)
r = raster.Raster(10)
xl, yl, zl = r.box_linspaces(box_)
N = len(xl) * len(yl) * len(zl)
print(len(xl))
A = H.HamiltonianOperatorCuPy(xl, yl, zl, extent=HYDROGEN_RADIUS)

# X, Y, Z = cp.meshgrid(cp.array(xl), cp.array(yl), cp.array(zl))
# gaussian_orbital = cp.exp(-ALPHA * (X ** 2 + Y ** 2 + Z ** 2), dtype=cp.float32)
# norm_factor = cp.sum(gaussian_orbital ** 2) * A.h ** 3
# gaussian_orbital /= cp.sqrt(norm_factor)

h_1s = hydrogen_1s(len(xl), HYDROGEN_RADIUS)
# h_2s = hydrogen_2s(len(xl), HYDROGEN_RADIUS)
# h_2px = hydrogen_2px(len(xl), HYDROGEN_RADIUS)
h2_py = hydrogen_2py(len(xl), HYDROGEN_RADIUS)
h2_pz = hydrogen_2pz(len(xl), HYDROGEN_RADIUS)


# v_init = cp.reshape(gaussian_orbital, (len(X) * len(Y) * len(Z), 1))
#v_init = cp.random.random((N, 1))


# Y1 = load_basic("../data/h100.cub")
# Y1 = cp.reshape(Y1, (N, 1), )
lambd = [cp.asarray([[-0.49857717]], dtype=cp.float32),
         cp.asarray([[-0.1251940]], dtype=cp.float32),
         cp.asarray([[-0.12494146]], dtype=cp.float32)
         ]
'''
         cp.asarray([[-0.4985772]], dtype=cp.float32),
         cp.asarray([[-0.12432906]], dtype=cp.float32),
         cp.asarray([[-0.12279524]], dtype=cp.float32),
         cp.asarray([[-0.02790804]], dtype=cp.float32),
         cp.asarray([[-0.03419532]], dtype=cp.float32),
'''
# Y1 = load_basic("eigenvector_1.cub")
# Y1 = Y1.reshape((N,1), )
# Y2 = load_basic("eigenvector_2.cub")
# Y2 = Y2.reshape((N,1), )
# Y3 = load_basic("eigenvector_3.cub")
# Y3 = Y3.reshape((N,1), )

'''Y5 = load_cub("h1xx4.cub")
Y5 = Y5.reshape((N,1), )
Y6 = load_cub("h1xx5.cub")
Y6 = Y5.reshape((N,1), )
Y7 = load_cub("h1xx6.cub")
Y7 = Y5.reshape((N,1), )
Y8 = load_cub("h1xx7.cub")
Y8 = Y5.reshape((N,1), )
'''
# Y = [Y1, Y2, Y3]

goal_gradient = GoalGradient(A=A, Y=None)

#Y=None
#lambd=None
# Find the lowest five eigenvalues
start_time = time.time()
eigenvalues, eigenvectors = find_lowest_eigenvalues(A, h_1s, num_eigenvalues=4, lr_x=1e-6, lr_lambda=1e-2, # e-3 dla pierwszych 3 wartosci
                                                    tol=1e-5, max_iter=10000, initial_Y=None, initial_lambd=None)
end_time = time.time()

# Display the results
for idx, eigenvalue in enumerate(eigenvalues):
    print(f"Eigenvalue {idx+1}: {eigenvalue}")

print("Czas wykonania:", end_time - start_time, "sekund")
