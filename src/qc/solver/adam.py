# import cupy as cp
#
#
# h = Hamiltonian()
#
# #Require:α: Stepsize
# alpha = 0.001
# #Require:β1,β2∈[0,1): Exponential decay rates for the moment estimates
# beta1 = 0.9
# beta2 = 0.999
# #Require:f(θ): Stochastic objective function with parametersθ
# hx = h.matvec(x_k)
# xhx = x_k.T.dot(hx)
# print(f"eigenvalue: {xhx}")
# xx = x_k.T.dot(x_k)
# p_k = hx - (xhx/xx)*x_k  # f(theta)
# #Require:θ0: Initial parameter vector
# x_k = cp.random.randn(N,1,dtype=cp.float32)
# x_k = x_k/cp.linalg.norm(x_k) # theta
# print(x_k.size)
# #m0←0(Initialize 1stmoment vector)
# m = cp.zeros(x_k.shape)
# #v0←0(Initialize 2ndmoment vector)
# v = cp.zeros(x_k.shape)
# #t←0(Initialize timestep)
# #to jest zrobione w range ponizej
#
# eps = 1e-8
# for i in range(1000):
#     hx = h.matvec(x_k)
#     xhx = x_k.T.dot(hx)
#     print(f"eigenvalue: {xhx}")
#     xx = x_k.T.dot(x_k)
#     #gt←∇θft(θt−1)(Get gradients w.r.t. stochastic objective at timestept)
#     p_k = hx - (xhx/xx)*x_k  # gt
#
#     #mt←β1·mt−1+ (1−β1)·gt(Update biased first moment estimate)
#     m = beta1 * m + (1 - beta1) * pk
#
#     #vt←β2·vt−1+ (1−β2)·g2t(Update biased second raw moment estimate)̂
#     v = beta2 * v + (1 - beta2) * pk * pk
#
#     #mt←mt/(1−βt1)(Compute bias-corrected first moment estimate)̂
#     m_corrected = m / (1 - beta1 ** i)
#
#     #vt←vt/(1−βt2)(Compute bias-corrected second raw moment estimate)
#     v_corrected = v / (1 - beta2 ** i)
#     #θt←θt−1−α·̂mt/(√̂vt+)(Update parameters)
#     x_k = x_k - alpha * m_corrected / (sqrt(v_corrected) + eps)