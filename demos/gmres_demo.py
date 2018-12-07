from AutoDiff.AutoDiff import Variable
from AutoDiff.gmres import A, b, grad, gmres_autodiff
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import sympy as sym

# function to construct Jacobian matrix
def Jacobian(v_str, f_list):
    vars = sym.symbols(v_str)
    f = sym.sympify(f_list)
    J = sym.zeros(len(f),len(vars))
    for i, fi in enumerate(f):
        for j, s in enumerate(vars):
            J[i,j] = sym.diff(fi, s)
    return J

# function to get matrix-vector product
def action_func(x):
    return np.dot(A, x)

# time the Jacobian matrix construction for matrices of different sizes
times_jacobian = []
for size_n in range(10, 110, 10):
    var = ' '.join([''.join(x) for x in list(zip(['x']*size_n, 
                   [str(x) for x in list(range(1, size_n+1, 1))]))])
    coefs = np.random.rand(size_n, size_n)
    f = [''] * size_n
    for i in range(coefs.shape[0]):
        for j in range(coefs.shape[1]):
            f[i] += str(coefs[i, j])+'*'+var.split(' ')[j]
            if j != coefs.shape[0]-1:
                f[i] += '+'
    start = time.time()
    Jacobian(var, f)
    end = time.time()
    times_jacobian.append(end-start)
    
# time the GMRES solving times
times_gmres = []
for i in range(10, 110, 10):
    # GMRES to solve Ax=b
    A = np.random.rand(i, i)
    b = np.random.rand(i)
    action = LinearOperator((i, i), matvec=action_func)
    start = time.time()
    x, exitcode = gmres(action, b)
    end = time.time()
    times_gmres.append(end-start)
    
print('Jacobian times', times_jacobian)
print('GMRES times', times_gmres)
