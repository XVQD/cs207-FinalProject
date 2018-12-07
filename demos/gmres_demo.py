from AutoDiff.AutoDiff import Variable
from AutoDiff.gmres import grad, gmres_autodiff
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import sympy as sym

# GMRES
# define variables and do AD
x1 = Variable(1, name='x1')
x2 = Variable(1, name='x2')
x3 = Variable(1, name='x3')
f1 = 2*x1+3*x2+2*x3
f2 = 3*x1+2*x2+1*x3
f3 = 3*x1+3*x2+3*x3
F = [f1, f2, f3]
b = np.array([1, 2, 3])

# define action on x
action = LinearOperator((3, 3), matvec=grad)

# GMRES to get x
x, exitcode = gmres(action, b)
print(x)

# function to construct Jacobian matrix
def Jacobian(v_str, f_list):
    vars = sym.symbols(v_str)
    f = sym.sympify(f_list)
    J = sym.zeros(len(f),len(vars))
    for i, fi in enumerate(f):
        for j, s in enumerate(vars):
            J[i,j] = sym.diff(fi, s)
    return J

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

print('Jacobian time', times_jacobian)
