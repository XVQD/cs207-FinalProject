from AutoDiff.AutoDiff import Variable
from AutoDiff.gmres import autograd, gmres_autodiff
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

# define action
grad = autograd(F)

# define LinearOperator
action = LinearOperator((3, 3), matvec=grad)

# GMRES to get x
x, exitcode = gmres(action, b)
print(x)