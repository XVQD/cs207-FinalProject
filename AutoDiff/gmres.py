from AutoDiff.AutoDiff import Variable
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from tests.GLOBAL_VARS import F, b


def grad(p):
    """Calculates derivatives of Ax where A is a matrix using forward mode automatic differentiation
    
    Parameters
    ==========
    p:  np.array
        the vector p in Jp matrix-vector product where J is the Jacobian

    Returns
    =======
    np.array
    dot product of J and p
    """
    dim = len(F)
    J = np.zeros((dim, dim))
    for i in range(dim):
        J[i] = [v[0] for k, v in sorted(F[i].der.items())]
    return np.dot(J, p)

def gmres_autodiff(b, grad):
    """Solves Ax=b using GMRes with automatic differentiation
    
    Parameters
    ==========
    b:     np.array (1d)
           RHS of Ax=b
    grad:  function
           function to calculate dot product of J and p
         
    Returns
    =======
    x:     np.array (1d)
           solution to Ax=b
    
    Examples
    ========
    >>> b = np.array([2, 3, 1])
    >>> x, exitcode = gmres_autodiff(b)
    >>> x
    array([ 1., -1.,  1.])
    """
    dim = b.shape[0]
    action = LinearOperator((dim, dim), matvec=grad)
    x, exitcode = gmres(action, b, atol='legacy')
    return x
