import AutoDiff.AutoDiff as ad
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

def autograd(F):
    """Calculates derivatives of Ax where A is a matrix using forward mode automatic differentiation

    Parameters
    ==========
    F:  list
        user-defined vector of vector functions
    p:  np.array
        the vector p in Jp matrix-vector product where J is the Jacobian

    Returns
    =======
    function
    function to get matrix-vector product
    """
    def grad(p):
        dim = len(F)
        J = np.zeros((dim, dim))
        for i in range(dim):
            J[i] = [v[0] for k, v in sorted(F[i].der.items())]
        return np.dot(J, p)
    return grad

def gmres_autodiff(F, b):
    """Solves Ax=b using GMRes with automatic differentiation
    
    Parameters
    ==========
    F:     list
           vector of vector functions
    b:     list
           RHS of Ax=b
         
    Returns
    =======
    x:     np.array (1d)
           solution to Ax=b
    
    Examples
    ========
    >>> b = [2, 3, 1]
    >>> x1 = ad.Variable(1, name='x1')
    >>> x2 = ad.Variable(2, name='x2')
    >>> x3 = ad.Variable(2, name='x3')
    >>> f1 = 2*x1+3*x2+2*x3
    >>> f2 = 3*x1+2*x2+1*x3
    >>> f3 = 3*x1+3*x2+3*x3
    >>> F = [f1, f2, f3]
    >>> x = gmres_autodiff(F, b)
    >>> x
    array([ 1., -1.,  1.])
    """
    dim = np.array(b).shape[0]
    grad = autograd(F)
    action = LinearOperator((dim, dim), matvec=grad)
    x, exitcode = gmres(action, b, atol='legacy')
    return x
