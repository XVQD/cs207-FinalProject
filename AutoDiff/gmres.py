from AutoDiff import Variable
import numpy as np
from scipy.sparse.linalg import gmres

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
    x = [Variable(1, name='x1')] * A.shape[0]
    f = [0] * A.shape[0]
    J = np.zeros(A.shape)
    for i in range(A.shape[0]):
        x[i] = Variable(1, name='x'+str(i))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            f[i] += A[i, j]*x[j]
        J[i] = [x[0] for x in list(f[i].der.values())]
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
    >>> array([ 1., -1.,  1.])
    """
    action = LinearOperator((b.shape[0], b.shape[0]), matvec=grad)
    x, exitcode = gmres(action, b)
    return x