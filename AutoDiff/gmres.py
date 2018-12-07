from AutoDiff.AutoDiff import Variable
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

A = np.array([[2, 3, 2], 
               [3, 2, 1],
               [3, 3, 3]])
b = np.array([1, 2, 3])
    
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
    x = [Variable(1, name='x1', der=p)] * A.shape[0]
    f = [0] * A.shape[0]
    J = np.zeros(A.shape)
    for i in range(A.shape[0]):
        x[i] = Variable(1, name='x'+str(i), der=p)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            f[i] += A[i, j]*x[j]
        J[i] = [v[0] for k, v in sorted(f[i].der.items())]
    return np.dot(J, p)

def gmres_autodiff(A, b, grad):
    """Solves Ax=b using GMRes with automatic differentiation
    
    Parameters
    ==========
    A:     np.array (1d)
           A in Ax=b
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
    action = LinearOperator((b.shape[0], b.shape[0]), matvec=grad)
    x, exitcode = gmres(action, b)
    return x
