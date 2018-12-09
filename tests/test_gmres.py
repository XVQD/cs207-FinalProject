import pytest
import numpy as np
from AutoDiff.AutoDiff import Variable
from AutoDiff.gmres import autograd, gmres_autodiff

    
def test_gmres_autodiff():
    b = np.array([1, 2, 3])
    p = np.array([1, 0, 0])
    x1 = Variable(1, name='x1')
    x2 = Variable(1, name='x2')
    x3 = Variable(1, name='x3')
    f1 = 2*x1+3*x2+2*x3
    f2 = 3*x1+2*x2+1*x3
    f3 = 3*x1+3*x2+3*x3
    F = [f1, f2, f3]
    grad = autograd(F)
    x = gmres_autodiff(b, grad)
    np.testing.assert_allclose(x, [1, -1, 1])
