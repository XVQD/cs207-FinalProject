import pytest
import numpy as np
from AutoDiff.AutoDiff import Variable
from AutoDiff.gmres import grad, gmres_autodiff, A, b
from AutoDiff import AutoDiff


def test_grad():
    p = np.array([1, 0, 0])
    np.testing.assert_allclose(grad(p), [2, 3, 3])
    
def test_gmres_autodiff():
    x = gmres_autodiff(A, b, grad)
    np.testing.assert_allclose(x, [1, -1, 1])