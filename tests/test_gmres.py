import pytest
import numpy as np
import AutoDiff.AutoDiff as ad
from AutoDiff.GMRes import gmres_autodiff

    
def test_gmres_autodiff():
    b = [1, 2, 3]
    x1 = ad.Variable(1, name='x1')
    x2 = ad.Variable(1, name='x2')
    x3 = ad.Variable(1, name='x3')
    f1 = 2*x1+3*x2+2*x3
    f2 = 3*x1+2*x2+1*x3
    f3 = 3*x1+3*x2+3*x3
    F = [f1, f2, f3]
    x = gmres_autodiff(F, b)
    np.testing.assert_allclose(x, [1, -1, 1])
