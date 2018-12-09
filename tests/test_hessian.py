import pytest
import numpy as np
from AutoDiff.AutoDiff import Variable
import AutoDiff.AutoDiff as ad

# a multi variable function
def f2(values):
    x1 = ad.Variable(values['x1'], name='x1')
    x2 = ad.Variable(values['x2'], name='x2')
    return 2 * (x1 ** 2) + ad.sin(x2)

def test_hessian():
    # define a starting point
    x = {'x1':5, 'x2':6}
    f2x=f2(x)
    hessian=f2x.hessian(["x1","x2"])
    assert np.allclose( hessian , np.array([[4.,0.],[0.,0.2794155]]) )

if __name__=="__main__":
    test_hessian()