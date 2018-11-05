import pytest
import numpy as np
from AutoDiff.AutoDiff import Variable
from AutoDiff import AutoDiff

class Test_Elementary_functions():

    def test_exp(self):
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        x3 = AutoDiff.exp(x1 + x2)
        assert x3.val == np.exp(1+1)
        assert x3.der == {'x1': np.exp(1+1), 'x2': np.exp(1+1)}

    def test_log(self):
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        x3 = AutoDiff.log(x1 + x2)
        assert x3.val == np.log(1+1)
        assert x3.der == {'x1': 1/2, 'x2': 1/2}

    def test_trigonometric(self):
        x1 = Variable(np.pi/4, name='x1')
        x2 = Variable(np.pi/4, name='x2')
        x3 = Variable(np.pi/4, name='x3')
        x4 = AutoDiff.sin(x1) + AutoDiff.cos(x2) + AutoDiff.tan(x3)
        assert x4.val == 2**0.5 + 1
        assert x4.der == {'x1': 0.70710678118654757, 
            'x2': -0.70710678118654746,
            'x3': 1.9999999999999998}

    def test_hyberbolic(self):
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        x3 = Variable(1, name='x3')
        x4 = AutoDiff.sinh(x1) + AutoDiff.cosh(x2) + AutoDiff.tanh(x3)
        assert x4.val == 3.4798759844148099
        assert x4.der == {'x1': 1.5430806348152437, 
            'x2': 1.1752011936438014,
            'x3': 0.41997434161402614}

    def test_inverse_trigonometric(self):
        x1 = Variable(0.1, name='x1')
        x2 = Variable(0.2, name='x2')
        x3 = Variable(0.3, name='x3')
        x4 = AutoDiff.arcsin(x1) + AutoDiff.arccos(x2) + AutoDiff.arctan(x3)
        assert x4.val == 1.7610626216439926
        assert x4.der == {'x1': 1.005037815259212, 
            'x2':  -1.0206207261596576,
            'x3': 0.9174311926605504}