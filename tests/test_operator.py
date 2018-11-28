import pytest
import numpy as np
from AutoDiff.AutoDiff import Variable

class Test_Operator():

    def test_add(self):
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        x3 = +x1 + x2 # also test __pos__
        x4 = -x1 + 5 # also test __neg__
        x5 = 5 + x1
        assert x3.val == 2
        assert x3.der == {'x1': 1, 'x2': 1}
        assert x4.val == 4
        assert x4.der == {'x1': -1}
        assert x5.val == 6
        assert x5.der == {'x1': 1}

    def test_sub(self):
        x1 = Variable(4, name='x1')
        x2 = Variable(1, name='x2')
        x3 = x1 - x2
        x4 = x2 - x1
        x5 = 4 - x1
        x6 = x1 - 4
        assert x3.val == 3
        assert x3.der == {'x1': 1, 'x2': -1}
        assert x4.val == -3
        assert x4.der == {'x1': -1, 'x2': 1}
        assert x5.val == 0
        assert x5.der == {'x1': -1}
        assert x6.val == 0
        assert x6.der == {'x1': 1}

    def test_mul(self):
        x1 = Variable(4, name='x1')
        x2 = Variable(1, name='x2')
        x3 = 3*x1*x2*3
        assert x3.val == 36
        assert x3.der == {'x1': 9, 'x2': 36}

    def test_div(self):
        x1 = Variable(1, name='x1')
        x2 = Variable(4, name='x2')
        x3 = x1/x2
        x4 = x1/3
        x5 = 3/x1
        assert x3.val == 1/4
        assert x3.der == {'x1': 1/4, 'x2': -1/16}
        assert x4.val == 1/3
        assert x4.der == {'x1': 1/3}  
        assert x5.val == 3
        assert x5.der == {'x1': -3}

    def test_pow(self):
        x1 = Variable(3, name='x1')
        x2 = Variable(2, name='x2')
        x3 = x1**x2
        x4 = x1**3 # test the case where x2 is a constant
        x5 = 3**x1 # test __rpow__
        assert x3.val == 9
        assert x3.der == {'x1': 6, 'x2': 9*np.log(3)}
        assert x4.val == 27
        assert x4.der == {'x1': 27}  
        assert x5.val == 27
        assert x5.der == {'x1': np.log(3)*3**3}

    def test_print(self):
        x1 = Variable(3, name='x1')
        print(x1)
        assert  1 == 1
