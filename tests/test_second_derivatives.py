import pytest
import numpy as np
from AutoDiff.AutoDiff import Variable
from AutoDiff import AutoDiff

class Test_second_derivatives():

    def test_add_sub(self):
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        x = AutoDiff.sin(x1)
        y = AutoDiff.sin(x2)
        z1 = x+y
        z2 = x-y
        z3 = +x
        z4 = -x
        assert z1.der2 == {'x1x1': np.array([-0.8414709848078965]), 
            'x2x2': np.array([-0.8414709848078965]),
            'x1x2':np.array([0.0]),'x2x1':np.array([0.0])}
        assert z2.der2 == {'x1x1': np.array([-0.8414709848078965]), 
            'x2x2': np.array([0.8414709848078965]),
            'x1x2': np.array([0.0]),'x2x1':np.array([0.0])}
        assert z3.der2 == {'x1x1': np.array([-0.8414709848078965])}
        assert z4.der2 == {'x1x1': np.array([0.8414709848078965])}

    def test_mul_div(self):
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        x = AutoDiff.sin(x1)
        y = AutoDiff.sin(x2)
        z1 = x*y
        z2 = x/y
        z3 = 1/x
        z4 = x/1
        assert z1.der2['x1x1'][0] == -0.70807341827357118
        assert z1.der2['x1x2'][0] == 0.29192658172642888
        assert z1.der2['x2x1'][0] == 0.29192658172642888
        assert z1.der2['x2x2'][0] == -0.70807341827357118

        assert z2.der2['x1x1'][0] == -1.0
        assert z2.der2['x1x2'][0] == -0.41228292743739203
        assert z2.der2['x2x1'][0] == -0.41228292743739203
        assert z2.der2['x2x2'][0] == 1.8245658548747841

        assert z3.der2['x1x1'][0] == 2.1683051321030673
        assert z4.der2['x1x1'][0] == -0.8414709848078965

    def test_pow(self):
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        x = AutoDiff.sin(x1)
        y = AutoDiff.sin(x2)
        z1 = x**y
        z2 = 2**x
        z3 = x**2

        assert z1.der2['x1x1'][0] == -0.77527835938046719
        assert z1.der2['x1x2'][0] == 0.25644885963975589
        assert z1.der2['x2x1'][0] == 0.25644885963975589
        assert z1.der2['x2x2'][0] == 0.13312782635352052

        assert z2.der2['x1x1'][0] == -0.79381233880392665
        assert z3.der2['x1x1'][0] == -0.83229367309428459

    def test_exp_log(self):
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        x = AutoDiff.exp(x1)
        y = AutoDiff.log(x2)
        z1 = x*y

        assert z1.der2['x1x1'][0] == 0.
        assert z1.der2['x1x2'][0] == 2.7182818284590451
        assert z1.der2['x2x1'][0] == 2.7182818284590451
        assert z1.der2['x2x2'][0] == -2.7182818284590451

    def test_trigonometric(self):
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        x = AutoDiff.sin(x1)
        y = AutoDiff.cos(x2)
        z = AutoDiff.tan(x2)
        z1 = x*y+z
        assert z1.der2['x1x1'][0] == -0.45464871341284091
        assert z1.der2['x1x2'][0] == -0.45464871341284091
        assert z1.der2['x2x1'][0] == -0.45464871341284091
        assert z1.der2['x2x2'][0] == 10.215210231562478

    def test_hyperbolics(self):
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        x = AutoDiff.sinh(x1)
        y = AutoDiff.cosh(x2)
        z = AutoDiff.tanh(x2)
        z1 = x*y+z
        assert z1.der2['x1x1'][0] == 1.8134302039235093
        assert z1.der2['x1x2'][0] == 1.8134302039235093
        assert z1.der2['x2x1'][0] == 1.8134302039235093
        assert z1.der2['x2x2'][0] == 1.1737301954742847

    def test_inverse_trigonometric(self):
        x1 = Variable(0.5, name='x1')
        x2 = Variable(0.6, name='x2')
        x = AutoDiff.arcsin(x1)
        y = AutoDiff.arccos(x2)
        z = AutoDiff.arctan(x2)
        z1 = x*y+z
        assert z1.der2['x1x1'][0] == 0.71383219164197809
        assert z1.der2['x1x2'][0] == -1.4433756729740643
        assert z1.der2['x2x1'][0] == -1.4433756729740643
        assert z1.der2['x2x2'][0] == -1.2623812424898966     
