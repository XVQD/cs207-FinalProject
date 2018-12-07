import pytest
import numpy as np
import AutoDiff.AutoDiff as ad
import AutoDiff.GradDesc as gd
import warnings
warnings.simplefilter('error')

# a single variable function
def f1(values):
	x1 = ad.Variable(values['x1'], name='x1')
	f = 2 * (x1 ** 2) + 5
	return f

# a multi variable function
def f2(values):
	x1 = ad.Variable(values['x1'], name='x1')
	x2 = ad.Variable(values['x2'], name='x2')
	f = 2 * (x1 ** 2) + ad.sin(x2)
	return f

def test_single_var_result():
	# define a starting point
	x = {'x1':5}

	# varying gammas/step-sizes
	a = gd.grad_desc(f1, x, gamma = 0.001, message = False)
	assert (round(a['point']['x1'][0], 5) == 0.00248)
	assert (a['iters'] == 1898)

	b = gd.grad_desc(f1, x, gamma = 0.01, message = False)
	assert (round(b['point']['x1'][0], 5) == 0.00024)
	assert (b['iters'] == 244)

	c = gd.grad_desc(f1, x, gamma = 0.1, message = False)
	assert (round(c['point']['x1'][0]) == 0.0)
	assert (c['iters'] == 25)

	# this will keep going back between 5 and -5, since gradient is always 20
	d = gd.grad_desc(f1, x, gamma = 0.5, message = False)
	assert (d['point']['x1'][0] == 5.)
	assert (d['iters'] == 10000)

	# defining a different starting point
	x2 = {'x1':-20}

	# varying gammas/step-sizes
	a = gd.grad_desc(f1, x2, gamma = 0.001, message = False)
	assert (round(a['point']['x1'][0], 5) == -0.00248)
	assert (a['iters'] == 2244)

	b = gd.grad_desc(f1, x2, gamma = 0.01, message = False)
	assert (round(b['point']['x1'][0], 5) == -0.00024)
	assert (b['iters'] == 278)

	c = gd.grad_desc(f1, x2, gamma = 0.1, message = False)
	assert (round(c['point']['x1'][0]) == 0.0)
	assert (c['iters'] == 28)

	# this will keep going back between -20 and 20, since gradient is always 80
	d = gd.grad_desc(f1, x2, gamma = 0.5, message = False)
	assert (d['point']['x1'][0] == -20.)
	assert (d['iters'] == 10000)

	# one less iteration will make it positive
	d = gd.grad_desc(f1, x2, gamma = 0.5, message = False, max_iters = 9999)
	assert (d['point']['x1'][0] == 20.)
	assert (d['iters'] == 9999)

def test_single_var_exe():
	# define a starting point
	x = {'x1':5}

	# step size too large
	with pytest.raises(RuntimeWarning):
		a = gd.grad_desc(f1, x, gamma = 1, message = False)

	# define a starting point, but with different variable name
	x = {'x2':5}

	with pytest.raises(KeyError):
		a = gd.grad_desc(f1, x, gamma = 0.01, message = False)

def test_multi_var_result():
	# should have local minimum when x1 = 0 and when sin(x2) = -1
	# define a starting point
	x = {'x1':5, 'x2':6}

	# varying gammas/step-sizes
	a = gd.grad_desc(f2, x, gamma = 0.001, message = False)
	assert (round(a['point']['x1'][0], 5) == 0.0)
	assert (round(a['point']['x2'][0], 5) == 4.72238)
	assert (a['iters'] == 5010)

	b = gd.grad_desc(f2, x, gamma = 0.01, message = False)
	assert (round(b['point']['x1'][0], 5) == 0.0)
	assert (round(b['point']['x2'][0], 5) == 4.71338)
	assert (b['iters'] == 729)

	c = gd.grad_desc(f2, x, gamma = 0.1, message = False)
	assert (round(c['point']['x1'][0], 5) == 0.0)
	assert (round(c['point']['x2'][0], 5) == 4.71247)
	assert (c['iters'] == 93)

	assert(round(np.sin(a['point']['x2'][0])) == -1.0)
	assert(round(np.sin(b['point']['x2'][0])) == -1.0)
	assert(round(np.sin(c['point']['x2'][0])) == -1.0)

	# defining a different starting point
	x = {'x1':-23, 'x2':23}

	# varying gammas/step-sizes, should find closest x2 such that sin(x2) = -1
	a = gd.grad_desc(f2, x, gamma = 0.001, message = False)
	assert (round(a['point']['x1'][0], 5) == -0.0)
	assert (round(a['point']['x2'][0], 5) == 23.55196)
	assert (a['iters'] == 4055)

	b = gd.grad_desc(f2, x, gamma = 0.01, message = False)
	assert (round(b['point']['x1'][0], 5) == -0.0)
	assert (round(b['point']['x2'][0], 5) == 23.56096)
	assert (b['iters'] == 634)

	c = gd.grad_desc(f2, x, gamma = 0.1, message = False)
	assert (round(c['point']['x1'][0], 5) == -0.0)
	assert (round(c['point']['x2'][0], 5) == 23.56186)
	assert (c['iters'] == 84)

	assert(round(np.sin(a['point']['x2'][0])) == -1.0)
	assert(round(np.sin(b['point']['x2'][0])) == -1.0)
	assert(round(np.sin(c['point']['x2'][0])) == -1.0)
	
def test_multi_var_exe():
	# define a starting point
	x = {'x1':5, 'x2':6}

	# step size too large
	with pytest.raises(RuntimeWarning):
		a = gd.grad_desc(f2, x, gamma = 1, message = False)

	# define a starting point, but with different variable name
	x = {'x1':5, 'x3':6}

	with pytest.raises(KeyError):
		a = gd.grad_desc(f2, x, gamma = 0.01, message = False)
