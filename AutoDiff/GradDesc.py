import AutoDiff as ad
import numpy as np

def grad_desc(f, point, gamma, precision = 0.00001, max_iters = 10000, message=True):
	'''Performs the gradient descent on a function or vector function of scalars.

	INPUT
	=========
	f         : function
	            the objective function that returns AutoDiff.Variable class;
	            has 1 argument -- a list of numbers that correspond to values at which
	            variable are evaluated
	point     : dictionary
			    variable string key:int/float value pair that represents the variables
			    at which they are evaluated.
			    assumes length matches the number of unique variables in f
	gamma     : int or float
	            step-size/learning rate
	precision : int or float
	            numerical threshold at which to stop the descent (default 0.00001)
	max_iters : int
	            maximum amount of iterations of descent 
	            if precision not met (default 10000)
	message   : Boolean
	            Prints summary of number of iterations and the point at which there is local
	            optimum (default True)

	OUTPUT
	=========
	point    : dictionary
	           two keys: 'point', whose value is the local optimum, and 'iters', whose value
	           is the number of iterations of the gradient descent.

	WARNINGS
	=========
	Will raise a RuntimeWarning if gamma (step-size) too large and causes overflow.

	EXAMPLES
	=========
	>>> import AutoDiff.AutoDiff as ad
	>>> import AutoDiff.GradDesc as gd
	>>> def f(values):
	...     x1 = ad.Variable(values['x1'], name='x1')
	...     f = 5 * (x1 ** 2)
	...     return f
	>>> x = {'x1':5}
	>>> a = gd.grad_desc(f, x, gamma = 0.01, message = False)
	>>> print(a['point'], a['iters'])
	{'x1': array([8.71346691e-05])} 104 # really near 0
	>>> def g(values):
	...     x1 = ad.Variable(values['x1'], name='x1')
	...     x2 = ad.Variable(values['x2'], name='x2')
	...     f = 2 * (x1 ** 2) + 5 * (x2 ** 2)
	...     return f # multi variable function
    >>> xm = {'x1':0.001, 'x2':-0.1}
    >>> b = gd.grad_desc(g, xm, gamma = 0.01, message = False)
    >>> print(b['point'], b['iters'])
    {'x1': array([6.48892289e-05]), 'x2': array([-8.59504456e-05])} 67
	'''

	iters = 0

	curr_point = point.copy()

	prev_diff = float('inf')

	while prev_diff > precision and iters < max_iters:

		# makes a copy of the current point as previous point
		prev_point = curr_point.copy()

		# find the partial derivatives evaluated at current point
		der_vals = f(prev_point).der

		# initializing list of differences between new point and old point
		diffs = []

		# update each coordinate and append difference
		for var in der_vals:
			curr_point[var] = curr_point[var] - (gamma * der_vals[var])
			diffs.append(curr_point[var] - prev_point[var])

		# finding the magnitude of the vectorized difference
		prev_diff = np.linalg.norm(np.array(diffs))

		iters += 1

	if message:
		print('Number of iterations: {}'.format(iters))
		print('The local minimum occurs at {}'.format(curr_point))

	return {'point':curr_point, 'iters':iters}
	

