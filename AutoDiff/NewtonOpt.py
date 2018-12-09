import AutoDiff as ad
import numpy as np

def NewtonOpt(f, init, precision = 0.00001, max_iters = 10000, message=True):
	'''Performs the Newton's optimization on a function or vector function of scalars.
	INPUT
	=========
	f         : function
	            the objective function that returns AutmoDiff.Variable class;
	            has 1 argument -- a list of numbers that correspond to values at which
	            variable are evaluated
	init      : dictionary
			    variable string key:int/float value pair that represents the variables
			    at which they are evaluated.
			    assumes length matches the number of unique variables in f

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

	curr_point = init.copy()#np.array(init.values())

	prev_diff = float('inf')

	while prev_diff > precision and iters < max_iters:

		# makes a copy of the current point as previous point
		prev_point = curr_point

		# find the partial derivatives evaluated at current point
		f2x= f(prev_point)
		gradf =[]
		for var in init.keys():
				gradf.append(f2x.der[var])
		gradf=np.array(gradf).flatten()
		
		# initializing list of differences between new point and old point

		hess=f2x.hessian(init.keys())
		#print(hess,curr_point)
		        
		sk=np.linalg.solve(hess,-gradf).reshape(-1)
		# update each coordinate and append difference
		#print(curr_point+sk)
		ii=0
		for var in init.keys():
           # print(curr_point)
			curr_point[var] = curr_point[var] + sk[ii]
			ii+=1
        
		# finding the magnitude of the vectorized difference
		prev_diff = np.linalg.norm(sk)

		iters += 1

	if message:
		print('Number of iterations: {}'.format(iters))
		print('The local minimum occurs at {}'.format(curr_point))

	return {'point':curr_point, 'iters':iters}