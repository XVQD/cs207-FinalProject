# This is a demo file for using GradDesc.py
import AutoDiff.AutoDiff as ad
import AutoDiff.GradDesc as gd 
import numpy as np

# define objective function that matches dictionary keys of objective function
def f(values):
    x1 = ad.Variable(values['x1'], name='x1')
    x2 = ad.Variable(values['x2'], name='x2')
    f = 2 * (x1 ** 2) + ad.sin(x2)
    return f

# define initial starting point
x = {'x1':-23, 'x2':23}

# store gradient descent results
grad_desc_f = gd.grad_desc(f = f, init = x, gamma = 0.001, message = False)

# can call dictionary keys for number of iterations and point at which gradient descent stops
print (grad_desc_f['iters'])
print (grad_desc_f['point'])