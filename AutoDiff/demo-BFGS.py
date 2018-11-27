import AutoDiff as ad
import numpy as np

# class Vector:
#     def __init__(alist,name=None):
#         if name: 
#             x=[ad.Variable(v,name=name+'_'+str(i)) for (i,v) in enumerate(alist)]
#         else:
#             x=[ad.Variable(v) for v in enumerate(alist)]


# if name

def f(x,y):
    """
    f takes in a len-2 list or array and outputs the Rosenbrock function value
    """
    
#     x=np.array([1,0])
    x=ad.Variable(x,name='x')
    y=ad.Variable(y,name='y')
    return 100.*(y-x**2)**2 + (1-x)**2 
print(f(1,2))

# f=lambda x: 100.*(x[1]-x[0]**2)**2 + (1-x[0])**2 