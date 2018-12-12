import AutoDiff.AutoDiff as ad
from AutoDiff.GMRes import gmres_autodiff
import numpy as np


# define variables and vector of vector functions
x1 = ad.Variable(1, name='x1')
x2 = ad.Variable(1, name='x2')
x3 = ad.Variable(1, name='x3')
f1 = 2*x1+3*x2+2*x3
f2 = 3*x1+2*x2+1*x3
f3 = 3*x1+3*x2+3*x3
F = [f1, f2, f3]
b = [1, 2, 3]

# GMRES to get x
x = gmres_autodiff(F, b)
print(x)