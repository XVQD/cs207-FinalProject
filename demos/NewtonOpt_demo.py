import pytest
import numpy as np
import AutoDiff.AutoDiff as ad 
from AutoDiff.NewtonOpt import NewtonOpt 
def f(value):
    """
     f takes in a dictionary of independent variables
    """
    X=ad.Variable(value['x'],name='x')
    Y=ad.Variable(value['y'],name='y')
    Z1 = ad.exp(-X**2 - Y**2)
    Z2 = ad.exp(-(X - 1)**2 - (Y - 1)**2)
    return (Z1 - Z2) * 2
initp ={'x':.8,'y':1.4}
result= NewtonOpt(f,initp,message=False)
print(result)