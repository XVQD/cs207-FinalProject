# -*- coding: utf-8 -*-
import numpy as np

class Variable:
    
    """Auto differentation class with overloaded operators
    
    Parameters
    ==========
    input_val (np.array): values to evaluate the expression at
    init_der (np.array): initial derivatives
    
    Methods
    =======
    overloaded operators
    
    Examples
    ========
    >>> from Autodiff import Variable
    >>> x1 = Variable(1, name='x1') # register independent variables by specifying names
    >>> x2 = x1 + 1
    >>> x3 = Variable(7, 'x3')
    >>> x4 =  x2 * x3
    >>> print(x4.val, v4.der)  # only calculate derivatives of registered(i.e.named) independent variables 
    14 {'x1': 7, 'x3': 2} 
    """

    def __init__(self, val, name=None , der=None):
        self.val = val
        if name!= None:
            self.der = {name:1}
        else:
            self.der=der

    def __add__(self, other):
        try:
            a=self.der
            b=other.der
            der={x: a.get(x, 0) + b.get(x, 0) for x in set(a).union(b)} #combine dictionaries and do arithmatics
            return Variable(self.val+other.val, der=der)
        except AttributeError:
            return Variable(self.val+other, der=self.der)

    def __radd__(self, other):
        try:
            a=self.der
            b=other.der
            der={x: a.get(x, 0) + b.get(x, 0) for x in set(a).union(b)} #combine dictionaries and do arithmatics
            return Variable(self.val+other.val, der=der)
        except AttributeError:
            return Variable(self.val+other, der=self.der)

    
    def __sub__(self, other):
        raise NotImplemented 
    def __rsub__(self, other):
        raise NotImplemented 
    def __mul__(self, other):
        raise NotImplemented         
    def __rmul__(self, other):
        raise NotImplemented 
    def __pow__(self, other):
        raise NotImplemented 
    def __rpow__(self,other):
        raise NotImplemented 
#y ** x and pow( y,x ) call x .__rpow__( y ), when y doesn’t have __pow__. There is no three-argument form in this case.

if __name__=="__main__":
    #test
    x1 = Variable(1, name='x1') # register independent variables by specifying names
    print(x1.val, x1.der) 
    x2 = x1 + 1
    print(x2.val, x2.der) 
    x3 = Variable(7, 'x3')
    x4 = x2+x3
    print(x4.val, x4.der) 
###### output #####
#1 {'x1': 1}
#2 {'x1': 1}
#9 {'x1': 1, 'x3': 1}
