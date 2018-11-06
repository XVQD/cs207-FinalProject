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
    >>> from AutoDiff import Variable
    >>> x1 = Variable(1, name='x1') # register independent variables by specifying names
    >>> x2 = x1 + 1
    >>> x3 = Variable(7, 'x3')
    >>> x4 =  x2 * x3
    >>> print(x4.val, x4.der)  # only calculate derivatives of registered(i.e.named) independent variables 
    14 {'x1': 7, 'x3': 2} 
    """

    def __init__(self, val, name=None , der=None):
        self.val = val
        if name!= None:
            self.der = {name:1}
        else:
            self.der=der

    def __pos__(self):
        return Variable(self.val, der=self.der)

    def __neg__(self):
        a=self.der
        der={x: -a.get(x, 0) for x in set(a)}
        return Variable(-self.val, der=der)

    def __add__(self, other):
        try:
            a=self.der
            b=other.der
            der={x: a.get(x, 0) + b.get(x, 0) for x in set(a).union(b)} #combine dictionaries and do arithmatics
            return Variable(self.val+other.val, der=der)
        except AttributeError:
            return Variable(self.val+other, der=self.der)

    def __radd__(self, other):
        # other is an constant otherwise other.__add__ is implemented
        return Variable(self.val+other, der=self.der)
   
    def __sub__(self, other):
        try:
            a=self.der
            b=other.der
            der={x: a.get(x, 0) - b.get(x, 0) for x in set(a).union(b)} #combine dictionaries and do arithmatics
            return Variable(self.val-other.val, der=der)
        except AttributeError:
            return Variable(self.val-other, der=self.der)

    def __rsub__(self, other):
        # other is an constant otherwise other.__sub__ is implemented
        a=self.der
        der={x: -a.get(x, 0) for x in set(a)}
        return Variable(other-self.val, der=der)

    def __mul__(self, other):
        a=self.der
        try:
            b=other.der
            der={x: other.val * a.get(x, 0) + self.val * b.get(x, 0) for x in set(a).union(b)} #combine dictionaries and do arithmatics
            return Variable(self.val * other.val, der=der)
        except AttributeError:
            der={x: other * a.get(x, 0) for x in set(a)}
            return Variable(self.val * other, der=der)

    def __rmul__(self, other):
        # other is an constant otherwise other.__mul__ is implemented
        a=self.der
        der={x: other * a.get(x, 0) for x in set(a)}
        return Variable(self.val * other, der=der)

    def __truediv__(self, other):
        a=self.der
        try:          
            b=other.der
            der={x: 1/other.val * a.get(x, 0) - self.val/other.val**2 * b.get(x, 0) for x in set(a).union(b)} #combine dictionaries and do arithmatics
            return Variable(self.val / other.val, der=der)
        except AttributeError:
            der={x: a.get(x, 0) / other for x in set(a)} 
            return Variable(self.val / other, der=der)

    def __rtruediv__(self, other):
        # other is an constant otherwise other.__itruediv__ is implemented
        a=self.der
        der={x: -other/self.val**2 * a.get(x, 0) for x in set(a)} 
        return Variable(other/self.val, der= der)

    def __pow__(self, other):
        a=self.der
        try:
            b=other.der
            der={x: other.val * self.val ** (other.val-1) * a.get(x, 0) 
                + np.log(self.val) * self.val ** other.val * b.get(x, 0) for x in set(a).union(b)} #combine dictionaries and do arithmatics
            return Variable(self.val ** other.val, der=der)
        except AttributeError:
            der={x: other*self.val**(other-1) * a.get(x, 0) for x in set(a)} 
            return Variable(self.val ** other, der= der)

    def __rpow__(self,other):
        # other is an constant otherwise other.__pow__ is implemented
        a=self.der
        der={x: np.log(other) * other ** self.val * a.get(x, 0) for x in set(a)} 
        return Variable(other**self.val, der= der)
#y ** x and pow( y,x ) call x .__rpow__( y ), when y doesn’t have __pow__. There is no three-argument form in this case.

# elementary functions

def exp(obj):
    a = obj.der
    der = {x: np.exp(obj.val) * a.get(x, 0) for x in set(a)}
    val = np.exp(obj.val)
    return Variable(val, der = der)

def log(obj):
    a = obj.der
    der = {x: 1/obj.val * a.get(x, 0) for x in set(a)}
    val = np.log(obj.val)
    return Variable(val, der = der)

# trigonometric functions
def sin(obj):
    a = obj.der
    der = {x: np.cos(obj.val) * a.get(x, 0) for x in set(a)}
    val = np.sin(obj.val)
    return Variable(val, der = der)

def cos(obj):
    a = obj.der
    der = {x: -np.sin(obj.val) * a.get(x, 0) for x in set(a)}
    val = np.cos(obj.val)
    return Variable(val, der = der)

def tan(obj):
    a = obj.der
    der = {x: (1+np.tan(obj.val)**2) * a.get(x, 0) for x in set(a)}
    val = np.tan(obj.val)
    return Variable(val, der = der)

# hyperbolic functions
def sinh(obj):
    a = obj.der
    der = {x: np.cosh(obj.val) * a.get(x, 0) for x in set(a)}
    val = np.sinh(obj.val)
    return Variable(val, der = der)

def cosh(obj):
    a = obj.der
    der = {x: np.sinh(obj.val) * a.get(x, 0) for x in set(a)}
    val = np.cosh(obj.val)
    return Variable(val, der = der)

def tanh(obj):
    a = obj.der
    der = {x: (1-np.tanh(obj.val)**2) * a.get(x, 0) for x in set(a)}
    val = np.tanh(obj.val)
    return Variable(val, der = der)

# Inverse trigonometric functions
def arcsin(obj):
    a = obj.der
    der = {x: (1-(obj.val)**2)**(-0.5) * a.get(x, 0) for x in set(a)}
    val = np.arcsin(obj.val)
    return Variable(val, der = der)

def arccos(obj):
    a = obj.der
    der = {x: -(1-(obj.val)**2)**(-0.5) * a.get(x, 0) for x in set(a)}
    val = np.arccos(obj.val)
    return Variable(val, der = der)

def arctan(obj):
    a = obj.der
    der = {x: 1/(1+(obj.val)**2) * a.get(x, 0) for x in set(a)}
    val = np.arctan(obj.val)
    return Variable(val, der = der)

# if __name__=="__main__":
#     #test
#     x1 = Variable(1, name='x1') # register independent variables by specifying names
#     print(x1.val, x1.der) 
#     x2 = x1 + 1
#     print(x2.val, x2.der) 
#     x3 = Variable(7, 'x3')
#     x4 = x2+x3
#     print(x4.val, x4.der) 
###### output ######
#1 {'x1': 1}
#2 {'x1': 1}
#9 {'x1': 1, 'x3': 1}
