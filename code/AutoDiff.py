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
        return Variable(other-self.val, der=self.der)

    def __mul__(self, other):
        try:
            a=self.der
            b=other.der
            der={x: other.val * a.get(x, 0) + self.val * b.get(x, 0) for x in set(a).union(b)} #combine dictionaries and do arithmatics
            return Variable(self.val * other.val, der=der)
        except AttributeError:
            return Variable(self.val * other, der=self.der * other)

    def __rmul__(self, other):
        # other is an constant otherwise other.__mul__ is implemented
        return Variable(self.val * other, der=self.der * other)

    def __pow__(self, other):
        try:
            a=self.der
            b=other.der
            der={x: other.val * self.val ** (other.val-1) * a.get(x, 0) 
                + np.log(self.val) * self.val ** other.val * b.get(x, 0) for x in set(a).union(b)} #combine dictionaries and do arithmatics
            return Variable(self.val ** other.val, der=der)
        except AttributeError:
            return Variable(self.val ** other, der=other*self.val**(other-1) * self.der)

    def __rpow__(self,other):
        # other is an constant otherwise other.__pow__ is implemented
        return Variable(other**self.val, der=np.log(other) * other ** self.val * self.der)
#y ** x and pow( y,x ) call x .__rpow__( y ), when y doesn’t have __pow__. There is no three-argument form in this case.

# elementary functions

def exp(obj):
    return Variable(np.exp(obj.val), np.exp(obj.val) * obj.der)

# trigonometric functions
def sin(obj):
    return Variable(np.sin(obj.val), np.cos(obj.val) * obj.der)

def cos(obj):
    return Variable(np.cos(obj.val), -np.sin(obj.val) * obj.der)

# hyperbolic functions
def sinh(obj):
    return Variable(np.sinh(obj.val), np.cosh(obj.val) * obj.der)

def cosh(obj):
    return Variable(np.cosh(obj.val), np.sinh(obj.val) * obj.der)

def tanh(obj):
    return Variable(np.tanh(obj.val), (1-np.tanh(obj.val)**2) * obj.der)

# Inverse trigonometric functions
def arcsin(obj):
    return Variable(np.arcsin(obj.val), (1-(obj.val)**2)**(-0.5) * obj.der)

def arccos(obj):
    return Variable(np.arccos(obj.val), -(1-(obj.val)**2)**(-0.5) * obj.der)

def arctan(obj):
    return Variable(np.arctan(x_new.val), 1/(1+(obj.val)**2) * obj.der)


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
