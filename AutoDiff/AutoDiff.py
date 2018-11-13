# -*- coding: utf-8 -*-
import numpy as np

class Variable:
    """A Variable class which contains the variable's value and derivatives and has
    overloaded operators to interact with other Variables.
    
    Parameters (instantiating a Variable object)
    ==========
    val      : float or int
               the value at we wish to evaluate this variable
    name     : (optional) string
               name of the variable; suggested that the name should match the name of the Variable created
               If name is not supplied, then a new variable has not been created, but rather a combination
               of other variables has occurred
    der      : (optional) dict of str keys and float/int values
               str keys correspond to variable names, and values are floats/ints corresponding to the partial
               derivative with respect to the name key
               
    Output
    ==========
    Variable : object
               the instantiation of the class Variable which represents the variable desired. Contains
               Variable.val, which is the variable's current value, and Variable.der, which is the variable's
               current dictionary of partial derivatives

    Methods
    ==========
    Overloaded binary operators for addition, subtraction, multiplication, division, and exponentiation
    were implemented to allow for operations between Variables.
    
    Examples
    ==========
    >>> from AutoDiff import Variable
    >>> x1 = Variable(1, name='x1') # register independent variables by specifying names
    >>> x2 = x1 + 1
    >>> x3 = Variable(7, 'x3')
    >>> x4 =  x2 * x3
    >>> print(x4.val, x4.der)  # only calculate derivatives of registered(i.e.named) independent variables 
    14 {'x1': 7, 'x3': 2} 
    """

    def __init__(self, val, name=None , der=None, der2=None):
        """Initializes Variable with a value and a derivative."""
        self.val = val
        
        # if a name is supplied, then create a new variable with its own derivative
        if name!= None:
            self.der = {name:1} # the first derivative of a variable is 1
            self.der2 = {name:0} # the first derivative of a variable is 0
        else:
            self.der = der
            self.der2 = der2

    def __pos__(self):
        """Returns the Variable itself. Does nothing to value or derivative."""
        return Variable(self.val, der=self.der, der2=self.der2)

    def __neg__(self):
        """Returns a Variable with negated value and derivative."""
        # first order
        a=self.der
        der={x: -a.get(x, 0) for x in set(a)}
        # second order
        a2 = self.der2
        der2={x: -a2.get(x, 0) for x in set(a2)}
        return Variable(-self.val, der=der, der2 = der2)

    def __add__(self, other):
        """Returns a Variable that adds a Variable with another Variable, or with a constant."""
        try:
            a=self.der
            b=other.der
            a2=self.der2
            b2=other.der2
            # combine derivative dictionaries by adding them
            der={x: a.get(x, 0) + b.get(x, 0) for x in set(a).union(b)}
            der2={x: a2.get(x, 0) + b2.get(x, 0) for x in set(a2).union(b2)}
            return Variable(self.val+other.val, der=der, der2 = der2)
        # other is not a Variable
        except AttributeError:
            return Variable(self.val+other, der=self.der, der2 = self.der2)

    def __radd__(self, other):
        """Returns a Variable that adds a constant with a Variable."""
        # other is an constant otherwise other.__add__ is implemented
        return Variable(self.val+other, der=self.der, der2 = self.der2)
   
    def __sub__(self, other):
        """Returns a Variable that subtracts a Variable from another Variable, or with a constant."""
        try:
            a=self.der
            b=other.der
            a2=self.der2
            b2=other.der2
            # combine derivative dictionaries by subtracting corresponding values
            der={x: a.get(x, 0) - b.get(x, 0) for x in set(a).union(b)}
            der2={x: a2.get(x, 0) - b2.get(x, 0) for x in set(a2).union(b2)}
            return Variable(self.val-other.val, der=der, der2 = der2)
        # other is not a Variable
        except AttributeError:
            return Variable(self.val-other, der=self.der, der2=self.der2)

    def __rsub__(self, other):
        """Returns a Variable that subtracts a constant from a Variable."""
        # other is an constant otherwise other.__sub__ is implemented
        a=self.der
        a2 = self.der2
        der={x: -a.get(x, 0) for x in set(a)}
        der2={x: -a2.get(x, 0) for x in set(a2)}
        return Variable(other-self.val, der=der, der2 = der2)

    def __mul__(self, other):
        """Returns a Variable that mulitplies a Variable with another Variable, or with a constant."""
        a=self.der
        a2=self.der2
        try:
            b=other.der
            b2=other.der2
            # combine derivative dictionaries by multiplying one variable's value with the other's derivatives
            der={x: other.val * a.get(x, 0) + self.val * b.get(x, 0) for x in set(a).union(b)} 
            return Variable(self.val * other.val, der=der)
        # other is not a Variable
        except AttributeError:
            der={x: other * a.get(x, 0) for x in set(a)}
            return Variable(self.val * other, der=der)

    def __rmul__(self, other):
        """Returns a Variable that mulitplies a constant with a Variable."""
        # other is an constant otherwise other.__mul__ is implemented
        a=self.der
        a2 = self.der2
        der={x: other * a.get(x, 0) for x in set(a)}
        return Variable(self.val * other, der=der)

    def __truediv__(self, other):
        """Returns a Variable that divides a Variable from another Variable, or with a constant."""
        a=self.der
        a2 = self.der2
        try:          
            b=other.der
            b2 = other.der2
            der={x: 1/other.val * a.get(x, 0) - self.val/other.val**2 * b.get(x, 0) for x in set(a).union(b)} #combine dictionaries and do arithmatics
            return Variable(self.val / other.val, der=der)
        except AttributeError:
            der={x: a.get(x, 0) / other for x in set(a)} 
            return Variable(self.val / other, der=der)

    def __rtruediv__(self, other):
        """Returns a Variable that divides a constant from a Variable."""
        # other is an constant otherwise other.__itruediv__ is implemented
        a=self.der
        a2 = self.der2
        der={x: -other/self.val**2 * a.get(x, 0) for x in set(a)} 
        return Variable(other/self.val, der= der)

    def __pow__(self, other):
        """Returns a Variable that raises a Variable to another Variable, or with a constant."""
        a=self.der
        a2 = self.der2
        try:
            b=other.der
            b2 = other.der2
            der={x: other.val * self.val ** (other.val-1) * a.get(x, 0) 
                + np.log(self.val) * self.val ** other.val * b.get(x, 0) for x in set(a).union(b)} #combine dictionaries and do arithmatics
            return Variable(self.val ** other.val, der=der)
        except AttributeError:
            der={x: other*self.val**(other-1) * a.get(x, 0) for x in set(a)} 
            return Variable(self.val ** other, der= der)

    def __rpow__(self,other):
        """Returns a Variable that raises a constant to a Variable."""
        # other is an constant otherwise other.__pow__ is implemented
        a=self.der
        a2 = self.der2
        der={x: np.log(other) * other ** self.val * a.get(x, 0) for x in set(a)} 
        return Variable(other**self.val, der= der)
#y ** x and pow( y,x ) call x .__rpow__( y ), when y doesn’t have __pow__. There is no three-argument form in this case.

# ELEMENTARY FUNCTIONS

def exp(obj):
    """Returns a Variable that is e raised to that Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: np.exp(obj.val) * a.get(x, 0) for x in set(a)}
    val = np.exp(obj.val)
    return Variable(val, der = der)

def log(obj):
    """Returns the log (base e) of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: 1/obj.val * a.get(x, 0) for x in set(a)}
    val = np.log(obj.val)
    return Variable(val, der = der)

# TRIGONOMETRIC FUNCTIONS
def sin(obj):
    """Returns the sine of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: np.cos(obj.val) * a.get(x, 0) for x in set(a)}
    val = np.sin(obj.val)
    return Variable(val, der = der)

def cos(obj):
    """Returns the cosine of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: -np.sin(obj.val) * a.get(x, 0) for x in set(a)}
    val = np.cos(obj.val)
    return Variable(val, der = der)

def tan(obj):
    """Returns the tangent of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: (1+np.tan(obj.val)**2) * a.get(x, 0) for x in set(a)}
    val = np.tan(obj.val)
    return Variable(val, der = der)

# HYPERBOLIC FUNCTIONS
def sinh(obj):
    """Returns the hyperbolic sine of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: np.cosh(obj.val) * a.get(x, 0) for x in set(a)}
    val = np.sinh(obj.val)
    return Variable(val, der = der)

def cosh(obj):
    """Returns the hyperbolic cosine of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: np.sinh(obj.val) * a.get(x, 0) for x in set(a)}
    val = np.cosh(obj.val)
    return Variable(val, der = der)

def tanh(obj):
    """Returns the hyperbolic tangent of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: (1-np.tanh(obj.val)**2) * a.get(x, 0) for x in set(a)}
    val = np.tanh(obj.val)
    return Variable(val, der = der)

# Inverse trigonometric functions
def arcsin(obj):
    """Returns the inverse sine of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: (1-(obj.val)**2)**(-0.5) * a.get(x, 0) for x in set(a)}
    val = np.arcsin(obj.val)
    return Variable(val, der = der)

def arccos(obj):
    """Returns the inverse cosine of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: -(1-(obj.val)**2)**(-0.5) * a.get(x, 0) for x in set(a)}
    val = np.arccos(obj.val)
    return Variable(val, der = der)

def arctan(obj):
    """Returns the inverse tangent of the Variable."""
    a = obj.der
    a2 = obj.der2
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
