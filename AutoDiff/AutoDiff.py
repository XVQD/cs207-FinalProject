# -*- coding: utf-8 -*-
import numpy as np
import copy

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

        self.name=name
        self.val = np.array(val).astype(float)
        try:
            lenn=len(self.val)
        except:
            lenn=1
        # if a name is supplied, then create a new variable with its own derivative
        if name!= None:
            self.der = {name: np.ones(lenn)} # the first derivative of a variable is 1
            self.der2 = {name+name: np.zeros(lenn)} # the first derivative of a variable is 0
        else:
            self.der = der
            self.der2 = der2
    def __str__(self):
#         if self.name==None:
#             return "ad.Variable(val={})".format(self.val )
#         else:
        return "ad.Variable(val={},\n name='{}', \n der={}, \n der2={})".format(self.val,self.name,self.der,self.der2 )
    def __pos__(self):
        """Returns the Variable itself. Does nothing to value or derivative."""
        return Variable(self.val, der=self.der, der2=self.der2)

    def __neg__(self):
        """Returns a Variable with negated value and derivative."""
        # first order
        a=copy.deepcopy(self.der)
        a2=copy.deepcopy(self.der2)
        der={x: -a.get(x, 0) for x in set(a)}
        # second order
        der2={x: -a2.get(x, 0) for x in set(a2)}
        return Variable(-self.val, der=der, der2 = der2)

    def __add__(self, other):
        """Returns a Variable that adds a Variable with another Variable, or with a constant."""
        try:
            a=copy.deepcopy(self.der)
            a2=copy.deepcopy(self.der2)
            b=copy.deepcopy(other.der)
            b2=copy.deepcopy(other.der2)
            # combine derivative dictionaries by adding them
            der={x: a.get(x, 0) + b.get(x, 0) for x in set(a).union(b)}
            a2,b2 = self._expand(a,b,a2,b2)
            der2={x: a2.get(x, 0) + b2.get(x, 0) for x in set(a2).union(b2)}
            print(der2)
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
            a=copy.deepcopy(self.der)
            a2=copy.deepcopy(self.der2)
            b=copy.deepcopy(other.der)
            b2=copy.deepcopy(other.der2)
            # combine derivative dictionaries by subtracting corresponding values
            der={x: a.get(x, 0) - b.get(x, 0) for x in set(a).union(b)}
            a2,b2 = self._expand(a,b,a2,b2)
            der2={x: a2.get(x, 0) - b2.get(x, 0) for x in set(a2).union(b2)}
            return Variable(self.val-other.val, der=der, der2 = der2)
        # other is not a Variable
        except AttributeError:
            return Variable(self.val-other, der=self.der, der2=self.der2)

    def __rsub__(self, other):
        """Returns a Variable that subtracts a constant from a Variable."""
        # other is an constant otherwise other.__sub__ is implemented
        a=copy.deepcopy(self.der)
        a2=copy.deepcopy(self.der2)
        der={x: -a.get(x, 0) for x in set(a)}
        der2={x: -a2.get(x, 0) for x in set(a2)}
        return Variable(other-self.val, der=der, der2 = der2)

    def __mul__(self, other):
        """Returns a Variable that mulitplies a Variable with another Variable, or with a constant."""
        a=copy.deepcopy(self.der)
        a2=copy.deepcopy(self.der2)
        try:
            b=copy.deepcopy(other.der)
            b2=copy.deepcopy(other.der2)
            # combine derivative dictionaries by multiplying one variable's value with the other's derivatives
            der={x: other.val * a.get(x, 0) + self.val * b.get(x, 0) for x in set(a).union(b)} 
            a2, b2 = self._expand(a, b, a2, b2)
            der2 = {}
            for x in set(a).union(b):
                for y in set(a).union(b):
                    der2[x+y] = (other.val * a2.get(x+y,0) + self.val * b2.get(x+y,0) 
                            + a.get(x,0)*b.get(y,0) + a.get(y,0)*b.get(x,0))
            return Variable(self.val * other.val, der=der, der2 = der2)
        # other is not a Variable
        except AttributeError:
            der={x: other * a.get(x, 0) for x in set(a)}
            der2={x: other * a2.get(x, 0) for x in set(a2)}
            return Variable(self.val * other, der=der, der2 = der2)

    def __rmul__(self, other):
        """Returns a Variable that mulitplies a constant with a Variable."""
        # other is an constant otherwise other.__mul__ is implemented
        a=self.der
        a2 = self.der2
        der={x: other * a.get(x, 0) for x in set(a)}
        der2={x: other * a2.get(x, 0) for x in set(a2)}
        return Variable(self.val * other, der=der, der2 = der2)

    def __truediv__(self, other):
        """Returns a Variable that divides a Variable from another Variable, or with a constant."""
        a=copy.deepcopy(self.der)
        a2=copy.deepcopy(self.der2)
        try:
            b=copy.deepcopy(other.der)
            b2=copy.deepcopy(other.der2)
            der={x: 1/other.val * a.get(x, 0) - self.val/other.val**2 * b.get(x, 0) for x in set(a).union(b)} #combine dictionaries and do arithmatics
            a2,b2 = self._expand(a,b,a2,b2)
            der2 = {}
            for x in set(a).union(b):
                for y in set(a).union(b):
                    der2[x+y] = (1/other.val*a2.get(x+y,0) - self.val/other.val**2*b2.get(x+y,0) 
                            +2*self.val/other.val**3*b.get(x,0)*b.get(y,0)
                            -1/other.val**2*a.get(x,0)*b.get(y,0)
                            -1/other.val**2*a.get(y,0)*b.get(x,0))
            return Variable(self.val / other.val, der=der, der2 = der2)
        except AttributeError:
            der={x: a.get(x, 0) / other for x in set(a)} 
            der2={x: a2.get(x, 0) / other for x in set(a2)} 
            return Variable(self.val / other, der=der, der2 = der2)

    def __rtruediv__(self, other):
        """Returns a Variable that divides a constant from a Variable."""
        # other is an constant otherwise other.__itruediv__ is implemented
        a=copy.deepcopy(self.der)
        a2=copy.deepcopy(self.der2)
        der={x: -other/self.val**2 * a.get(x, 0) for x in set(a)} 
        der2 = {}
        for x in set(a):
            for y in set(a):
                der2[x+y] = -other/self.val**2 * a2.get(x+y,0) + 2*other/self.val**3*a.get(x,0)*a.get(y,0)
        return Variable(other/self.val, der= der, der2 = der2)

    def __pow__(self, other):
        """Returns a Variable that raises a Variable to another Variable, or with a constant."""
        a=copy.deepcopy(self.der)
        a2=copy.deepcopy(self.der2)
        try:
            b=copy.deepcopy(other.der)
            b2=copy.deepcopy(other.der2)
            der={x: other.val * self.val ** (other.val-1) * a.get(x, 0) 
                + np.log(self.val) * self.val ** other.val * b.get(x, 0) for x in set(a).union(b)} #combine dictionaries and do arithmatics
            a2,b2 = self._expand(a,b,a2,b2)
            der2 = {}
            for x in set(a).union(b):
                for y in set(a).union(b):
                    der2[x+y] = (other.val * self.val ** (other.val-1) * a2.get(x+y, 0) 
                        + np.log(self.val) * self.val ** other.val * b2.get(x+y, 0)
                        + a.get(x,0)*a.get(y,0)*(self.val**(other.val-2)*(other.val**2-other.val))
                        + a.get(x,0)*b.get(y,0)*(self.val**(other.val-1)+np.log(self.val)*self.val**(other.val-1)*other.val)
                        + a.get(y,0)*b.get(x,0)*(self.val**(other.val-1)+np.log(self.val)*self.val**(other.val-1)*other.val)
                        + (np.log(self.val))**2*self.val**other.val*b.get(x,0)*b.get(y,0))
            return Variable(self.val ** other.val, der=der, der2 = der2)
        except AttributeError:
            der={x: other*self.val**(other-1) * a.get(x, 0) for x in set(a)} 
            der2 = {}
            for x in set(a):
                for y in set(a):
                    der2[x+y] = other*self.val**(other-2)*((other-1)*a.get(x, 0)*a.get(y,0) + self.val*a2.get(x+y, 0))
            return Variable(self.val ** other, der= der, der2 = der2)

    def __rpow__(self,other):
        """Returns a Variable that raises a constant to a Variable."""
        # other is an constant otherwise other.__pow__ is implemented
        a=copy.deepcopy(self.der)
        a2=copy.deepcopy(self.der2)
        der={x: np.log(other) * other ** self.val * a.get(x, 0) for x in set(a)} 
        der2 = {}
        for x in set(a):
            for y in set(a):
                der2[x+y] = other**self.val*np.log(other)*(np.log(other)*a.get(x, 0)*a.get(y,0)+a2.get(x+y, 0))
        return Variable(other**self.val, der= der, der2 = der2)
#y ** x and pow( y,x ) call x .__rpow__( y ), when y doesn’t have __pow__. There is no three-argument form in this case.

    def _expand(self,a,b,a2,b2):
        for x in set(a).union(b):
            for y in set(a).union(b):
                try:
                    temp = a2[x+y]
                except KeyError:
                    a2[x+y] = np.array([0.0])
                try:
                    temp = b2[x+y]
                except KeyError:
                    b2[x+y] = np.array([0.0])
        return a2, b2


# ELEMENTARY FUNCTIONS

def exp(obj):
    """Returns a Variable that is e raised to that Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: np.exp(obj.val) * a.get(x, 0) for x in set(a)}
    der2 = {x: np.exp(obj.val) * (a.get(x, 0)**2+a2.get(x,0)) for x in set(a2)}
    der2 = {}
    for x in set(a):
        for y in set(a):
            der2[x+y] = np.exp(obj.val)*(a.get(x, 0)*a.get(y, 0)+a2.get(x+y,0))
    val = np.exp(obj.val)
    return Variable(val, der = der, der2 = der2)

def log(obj):
    """Returns the log (base e) of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: 1/obj.val * a.get(x, 0) for x in set(a)}
    der2 = {}
    for x in set(a):
        for y in set(a):
            der2[x+y] = (-a.get(x, 0)*a.get(y, 0)+obj.val*a2.get(x, 0))/obj.val**2
    val = np.log(obj.val)
    return Variable(val, der = der, der2 = der2)

# TRIGONOMETRIC FUNCTIONS
def sin(obj):
    """Returns the sine of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: np.cos(obj.val) * a.get(x, 0) for x in set(a)}
    der2 = {}
    for x in set(a):
        for y in set(a):
            der2[x+y] = np.cos(obj.val)*a2.get(x+y, 0)-np.sin(obj.val)*a.get(x,0)*a.get(y,0)
    val = np.sin(obj.val)
    return Variable(val, der = der, der2 = der2)

def cos(obj):
    """Returns the cosine of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: -np.sin(obj.val) * a.get(x, 0) for x in set(a)}
    der2 = {}
    for x in set(a):
        for y in set(a):
            der2[x+y] = -np.sin(obj.val)*a2.get(x, 0) - np.cos(obj.val)*a.get(x,0)*a.get(y,0)
    val = np.cos(obj.val)
    return Variable(val, der = der, der2 = der2)

def tan(obj):
    """Returns the tangent of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: (1+np.tan(obj.val)**2) * a.get(x, 0) for x in set(a)}
    der2 = {}
    for x in set(a):
        for y in set(a):
            der2[x+y] = (1+np.tan(obj.val)**2)*(a2.get(x, 0)+2*np.tan(obj.val)*a.get(x,0)*a.get(y,0))
    val = np.tan(obj.val)
    return Variable(val, der = der, der2 = der2)

# HYPERBOLIC FUNCTIONS
def sinh(obj):
    """Returns the hyperbolic sine of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: np.cosh(obj.val) * a.get(x, 0) for x in set(a)}
    der2 = {}
    for x in set(a):
        for y in set(a):
            der2[x+y] = np.cosh(obj.val)*a2.get(x, 0) + np.sinh(obj.val)*a.get(x, 0)*a.get(y,0)
    val = np.sinh(obj.val)
    return Variable(val, der = der, der2 = der2)

def cosh(obj):
    """Returns the hyperbolic cosine of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: np.sinh(obj.val) * a.get(x, 0) for x in set(a)}
    der2 = {}
    for x in set(a):
        for y in set(a):
            der2[x+y] = np.sinh(obj.val)*a2.get(x, 0) + np.cosh(obj.val)*a.get(x, 0)*a.get(y,0)
    val = np.cosh(obj.val)
    return Variable(val, der = der, der2 = der2)

def tanh(obj):
    """Returns the hyperbolic tangent of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: (1-np.tanh(obj.val)**2) * a.get(x, 0) for x in set(a)}
    der2 = {}
    for x in set(a):
        for y in set(a):
            der2[x+y] = (1-np.tanh(obj.val)**2)*(a2.get(x, 0)-2*np.tanh(obj.val)*a.get(x, 0)*a.get(y,0))
    val = np.tanh(obj.val)
    return Variable(val, der = der, der2 = der2)

# Inverse trigonometric functions
def arcsin(obj):
    """Returns the inverse sine of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: (1-(obj.val)**2)**(-0.5) * a.get(x, 0) for x in set(a)}
    der2 = {}
    for x in set(a):
        for y in set(a):
            der2[x+y] = (1-(obj.val)**2)**(-1.5)*(obj.val*a.get(x, 0)*a.get(y,0)-(obj.val**2-1)*a2.get(x, 0))
    val = np.arcsin(obj.val)
    return Variable(val, der = der, der2 = der2)

def arccos(obj):
    """Returns the inverse cosine of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: -(1-(obj.val)**2)**(-0.5) * a.get(x, 0) for x in set(a)}
    der2 = {}
    for x in set(a):
        for y in set(a):
            der2[x+y] = -(1-(obj.val)**2)**(-1.5)*(obj.val*a.get(x, 0)*a.get(y,0)+a2.get(x, 0)-obj.val**2*a2.get(x, 0))
    val = np.arccos(obj.val)
    return Variable(val, der = der, der2 = der2)

def arctan(obj):
    """Returns the inverse tangent of the Variable."""
    a = obj.der
    a2 = obj.der2
    der = {x: 1/(1+(obj.val)**2) * a.get(x, 0) for x in set(a)}
    der2 = {}
    for x in set(a):
        for y in set(a):
            der2[x+y] = (1+(obj.val)**2)**(-2)*(-2*obj.val*a.get(x, 0)*a.get(y,0)+(obj.val**2+1)*a2.get(x, 0))
    val = np.arctan(obj.val)
    return Variable(val, der = der, der2 = der2)

