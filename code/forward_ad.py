import numpy as np

class AutoDiff:
    
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
    >>> x1 = AutoDiff(1, name='x1') # register independent variables by specifying names
    >>> x2 = x1 + 1
    >>> x3 = AutoDiff(7,  name='x3')
    >>> x4 =  x2 * x3
    >>> print(x4.val, v4.der)  # only calculate derivatives of registered(i.e.named) independent variables 
    14 {'x1': 7, 'x2': 2} 
    """

    def __init__(self, input_val, init_der):
        self.val = input_val
        self.der = init_der

    def __add__(self, other):
        try:
            x_new = AutoDiff(self.val, self.der)
            x_new.val += other.val
            x_new.der += other.der
        except AttributeError:
            x_new.val += other
        return x_new
        
    def __radd__(self, other):
        try:
            x_new = AutoDiff(self.val, self.der)
            x_new.val += other.val
            x_new.der += other.der
        except AttributeError:
            x_new.val += other
        return x_new
    
    def __sub__(self, other):
        try:
            x_new = AutoDiff(self.val, self.der)
            x_new.val -= other.val
            x_new.der -= other.der
        except AttributeError:
            x_new.val -= other
        return x_new
    
    def __rsub__(self, other):
        try:
            x_new = AutoDiff(self.val, self.der)
            x_new.val -= other.val
            x_new.der -= other.der
        except AttributeError:
            x_new.val -= other
        return x_new
        
    def __mul__(self, other):
        try:
            x_new = AutoDiff(self.val, self.der)
            x_new.val *= other.val
            x_new.der = other.val
        except AttributeError:
            x_new.val *= other
        return x_new
        
    def __rmul__(self, other):
        try:
            x_new = AutoDiff(self.val, self.der)
            x_new.val *= other.val
            x_new.der = other.val
        except AttributeError:
            x_new.val *= other
        return x_new
        
    def __truediv(self, other):
        pass

    def __pow__(self, other):
        x_new = AutoDiff(self.val, self.der)
        try:
            x_new.val = self.val ** other.val
            # x_new.der = np.array([other.val/self.val * self.val ** other.val, 
            #                       np.log(self.val) * self.val ** other.val])
            # # should it be:
            x_new.der = other.val * self.val ** (other.val-1) * self.der + 
                     np.log(self.val) * self.val ** other.val * other.der
        except AttributeError:
            x_new.der = other * x_new.val ** (other - 1)
            x_new.val = x_new.val**other
 
        return x_new
