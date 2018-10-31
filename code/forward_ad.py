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
    >>> a = 2
    >>> b = 3
    >>> c = 4
    >>> x1 = AutoDiff(a, np.array([1, 0, 0]))
    >>> x2 = AutoDiff(b, np.array([0, 1, 0]))
    >>> x3 = AutoDiff(c, np.array([0, 0, 1]))
    >>> x3 = x1 + x2 + x3
    >>> x3.val, x3.der
    (9, array([1, 1, 1]))
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
            x_new.der = np.array([other.val/self.val * self.val ** other.val, 
                                  np.log(self.val) * self.val ** other.val])
            # # should it be:
            # x_new.der = other.val * self.val ** other.val * self.der + 
            #          np.log(self.val) * self.val ** other.val * other.der
        except AttributeError:
            x_new.der = other * x_new.val ** (other - 1)
            x_new.val = x_new.val**other
 
        return x_new
