from forward_ad import AutoDiff

def exp(autodiff_obj):
    x_new = AutoDiff(autodiff_obj.val, autodiff_obj.der)
    x_new.val = np.exp(x_new.val)
    x_new.der = x_new.val * autodiff_obj.der
    return x_new

# trigonometric functions
def sin(autodiff_obj):
    x_new = AutoDiff(autodiff_obj.val, autodiff_obj.der)
    x_new.val = np.sin(x_new.val)
    x_new.der = np.cos(autodiff_obj.val) * autodiff_obj.der
    return x_new

def cos(autodiff_obj):
    x_new = AutoDiff(autodiff_obj.val, autodiff_obj.der)
    x_new.val = np.cos(x_new.val)
    x_new.der = -np.sin(autodiff_obj.val) * autodiff_obj.der
    return x_new

# hyperbolic functions
def sinh(autodiff_obj):
    x_new = AutoDiff(autodiff_obj.val, autodiff_obj.der)
    x_new.val = np.sinh(x_new.val)
    x_new.der = np.cosh(x_new.val) * autodiff_obj.der
    return x_new

def cosh(autodiff_obj):
    x_new = AutoDiff(autodiff_obj.val, autodiff_obj.der)
    x_new.val = np.cosh(x_new.val)
    x_new.der = np.sinh(x_new.val) * autodiff_obj.der
    return x_new

def tanh(autodiff_obj):
    x_new = AutoDiff(autodiff_obj.val, autodiff_obj.der)
    x_new.val = np.tanh(x_new.val)
    x_new.der = (1-np.tanh(x_new.val)**2) * autodiff_obj.der
    return x_new

# Inverse trigonometric functions
def arcsin(autodiff_obj):
    x_new = AutoDiff(autodiff_obj.val, autodiff_obj.der)
    x_new.val = np.arcsin(x_new.val)
    x_new.der = (1-(x_new.val)**2)**(-0.5) * autodiff_obj.der
    return x_new

def arccos(autodiff_obj):
    x_new = AutoDiff(autodiff_obj.val, autodiff_obj.der)
    x_new.val = np.arccos(x_new.val)
    x_new.der = -(1-(x_new.val)**2)**(-0.5) * autodiff_obj.der
    return x_new

def arctan(autodiff_obj):
    x_new = AutoDiff(autodiff_obj.val, autodiff_obj.der)
    x_new.val = np.arctan(x_new.val)
    x_new.der = 1/(1+(x_new.val)**2) * autodiff_obj.der
    return x_new

