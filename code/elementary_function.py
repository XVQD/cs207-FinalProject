from forward_ad import AutoDiff

def exp(autodiff_obj):
    return AutoDiff(np.exp(autodiff_obj.val), 
        np.exp(autodiff_obj.val) * autodiff_obj.der)

# trigonometric functions
def sin(autodiff_obj):
    return AutoDiff(np.sin(autodiff_obj.val), 
        np.cos(autodiff_obj.val) * autodiff_obj.der)

def cos(autodiff_obj):
    return AutoDiff(np.cos(autodiff_obj.val), 
        -np.sin(autodiff_obj.val) * autodiff_obj.der)

# hyperbolic functions
def sinh(autodiff_obj):
    return AutoDiff(np.sinh(autodiff_obj.val), 
        np.cosh(autodiff_obj.val) * autodiff_obj.der)

def cosh(autodiff_obj):
    return AutoDiff(np.cosh(autodiff_obj.val), 
        np.sinh(autodiff_obj.val) * autodiff_obj.der)

def tanh(autodiff_obj):
    return AutoDiff(np.tanh(autodiff_obj.val), 
        (1-np.tanh(autodiff_obj.val)**2) * autodiff_obj.der)

# Inverse trigonometric functions
def arcsin(autodiff_obj):
    return AutoDiff(np.arcsin(autodiff_obj.val), 
        (1-(autodiff_obj.val)**2)**(-0.5) * autodiff_obj.der)

def arccos(autodiff_obj):
    return AutoDiff(np.arccos(autodiff_obj.val), 
        -(1-(autodiff_obj.val)**2)**(-0.5) * autodiff_obj.der)

def arctan(autodiff_obj):
    return AutoDiff(np.arctan(x_new.val), 
        1/(1+(autodiff_obj.val)**2) * autodiff_obj.der)