from forward_ad import AutoDiff

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

def exp(autodiff_obj):
    x_new = AutoDiff(autodiff_obj.val, autodiff_obj.der)
    x_new.val = np.exp(x_new.val)
    x_new.der = x_new.val * autodiff_obj.der
    return x_new
