
import numpy as np
import numba
import lmfit
print('numpy: %s' % np.__version__)
print('numba: %s' % numba.__version__)
print('lmfit: %s' % lmfit.__version__)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Model functions
#
def exp_func(t, t0, tau, init_value, final_value):
    """Simple exponential transient.
    """
    t = np.asfarray(t)
    A = final_value - init_value
    y = np.full_like(t, init_value)
    m = t > t0
    if init_value < final_value:
        y[m] = A * (1 - np.exp(-(t[m] - t0)/tau)) + init_value
    else:
        y[m] = -A * (np.exp(-(t[m] - t0)/tau)) + final_value
    return y

def expwindec_func(t, t0, t_window, tau, init_value, final_value,
                   decimation=100, sigma=0):
    """Exponential transient with integrating window smoothing."""
    t = np.asfarray(t)
    t_step = t[1] - t[0]
    nwindow = t_window / t_step
    t_step_fine = t_step / decimation
    nwindow_fine = int(t_window / t_step_fine)
    #
    # Visual Example on computing size of fine time axis (t_fine)
    # -----------------------------------------------------------
    # t.size = 5
    # nwindow = 2
    # decimation = 3
    #
    # Time axis:
    #
    #   o-----o-----o-----o-----o------------------->  t.size
    #         '----------'                             integration window   
    #   o-x-x-o-x-x-o-x-x-o-x-x-o-x-x--------------->  t.size * decimation
    #                           '----------'           (last window unfilled)
    #
    #   o-x-x-o-x-x-o-x-x-o-x-x-o-x-x-O-+-+-------->   (t.size + nwindow - 1) * decimation
    #                           '----------'           (last window filled)
    #
    t_fine = np.arange((t.size + nwindow - 1)*decimation)*t_step_fine + t[0]
    y_fine = exp_func(t_fine, t0=t0, tau=tau,
                      init_value=init_value, final_value=final_value)

    if sigma > 0:
        y_fine += np.random.randn(t_fine.size) * sigma * np.sqrt(nwindow_fine)

    return _window_avg(t_fine, y_fine, nwindow_fine, decimation, t.size)

@numba.jit
def _window_avg(t_fine, y_fine, nwindow_fine, decimation, tsize):
    yw = np.zeros(tsize, dtype=float)
    for i, i_start in enumerate(range(0, t_fine.size - nwindow_fine + 1, decimation)):
        yw[i] = y_fine[i_start:i_start + nwindow_fine].mean()
    return yw

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Fitting Models
#
def factory_model_exp(tau=45, init_value=0.1, final_value=0.9,
                      t0=0, t0_vary=True):
    """Returns initialized simple exponential fitting model.
    """
    model = lmfit.model.Model(exp_func)
    model.set_param_hint('t0', value=t0, vary=t0_vary)
    model.set_param_hint('tau', value=tau, min=0)
    model.set_param_hint('init_value', value=init_value, min=0, max=1)
    model.set_param_hint('final_value', value=final_value, min=0, max=1)
    return model

def factory_model_expwin(t_window, decimation, tau=45,
                         init_value=0.1, final_value=0.9, t0=0, t0_vary=True):
    """Returns exponential + integrating window fitting model.
    """
    model = lmfit.model.Model(expwindec_func)
    model.set_param_hint('t_window', value=t_window, vary=False)
    model.set_param_hint('decimation', value=decimation, vary=False)
    model.set_param_hint('sigma', value=0, vary=False)
    model.set_param_hint('t0', value=t0, vary=t0_vary)
    model.set_param_hint('tau', value=tau, min=0)
    model.set_param_hint('init_value', value=init_value, min=0, max=1)
    model.set_param_hint('final_value', value=final_value, min=0, max=1)
    return model
