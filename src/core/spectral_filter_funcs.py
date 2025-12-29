import jax.numpy as jnp


def gaussian_filter(frequency, parameters, filter_dict):
    a, f0, fwhm, p = parameters
    y = a*jnp.exp(-jnp.log(2)*(4*(frequency-f0)**2/fwhm**2)**p)
    return y


def lorentzian_filter(frequency, parameters, filter_dict):
    a, f0, fwhm, p = parameters
    y = a/(1+jnp.abs(2*(frequency-f0)/fwhm)**(2*p))
    return y


def rectangular_filter(frequency, parameters, filter_dict):
    y = jnp.arange(jnp.size(frequency))

    a, f0, width = parameters
    idx1 = jnp.argmin(jnp.abs(frequency-(f0-width/2)))
    idx2 = jnp.argmin(jnp.abs(frequency-(f0+width/2)))
    y1 = jnp.where(y<idx1, 0, 1)
    y2 = jnp.where(y>idx2, 0, 1)
    y = a*y1*y2
    return y


def multi_filter(frequency, parameters, filter_dict):
    N = len(parameters)
    y = jnp.zeros(jnp.size(frequency))
    for i in range(N):
        filter_func = parameters[i][0]
        y = y + filter_dict[filter_func](frequency, parameters[i][1:], filter_dict)
    return y


def get_filter(filter_func, frequency, parameters, custom_func=None):
    """ 
    Generate a spectral filter.

    Args:
        filter_func (str, tuple[str]): can be one of gaussian, lorentzian, rectangular, multi or custom. 
                                        For multi, parameters needs to specify the repsective filetr function
        frequency (jnp.array): the frequency axis
        parameters (tuple): the parameters required by the filter function
        custom_func (Callable, None): in case of filter_func="custom" the custom filter function needs to be provided here.
    
    Returns:
        jnp.array, the spectral filter on the frequency axis
    """
    filter_dict = dict(gaussian=gaussian_filter,
                       lorentzian=lorentzian_filter,
                       rectangular=rectangular_filter,
                       multi=multi_filter,
                       custom=custom_func)
    y = filter_dict[filter_func](frequency, parameters, filter_dict)
    return y/jnp.abs(jnp.max(y))