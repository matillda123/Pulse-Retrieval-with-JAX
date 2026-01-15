import jax.numpy as jnp
from .phase_matrix_funcs import calc_group_delay_phase
from scipy.constants import c as c0
import refractiveindex


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


def get_filter(filter_func, frequency, parameters, custom_func=None, 
               refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
               material_thickness=0):
    """ 
    Generate a spectral filter.

    Args:
        filter_func (str, tuple[str]): can be one of ```gaussian, lorentzian, rectangular, multi or custom```. 
                                        For multi, parameters needs to specify the repsective filetr function
        frequency (jnp.array): the frequency axis
        parameters (tuple): the parameters required by the filter function, have the input form (a, f0, fwhm, p)
        custom_func (Callable, None): in case of filter_func="custom" the custom filter function needs to be provided here.
        refractive_index (refractiveindex.RefractiveIndexMaterial): provides the refractive index
        material_thickness (float, int): the material thickness in millimeters
    
    Returns:
        jnp.array, the spectral filter on the frequency axis
    """
    filter_dict = dict(gaussian=gaussian_filter,
                       lorentzian=lorentzian_filter,
                       rectangular=rectangular_filter,
                       multi=multi_filter,
                       custom=custom_func)
    y = filter_dict[filter_func](frequency, parameters, filter_dict)
    y = y/jnp.abs(jnp.max(y))

    if material_thickness>0:
        f0 = jnp.sum(frequency*y)/jnp.sum(y)
        phase_matrix = get_phase_matrix(refractive_index, material_thickness, frequency, f0)
        y = y*jnp.exp(1j*phase_matrix)

    return y







def get_phase_matrix(refractive_index, material_thickness, frequency, central_frequency):
    """ 
    Calculates the phase matrix that is applied of a pulse passes through a material.
    """
    wavelength = c0/frequency*1e-6 # wavelength in nm
    n_arr = refractive_index.material.getRefractiveIndex(jnp.abs(wavelength) + 1e-9, bounds_error=False) # wavelength needs to be in nm
    n_arr = jnp.where(jnp.isnan(n_arr)==False, n_arr, 1.0)
    k0_arr = 2*jnp.pi/(wavelength*1e-6 + 1e-9) #wavelength is needed in mm
    k_arr = k0_arr*n_arr

    wavelength_0 = c0/(central_frequency + 1e-9)*1e-6 
    Tg_phase = calc_group_delay_phase(refractive_index, n_arr, k0_arr, wavelength_0, wavelength)
    phase_matrix = material_thickness*(k_arr-Tg_phase)
    return phase_matrix