import jax.numpy as jnp
import jax
import refractiveindex

from scipy.constants import c as c0


def get_parameters_material_scan(shelf="main", book="SiO2", page="Malitson"):
    return refractiveindex.RefractiveIndexMaterial(shelf=shelf, book=book, page=page)


def calc_group_delay_phase(refractive_index, n_arr, k0_arr, wavelength_0, wavelength):
    """ 
    Uses finite differences to get the group index at a specfific wavelength and 
    returns the linearized group delay per mm at that wavelength. 
    """
    
    dlambda = 0.5 #nm
    n0 = refractive_index.material.getRefractiveIndex(jnp.abs(wavelength_0)-dlambda + 1e-9, bounds_error=False)
    n2 = refractive_index.material.getRefractiveIndex(jnp.abs(wavelength_0)+dlambda + 1e-9, bounds_error=False)
    dndl = (n2-n0)/(2*dlambda)

    idx = jnp.argmin(jnp.abs(wavelength-wavelength_0))
    ng = n_arr[idx] - wavelength_0*dndl
    k_0 = 2*jnp.pi/(wavelength_0*1e-6 + 1e-9) # in mm 
    return ng*(k0_arr - k_0)


def calculate_phase_matrix_material(measurement_info, parameters):
    """ 
    Calculates a phase matrix via material dispersion. Not differentiable due to usage of refractiveindex.

    Args:
        measurement_info (Pytree): holds measurement data and parameters, needs to contain the material thickness theta in mm.
        parameters (refractiveindex.RefractiveIndexMaterial): an object providing the refractive index, the speed of light in m/s

    Returns:
        jnp.array, the calculated phase matrix
    """
    # theta needs to be in mm, is material thickness not translation
    refractive_index = parameters
    theta, frequency = measurement_info.theta, measurement_info.frequency

    wavelength = c0/frequency*1e-6 # wavelength in nm
    n_arr = refractive_index.material.getRefractiveIndex(jnp.abs(wavelength) + 1e-9, bounds_error=False) # wavelength needs to be in nm
    n_arr = jnp.where(jnp.isnan(n_arr)==False, n_arr, 1.0)
    k0_arr = 2*jnp.pi/(wavelength*1e-6 + 1e-9) #wavelength is needed in mm
    k_arr = k0_arr*n_arr

    wavelength_0 = c0/(measurement_info.central_frequency + 1e-9)*1e-6 
    Tg_phase = calc_group_delay_phase(refractive_index, n_arr, k0_arr, wavelength_0, wavelength)

    phase_matrix = theta[:, jnp.newaxis]*(k_arr[jnp.newaxis, :] - Tg_phase[jnp.newaxis,:])
    return phase_matrix






def calc_sine_phase(omega, phase_shift, parameters):
    alpha, gamma, central_frequency = parameters
    omega = omega - 2*jnp.pi*central_frequency
    return alpha*jnp.sin(gamma*omega - phase_shift)


def calc_tanh_phase(omega, phase_shift, parameters):
    alpha, gamma, central_frequency = parameters
    omega = omega - 2*jnp.pi*central_frequency
    return alpha*jnp.pi/2*jnp.tanh(gamma*(omega - phase_shift))


def calc_gaussian_phase(omega, phase_shift, parameters):
    # the factor of 1j causes a change in amplitude, as intended by GMIIPS
    gamma, sigma, central_frequency = parameters
    omega = omega - 2*jnp.pi*central_frequency
    return 1j*(gamma*omega - phase_shift)**2/(2*sigma**2)



def calc_MIIPS_phase(omega, phase_shift, parameters):
    return calc_sine_phase(omega, phase_shift, parameters)


def calc_G_MIIPS_phase(omega, phase_shift, parameters):
    alpha, gamma, central_frequency, sigma = parameters
    return calc_gaussian_phase(omega, phase_shift, (gamma, sigma, central_frequency)) + calc_sine_phase(omega, phase_shift, (alpha, gamma, central_frequency))




phase_func_dict = {"sine": calc_sine_phase,
                     "tanh": calc_tanh_phase,
                     "gaussian": calc_gaussian_phase,
                     "MIIPS": calc_MIIPS_phase,
                     "GMIIPS": calc_G_MIIPS_phase}








def calculate_phase_matrix(measurement_info, parameters, phase_func=calc_MIIPS_phase):
    """ 
    Calculates a phase matrix using a specified phase function.

    Args:
        measurement_info (Pytree): holds measurement data and parameters, needs to contain the shift-values in appropriate units
        parameters (tuple): the parameters which are expected by phase_func
        phase_func (Callable): defines how the phase is calculated

    Returns:
        jnp.array, the calculated phase matrix
    """
    
    theta, omega = measurement_info.theta, 2*jnp.pi*measurement_info.frequency
    phase_matrix = phase_func(omega[jnp.newaxis,:], theta[:, jnp.newaxis], parameters)
    return phase_matrix





def calc_GDD(omega, phase_shift, parameters, phase_func=calc_sine_phase):
    """ Uses jax.grad to get the GDD of phase_func. Thus doesnt work with non-differentiable functions. Assumes phase_func is holomorphic. """
    # i dont like the use of holomorphic and conversion to complex values, but it should be fine. 
    omega = jnp.asarray(omega, dtype=jnp.complex64)
    calc_2nd_grad = jax.grad(jax.grad(phase_func, argnums=0, holomorphic=True), argnums=0, holomorphic=True)
    return jax.vmap(calc_2nd_grad, in_axes=(0,0,None))(omega, phase_shift, parameters)
