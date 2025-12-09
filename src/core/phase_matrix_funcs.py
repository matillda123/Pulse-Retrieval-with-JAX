import jax.numpy as jnp
import jax
import refractiveindex

from scipy.constants import c as c0



parameters_material_scan = (refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), c0)
def calculate_phase_matrix_material(measurement_info, parameters):
    """ 
    Calculates a phase matrix via material dispersion. Not differentiable due to usage of refractiveindex.

    Args:
        measurement_info: Pytree, holds measurement data and parameters, needs to contain the material thickness z_arr in mm.
        parameters: tuple[refractiveindex.RefractiveIndexMaterial, float], an object providing the refractive index, the speed of light in m/s

    Returns:
        jnp.array, the calculated phase matrix
    """
    # z_arr needs to be in mm, is material thickness not translation
    refractive_index, c0 = parameters
    z_arr, frequency = measurement_info.z_arr, measurement_info.frequency

    wavelength = c0/frequency*1e-6 # wavelength in nm
    n_arr = refractive_index.material.getRefractiveIndex(jnp.abs(wavelength) + 1e-9, bounds_error=False) # wavelength needs to be in nm
    n_arr = jnp.where(jnp.isnan(n_arr)==False, n_arr, 1.0)
    k_arr = 2*jnp.pi/(wavelength*1e-6 + 1e-9)*n_arr #wavelength is needed in mm
    phase_matrix = z_arr[:, jnp.newaxis]*k_arr[jnp.newaxis, :]
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




# def calc_taylor_phase(omega, phase_amp, parameters):
#     n, central_frequency = parameters
#     assert n>=2, "the phase should not include a linear component"
#     return phase_amp*(omega-2*jnp.pi*central_frequency)**n



def calc_taylor_phase(omega, phase_shift, parameters):
    a, n, central_frequency = parameters
    assert n>=2, "the phase should not include a linear component"
    omega = omega-2*jnp.pi*central_frequency
    return a*(omega+phase_shift)**n





phase_func_dict = {"sine": calc_sine_phase,
                     "tanh": calc_tanh_phase,
                     "gaussian": calc_gaussian_phase,
                     "MIIPS": calc_MIIPS_phase,
                     "GMIIPS": calc_G_MIIPS_phase,
                     "taylor": calc_taylor_phase}








def calculate_phase_matrix(measurement_info, parameters, phase_func=calc_MIIPS_phase):
    """ 
    Calculates a phase matrix using a specified phase function.

    Args:
        measurement_info: Pytree, holds measurement data and parameters, needs to contain the shift-values in appropriate units
        parameters: tuple, the parameters which are expected by phase_func
        phase_func: Callable, defines how the phase is calculated

    Returns:
        jnp.array, the calculated phase matrix
    """
    
    z_arr, omega = measurement_info.z_arr, 2*jnp.pi*measurement_info.frequency
    phase_matrix = phase_func(omega[jnp.newaxis,:], z_arr[:, jnp.newaxis], parameters)
    return phase_matrix





def calc_GDD(omega, phase_shift, parameters, phase_func=calc_sine_phase):
    """ Uses jax.grad to get the GDD of phase_func. Thus doesnt work with non-differentiable functions. Assumes phase_func is holomorphic. """
    # i dont like the use of holomorphic and conversion to complex values, but it should be fine. 
    omega = jnp.asarray(omega, dtype=jnp.complex64)
    calc_2nd_grad = jax.grad(jax.grad(phase_func, argnums=0, holomorphic=True), argnums=0, holomorphic=True)
    return jax.vmap(calc_2nd_grad, in_axes=(0,0,None))(omega, phase_shift, parameters)
