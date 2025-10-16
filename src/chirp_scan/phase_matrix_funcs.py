import jax.numpy as jnp
import refractiveindex

from scipy.constants import c as c0



parameters_material_scan = (refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), c0)
def calculate_phase_matrix_material(measurement_info, parameters):
    """ 
    Calculates a phase matrix via material disperson.

    Args:
        measurement_info: Pytree, holds measurement data and parameters, needs to contain the material thickness z_arr in mm.
        parameters: tuple[refractiveindex.RefractiveIndexMaterial, float], an object providing the refractive index, the speed of light in m/s

    Returns:
        jnp.array, the calculated phase matrix
    """
    # z_arr needs to be in mm, is material thickness not translation
    refractive_index, c0 = parameters
    z_arr, frequency = measurement_info.z_arr, measurement_info.frequency

    c0 = c0*1e-12 # speed of light in mm/fs
    wavelength = c0/frequency
    n_arr = refractive_index.material.getRefractiveIndex(jnp.abs(wavelength)*1e6 + 1e-9, bounds_error=False) # wavelength needs to be in nm
    n_arr = jnp.where(jnp.isnan(n_arr)==False, n_arr, 1.0)
    k_arr = 2*jnp.pi/(wavelength + 1e-9)*n_arr
    phase_matrix = jnp.outer(z_arr, k_arr)
    return phase_matrix



#parameters=(0.75, 5, central_f.item())
def calculate_phase_matrix_miips(measurement_info, parameters):
    """ 
    Calculates a phase matrix through a sinusoidal phase.

    Args:
        measurement_info: Pytree, holds measurement data and parameters, needs to contain the material thickness z_arr in mm.
        parameters: tuple[float, float, float], the amplitude, frequency of the sin-function, the central-frequency of the spectrum

    Returns:
        jnp.array, the calculated phase matrix
    """
    
    alpha, gamma, central_frequency = parameters
    z_arr, frequency = measurement_info.z_arr, measurement_info.frequency
    
    omega = 2*jnp.pi * (frequency - central_frequency)
    phase_matrix = alpha*jnp.sin(gamma*omega[jnp.newaxis, :] - z_arr[:, jnp.newaxis])
    return phase_matrix




#parameters = (10, 0, central_f.item())
def calculate_phase_matrix_tanh(measurement_info, parameters):
    """ 
    Calculates a phase matrix through a hyperbolic tangetn.

    Args:
        measurement_info: Pytree, holds measurement data and parameters, needs to contain the material thickness z_arr in mm.
        parameters: tuple[float, float, float], the slope, the offset, the central-frequency of the spectrum

    Returns:
        jnp.array, the calculated phase matrix
    """
    a, b, central_f = parameters
    z_arr, frequency = measurement_info.z_arr, measurement_info.frequency
    
    phase_matrix = z_arr[:,jnp.newaxis]*jnp.tanh(a*(frequency[jnp.newaxis, :] - central_f)) + b
    return phase_matrix

