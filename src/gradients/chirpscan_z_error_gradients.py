import jax.numpy as jnp
from src.utilities import do_fft



def Z_gradient_shg(pulse_t_dispersed, difference_signal_t, sk, rn):
    grad = 2*do_fft(jnp.conjugate(pulse_t_dispersed)*difference_signal_t, sk, rn)
    return grad

def Z_gradient_thg(pulse_t_dispersed, difference_signal_t, sk, rn):
    grad = 3*do_fft(jnp.conjugate(pulse_t_dispersed**2)*difference_signal_t, sk, rn)
    return grad


def Z_gradient_pg(pulse_t_dispersed, difference_signal_t, sk, rn):
    term1 = 2*jnp.conjugate(pulse_t_dispersed)*difference_signal_t
    term2 = pulse_t_dispersed*jnp.conjugate(difference_signal_t)
    grad = do_fft(pulse_t_dispersed*(term1 + term2), sk, rn)
    return grad



def calculate_Z_gradient(pulse_t_dispersed, signal_t, signal_t_new, phase_matrix, measurement_info):
    """ 
    Calculates the analytical Z-error gradient of a chirp-scan method and a given nonlinear method. 
    The gradient is calculated in the frequency domain.
    
    Args:
        pulse_t_dispersed: jnp.array, the current guess after phase_matrix was applied
        signal_t: jnp.array, the signal field of the current guess
        signal_t_new: jnp.array, the signal field after projection onto the measured intensity
        phase_matrix: jnp.array, the phase matrix which was applied
        measurement_info: Pytree, contains measurement data and parameters

    Returns:
        jnp.array, the Z-error gradient
    
    """
    nonlinear_method, sk, rn = measurement_info.nonlinear_method, measurement_info.sk, measurement_info.rn
    difference_signal_t = signal_t_new-signal_t

    # sd, pg and tg are all the same
    grad_func_dict={"shg": Z_gradient_shg,
                    "thg": Z_gradient_thg,
                    "pg": Z_gradient_pg,
                    "sd": Z_gradient_pg,
                    "tg": Z_gradient_pg,}
    
    grad=grad_func_dict[nonlinear_method](pulse_t_dispersed, difference_signal_t, sk, rn)
    return -2*grad*jnp.exp(-1*1j*phase_matrix)

