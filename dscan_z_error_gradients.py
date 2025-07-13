import jax.numpy as jnp
from utilities import do_fft



def Z_gradient_shg(pulse_t_dispersed, difference_signal_t, sk, rn):
    grad=2*do_fft(jnp.conjugate(pulse_t_dispersed)*difference_signal_t, sk, rn)
    return grad

def Z_gradient_thg(pulse_t_dispersed, difference_signal_t, sk, rn):
    grad=3*do_fft(jnp.conjugate(pulse_t_dispersed**2)*difference_signal_t, sk, rn)
    return grad


def Z_gradient_pg(pulse_t_dispersed, difference_signal_t, sk, rn):
    term1=2*jnp.conjugate(pulse_t_dispersed)*difference_signal_t
    term2=pulse_t_dispersed*jnp.conjugate(difference_signal_t)
    grad=do_fft(pulse_t_dispersed*(term1 + term2), sk, rn)
    return grad



def calculate_Z_gradient(pulse_t_dispersed, signal_t, signal_t_new, phase_matrix, measurement_info):
    nonlinear_method, sk, rn = measurement_info.nonlinear_method, measurement_info.sk, measurement_info.rn
    difference_signal_t = signal_t_new-signal_t

    grad_func_dict={"shg": Z_gradient_shg,
                    "thg": Z_gradient_thg,
                    "pg": Z_gradient_pg,
                    "sd": Z_gradient_pg,
                    "tg": Z_gradient_pg,}
    
    grad=grad_func_dict[nonlinear_method](pulse_t_dispersed, difference_signal_t, sk, rn)
    return -2*grad*jnp.exp(-1*1j*phase_matrix)

