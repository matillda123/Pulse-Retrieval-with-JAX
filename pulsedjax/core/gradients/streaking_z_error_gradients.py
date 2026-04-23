import jax.numpy as jnp
from pulsedjax.utilities import do_fft



def Z_gradient_EUV_pulse(): # frequency domain
    pass


def Z_gradient_Vectorpotential(): # frequency domain
    pass


def Z_gradient_DTME(): # momentum domain
    pass





def calculate_Z_gradient(signal_t, signal_t_new, tau_arr, measurement_info, pulse_or_gate):
    """
    Calculates the Z-error gradient with respect to the pulse, gate-pulse or the DTME for a given Streaking measurement. 
    The gradient is calculated in the frequency domain.

    Args:
        signal_t (jnp.array): the current signal field
        signal_t_new (jnp.array): the current signal field projected onto the measured intensity
        pulse_t (jnp.array): the current guess
        pulse_t_shifted (jnp.array): the current guess translated on the time axis
        gate_shifted (jnp.array): the current gate translated on the time axis
        tau_arr (jnp.array): the delays
        measurement_info (Pytree): contains measurement data and parameters
        pulse_or_gate (str): whether the gradient is calculated with respect to the pulse or the gate-pulse

    Returns:
        jnp.array, the Z-error gradient
    """
    omega_arr = 2*jnp.pi*measurement_info.frequency
    exp_arr = jnp.exp(1j*jnp.outer(tau_arr, omega_arr))

    deltaS = signal_t_new - signal_t



    calculate_Z_gradient_dict={"pulse": Z_gradient_Vectorpotential,
                               "gate": Z_gradient_EUV_pulse, 
                               "dtme": Z_gradient_DTME}
    return calculate_Z_gradient_dict[pulse_or_gate](signal_t, signal_t_new, tau_arr, measurement_info)