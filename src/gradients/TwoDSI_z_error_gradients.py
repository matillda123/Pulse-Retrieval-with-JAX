import jax.numpy as jnp
from src.utilities import do_fft




def Z_gradient_pulse(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn):
    # gradient with respect to pulse, is the same for all nonlinear methods
    grad=do_fft(deltaS*jnp.conjugate(gate), sk, rn)
    return -2*grad





def Z_gradient_gate_shg(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn):
    grad = (1 + exp_arr)*do_fft(deltaS*jnp.conjugate(pulse_t), sk, rn)
    return -2*grad



def Z_gradient_gate_thg(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn):
    grad = 2*(1 + exp_arr)*do_fft(deltaS*jnp.conjugate(pulse_t*gate_pulses), sk, rn)
    return -2*grad



def Z_gradient_gate_pg(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn):
    grad = 2*(1 + exp_arr)*do_fft(jnp.real(jnp.conjugate(pulse_t)*deltaS)*gate_pulses, sk, rn)
    return -2*grad



def Z_gradient_gate_sd(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn):
    grad = 2*(1 + exp_arr)*do_fft(deltaS*pulse_t*jnp.conjugate(gate_pulses), sk, rn)
    return -2*grad








def calculate_Z_gradient_pulse(signal_t, signal_t_new, pulse_t, gate_pulses, gate, tau_arr, measurement_info):
    frequency, sk, rn = measurement_info.frequency, measurement_info.sk, measurement_info.rn

    omega_arr=2*jnp.pi*frequency
    exp_arr=jnp.exp(1j*jnp.outer(tau_arr, omega_arr))

    deltaS = signal_t_new - signal_t

    grad = Z_gradient_pulse(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn)
    return grad




def calculate_Z_gradient_gate(signal_t, signal_t_new, pulse_t, gate_pulses, gate, tau_arr, measurement_info):
    frequency, sk, rn = measurement_info.frequency, measurement_info.sk, measurement_info.rn
    nonlinear_method = measurement_info.nonlinear_method

    omega_arr=2*jnp.pi*frequency
    exp_arr=jnp.exp(1j*jnp.outer(tau_arr, omega_arr))

    deltaS = signal_t_new-signal_t

    grad_func={"shg": Z_gradient_gate_shg, "thg": Z_gradient_gate_thg, "pg": Z_gradient_gate_pg, "sd": Z_gradient_gate_sd}

    grad=grad_func[nonlinear_method](deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn)
    return grad






def calculate_Z_gradient(signal_t, signal_t_new, pulse_t, gate_pulses, gate, tau_arr, measurement_info, pulse_or_gate):
    """
    Calculates the Z-error gradient with respect to the pulse or the gate-pulse for a given 2DSI measurement. 
    The gradient is calculated in the frequency domain.

    Args:
        signal_t: jnp.array, the current signal field
        signal_t_new: jnp.array, the current signal field projected onto the measured intensity
        pulse_t: jnp.array, the current guess
        gate_pulses: jnp.array, the current gate-pulse guess
        gate: jnp.array, the current gate
        tau_arr: jnp.array, the delays
        measurement_info: Pytree, contains measurement data and parameters
        pulse_or_gate: str, whether the gradient is calculated with respect to the pulse or the gate-pulse

    Returns:
        jnp.array, the Z-error gradient
    """
        
    calculate_Z_gradient_dict={"pulse": calculate_Z_gradient_pulse,
                               "gate": calculate_Z_gradient_gate}
    return calculate_Z_gradient_dict[pulse_or_gate](signal_t, signal_t_new, pulse_t, gate_pulses, gate, tau_arr, measurement_info)