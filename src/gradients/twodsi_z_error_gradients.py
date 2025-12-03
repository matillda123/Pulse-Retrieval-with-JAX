import jax.numpy as jnp
from src.utilities import do_fft



def Z_gradient_shg_pulse(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn):
    term1 = do_fft(deltaS*jnp.conjugate(gate), sk, rn)
    term2 = do_fft(deltaS*jnp.conjugate(pulse_t), sk, rn)
    grad = term1 + exp_arr*term2
    return -2*grad


def Z_gradient_thg_pulse(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn):
    term1 = do_fft(deltaS*jnp.conjugate(gate), sk, rn)
    term2 = do_fft(deltaS*jnp.conjugate(pulse_t*gate_pulses), sk, rn)
    grad = term1 + 2*exp_arr*term2
    return -2*grad


def Z_gradient_pg_pulse(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn):
    term1 = do_fft(deltaS*gate, sk, rn)
    term2 = do_fft(jnp.real(jnp.conjugate(deltaS)*(pulse_t*gate_pulses)), sk, rn)
    grad = term1 + 2*exp_arr*term2
    return -2*grad


def Z_gradient_sd_pulse(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn):
    term1 = do_fft(deltaS*jnp.conjugate(gate), sk, rn)
    term2 = do_fft(pulse_t*jnp.conjugate(deltaS*gate_pulses), sk, rn)
    grad = term1 + 2*exp_arr*term2
    return -2*grad







def Z_gradient_cross_correlation_pulse(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn):
    # gradient with respect to pulse, is the same for all nonlinear methods
    grad = do_fft(deltaS*jnp.conjugate(gate), sk, rn)
    return -2*grad





def Z_gradient_shg_cross_correlation_gate(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn):
    grad = exp_arr*do_fft(deltaS*jnp.conjugate(pulse_t), sk, rn)
    return -2*grad



def Z_gradient_thg_cross_correlation_gate(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn):
    grad = 2*exp_arr*do_fft(deltaS*jnp.conjugate(pulse_t*gate_pulses), sk, rn)
    return -2*grad



def Z_gradient_pg_cross_correlation_gate(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn):
    grad = 2*exp_arr*do_fft(gate_pulses*jnp.real(jnp.conjugate(pulse_t)*deltaS), sk, rn)
    return -2*grad



def Z_gradient_sd_cross_correlation_gate(deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn):
    grad = 2*exp_arr*do_fft(pulse_t*jnp.conjugate(deltaS*gate_pulses), sk, rn)
    return -2*grad








def calculate_Z_gradient_pulse(signal_t, signal_t_new, pulse_t, gate_pulses, gate, tau_arr, delay, measurement_info, is_vampire):
    frequency, sk, rn = measurement_info.frequency, measurement_info.sk, measurement_info.rn
    nonlinear_method = measurement_info.nonlinear_method

    omega_arr = 2*jnp.pi*frequency
    
    if is_vampire==True:
        tau, phase_matrix = measurement_info.tau_interferometer, measurement_info.phase_matrix
        exp_arr = jnp.exp(1j*jnp.outer(tau_arr, omega_arr))*(jnp.exp(1j*omega_arr*tau) + jnp.exp(-1j*(phase_matrix-omega_arr*delay)))
    else:
        spectral_filter1, spectral_filter2 = jnp.conjugate(measurement_info.spectral_filter1), jnp.conjugate(measurement_info.spectral_filter2)
        tau, phase_matrix = measurement_info.tau_pulse_anc1, measurement_info.phase_matrix
        exp_arr = (spectral_filter1*jnp.exp(1j*omega_arr*tau) + spectral_filter2*jnp.exp(1j*jnp.outer(tau_arr, omega_arr)))*jnp.exp(-1j*(phase_matrix-omega_arr*delay))

    deltaS = signal_t_new - signal_t


    if measurement_info.doubleblind==True or measurement_info.cross_correlation==True:
        xcorr = True
    else:
        xcorr = False

    grad_func_ac = {"shg": Z_gradient_shg_pulse, "thg": Z_gradient_thg_pulse, 
                    "pg": Z_gradient_pg_pulse, "sd": Z_gradient_sd_pulse}
    
    grad_func_xcorr = {"shg": Z_gradient_cross_correlation_pulse, "thg": Z_gradient_cross_correlation_pulse, 
                       "pg": Z_gradient_cross_correlation_pulse, "sd": Z_gradient_cross_correlation_pulse}
    
    grad_func = {True: grad_func_xcorr,
                 False: grad_func_ac}

    grad = grad_func[xcorr][nonlinear_method](deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn)
    return grad




def calculate_Z_gradient_gate(signal_t, signal_t_new, pulse_t, gate_pulses, gate, tau_arr, delay, measurement_info, is_vampire):
    frequency, sk, rn = measurement_info.frequency, measurement_info.sk, measurement_info.rn
    nonlinear_method = measurement_info.nonlinear_method

    omega_arr = 2*jnp.pi*frequency
    
    if is_vampire==True:
        tau, phase_matrix = measurement_info.tau_interferometer, measurement_info.phase_matrix
        exp_arr = jnp.exp(1j*jnp.outer(tau_arr, omega_arr))*(jnp.exp(1j*omega_arr*tau) + jnp.exp(-1j*(phase_matrix-omega_arr*delay)))
    else:
        spectral_filter1, spectral_filter2 = jnp.conjugate(measurement_info.spectral_filter1), jnp.conjugate(measurement_info.spectral_filter2)
        tau, phase_matrix = measurement_info.tau_pulse_anc1, measurement_info.phase_matrix
        exp_arr = (spectral_filter1*jnp.exp(1j*omega_arr*tau) + spectral_filter2*jnp.exp(1j*jnp.outer(tau_arr, omega_arr)))*jnp.exp(-1j*(phase_matrix-omega_arr*delay))

    deltaS = signal_t_new-signal_t

    grad_func = {"shg": Z_gradient_shg_cross_correlation_gate, "thg": Z_gradient_thg_cross_correlation_gate, 
                 "pg": Z_gradient_pg_cross_correlation_gate, "sd": Z_gradient_sd_cross_correlation_gate}

    grad = grad_func[nonlinear_method](deltaS, pulse_t, gate_pulses, gate, exp_arr, sk, rn)
    return grad






def calculate_Z_gradient(signal_t, signal_t_new, pulse_t, gate_pulses, gate, tau_arr, delay, measurement_info, pulse_or_gate, is_vampire=False):
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
    return calculate_Z_gradient_dict[pulse_or_gate](signal_t, signal_t_new, pulse_t, gate_pulses, gate, tau_arr, delay, measurement_info, is_vampire)