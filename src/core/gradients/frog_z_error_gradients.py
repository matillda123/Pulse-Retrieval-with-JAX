import jax.numpy as jnp
from src.utilities import do_fft



def Z_gradient_shg(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    term1=do_fft(deltaS*jnp.conjugate(pulse_t_shifted), sk, rn)
    term2=do_fft(deltaS*jnp.conjugate(pulse_t), sk, rn)
    grad=term1+term2*exp_arr
    return -2*grad

def Z_gradient_thg(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    term1=do_fft(deltaS*jnp.conjugate(pulse_t_shifted)**2, sk, rn)
    term2=do_fft(deltaS*jnp.conjugate(pulse_t*pulse_t_shifted), sk, rn)
    grad=term1+2*term2*exp_arr
    return -2*grad

def Z_gradient_pg(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    term1=do_fft(deltaS*jnp.abs(pulse_t_shifted)**2, sk, rn)
    term2=do_fft(pulse_t_shifted*jnp.real(deltaS*jnp.conjugate(pulse_t)), sk, rn)
    grad=term1+2*term2*exp_arr
    return -2*grad

def Z_gradient_sd(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    term1=do_fft(deltaS*pulse_t_shifted**2, sk, rn)
    term2=do_fft(jnp.conjugate(deltaS*pulse_t_shifted)*pulse_t, sk, rn)
    grad=term1+2*term2*exp_arr
    return -2*grad

def Z_gradient_nhg(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    term1=do_fft(deltaS*jnp.conjugate(pulse_t_shifted)**(n-1), sk, rn)
    term2=do_fft(deltaS*jnp.conjugate(pulse_t*pulse_t_shifted**(n-2)), sk, rn)
    grad=term1+(n-1)*term2*exp_arr
    return -2*grad






def Z_gradient_cross_correlation_pulse(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    # gradient with respect to pulse, is the same for all nonlinear methods
    grad=do_fft(deltaS*jnp.conjugate(gate_shifted), sk, rn)
    return -2*grad






def Z_gradient_shg_cross_correlation_gate(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    grad=exp_arr*do_fft(deltaS*jnp.conjugate(pulse_t), sk, rn)
    return -2*grad


def Z_gradient_thg_cross_correlation_gate(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    grad=2*exp_arr*do_fft(deltaS*jnp.conjugate(pulse_t*pulse_t_shifted), sk, rn)
    return -2*grad


def Z_gradient_pg_cross_correlation_gate(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    grad=2*exp_arr*do_fft(pulse_t_shifted*jnp.real(deltaS*jnp.conjugate(pulse_t)), sk, rn)
    return -2*grad


def Z_gradient_sd_cross_correlation_gate(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    grad=2*exp_arr*do_fft(pulse_t*jnp.conjugate(deltaS*pulse_t_shifted), sk, rn)
    return -2*grad

def Z_gradient_nhg_cross_correlation_gate(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    grad = (n-1)*exp_arr*do_fft(deltaS*jnp.conjugate(pulse_t*pulse_t_shifted**(n-1)), sk, rn)
    return -2*grad










def Z_gradient_shg_interferometric(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    grad=2*(1+exp_arr)*do_fft(deltaS*jnp.conjugate(pulse_t+pulse_t_shifted), sk, rn)
    return -2*grad



def Z_gradient_thg_interferometric(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    grad=3*(1+exp_arr)*do_fft(deltaS*jnp.conjugate(pulse_t+pulse_t_shifted)**2, sk, rn)
    return -2*grad



def Z_gradient_pg_interferometric(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    # is the same as sd
    term1=jnp.conjugate(deltaS)*(pulse_t+pulse_t_shifted)**2
    term2=deltaS*jnp.abs(pulse_t+pulse_t_shifted)**2
    grad=(1+exp_arr)*do_fft(term1+2*term2, sk, rn)
    return -2*grad


def Z_gradient_nhg_interferometric(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    grad=n*(1+exp_arr)*do_fft(deltaS*jnp.conjugate(pulse_t+pulse_t_shifted)**(n-1), sk, rn)
    return -2*grad






def Z_gradient_shg_interferometric_cross_correlation_pulse(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    grad=2*do_fft(deltaS*jnp.conjugate(pulse_t+pulse_t_shifted), sk, rn)
    return -2*grad



def Z_gradient_thg_interferometric_cross_correlation_pulse(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    grad=3*do_fft(deltaS*jnp.conjugate(pulse_t+pulse_t_shifted)**2, sk, rn)
    return -2*grad



def Z_gradient_pg_interferometric_cross_correlation_pulse(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    # is the same as sd
    term1=jnp.conjugate(deltaS)*(pulse_t+pulse_t_shifted)**2
    term2=deltaS*jnp.abs(pulse_t+pulse_t_shifted)**2
    grad=do_fft(term1+2*term2, sk, rn)
    return -2*grad


def Z_gradient_nhg_interferometric_cross_correlation_pulse(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    grad=n*do_fft(deltaS*jnp.conjugate(pulse_t+pulse_t_shifted)**(n-1), sk, rn)
    return -2*grad






def Z_gradient_shg_interferometric_cross_correlation_gate(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    grad=2*exp_arr*do_fft(deltaS*jnp.conjugate(pulse_t+pulse_t_shifted), sk, rn)
    return -2*grad


def Z_gradient_thg_interferometric_cross_correlation_gate(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    grad=3*exp_arr*do_fft(deltaS*jnp.conjugate(pulse_t+pulse_t_shifted)**2, sk, rn)
    return -2*grad


def Z_gradient_pg_interferometric_cross_correlation_gate(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    # is the same as sd
    term1=jnp.conjugate(deltaS)*(pulse_t+pulse_t_shifted)**2
    term2=deltaS*jnp.abs(pulse_t+pulse_t_shifted)**2
    grad=exp_arr*do_fft(term1+2*term2, sk, rn)
    return -2*grad


def Z_gradient_nhg_interferometric_cross_correlation_gate(deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n):
    grad=n*exp_arr*do_fft(deltaS*jnp.conjugate(pulse_t+pulse_t_shifted)**(n-1), sk, rn)
    return -2*grad






















def calculate_Z_gradient_pulse(signal_t, signal_t_new, pulse_t, pulse_t_shifted, gate_shifted, tau_arr, measurement_info, is_tdp):
    frequency, sk, rn = measurement_info.frequency, measurement_info.sk, measurement_info.rn
    cross_correlation, doubleblind, interferometric, frogmethod = measurement_info.cross_correlation, measurement_info.doubleblind, measurement_info.interferometric, measurement_info.nonlinear_method

    omega_arr = 2*jnp.pi*frequency
    exp_arr = jnp.exp(1j*jnp.outer(tau_arr, omega_arr))

    if is_tdp==True:
        exp_arr = exp_arr*jnp.conjugate(measurement_info.spectral_filter)
    else:
        pass

    deltaS = signal_t_new - signal_t


    if cross_correlation==True or doubleblind==True:
        cross_correlation=True

    if frogmethod[-2:]=="hg" and frogmethod!="shg" and frogmethod!="thg":
        n = int(frogmethod[0])
        frogmethod="nhg"
    else:
        n = None


    grad_func_interferometric_False_cross_correlation_False={"shg": Z_gradient_shg, "thg": Z_gradient_thg, "pg": Z_gradient_pg, "sd": Z_gradient_sd, "nhg": Z_gradient_nhg}
    grad_func_interferometric_False_cross_correlation_True={"shg": Z_gradient_cross_correlation_pulse, "thg": Z_gradient_cross_correlation_pulse, "pg": Z_gradient_cross_correlation_pulse, "sd": Z_gradient_cross_correlation_pulse, "nhg": Z_gradient_cross_correlation_pulse}

    grad_func_interferometric_True_cross_correlation_False={"shg": Z_gradient_shg_interferometric, "thg": Z_gradient_thg_interferometric, "pg": Z_gradient_pg_interferometric, "nhg": Z_gradient_nhg_interferometric}
    grad_func_interferometric_True_cross_correlation_True={"shg": Z_gradient_shg_interferometric_cross_correlation_pulse, "thg": Z_gradient_thg_interferometric_cross_correlation_pulse, "pg": Z_gradient_pg_interferometric_cross_correlation_pulse, "nhg": Z_gradient_nhg_interferometric_cross_correlation_pulse}

    grad_func_interferometric_False={False: grad_func_interferometric_False_cross_correlation_False,
                           True: grad_func_interferometric_False_cross_correlation_True}
    

    grad_func_interferometric_True={False: grad_func_interferometric_True_cross_correlation_False,
                          True: grad_func_interferometric_True_cross_correlation_True}
    
    grad_func={False: grad_func_interferometric_False,
               True: grad_func_interferometric_True}
    
    grad = grad_func[interferometric][cross_correlation][frogmethod](deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n)
    return grad









def calculate_Z_gradient_gate(signal_t, signal_t_new, pulse_t, pulse_t_shifted, gate_shifted, tau_arr, measurement_info, is_tdp):
    frequency, sk, rn = measurement_info.frequency, measurement_info.sk, measurement_info.rn
    interferometric, frogmethod = measurement_info.interferometric, measurement_info.nonlinear_method

    omega_arr = 2*jnp.pi*frequency
    exp_arr = jnp.exp(1j*jnp.outer(tau_arr, omega_arr))

    if is_tdp==True:
        exp_arr = exp_arr*jnp.conjugate(measurement_info.spectral_filter)
    else:
        pass

    deltaS = signal_t_new-signal_t

    if frogmethod[-2:]=="hg" and frogmethod!="shg" and frogmethod!="thg":
        n = int(frogmethod[0])
        frogmethod="nhg"
    else:
        n = None


    grad_func_interferometric_False_cross_correlation_gate={"shg": Z_gradient_shg_cross_correlation_gate, 
                                      "thg": Z_gradient_thg_cross_correlation_gate, 
                                      "pg": Z_gradient_pg_cross_correlation_gate, 
                                      "sd": Z_gradient_sd_cross_correlation_gate,
                                      "nhg": Z_gradient_nhg_cross_correlation_gate}
    
    grad_func_interferometric_True_cross_correlation_gate={"shg": Z_gradient_shg_interferometric_cross_correlation_gate, 
                                     "thg": Z_gradient_thg_interferometric_cross_correlation_gate, 
                                     "pg": Z_gradient_pg_interferometric_cross_correlation_gate,
                                     "nhg": Z_gradient_nhg_interferometric_cross_correlation_gate}
    
    grad_func={False: grad_func_interferometric_False_cross_correlation_gate,
               True: grad_func_interferometric_True_cross_correlation_gate}
    
    grad = grad_func[interferometric][frogmethod](deltaS, pulse_t, pulse_t_shifted, gate_shifted, exp_arr, sk, rn, n)
    return grad













def calculate_Z_gradient(signal_t, signal_t_new, pulse_t, pulse_t_shifted, gate_shifted, tau_arr, measurement_info, pulse_or_gate, is_tdp=False):
    """
    Calculates the Z-error gradient with respect to the pulse or the gate-pulse for a given FROG measurement. 
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
    calculate_Z_gradient_dict={"pulse": calculate_Z_gradient_pulse,
                               "gate": calculate_Z_gradient_gate}
    return calculate_Z_gradient_dict[pulse_or_gate](signal_t, signal_t_new, pulse_t, pulse_t_shifted, gate_shifted, tau_arr, measurement_info, is_tdp)