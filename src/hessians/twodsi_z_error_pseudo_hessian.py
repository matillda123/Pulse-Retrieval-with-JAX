import jax.numpy as jnp
import jax
from jax.tree_util import Partial

from src.utilities import calculate_newton_direction, scan_helper



def calc_Z_error_pseudo_hessian_subelement_shg_pulse(pulse_t, gate_pulses_m, gate_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k = 0.5*jnp.conjugate(gate_m + pulse_t*exp_arr_mn)*(gate_m + pulse_t*exp_arr_mp)
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_thg_pulse(pulse_t, gate_pulses_m, gate_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k = 0.5*jnp.conjugate(gate_m + 2*pulse_t*gate_pulses_m*exp_arr_mn)*(gate_m + 2*pulse_t*gate_pulses_m*exp_arr_mp)
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_pg_pulse(pulse_t, gate_pulses_m, gate_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    term1 = gate_m + jnp.conjugate(pulse_t)*gate_pulses_m*jnp.conjugate(exp_arr_mn)
    term2 = gate_m + pulse_t*jnp.conjugate(gate_pulses_m)*exp_arr_mp
    term3 = jnp.conjugate(gate_pulses_m)*exp_arr_mp*deltaS_m
    term4 = gate_pulses_m*jnp.conjugate(exp_arr_mn)*jnp.conjugate(deltaS_m)
    Uzz_k = 0.5*(term1*term2 + jnp.abs(pulse_t)**2*gate_m*jnp.conjugate(exp_arr_mn)*exp_arr_mp)
    Vzz_k = 0.5*(term3 + term4 + 2*jnp.real(jnp.conjugate(pulse_t)*deltaS_m)*jnp.conjugate(exp_arr_mn)*exp_arr_mp)
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_sd_pulse(pulse_t, gate_pulses_m, gate_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    term1 = 2*gate_pulses_m*deltaS_m
    Uzz_k = 0.5*(jnp.abs(gate_pulses_m)**4 + 4*jnp.abs(pulse_t)**2*jnp.abs(gate_pulses_m)**2*jnp.conjugate(exp_arr_mn)*exp_arr_mp)
    Vzz_k = 0.5*(term1*exp_arr_mp + jnp.conjugate(term1*exp_arr_mn))
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val







def calc_Z_error_pseudo_hessian_subelement_cross_correlation_pulse(pulse_t, gate_pulses_m, gate_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k = 0.5*jnp.abs(gate_m)**2
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k
    
    val = D_arr_pn*Hzz_k
    return val





def calc_Z_error_pseudo_hessian_subelement_shg_cross_correlation_gate(pulse_t, gate_pulses_m, gate_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k = 0.5*jnp.conjugate(exp_arr_mn)*exp_arr_mp*jnp.abs(pulse_t)**2
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_thg_cross_correlation_gate(pulse_t, gate_pulses_m, gate_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k = 2*jnp.conjugate(exp_arr_mn)*exp_arr_mp*jnp.abs(pulse_t*gate_pulses_m)**2
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_pg_cross_correlation_gate(pulse_t, gate_pulses_m, gate_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k = jnp.conjugate(exp_arr_mn)*exp_arr_mp*jnp.abs(pulse_t*gate_pulses_m)**2
    Vzz_k = jnp.real(deltaS_m*jnp.conjugate(pulse_t))*jnp.conjugate(exp_arr_mn)*exp_arr_mp
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_sd_cross_correlation_gate(pulse_t, gate_pulses_m, gate_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k = 2*jnp.conjugate(exp_arr_mn)*exp_arr_mp*jnp.abs(pulse_t*gate_pulses_m)**2
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val








def calc_Z_error_pseudo_hessian_element_pulse(exp_arr_mp, exp_arr_mn, omega_p, omega_n, time_k, pulse_t, gate_pulses_m, gate_m, deltaS_m, 
                                              nonlinear_method, cross_correlation):
    """ Sum over time axis via jax.lax.scan. Does not use jax.vmap because of memory limits. """
    
    D_arr_pn=jnp.exp(1j*time_k*(omega_p-omega_n))

    func_dict_xcorr = {"shg": calc_Z_error_pseudo_hessian_subelement_cross_correlation_pulse, "thg": calc_Z_error_pseudo_hessian_subelement_cross_correlation_pulse, 
                       "pg": calc_Z_error_pseudo_hessian_subelement_cross_correlation_pulse, "sd": calc_Z_error_pseudo_hessian_subelement_cross_correlation_pulse}
    
    func_dict_ac = {"shg": calc_Z_error_pseudo_hessian_subelement_shg_pulse, "thg": calc_Z_error_pseudo_hessian_subelement_thg_pulse, 
                    "pg": calc_Z_error_pseudo_hessian_subelement_pg_pulse, "sd": calc_Z_error_pseudo_hessian_subelement_sd_pulse}

    func_dict = {True: func_dict_xcorr,
                 False: func_dict_ac}

    calc_subelement=Partial(func_dict[cross_correlation][nonlinear_method], exp_arr_mn=exp_arr_mn, exp_arr_mp=exp_arr_mp)

    hessian_element_arr = jax.vmap(calc_subelement, in_axes=(0,0,0,0,0))(pulse_t, gate_pulses_m, gate_m, deltaS_m, D_arr_pn)
    hessian_element = jnp.sum(hessian_element_arr)

    return hessian_element







def calc_Z_error_pseudo_hessian_element_gate(exp_arr_mp, exp_arr_mn, omega_p, omega_n, time_k, pulse_t, gate_pulses_m, gate_m, deltaS_m, 
                                              nonlinear_method, cross_correlation):
    
    """ Sum over time axis via jax.lax.scan. Does not use jax.vmap because of memory limits. """
    
    D_arr_pn=jnp.exp(1j*time_k*(omega_p-omega_n))

    calc_hessian_subelement={"shg": calc_Z_error_pseudo_hessian_subelement_shg_cross_correlation_gate,
                             "thg": calc_Z_error_pseudo_hessian_subelement_thg_cross_correlation_gate,
                             "pg": calc_Z_error_pseudo_hessian_subelement_pg_cross_correlation_gate,
                             "sd": calc_Z_error_pseudo_hessian_subelement_sd_cross_correlation_gate}

    calc_subelement=Partial(calc_hessian_subelement[nonlinear_method], exp_arr_mn=exp_arr_mn, exp_arr_mp=exp_arr_mp)

    hessian_element_arr = jax.vmap(calc_subelement, in_axes=(0,0,0,0,0))(pulse_t, gate_pulses_m, gate_m, deltaS_m, D_arr_pn)
    hessian_element = jnp.sum(hessian_element_arr)

    return hessian_element






def calc_Z_error_pseudo_hessian_one_m(dummy, exp_arr_m, gate_pulses_m, gate_m, deltaS_m, 
                                      pulse_t, time, omega, nonlinear_method, cross_correlation, full_or_diagonal, pulse_or_gate):
    """ jax.vmap ovet the frequency axis """

    calc_Z_error_pseudo_hessian_element = {"pulse": calc_Z_error_pseudo_hessian_element_pulse,
                                           "gate": calc_Z_error_pseudo_hessian_element_gate}
    
    calc_hessian_partial=Partial(calc_Z_error_pseudo_hessian_element[pulse_or_gate], time_k=time, pulse_t=pulse_t, gate_pulses_m=gate_pulses_m, 
                                 gate_m=gate_m, deltaS_m=deltaS_m, nonlinear_method=nonlinear_method, 
                                 cross_correlation=cross_correlation)

    if full_or_diagonal=="full":
        calc_hessian=jax.vmap(jax.vmap(calc_hessian_partial, in_axes=(0, None, 0, None)), in_axes=(None, 0, None, 0))

    elif full_or_diagonal=="diagonal":
        calc_hessian=jax.vmap(calc_hessian_partial, in_axes=(0, 0, 0, 0))
    
    hessian=calc_hessian(exp_arr_m, exp_arr_m, omega, omega)
    return dummy, hessian



def calc_Z_error_pseudo_hessian_all_m(pulse_t, gate_pulses, gate, deltaS, tau_arr, delay, measurement_info, full_or_diagonal, pulse_or_gate, is_vampire):
    """ jax.vmap along the delays """
    time, omega = measurement_info.time, 2*jnp.pi*measurement_info.frequency
    nonlinear_method = measurement_info.nonlinear_method

    if measurement_info.cross_correlation==True or measurement_info.doubleblind==True:
        cross_correlation=True
    else:
        cross_correlation=False

    if is_vampire==True:
        tau, phase_matrix = measurement_info.tau_interferometer, measurement_info.phase_matrix
        exp_arr = jnp.exp(-1j*jnp.outer(tau_arr, omega))*(jnp.exp(-1j*omega*tau) + jnp.exp(1j*(phase_matrix-omega*delay)))
    else:
        spectral_filter1, spectral_filter2 = measurement_info.spectral_filter1, measurement_info.spectral_filter2
        tau, phase_matrix = measurement_info.tau_pulse_anc1, measurement_info.phase_matrix
        exp_arr = (spectral_filter1*jnp.exp(-1j*omega*tau) + spectral_filter2*jnp.exp(-1j*jnp.outer(tau_arr, omega)))*jnp.exp(1j*(phase_matrix-omega*delay))


    hessian_all_m=Partial(calc_Z_error_pseudo_hessian_one_m, pulse_t=pulse_t, time=time, omega=omega, nonlinear_method=nonlinear_method, 
                          cross_correlation=cross_correlation,
                          full_or_diagonal=full_or_diagonal, pulse_or_gate=pulse_or_gate)

    carry = jnp.zeros(1)
    xs = (exp_arr, gate_pulses, gate, deltaS)
    get_hessian_all_m = Partial(scan_helper, actual_function=hessian_all_m, number_of_args=1, number_of_xs=4)
    _, h_all_m = jax.lax.scan(get_hessian_all_m, carry, xs)

    return h_all_m




def get_pseudo_newton_direction_Z_error(grad_m, pulse_t, gate_pulses, gate, signal_t, signal_t_new, tau_arr, delay, measurement_info, 
                                        newton_state, newton_info, full_or_diagonal, pulse_or_gate):
    
    """
    Calculates the pseudo-newton direction for the Z-error of a 2DSI measurement.
    The direction is calculated in the frequency domain.

    Args:
        grad_m: jnp.array, the current Z-error gradient
        pulse_t: jnp.array, the current guess
        gate_pulses: jnp.array, the currently guessed gate-pulses
        gate: jnp.array, the current gate
        signal_t: jnp.array, the current signal field
        signal_t_new: jnp.array, the current signal field projected onto the measured intensity
        tau_arr: jnp.array, the applied delays
        measurement_info: Pytree, contains measurement data and parameters
        newton_state: Pytree, contains the current state of the hessian calculation, e.g. the previous newton direction
        newton_info: Pytree, contains parameters for the pseudo-newton direction calculation
        full_or_diagonal: str, calculate using the full or diagonal pseudo hessian?
        pulse_or_gate: str, whether the direction is calculated for the pulse or the gate-pulse

    Returns:
        tuple[jnp.array, Pytree], the pseudo-newton direction and the updated newton_state
    
    """
     

    lambda_lm = newton_info.lambda_lm
    solver = newton_info.linalg_solver
    newton_direction_prev = getattr(newton_state, pulse_or_gate).newton_direction_prev  
    deltaS = signal_t_new-signal_t

    # vmap over population here -> only for small populations since memory will explode. 
    calc_hessian = Partial(calc_Z_error_pseudo_hessian_all_m, is_vampire=False)
    hessian_m=jax.vmap(calc_hessian, in_axes=(0,0,0,0,0,0,None,None,None))(pulse_t, gate_pulses, gate, deltaS, 
                                                                                                tau_arr, delay, measurement_info, full_or_diagonal, pulse_or_gate)

    return calculate_newton_direction(grad_m, hessian_m, lambda_lm, newton_direction_prev, solver, full_or_diagonal)
        