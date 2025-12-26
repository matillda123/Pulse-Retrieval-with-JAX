import jax.numpy as jnp
import jax
from jax.tree_util import Partial

from src.utilities import calculate_newton_direction, scan_helper



def calc_Z_error_pseudo_hessian_subelement_shg(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    term1 = pulse_t_shifted_m+pulse_t*exp_arr_mn
    term2 = pulse_t_shifted_m+pulse_t*exp_arr_mp
    Uzz_k = 0.5*jnp.conjugate(term1)*term2
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_thg(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    term1 = pulse_t_shifted_m+2*pulse_t*exp_arr_mn
    term2 = pulse_t_shifted_m+2*pulse_t*exp_arr_mp
    Uzz_k = 0.5*jnp.abs(pulse_t_shifted_m)**2*jnp.conjugate(term1)*term2
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val



def calc_Z_error_pseudo_hessian_subelement_pg(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    term1 = pulse_t_shifted_m+pulse_t*exp_arr_mn
    term2 = pulse_t_shifted_m+pulse_t*exp_arr_mp
    Uzz_k = 0.5*(jnp.abs(pulse_t_shifted_m)**2*jnp.conjugate(term1)*term2 + jnp.abs(pulse_t*pulse_t_shifted_m)*2*jnp.conjugate(exp_arr_mn)*exp_arr_mp)

    term3 = pulse_t_shifted_m+pulse_t*exp_arr_mn
    term4 = pulse_t_shifted_m+pulse_t*exp_arr_mp
    Vzz_k = 0.5*(jnp.conjugate(term3)*deltaS_m*exp_arr_mp + term4*jnp.conjugate(deltaS_m*exp_arr_mn))

    Hzz_k = Uzz_k-Vzz_k
    val = D_arr_pn*Hzz_k
    return val



def calc_Z_error_pseudo_hessian_subelement_sd(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 0.5*(jnp.abs(pulse_t_shifted_m)**4+4*jnp.abs(pulse_t*pulse_t_shifted_m)**2*jnp.conjugate(exp_arr_mn)*exp_arr_mp)
    Vzz_k = pulse_t_shifted_m*deltaS_m*exp_arr_mp+jnp.conjugate(pulse_t_shifted_m*deltaS_m*exp_arr_mn)
    Hzz_k = Uzz_k-Vzz_k
    
    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_nhg(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    # should be correct -> pulse_t_shifted_m^(n-2) can be factored put of term1/2
    term1 = pulse_t_shifted_m+(n-1)*pulse_t*exp_arr_mn
    term2 = pulse_t_shifted_m+(n-1)*pulse_t*exp_arr_mp
    Uzz_k = 0.5*jnp.abs(pulse_t_shifted_m)**(2*n-4)*jnp.conjugate(term1)*term2
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val







def calc_Z_error_pseudo_hessian_subelement_cross_correlation_pulse(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 0.5*jnp.abs(gate_shifted_m)**2
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k
    
    val = D_arr_pn*Hzz_k
    return val





def calc_Z_error_pseudo_hessian_subelement_shg_cross_correlation_gate(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 0.5*jnp.conjugate(exp_arr_mn)*exp_arr_mp*jnp.abs(pulse_t)**2
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_thg_cross_correlation_gate(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 2*jnp.conjugate(exp_arr_mn)*exp_arr_mp*jnp.abs(pulse_t)**2*jnp.abs(pulse_t_shifted_m)**2
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_pg_cross_correlation_gate(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 0.5*jnp.conjugate(exp_arr_mn)*exp_arr_mp*jnp.abs(pulse_t)**2*jnp.abs(pulse_t_shifted_m)**2
    Vzz_k = jnp.real(deltaS_m*jnp.conjugate(pulse_t))*jnp.conjugate(exp_arr_mn)*exp_arr_mp
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_sd_cross_correlation_gate(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 2*jnp.conjugate(exp_arr_mn)*exp_arr_mp*jnp.abs(pulse_t)**2*jnp.abs(pulse_t_shifted_m)**2
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_nhg_cross_correlation_gate(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 0.5*(n-1)**2*jnp.conjugate(exp_arr_mn)*exp_arr_mp*jnp.abs(pulse_t)**2*jnp.abs(pulse_t_shifted_m)**(2*n-4)
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val








def calc_Z_error_pseudo_hessian_subelement_shg_interferometric(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 2*jnp.conjugate(1+exp_arr_mn)*(1+exp_arr_mp)*jnp.abs(pulse_t+pulse_t_shifted_m)**2
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val

def calc_Z_error_pseudo_hessian_subelement_thg_interferometric(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 4.5*jnp.conjugate(1+exp_arr_mn)*(1+exp_arr_mp)*jnp.abs(pulse_t+pulse_t_shifted_m)**4  
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_pg_interferometric(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 2.5*jnp.conjugate(1+exp_arr_mn)*(1+exp_arr_mp)*jnp.abs(pulse_t+pulse_t_shifted_m)**4
    Vzz_k = 2*jnp.conjugate(1+exp_arr_mn)*(1+exp_arr_mp)*jnp.real((pulse_t+pulse_t_shifted_m)*jnp.conjugate(deltaS_m))
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_nhg_interferometric(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 0.5*n**2*jnp.conjugate(1+exp_arr_mn)*(1+exp_arr_mp)*jnp.abs(pulse_t+pulse_t_shifted_m)**(2*n-2)
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val






def calc_Z_error_pseudo_hessian_subelement_shg_interferometric_cross_correlation_pulse(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 2*jnp.abs(pulse_t+pulse_t_shifted_m)**2
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val

def calc_Z_error_pseudo_hessian_subelement_thg_interferometric_cross_correlation_pulse(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 4.5*jnp.abs(pulse_t+pulse_t_shifted_m)**4
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val

def calc_Z_error_pseudo_hessian_subelement_pg_interferometric_cross_correlation_pulse(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 2.5*jnp.abs(pulse_t+pulse_t_shifted_m)**4
    Vzz_k = 2*jnp.real(jnp.conjugate(pulse_t+pulse_t_shifted_m)*deltaS_m)
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_nhg_interferometric_cross_correlation_pulse(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 0.5*n**2*jnp.abs(pulse_t+pulse_t_shifted_m)**(2*n-2)
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val







def calc_Z_error_pseudo_hessian_subelement_shg_interferometric_cross_correlation_gate(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 2*jnp.abs(pulse_t+pulse_t_shifted_m)**2*jnp.conjugate(exp_arr_mn)*exp_arr_mp
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val

def calc_Z_error_pseudo_hessian_subelement_thg_interferometric_cross_correlation_gate(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 4.5*jnp.abs(pulse_t+pulse_t_shifted_m)**4*jnp.conjugate(exp_arr_mn)*exp_arr_mp
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val

def calc_Z_error_pseudo_hessian_subelement_pg_interferometric_cross_correlation_gate(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 2.5*jnp.abs(pulse_t+pulse_t_shifted_m)**4*jnp.conjugate(exp_arr_mn)*exp_arr_mp
    Vzz_k = 2*jnp.real(jnp.conjugate(pulse_t+pulse_t_shifted_m)*deltaS_m)*jnp.conjugate(exp_arr_mn)*exp_arr_mp
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_nhg_interferometric_cross_correlation_gate(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k = 0.5*n**2*jnp.abs(pulse_t+pulse_t_shifted_m)**(2*n-2)*jnp.conjugate(exp_arr_mn)*exp_arr_mp
    Vzz_k = 0
    Hzz_k = Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val






def calc_Z_error_pseudo_hessian_element_pulse(exp_arr_mp, exp_arr_mn, omega_p, omega_n, time_k, pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, 
                                              frogmethod, cross_correlation, interferometric):
    
    """ Sum over time axis via jax.lax.scan. Does not use jax.vmap because of memory limits. """
    
    D_arr_pn=jnp.exp(1j*time_k*(omega_p-omega_n))


    if cross_correlation==True or cross_correlation=="doubleblind":
        cross_correlation=True
    else:
        cross_correlation=False

    if frogmethod[-2:]=="hg" and frogmethod!="shg" and frogmethod!="thg":
        n = int(frogmethod[0])
        frogmethod = "nhg"
    else:
        n = None


    hess_func_interferometric_False_cross_correlation_False={"shg": calc_Z_error_pseudo_hessian_subelement_shg, 
                                       "thg": calc_Z_error_pseudo_hessian_subelement_thg, 
                                       "pg": calc_Z_error_pseudo_hessian_subelement_pg, 
                                       "sd": calc_Z_error_pseudo_hessian_subelement_sd,
                                       "nhg": calc_Z_error_pseudo_hessian_subelement_nhg}
    
    hess_func_interferometric_False_cross_correlation_True={"shg": calc_Z_error_pseudo_hessian_subelement_cross_correlation_pulse, 
                                       "thg": calc_Z_error_pseudo_hessian_subelement_cross_correlation_pulse, 
                                       "pg": calc_Z_error_pseudo_hessian_subelement_cross_correlation_pulse, 
                                       "sd": calc_Z_error_pseudo_hessian_subelement_cross_correlation_pulse,
                                       "nhg": calc_Z_error_pseudo_hessian_subelement_cross_correlation_pulse}

    hess_func_interferometric_True_cross_correlation_False={"shg": calc_Z_error_pseudo_hessian_subelement_shg_interferometric,
                                     "thg": calc_Z_error_pseudo_hessian_subelement_thg_interferometric,
                                     "pg": calc_Z_error_pseudo_hessian_subelement_pg_interferometric,
                                     "nhg": calc_Z_error_pseudo_hessian_subelement_nhg_interferometric}
    
    hess_func_interferometric_True_cross_correlation_True={"shg": calc_Z_error_pseudo_hessian_subelement_shg_interferometric_cross_correlation_pulse,
                                     "thg": calc_Z_error_pseudo_hessian_subelement_thg_interferometric_cross_correlation_pulse,
                                     "pg": calc_Z_error_pseudo_hessian_subelement_pg_interferometric_cross_correlation_pulse,
                                     "nhg": calc_Z_error_pseudo_hessian_subelement_nhg_interferometric_cross_correlation_pulse}


    hess_func_interferometric_False={False: hess_func_interferometric_False_cross_correlation_False,
                           True: hess_func_interferometric_False_cross_correlation_True}
    

    hess_func_interferometric_True={False: hess_func_interferometric_True_cross_correlation_False,
                          True: hess_func_interferometric_True_cross_correlation_True}
    
    
    calc_hessian_subelement={False: hess_func_interferometric_False,
                             True: hess_func_interferometric_True}

    
    calc_subelement = Partial(calc_hessian_subelement[interferometric][cross_correlation][frogmethod], exp_arr_mn=exp_arr_mn, exp_arr_mp=exp_arr_mp, n=n)

    hessian_element_arr = jax.vmap(calc_subelement, in_axes=(0,0,0,0,0))(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn)
    hessian_element = jnp.sum(hessian_element_arr)
    return hessian_element







def calc_Z_error_pseudo_hessian_element_gate(exp_arr_mp, exp_arr_mn, omega_p, omega_n, time_k, pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, 
                                              frogmethod, cross_correlation, interferometric):
    
    """ Sum over time axis via jax.lax.scan. """
    
    D_arr_pn=jnp.exp(1j*time_k*(omega_p-omega_n))

    if frogmethod[-2:]=="hg" and frogmethod!="shg" and frogmethod!="thg":
        n = int(frogmethod[0])
        frogmethod = "nhg"
    else:
        n = None

    
    hess_func_interferometric_False_cross_correlation_gate={"shg": calc_Z_error_pseudo_hessian_subelement_shg_cross_correlation_gate, 
                                       "thg": calc_Z_error_pseudo_hessian_subelement_thg_cross_correlation_gate, 
                                       "pg": calc_Z_error_pseudo_hessian_subelement_pg_cross_correlation_gate, 
                                       "sd": calc_Z_error_pseudo_hessian_subelement_sd_cross_correlation_gate,
                                       "nhg": calc_Z_error_pseudo_hessian_subelement_nhg_cross_correlation_gate}
    
    
    hess_func_interferometric_True_cross_correlation_gate={"shg": calc_Z_error_pseudo_hessian_subelement_shg_interferometric_cross_correlation_gate,
                                     "thg": calc_Z_error_pseudo_hessian_subelement_thg_interferometric_cross_correlation_gate,
                                     "pg": calc_Z_error_pseudo_hessian_subelement_pg_interferometric_cross_correlation_gate,
                                     "nhg": calc_Z_error_pseudo_hessian_subelement_nhg_interferometric_cross_correlation_gate}
    
    
    calc_hessian_subelement={False: hess_func_interferometric_False_cross_correlation_gate,
                             True: hess_func_interferometric_True_cross_correlation_gate}

    
    calc_subelement=Partial(calc_hessian_subelement[interferometric][frogmethod], exp_arr_mn=exp_arr_mn, exp_arr_mp=exp_arr_mp, n=n)

    hessian_element_arr = jax.vmap(calc_subelement, in_axes=(0,0,0,0,0))(pulse_t, pulse_t_shifted_m, gate_shifted_m, deltaS_m, D_arr_pn)
    hessian_element = jnp.sum(hessian_element_arr)
    return hessian_element






def calc_Z_error_pseudo_hessian_one_m(dummy, exp_arr_m, pulse_t_shifted_m, gate_shifted_m, deltaS_m, 
                                      pulse_t, time, omega, frogmethod, cross_correlation, interferometric, full_or_diagonal, pulse_or_gate):
    """ jax.vmap over frequency axis """

    calc_Z_error_pseudo_hessian_element = {"pulse": calc_Z_error_pseudo_hessian_element_pulse,
                                           "gate": calc_Z_error_pseudo_hessian_element_gate}
    calc_hessian_partial=Partial(calc_Z_error_pseudo_hessian_element[pulse_or_gate], time_k=time, pulse_t=pulse_t, pulse_t_shifted_m=pulse_t_shifted_m, 
                                 gate_shifted_m=gate_shifted_m, deltaS_m=deltaS_m, frogmethod=frogmethod, cross_correlation=cross_correlation, interferometric=interferometric)

    if full_or_diagonal=="full":
        calc_hessian=jax.vmap(jax.vmap(calc_hessian_partial, in_axes=(0, None, 0, None)), in_axes=(None, 0, None, 0))

    elif full_or_diagonal=="diagonal":
        calc_hessian=jax.vmap(calc_hessian_partial, in_axes=(0, 0, 0, 0))
    
    hessian=calc_hessian(exp_arr_m, exp_arr_m, omega, omega)
    return dummy, hessian



def calc_Z_error_pseudo_hessian_all_m(pulse_t, pulse_t_shifted, gate_shifted, deltaS, tau_arr, measurement_info, full_or_diagonal, pulse_or_gate, is_tdp):
    """ Loop over shifts to get hessian for each. Does not use jax.vmap because of memory limits. """

    time, omega = measurement_info.time, 2*jnp.pi*measurement_info.frequency
    cross_correlation, interferometric, frogmethod = measurement_info.cross_correlation, measurement_info.interferometric, measurement_info.nonlinear_method

    exp_arr = jnp.exp(-1j*jnp.outer(tau_arr, omega))
    if is_tdp==True:
        exp_arr = exp_arr*measurement_info.spectral_filter
    else:
        pass

    hessian_all_m = Partial(calc_Z_error_pseudo_hessian_one_m, pulse_t=pulse_t, time=time, omega=omega, frogmethod=frogmethod, cross_correlation=cross_correlation, interferometric=interferometric, 
                          full_or_diagonal=full_or_diagonal, pulse_or_gate=pulse_or_gate)

    carry = jnp.zeros(1)
    xs = (exp_arr, pulse_t_shifted, gate_shifted, deltaS)

    get_hessian_all_m = Partial(scan_helper, actual_function=hessian_all_m, number_of_args=1, number_of_xs=4)
    _, h_all_m = jax.lax.scan(get_hessian_all_m, carry, xs)
    return h_all_m




def get_pseudo_newton_direction_Z_error(grad_m, pulse_t, pulse_t_shifted, gate_shifted, signal_t, signal_t_new, tau_arr, measurement_info, 
                                        newton_state, newton_info, full_or_diagonal, pulse_or_gate):
    
    """
    Calculates the pseudo-newton direction for the Z-error of a FROG measurement.
    The direction is calculated in the frequency domain.

    Args:
        grad_m (jnp.array): the current Z-error gradient
        pulse_t (jnp.array): the current guess
        pulse_t_shifted (jnp.array): the current guess shifted along the time axis
        gate_shifted (jnp.array): the current gate guess shifted along the time axis
        signal_t (jnp.array): the current signal field
        signal_t_new (jnp.array): the current signal field projected onto the measured intensity
        tau_arr (jnp.array): the time delays
        measurement_info (Pytree): contains measurement data and parameters
        newton_state (Pytree): contains the current state of the hessian calculation, e.g. the previous newton direction
        newton_info (Pytree): contains parameters for the pseudo-newton direction calculation
        full_or_diagonal (str): calculate using the full or diagonal pseudo hessian?
        pulse_or_gate (str): whether the direction is calculated for the pulse or the gate-pulse

    Returns:
        tuple[jnp.array, Pytree], the pseudo-newton direction and the updated newton_state
    
    """


    lambda_lm = newton_info.lambda_lm
    solver = newton_info.linalg_solver
    newton_direction_prev = getattr(newton_state, pulse_or_gate).newton_direction_prev  
    deltaS = signal_t_new-signal_t

    # vmap over population here -> only for small populations since memory will explode. 
    calc_hessian = Partial(calc_Z_error_pseudo_hessian_all_m, is_tdp=False)
    hessian_m=jax.vmap(calc_hessian, in_axes=(0,0,0,0,0,None,None,None))(pulse_t, pulse_t_shifted, gate_shifted, deltaS, 
                                                                        tau_arr, measurement_info, full_or_diagonal, pulse_or_gate)
    
    return calculate_newton_direction(grad_m, hessian_m, lambda_lm, newton_direction_prev, solver, full_or_diagonal)

        











