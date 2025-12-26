import jax.numpy as jnp
import jax
from jax.tree_util import Partial

from src.utilities import scan_helper, calculate_newton_direction




def calc_Z_error_pseudo_hessian_subelement_shg(pulse_t_dispersed, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k=2*jnp.abs(pulse_t_dispersed)**2*exp_arr_mn*jnp.conjugate(exp_arr_mp)
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val


def calc_Z_error_pseudo_hessian_subelement_thg(pulse_t_dispersed, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k=4.5*jnp.abs(pulse_t_dispersed)**4*exp_arr_mn*jnp.conjugate(exp_arr_mp)
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val



def calc_Z_error_pseudo_hessian_subelement_pg(pulse_t_dispersed, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k=2.5*jnp.abs(pulse_t_dispersed)**4
    Vzz_k=2*jnp.real(jnp.conjugate(pulse_t_dispersed)*deltaS_m)
    Hzz_k=Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k*exp_arr_mn*jnp.conjugate(exp_arr_mp)
    return val


def calc_Z_error_pseudo_hessian_subelement_nhg(pulse_t_dispersed, deltaS_m, D_arr_pn, exp_arr_mn, exp_arr_mp, n):
    Uzz_k=0.5*n**2*jnp.abs(pulse_t_dispersed)**(2*n-2)*exp_arr_mn*jnp.conjugate(exp_arr_mp)
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    val = D_arr_pn*Hzz_k
    return val






def calc_Z_error_pseudo_hessian_element(exp_arr_mp, exp_arr_mn, omega_p, omega_n, time, pulse_t_dispersed, deltaS_m, nonlinear_method):
    """ Sum over time axis via jax.lax.scan. """
    
    D_arr_pn=jnp.exp(1j*time*(omega_p-omega_n))
    
    if nonlinear_method[-2:]=="hg" and nonlinear_method!="shg" and nonlinear_method!="thg":
        n = int(nonlinear_method[0])
        nonlinear_method = "nhg"
    else:
        n = None

    calc_hessian_subelement={"shg": calc_Z_error_pseudo_hessian_subelement_shg,
                             "thg": calc_Z_error_pseudo_hessian_subelement_thg,
                             "pg": calc_Z_error_pseudo_hessian_subelement_pg,
                             "sd": calc_Z_error_pseudo_hessian_subelement_pg,
                             "tg": calc_Z_error_pseudo_hessian_subelement_pg,
                             "nhg": calc_Z_error_pseudo_hessian_subelement_nhg}


    calc_subelement = Partial(calc_hessian_subelement[nonlinear_method], exp_arr_mn=exp_arr_mn, exp_arr_mp=exp_arr_mp, n=n)
    hessian_element_arr = jax.vmap(calc_subelement, in_axes=(0,0,0))(pulse_t_dispersed, deltaS_m, D_arr_pn)
    hessian_element = jnp.sum(hessian_element_arr)
    return hessian_element



def calc_Z_error_pseudo_hessian_one_m(dummy, exp_arr_m, pulse_t_dispersed, deltaS_m, time, omega, nonlinear_method, full_or_diagonal):
    """ jax.vmap over frequency axis """

    calc_hessian_partial=Partial(calc_Z_error_pseudo_hessian_element, time=time, pulse_t_dispersed=pulse_t_dispersed, deltaS_m=deltaS_m, 
                                 nonlinear_method=nonlinear_method)

    if full_or_diagonal=="full":
        calc_hessian=jax.vmap(jax.vmap(calc_hessian_partial, in_axes=(0, None, 0, None)), in_axes=(None, 0, None, 0))

    elif full_or_diagonal=="diagonal":
        calc_hessian=jax.vmap(calc_hessian_partial, in_axes=(0, 0, 0, 0))
    
    hessian=calc_hessian(exp_arr_m, exp_arr_m, omega, omega)
    return dummy, hessian



def calc_Z_error_pseudo_hessian_all_m(pulse_t_dispersed, deltaS, phase_matrix, measurement_info, full_or_diagonal):
    """ Loop over shifts to get hessian for each. Does not use jax.vmap because of memory limits. """
    time, omega, nonlinear_method = measurement_info.time, 2*jnp.pi*measurement_info.frequency, measurement_info.nonlinear_method

    exp_arr = jnp.exp(-1j*phase_matrix)

    hessian_all_m = Partial(calc_Z_error_pseudo_hessian_one_m, time=time, omega=omega, nonlinear_method=nonlinear_method, full_or_diagonal=full_or_diagonal)

    carry = jnp.zeros(1)
    xs = (exp_arr, pulse_t_dispersed, deltaS)
    get_hessian_all_m = Partial(scan_helper, actual_function=hessian_all_m, number_of_args=1, number_of_xs=3)
    _, h_all_m = jax.lax.scan(get_hessian_all_m, carry, xs)
    return h_all_m




def get_pseudo_newton_direction_Z_error(grad_m, pulse_t_dispersed, signal_t, signal_t_new, phase_matrix, measurement_info, newton_state, newton_info, 
                                        full_or_diagonal):
    
    """
    Calculates the pseudo-newton direction for the Z-error of a chirp-scan.
    The direction is calculated in the frequency domain.

    Args:
        grad_m (jnp.array): the current Z-error gradient
        pulse_t_dispersed (jnp.array): the current guess after phase_matrix was applied
        signal_t (jnp.array): the current signal field
        signal_t_new (jnp.array): the current signal field projected onto the measured intensity
        phase_matrix (jnp.array): the applied phases
        measurement_info (Pytree): contains measurement data and parameters
        newton_state (Pytree): contains the current state of the hessian calculation, e.g. the previous newton direction
        newton_info (Pytree): contains parameters for the pseudo-newton direction calculation
        full_or_diagonal (str): calculate using the full or diagonal pseudo hessian?

    Returns:
        tuple[jnp.array, Pytree], the pseudo-newton direction and the updated newton_state
    
    """

    lambda_lm = newton_info.lambda_lm
    solver = newton_info.linalg_solver
    newton_direction_prev = newton_state.pulse.newton_direction_prev
    deltaS = signal_t_new-signal_t

    # vmap over population here -> only for small populations since memory will explode. 
    hessian_m=jax.vmap(calc_Z_error_pseudo_hessian_all_m, in_axes=(0,0,0,None,None))(pulse_t_dispersed, deltaS, phase_matrix, measurement_info, 
                                                                                       full_or_diagonal)

    return calculate_newton_direction(grad_m, hessian_m, lambda_lm, newton_direction_prev, solver, full_or_diagonal)
