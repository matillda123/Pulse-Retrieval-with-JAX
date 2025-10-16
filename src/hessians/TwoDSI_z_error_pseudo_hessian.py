import jax.numpy as jnp
import jax
from jax.tree_util import Partial

from src.utilities import calculate_newton_direction, scan_helper



def calc_Z_error_pseudo_hessian_subelement_pulse(dummy_element, pulse_t, gate_pulses_m, gate_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=0.5*jnp.abs(gate_m)**2
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k
    
    res=D_arr_pn*Hzz_k
    return dummy_element + res, None





def calc_Z_error_pseudo_hessian_subelement_shg_gate(dummy_element, pulse_t, gate_pulses_m, gate_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=0.5*(1+jnp.conjugate(exp_arr_mn))*(1+exp_arr_mp)*jnp.abs(pulse_t)**2
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None


def calc_Z_error_pseudo_hessian_subelement_thg_gate(dummy_element, pulse_t, gate_pulses_m, gate_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=2*(1+jnp.conjugate(exp_arr_mn))*(1+exp_arr_mp)*jnp.abs(pulse_t*gate_pulses_m)**2
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None


def calc_Z_error_pseudo_hessian_subelement_pg_gate(dummy_element, pulse_t, gate_pulses_m, gate_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=(1+jnp.conjugate(exp_arr_mn))*(1+exp_arr_mp)*jnp.abs(pulse_t*gate_pulses_m)**2
    Vzz_k=jnp.real((signal_t_new_m-signal_t_m)*jnp.conjugate(pulse_t))*(1+jnp.conjugate(exp_arr_mn))*(1+exp_arr_mp)
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None


def calc_Z_error_pseudo_hessian_subelement_sd_gate(dummy_element, pulse_t, gate_pulses_m, gate_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=2*(1+jnp.conjugate(exp_arr_mn))*(1+exp_arr_mp)*jnp.abs(pulse_t*gate_pulses_m)**2
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None








def calc_Z_error_pseudo_hessian_element_pulse(exp_arr_mp, exp_arr_mn, omega_p, omega_n, time_k, pulse_t, gate_pulses_m, gate_m, signal_t_m, signal_t_new_m, 
                                              nonlinear_method):
    """ Sum over time axis via jax.lax.scan. Does not use jax.vmap because of memory limits. """
    
    D_arr_pn=jnp.exp(1j*time_k*(omega_p-omega_n))

    calc_subelement=Partial(calc_Z_error_pseudo_hessian_subelement_pulse, exp_arr_mn=exp_arr_mn, exp_arr_mp=exp_arr_mp)
    element_scan=Partial(scan_helper, actual_function=calc_subelement, number_of_args=1, number_of_xs=6)

    carry=0+0j
    xs=(pulse_t, gate_pulses_m, gate_m, signal_t_m, signal_t_new_m, D_arr_pn)
    hessian_element, _ =jax.lax.scan(element_scan, carry, xs)

    return hessian_element







def calc_Z_error_pseudo_hessian_element_gate(exp_arr_mp, exp_arr_mn, omega_p, omega_n, time_k, pulse_t, gate_pulses_m, gate_m, signal_t_m, signal_t_new_m, 
                                              nonlinear_method):
    
    """ Sum over time axis via jax.lax.scan. Does not use jax.vmap because of memory limits. """
    
    D_arr_pn=jnp.exp(1j*time_k*(omega_p-omega_n))

    calc_hessian_subelement={"shg": calc_Z_error_pseudo_hessian_subelement_shg_gate,
                             "thg": calc_Z_error_pseudo_hessian_subelement_thg_gate,
                             "pg": calc_Z_error_pseudo_hessian_subelement_pg_gate,
                             "sd": calc_Z_error_pseudo_hessian_subelement_sd_gate}

    calc_subelement=Partial(calc_hessian_subelement[nonlinear_method], exp_arr_mn=exp_arr_mn, exp_arr_mp=exp_arr_mp)
    element_scan=Partial(scan_helper, actual_function=calc_subelement, number_of_args=1, number_of_xs=6)

    carry=0+0j
    xs=(pulse_t, gate_pulses_m, gate_m, signal_t_m, signal_t_new_m, D_arr_pn)
    hessian_element, _ =jax.lax.scan(element_scan, carry, xs)

    return hessian_element






def calc_Z_error_pseudo_hessian_one_m(exp_arr_m, gate_pulses_m, gate_m, signal_t_m, signal_t_new_m, 
                                      pulse_t, time, omega, nonlinear_method, full_or_diagonal, pulse_or_gate):
    """ jax.vmap ovet the frequency axis """

    calc_Z_error_pseudo_hessian_element = {"pulse": calc_Z_error_pseudo_hessian_element_pulse,
                                           "gate": calc_Z_error_pseudo_hessian_element_gate}
    
    calc_hessian_partial=Partial(calc_Z_error_pseudo_hessian_element[pulse_or_gate], time_k=time, pulse_t=pulse_t, gate_pulses_m=gate_pulses_m, 
                                 gate_m=gate_m, signal_t_m=signal_t_m, signal_t_new_m=signal_t_new_m, nonlinear_method=nonlinear_method)

    if full_or_diagonal=="full":
        calc_hessian=jax.vmap(jax.vmap(calc_hessian_partial, in_axes=(0, None, 0, None)), in_axes=(None, 0, None, 0))

    elif full_or_diagonal=="diagonal":
        calc_hessian=jax.vmap(calc_hessian_partial, in_axes=(0, 0, 0, 0))
    
    hessian=calc_hessian(exp_arr_m, exp_arr_m, omega, omega)
    return hessian



def calc_Z_error_pseudo_hessian_all_m(pulse_t, gate_pulses, gate, signal_t, signal_t_new, tau_arr, measurement_info, full_or_diagonal, pulse_or_gate):
    """ jax.vmap along the delays """
    time, omega = measurement_info.time, 2*jnp.pi*measurement_info.frequency
    nonlinear_method = measurement_info.nonlinear_method

    exp_arr = jnp.exp(-1j*jnp.outer(tau_arr, omega))


    hessian_all_m=Partial(calc_Z_error_pseudo_hessian_one_m, pulse_t=pulse_t, time=time, omega=omega, nonlinear_method=nonlinear_method, 
                          full_or_diagonal=full_or_diagonal, pulse_or_gate=pulse_or_gate)
    h_all_m=jax.vmap(hessian_all_m, in_axes=(0,0,0,0,0))(exp_arr, gate_pulses, gate, signal_t, signal_t_new)

    return h_all_m




def get_pseudo_newton_direction_Z_error(grad_m, pulse_t, gate_pulses, gate, signal_t, signal_t_new, tau_arr, measurement_info, 
                                        hessian_state, hessian_info, full_or_diagonal, pulse_or_gate):
    
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
        hessian_state: Pytree, contains the current state of the hessian calculation, e.g. the previous newton direction
        hessian_info: Pytree, contains parameters for the pseudo-newton direction calculation
        full_or_diagonal: str, calculate using the full or diagonal pseudo hessian?
        pulse_or_gate: str, whether the direction is calculated for the pulse or the gate-pulse

    Returns:
        tuple[jnp.array, Pytree], the pseudo-newton direction and the updated hessian_state
    
    """
     

    lambda_lm = hessian_info.lambda_lm
    solver = hessian_info.linalg_solver
    newton_direction_prev = getattr(hessian_state, pulse_or_gate).newton_direction_prev  

    # vmap over population here -> only for small populations since memory will explode. 
    hessian_m=jax.vmap(calc_Z_error_pseudo_hessian_all_m, in_axes=(0,0,0,0,0,0,None,None,None))(pulse_t, gate_pulses, gate, signal_t, signal_t_new, 
                                                                                                tau_arr, measurement_info, full_or_diagonal, pulse_or_gate)
    
    # hessian=jnp.sum(hessian_m, axis=1)
    # grad=jnp.sum(grad_m, axis=1)
    
    # if full_or_diagonal=="full":
        
    #     idx=jax.vmap(jnp.diag_indices_from)(hessian)
    #     hessian=jax.vmap(lambda x,y: x.at[y].add(lambda_lm*jnp.abs(x[y])))(hessian, idx)

    #     newton_direction=solve_linear_system(hessian, grad, newton_direction_prev, solver)

    # elif full_or_diagonal=="diagonal":
    #     hessian = hessian + lambda_lm*jnp.max(jnp.abs(hessian), axis=1)[:, jnp.newaxis]
    #     newton_direction=grad/hessian
        
    # else:
    #     raise ValueError(f"full_or_diagonal needs to be full or diagonal. Not {full_or_diagonal}")

    # hessian_state = MyNamespace(newton_direction_prev = newton_direction)
    # return -1*newton_direction, hessian_state
    return calculate_newton_direction(grad_m, hessian_m, lambda_lm, newton_direction_prev, solver, full_or_diagonal)
        