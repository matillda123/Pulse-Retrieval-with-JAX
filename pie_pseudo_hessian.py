import jax.numpy as jnp
import jax
from jax.tree_util import Partial
from utilities import scan_helper, calculate_newton_direction, MyNamespace



def PIE_get_pseudo_hessian_subelement(dummy, signal_f, measured_trace, D_arr_kj):
    """ Calculates one term of one hessian element. Sum over all terms yields one matrix element."""
    res = D_arr_kj*(2 - jnp.sign(measured_trace)*jnp.sqrt(jnp.abs(measured_trace))/(jnp.abs(signal_f) + 1e-9))
    return dummy + res, None


def PIE_get_pseudo_hessian_element(probe_k, probe_j, time_k, time_j, omega, signal_f, measured_trace):
    """ Sum over frequency axis via jax.lax.scan. Does not use jax.vmap because of memory limits. """

    D_arr_kj=jnp.exp(1j*omega*(time_k-time_j))
    get_subelement=Partial(scan_helper, actual_function=PIE_get_pseudo_hessian_subelement, number_of_args=1, number_of_xs=3)

    carry = 0+0j
    xs = (signal_f, measured_trace, D_arr_kj)
    val_subelement, _ = jax.lax.scan(get_subelement, carry, xs)

    hess_element = 0.25*jnp.conjugate(probe_k)*probe_j*val_subelement
    return hess_element



def PIE_get_pseudo_hessian_one_m(probe, signal_f, measured_trace, measurement_info, use_hessian):
    """ jax.vmap over time axis """

    time, frequency = measurement_info.time, measurement_info.frequency
    get_hessian=Partial(PIE_get_pseudo_hessian_element, omega=2*jnp.pi*frequency, signal_f=signal_f, measured_trace=measured_trace)

    if use_hessian=="full":
        hessian=jax.vmap(jax.vmap(get_hessian, in_axes=(None, 0, None, 0)), in_axes=(0, None, 0, None))(probe, probe, time, time)

    elif use_hessian=="diagonal":
        hessian=jax.vmap(get_hessian, in_axes=(0,0,0,0))(probe, probe, time, time)

    return hessian



def PIE_get_pseudo_hessian_all_m(probe_all_m, signal_f, measured_trace, measurement_info, use_hessian):
    """ jax.vmap over delays/shifts """
    get_hessian=Partial(PIE_get_pseudo_hessian_one_m, measurement_info=measurement_info, use_hessian=use_hessian)
    hessian_all_m=jax.vmap(get_hessian, in_axes=(0,0,0))(probe_all_m, signal_f, measured_trace)

    return hessian_all_m
    





def PIE_get_pseudo_newton_direction(grad, probe, signal_f, transform_arr, measured_trace, reverse_transform, newton_direction_prev, 
                                    measurement_info, descent_info, pulse_or_gate, local_or_global):
    
    """
    Calculates the pseudo-newton direction for the PIE loss function. Is the same for all methods.
    The direction is calculated in the time domain.

    Args:
        grad: jnp.array, the current (weighted) gradient
        probe: jnp.array, the PIE probe
        signal_f: jnp.array, the signal field in the frequency domain
        transform_arr: jnp.array, the delays or phase matrix
        measured_trace: jnp.array, the measured intensity
        reverse_transform: Callable, unused
        newton_direction_prev: jnp.array, the previous pseudo-newton direction
        measurement_info: Pytree, holds measurement data and parameters
        descent_info: Pytree, holds algorithm parameters
        pulse_or_gate: str, pulse or gate?
        local_or_global: str, local or global iteration?

    Returns:
        tuple[jnp.array, Pytree]
    
    """


    hessian = descent_info.hessian
    lambda_lm, solver = hessian.lambda_lm, hessian.linalg_solver
    full_or_diagonal = getattr(hessian, local_or_global)

    # vmap over population
    hessian_all_m=jax.vmap(PIE_get_pseudo_hessian_all_m, in_axes=(0,0,0,None,None))(probe, signal_f, measured_trace, measurement_info, full_or_diagonal)


    # if pulse_or_gate=="gate":
    #     hessian_all_m = jax.vmap(reverse_transform, in_axes=(0,0))(hessian_all_m, transform_arr)

    # grad = jnp.sum(grad, axis=1)
    # hessian = jnp.sum(hessian_all_m, axis=1)

    # if full_or_diagonal=="full":
    #     idx=jax.vmap(jnp.diag_indices_from)(hessian)
    #     hessian=jax.vmap(lambda x,y: x.at[y].add(lambda_lm*jnp.abs(x[y])))(hessian, idx)

    #     newton_direction=solve_linear_system(hessian, grad, newton_direction_prev, solver)

    # elif full_or_diagonal=="diagonal":
    #     hessian = hessian + lambda_lm*jnp.max(jnp.abs(hessian), axis=1)[:,jnp.newaxis]
    #     newton_direction = grad/hessian

    # else:
    #     raise ValueError(f"full_or_diagonal needs to be full or diagonal. Not {full_or_diagonal}")

    # hessian_state = MyNamespace(newton_direction_prev = newton_direction)
    # return -1*newton_direction, hessian_state
    return calculate_newton_direction(grad, hessian_all_m, lambda_lm, newton_direction_prev, solver, full_or_diagonal)
