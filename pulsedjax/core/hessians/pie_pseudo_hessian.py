import jax.numpy as jnp
import jax
from jax.tree_util import Partial

from pulsedjax.utilities import scan_helper, calculate_newton_direction




# def PIE_get_pseudo_hessian_element(probe_k, probe_j, time_k, time_j, omega, signal_f, measured_trace):
#     """ Sum over frequency axis via jax.lax.scan."""

#     D_arr_kj = jnp.exp(1j*omega*(time_k-time_j))

#     val_subelement_arr = D_arr_kj*(2 - jnp.sqrt(jnp.abs(measured_trace))/(jnp.abs(signal_f) + 1e-15))
#     val_subelement = jnp.sum(val_subelement_arr)

#     hess_element = 0.25*jnp.conjugate(probe_k)*probe_j*val_subelement
#     return hess_element



# def PIE_get_pseudo_hessian_one_m(dummy, probe, signal_f, measured_trace, measurement_info, use_hessian):
#     """ jax.vmap over time axis """

#     time, frequency = measurement_info.time, measurement_info.frequency
#     get_hessian = Partial(PIE_get_pseudo_hessian_element, omega=2*jnp.pi*frequency, signal_f=signal_f, measured_trace=measured_trace)

#     if use_hessian=="full":
#         hessian = jax.vmap(jax.vmap(get_hessian, in_axes=(None, 0, None, 0)), in_axes=(0, None, 0, None))(probe, probe, time, time)

#     elif use_hessian=="diagonal":
#         hessian = jax.vmap(get_hessian, in_axes=(0,0,0,0))(probe, probe, time, time)

#     return dummy, hessian



# def PIE_get_pseudo_hessian_all_m(probe_all_m, signal_f, measured_trace, measurement_info, use_hessian):
#     """ Loop over delays/shifts to get hessian for each. Does not use jax.vmap because of memory limits. """
#     get_hessian = Partial(PIE_get_pseudo_hessian_one_m, measurement_info=measurement_info, use_hessian=use_hessian)
    
#     xs = (probe_all_m, signal_f, measured_trace)
#     carry = jnp.zeros(1)
#     get_hessian_all_m = Partial(scan_helper, actual_function=get_hessian, number_of_args=1, number_of_xs=3)
#     _, hessian_all_m = jax.lax.scan(get_hessian_all_m, carry, xs)
#     return hessian_all_m
    


def PIE_get_full_pseudo_hessian_all_m(probe, signal_f, transform_arr, measured_trace, measurement_info, pulse_or_gate):
    """ Calculates the full pseudo hessian through jnp.einsum(). """

    time, omega = measurement_info.time, 2*jnp.pi*measurement_info.frequency
    Dkn = jnp.exp(1j*(time[:,jnp.newaxis]*omega[jnp.newaxis,:]))/jnp.sqrt(jnp.size(time))
    subelement = (2 - jnp.sqrt(jnp.abs(measured_trace))/(jnp.abs(signal_f) + 1e-15))

    if pulse_or_gate=="pulse":
        hessian_all_m = jnp.einsum("kn,Nmk,Nmj,jn,Nmn -> Nmkj", Dkn.conj(), probe, probe.conj(), Dkn, subelement)
    
    elif pulse_or_gate=="gate":
        # is here to test which shape transform arr has
        jax.debug.print("transform_arr {error}", error=jnp.shape(transform_arr))
        phase_matrix = jnp.exp(-1j*transform_arr[:,jnp.newaxis]*omega[jnp.newaxis,:])
        H = jnp.einsum("kn,Nmn,jn -> Nkj", Dkn, phase_matrix, Dkn.conj())
        hessian_all_m = jnp.einsum("Nju,un,Nmu,Nmi,in,Nki,Nmn -> Nmkj", H, Dkn, probe.conj(), probe, Dkn.conj(), H.conj(), subelement)
    
    elif pulse_or_gate=="chirpscan":
        phase_matrix = transform_arr
        H = jnp.einsum("kn,Nmn,jn -> Nkj", Dkn, phase_matrix, Dkn.conj())
        hessian_all_m = jnp.einsum("Nju,un,Nmu,Nmi,in,Nki,Nmn -> Nmkj", H, Dkn, probe.conj(), probe, Dkn.conj(), H.conj(), subelement)
    
    else:
        raise ValueError

    return hessian_all_m




def PIE_get_diagonal_pseudo_hessian_all_m(probe, signal_f, transform_arr, measured_trace, measurement_info, pulse_or_gate):
    """ Calculates the diagonal pseudo hessian through jnp.einsum(). """
    
    time, omega = measurement_info.time, 2*jnp.pi*measurement_info.frequency
    Dkn = jnp.exp(1j*(time[:,jnp.newaxis]*omega[jnp.newaxis,:]))/jnp.sqrt(jnp.size(time))
    subelement = (2 - jnp.sqrt(jnp.abs(measured_trace))/(jnp.abs(signal_f) + 1e-15))

    if pulse_or_gate=="pulse":
        hessian_all_m = jnp.einsum("kn,Nmk,Nmk,kn,Nmn -> Nmk", Dkn.conj(), probe, probe.conj(), Dkn, subelement)

    elif pulse_or_gate=="gate":
        jax.debug.print("transform_arr {error}", error=jnp.shape(transform_arr))
        phase_matrix = jnp.exp(-1j*transform_arr[:,jnp.newaxis]*omega[jnp.newaxis,:])
        H = jnp.einsum("kn,Nmn,jn -> Nkj", Dkn, phase_matrix, Dkn.conj())
        hessian_all_m = jnp.einsum("Nki,in,Nmi,Nmi,in,Nik,Nmn -> Nmk", H, Dkn, probe.conj(), probe, Dkn.conj(), H.conj(), subelement)
    
    elif pulse_or_gate=="chirpscan":
        phase_matrix = transform_arr
        H = jnp.einsum("kn,Nmn,jn -> Nkj", Dkn, phase_matrix, Dkn.conj())
        hessian_all_m = jnp.einsum("Nki,in,Nmi,Nmi,in,Nik,Nmn -> Nmk", H, Dkn, probe.conj(), probe, Dkn.conj(), H.conj(), subelement)
    
    else:
        raise ValueError

    return hessian_all_m



def PIE_get_pseudo_hessian_all_m(probe, signal_f, transform_arr, measured_trace, measurement_info, full_or_diagonal, pulse_or_gate):
    """ Just an intermediary to call full or diagonal hessian. """

    if full_or_diagonal=="full":
        hessian_all_m = PIE_get_full_pseudo_hessian_all_m(probe, signal_f, transform_arr, measured_trace, measurement_info, pulse_or_gate)
    elif full_or_diagonal=="diagonal":
        hessian_all_m = PIE_get_diagonal_pseudo_hessian_all_m(probe, signal_f, transform_arr, measured_trace, measurement_info, pulse_or_gate)
    else:
        raise ValueError
    
    return hessian_all_m




def PIE_get_pseudo_newton_direction(grad, probe, signal_f, transform_arr, measured_trace, newton_direction_prev, 
                                    measurement_info, descent_info, pulse_or_gate, local_or_global):
    
    """
    Calculates the pseudo-newton direction for the PIE loss function. Is the same for all methods. 
    Except for those which cannot be used with PIE, which are not available.
    The direction is calculated in the time domain.

    Args:
        grad (jnp.array): the current (weighted) gradient
        probe (jnp.array): the PIE probe or modified probe/object for hessian
        signal_f (jnp.array): the signal field in the frequency domain
        transform_arr (jnp.array): the delays or phase matrix
        measured_trace (jnp.array): the measured intensity
        newton_direction_prev (jnp.array): the previous pseudo-newton direction
        measurement_info (Pytree): holds measurement data and parameters
        descent_info (Pytree): holds algorithm parameters
        pulse_or_gate (str): pulse or gate, (or chirpscan)
        local_or_global (str): local or global iteration?

    Returns:
        tuple[jnp.array, Pytree]
    
    """

    newton = descent_info.newton
    lambda_lm, solver = newton.lambda_lm, newton.linalg_solver
    full_or_diagonal = getattr(newton, local_or_global)
    hessian_all_m = PIE_get_pseudo_hessian_all_m(probe, signal_f, transform_arr, measured_trace, measurement_info, full_or_diagonal, pulse_or_gate)

    return calculate_newton_direction(grad, hessian_all_m, lambda_lm, newton_direction_prev, solver, full_or_diagonal)
