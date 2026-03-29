import jax.numpy as jnp

from pulsedjax.utilities import calculate_newton_direction

    


def PIE_get_full_pseudo_hessian_all_m(probe, subelement, transform_arr, time, omega, Dkn, measurement_info, pulse_or_gate):
    """ Calculates the full pseudo hessian through jnp.einsum(). """

    if pulse_or_gate=="pulse":
        hessian_all_m = jnp.einsum("kn, Nmk, Nmj, jn, Nmn -> Nmkj", Dkn.conj(), probe, probe.conj(), Dkn, subelement)
    
    elif pulse_or_gate=="gate":
        phase_matrix = jnp.exp(-1j*transform_arr[:,:,jnp.newaxis]*omega[jnp.newaxis, jnp.newaxis,:])

        N = jnp.size(time)
        time_new = jnp.concatenate([time, time + (N+1)*jnp.mean(jnp.diff(time))])
        Dkn = jnp.exp(1j*(time_new[:,jnp.newaxis]*omega[jnp.newaxis,:]))/jnp.sqrt(N)
        H = jnp.einsum("kn, Nmn, jn -> Nmkj", Dkn, phase_matrix, Dkn.conj())
        H = H[:,:,:N,:N]
        hessian_all_m = jnp.einsum("Nmju, un, Nmu, Nmi, in, Nmki, Nmn -> Nmkj", H, Dkn, probe.conj(), probe, Dkn.conj(), H.conj(), subelement)

    elif pulse_or_gate=="chirpscan":
        phase_matrix = transform_arr
        H = jnp.einsum("kn, Nmn, jn -> Nmkj", Dkn, phase_matrix, Dkn.conj())
        hessian_all_m = jnp.einsum("Nmju, un, Nmu, Nmi, in, Nmki, Nmn -> Nmkj", H, Dkn, probe.conj(), probe, Dkn.conj(), H.conj(), subelement)
    
    else:
        raise ValueError

    return hessian_all_m




def PIE_get_diagonal_pseudo_hessian_all_m(probe, subelement, transform_arr, time, omega, Dkn, measurement_info, pulse_or_gate):
    """ Calculates the diagonal pseudo hessian through jnp.einsum(). """

    if pulse_or_gate=="pulse":
        hessian_all_m = jnp.einsum("kn, Nmk, Nmk, kn, Nmn -> Nmk", Dkn.conj(), probe, probe.conj(), Dkn, subelement)

    elif pulse_or_gate=="gate":
        phase_matrix = jnp.exp(-1j*transform_arr[:,:,jnp.newaxis]*omega[jnp.newaxis, jnp.newaxis,:])

        N = jnp.size(time)
        time_new = jnp.concatenate([time, time + (N+1)*jnp.mean(jnp.diff(time))])
        Dkn_new = jnp.exp(1j*(time_new[:,jnp.newaxis]*omega[jnp.newaxis,:]))/jnp.sqrt(N)
        H = jnp.einsum("kn, Nmn, jn -> Nmkj", Dkn_new, phase_matrix, Dkn_new.conj())
        H = H[:,:,:N,:N]
        hessian_all_m = jnp.einsum("Nmki, in, Nmi, Nmi, in, Nmik, Nmn -> Nmk", H, Dkn, probe.conj(), probe, Dkn.conj(), H.conj(), subelement)

    elif pulse_or_gate=="chirpscan":
        phase_matrix = transform_arr
        H = jnp.einsum("kn, Nmn, jn -> Nmkj", Dkn, phase_matrix, Dkn.conj())
        hessian_all_m = jnp.einsum("Nmki, in, Nmi, Nmi, in, Nmik, Nmn -> Nmk", H, Dkn, probe.conj(), probe, Dkn.conj(), H.conj(), subelement)
    
    else:
        raise ValueError

    return hessian_all_m



def PIE_get_pseudo_hessian_all_m(probe, signal_f, transform_arr, measured_trace, measurement_info, full_or_diagonal, pulse_or_gate):
    """ Just an intermediary to call full or diagonal hessian. """

    if measurement_info.real_fields==False:
        time, omega = measurement_info.time, 2*jnp.pi*measurement_info.frequency
    else:
        time, omega = measurement_info.time_big, 2*jnp.pi*measurement_info.frequency_big


    N = jnp.size(time)
    Dkn = jnp.exp(1j*(time[:,jnp.newaxis]*omega[jnp.newaxis,:]))/jnp.sqrt(N)
    subelement = (2 - jnp.sqrt(jnp.abs(measured_trace))/(jnp.abs(signal_f) + 1e-15))
    
    hessian_func_dict = {"full": PIE_get_full_pseudo_hessian_all_m,
                         "diagonal": PIE_get_diagonal_pseudo_hessian_all_m}
    
    hessian_all_m = hessian_func_dict[full_or_diagonal](probe, subelement, transform_arr, time, omega, Dkn, measurement_info, pulse_or_gate)
    return hessian_all_m




def PIE_get_pseudo_newton_direction(grad, probe, signal_f, transform_arr, measured_trace, descent_state, 
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
        descent_state (Pytree):
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

    if pulse_or_gate=="gate" and measurement_info.nonlinear_method=="sd":
        # this should be correct, one can pull the conjugate out, then one gets the same formula
        # so one calcualtes the hessian as with other methods, and then conjugates 
        hessian_all_m = jnp.conjugate(hessian_all_m)

    if pulse_or_gate=="chirpscan":
        newton_direction_prev = descent_state.newton.pulse.newton_direction_prev
    else:
        newton_direction_prev = getattr(descent_state.newton, pulse_or_gate).newton_direction_prev
    return calculate_newton_direction(grad, hessian_all_m, lambda_lm, newton_direction_prev, solver, full_or_diagonal)
