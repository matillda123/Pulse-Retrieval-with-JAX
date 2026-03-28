import jax
from functools import partial as Partial

from pulsedjax.utilities import calculate_newton_direction
from pulsedjax.core.hessians.frog_z_error_pseudo_hessian import calc_Z_error_pseudo_hessian_all_m






def get_pseudo_newton_direction_Z_error(grad_m, pulse_t, pulse_t_shifted, gate_shifted, signal_t, signal_t_new, tau_arr, 
                                        descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
    
    """
    Calculates the pseudo-newton direction for the Z-error of a Time-Domain-Ptychography measurement.
    The direction is calculated in the frequency domain.

    Args:
        grad_m (jnp.array): the current Z-error gradient
        pulse_t (jnp.array): the current guess
        pulse_t_shifted (jnp.array): the current guess shifted along the time axis
        gate_shifted (jnp.array): the current gate guess shifted along the time axis
        signal_t (jnp.array): the current signal field
        signal_t_new (jnp.array): the current signal field projected onto the measured intensity
        tau_arr (jnp.array): the time delays
        descent_state (pytree):
        measurement_info (pytree):
        descent_info (pytree):
        full_or_diagonal (str): calculate using the full or diagonal pseudo hessian?
        pulse_or_gate (str): whether the direction is calculated for the pulse or the gate-pulse

    Returns:
        tuple[jnp.array, Pytree], the pseudo-newton direction and the updated newton_state
    
    """


    lambda_lm = descent_info.newton.lambda_lm
    solver = descent_info.newton.linalg_solver
    newton_direction_prev = getattr(descent_state.newton, pulse_or_gate).newton_direction_prev  
    deltaS = signal_t_new - signal_t

    # vmap over population here -> only for small populations since memory will explode. 
    calc_hessian = Partial(calc_Z_error_pseudo_hessian_all_m, is_tdp=True)
    hessian_m=jax.vmap(calc_hessian, in_axes=(0,0,0,0,0,None,None,None))(pulse_t, pulse_t_shifted, gate_shifted, deltaS, 
                                                                        tau_arr, measurement_info, full_or_diagonal, pulse_or_gate)
    
    if getattr(descent_info.measured_spectrum_is_provided, pulse_or_gate)==True:
        spectral_amplitude = getattr(measurement_info.spectral_amplitude, pulse_or_gate)
        if full_or_diagonal=="full":
            hessian_m = hessian_m*spectral_amplitude[None,None,None,:]*spectral_amplitude[None,None,:,None]

        elif full_or_diagonal=="diagonal":
            hessian_m = hessian_m*spectral_amplitude[None,None,:]**2 # square because n=p

    return calculate_newton_direction(grad_m, hessian_m, lambda_lm, newton_direction_prev, solver, full_or_diagonal)
