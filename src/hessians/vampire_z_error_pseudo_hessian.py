import jax
from jax.tree_util import Partial

from src.utilities import calculate_newton_direction
from .twodsi_z_error_pseudo_hessian import calc_Z_error_pseudo_hessian_all_m




def get_pseudo_newton_direction_Z_error(grad_m, pulse_t, gate_pulses, gate, signal_t, signal_t_new, tau_arr, measurement_info, 
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
    calc_hessian = Partial(calc_Z_error_pseudo_hessian_all_m, is_vampire=True)
    hessian_m=jax.vmap(calc_hessian, in_axes=(0,0,0,0,0,None,None,None))(pulse_t, gate_pulses, gate, deltaS, 
                                                                                                tau_arr, measurement_info, full_or_diagonal, pulse_or_gate)

    return calculate_newton_direction(grad_m, hessian_m, lambda_lm, newton_direction_prev, solver, full_or_diagonal)
        