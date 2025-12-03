from .twodsi_z_error_gradients import calculate_Z_gradient as calculate_Z_gradient_2DSI



def calculate_Z_gradient(signal_t, signal_t_new, pulse_t, gate_pulses, gate, tau_arr, measurement_info, pulse_or_gate, is_vampire=True):
    """
    Calculates the Z-error gradient with respect to the pulse or the gate-pulse for a given 2DSI measurement. 
    The gradient is calculated in the frequency domain.

    Args:
        signal_t: jnp.array, the current signal field
        signal_t_new: jnp.array, the current signal field projected onto the measured intensity
        pulse_t: jnp.array, the current guess
        gate_pulses: jnp.array, the current gate-pulse guess
        gate: jnp.array, the current gate
        tau_arr: jnp.array, the delays
        measurement_info: Pytree, contains measurement data and parameters
        pulse_or_gate: str, whether the gradient is calculated with respect to the pulse or the gate-pulse

    Returns:
        jnp.array, the Z-error gradient
    """

    return calculate_Z_gradient_2DSI(signal_t, signal_t_new, pulse_t, gate_pulses, gate, tau_arr, measurement_info, pulse_or_gate, is_vampire)