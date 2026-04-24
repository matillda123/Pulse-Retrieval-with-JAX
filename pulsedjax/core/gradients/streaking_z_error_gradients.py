import jax.numpy as jnp
from pulsedjax.utilities import do_fft



def Z_gradient_EUV_pulse(signal_t, signal_t_new, tau_arr, pulse_t_euv_shifted, pulse_t_nir_vectorpotential, 
                         dtme_position, dtme_shift_and_volkov_phase0, volkov_phase0, volkov_phase1, measurement_info): # frequency domain
    dt = jnp.mean(jnp.diff(measurement_info.time))
    sk, rn = measurement_info.sk, measurement_info.rn
    omega = 2*jnp.pi*measurement_info.frequency
    exp_arr = jnp.exp(1j*tau_arr[:,None]*omega[None,:])

    delta_S_mq = signal_t_new - signal_t

    term1 = do_fft(dtme_shift_and_volkov_phase0*jnp.exp(1j*volkov_phase1), sk, rn)

    # term1 -> (q,n)
    # delta_S_mq -> (m,q)

    grad_temp = jnp.einsum("mq, qn -> mn", delta_S_mq, term1)
    return -2*1j*dt*exp_arr*grad_temp







def _get_gradient_of_dtme_with_respect_to_vectorpotential(dtme_position, pulse_t_nir_vectorpotential, measurement_info):
    if measurement_info.dtme_momentum is None:
            dtme_momentum = jnp.zeros((measurement_info.no_channels, jnp.size(dtme_position), jnp.size(pulse_t_nir_vectorpotential)))
    else: 
        r = measurement_info.position
        sk, rn = measurement_info.sk_position_momentum, measurement_info.rn_position_momentum

        # -1 because one wants to shift to positive values
        # but is this consistent with atomic units convention for elementary-charge?
        momentum_shift = -1*jnp.real(pulse_t_nir_vectorpotential)
        dtme_position = dtme_position[:,None,:]*jnp.exp(-1j*2*jnp.pi*r[None,None,:]*momentum_shift[None,:,None]) 
        dtme_position = dtme_position*r[None,None,:]

        dr = jnp.mean(jnp.diff(r))
        phase_correction = jnp.exp(1j*2*jnp.pi*dr*momentum_shift)
        dtme_position = dtme_position*phase_correction[None,:,None]

        dtme_momentum = do_fft(dtme_position, sk, rn)
        dtme_momentum = 2*jnp.pi*1j*jnp.transpose(dtme_momentum, (0,2,1)) # -> such that output is (C,b,k)

    return dtme_momentum


def Z_gradient_vectorpotential(signal_t, signal_t_new, tau_arr, pulse_t_euv_shifted, pulse_t_nir_vectorpotential, 
                               dtme_position, dtme_shift_and_volkov_phase0, volkov_phase0, volkov_phase1, measurement_info): # frequency domain
    gradient_dtme_Cbk = _get_gradient_of_dtme_with_respect_to_vectorpotential(dtme_position, pulse_t_nir_vectorpotential, measurement_info)
    gradient_dtme_Cbk = jnp.conjugate(gradient_dtme_Cbk)

    dt = jnp.mean(jnp.diff(measurement_info.time))
    position, momentum = measurement_info.position, measurement_info.momentum
    N = jnp.size(measurement_info.time)
    
    # volkov_phase0 is channel dependent 
    term1 = jnp.einsum("Cbk, Cbk -> bk", gradient_dtme_Cbk, jnp.exp(1j*volkov_phase0))
    term2 = 1j*dt*jnp.einsum("mk, bk, bk -> mbk", jnp.conjugate(pulse_t_euv_shifted), jnp.exp(1j*(volkov_phase1)), term1)


    grad_volkov1 = dt*(momentum[:,None] + jnp.real(pulse_t_nir_vectorpotential)[None,:])
    grad_volkov1 = jnp.broadcast_to(grad_volkov1, (jnp.size(momentum), N, N))
    mask = jnp.triu(jnp.ones((N, N)))
    grad_volkov1 = grad_volkov1*mask

    term4 = -1*dt*jnp.einsum("mk, bk, bkj, bj", jnp.conjugate(pulse_t_euv_shifted), jnp.exp(volkov_phase1), grad_volkov1, dtme_shift_and_volkov_phase0)


    delta_S_mq = signal_t_new - signal_t
    Dbq = jnp.exp(-1j*position[None,:]*momentum[:,None])
    term5 = term2 + term4

    grad = jnp.real(jnp.einsum("mq, bq, bk -> mk", delta_S_mq, Dbq, term5))

    sk, rn = measurement_info.sk, measurement_info.rn
    grad = do_fft(grad, sk, rn)
    return -2*grad







def Z_gradient_DTME(signal_t, signal_t_new, tau_arr, pulse_t_euv_shifted, pulse_t_nir_vectorpotential, 
                    dtme_position, dtme_shift_and_volkov_phase0, volkov_phase0, volkov_phase1, measurement_info): # momentum domain, need to check indices, what about multichannel?
    dt = jnp.mean(jnp.diff(measurement_info.time))
    sk_position_momentum, rn_position_momentum = measurement_info.sk_position_momentum, measurement_info.rn_position_momentum
    position = measurement_info.position

    delta_S_mq = signal_t_new - signal_t

    pulse_t_nir_vectorpotential = jnp.real(pulse_t_nir_vectorpotential)
    momentum_shift_term = jnp.exp(-1j*2*jnp.pi*position[:,None]*pulse_t_nir_vectorpotential[None,:])
    term1 = jnp.einsum("mk, Cbk, bk, qk -> Cmq", jnp.conj(pulse_t_euv_shifted), jnp.exp(1j*volkov_phase0), jnp.exp(1j*volkov_phase1), momentum_shift_term)
    
    grad_temp_q = delta_S_mq*term1 # maybe there needs to be a 1/N here becuae Dbq and Dqb cancel?
    grad_temp_b = do_fft(grad_temp_q, sk_position_momentum, rn_position_momentum)
    return -2*1j*dt*grad_temp_b # with shape (C,m,b) -> will cauuse issues in some algorihms because its not dimension (m,n)






def calculate_Z_gradient(signal_t, signal_t_new, tau_arr, measurement_info, pulse_or_gate):
    """
    Calculates the Z-error gradient with respect to the pulse, gate-pulse or the DTME for a given Streaking measurement. 
    The gradient is calculated in the frequency domain.

    Args:
        signal_t (jnp.array): the current signal field
        signal_t_new (jnp.array): the current signal field projected onto the measured intensity


        tau_arr (jnp.array): the delays
        measurement_info (Pytree): contains measurement data and parameters
        pulse_or_gate (str): whether the gradient is calculated with respect to the pulse or the gate-pulse

    Returns:
        jnp.array, the Z-error gradient
    """
    calculate_Z_gradient_dict={"pulse": Z_gradient_vectorpotential,
                               "gate": Z_gradient_EUV_pulse, 
                               "dtme": Z_gradient_DTME}
    
    pulse_t_euv_shifted, pulse_t_nir_vectorpotential = signal_t.pulse_t_euv_shifted, signal_t.pulse_t_nir_vectorpotential
    volkov_phase0, volkov_phase1 = signal_t.volkov_phase0, signal_t.volkov_phase1
    dtme_position, dtme_shift_and_volkov_phase0 = signal_t.dtme_position, signal_t.dtme_shift_and_volkov_phase0

    return calculate_Z_gradient_dict[pulse_or_gate](signal_t, signal_t_new, tau_arr, pulse_t_euv_shifted, pulse_t_nir_vectorpotential, 
                                                    dtme_position, dtme_shift_and_volkov_phase0, volkov_phase0, volkov_phase1, measurement_info)