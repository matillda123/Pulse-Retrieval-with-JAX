import jax.numpy as jnp
from pulsedjax.utilities import do_fft

import jax



def Z_gradient_EUV_pulse(signal_t, signal_t_new, tau_arr, measurement_info): # frequency domain
    dtme_shifted_and_volkov_phase0, volkov_phase1 = signal_t.dtme_shifted_and_volkov_phase0, signal_t.volkov_phase1
    dt = jnp.mean(jnp.diff(measurement_info.time))
    omega = 2*jnp.pi*measurement_info.frequency
    exp_arr = jnp.exp(1j*tau_arr[:,None]*omega[None,:])

    delta_S_mq = signal_t_new - signal_t.signal_t
    delta_S_mb = do_fft(delta_S_mq, measurement_info.sk_position_momentum, measurement_info.rn_position_momentum)

    term1 = jnp.conjugate(dtme_shifted_and_volkov_phase0)*jnp.exp(1j*volkov_phase1)
    grad_temp = jnp.einsum("mb, bn -> mn", delta_S_mb, term1)
    return -2*1j*dt*exp_arr*grad_temp







def _get_gradient_of_dtme_with_respect_to_vectorpotential(dtme_position, pulse_t_nir_vectorpotential, measurement_info):
    if measurement_info.dtme_momentum is None:
            dtme_momentum = jnp.zeros((measurement_info.no_channels, jnp.size(measurement_info.momentum), jnp.size(pulse_t_nir_vectorpotential)))
    else: 
        r = measurement_info.position
        sk, rn = measurement_info.sk_position_momentum, measurement_info.rn_position_momentum

        momentum_shift = -1*jnp.real(pulse_t_nir_vectorpotential)
        dtme_position = dtme_position[:,None,:]*jnp.exp(-1j*2*jnp.pi*r[None,None,:]*momentum_shift[None,:,None]) 
        dtme_position = dtme_position*r[None,None,:]

        dr = jnp.mean(jnp.diff(r))
        phase_correction = jnp.exp(1j*2*jnp.pi*dr*momentum_shift)
        dtme_position = dtme_position*phase_correction[None,:,None]

        dtme_momentum = do_fft(dtme_position, sk, rn)
        dtme_momentum = 2*jnp.pi*1j*jnp.transpose(dtme_momentum, (0,2,1)) # -> such that output is (C,b,k)

    return dtme_momentum



def Z_gradient_vectorpotential(signal_t, signal_t_new, tau_arr, measurement_info): # frequency domain
    pulse_t_euv_shifted, pulse_t_nir_vectorpotential = signal_t.pulse_t_euv_shifted, signal_t.pulse_t_nir_vectorpotential
    dtme_position, dtme_shifted_and_volkov_phase0 = signal_t.dtme_position, signal_t.dtme_shifted_and_volkov_phase0
    volkov_phase0, volkov_phase1 = signal_t.volkov_phase0, signal_t.volkov_phase1
    
    gradient_dtme_Cbk = _get_gradient_of_dtme_with_respect_to_vectorpotential(dtme_position, pulse_t_nir_vectorpotential, measurement_info)

    dt = jnp.mean(jnp.diff(measurement_info.time))
    sk, rn = measurement_info.sk, measurement_info.rn
    position, momentum = measurement_info.position, measurement_info.momentum
    N = jnp.size(measurement_info.time)
    
    term1 = jnp.einsum("Cbk, Cbk -> bk", jnp.conjugate(gradient_dtme_Cbk), jnp.exp(1j*volkov_phase0))
    term2 = 1j*dt*jnp.einsum("mk, bk, bk -> mbk", jnp.conjugate(pulse_t_euv_shifted), jnp.exp(1j*volkov_phase1), term1)


    grad_volkov1 = dt*(momentum[:,None] + jnp.real(pulse_t_nir_vectorpotential)[None,:]) 
    dtme_shifted_and_volkov_phase01 = jnp.conjugate(dtme_shifted_and_volkov_phase0)*jnp.exp(1j*volkov_phase1)
    mask = jnp.triu(jnp.ones((N,N))) 

    def mask_grad_volkov1(dummy, mask):
         val = jnp.einsum("mk, bk, k, bk -> mb", 
                          jnp.conjugate(pulse_t_euv_shifted), grad_volkov1, mask, dtme_shifted_and_volkov_phase01)
         return dummy, val
    
    _, term3 = jax.lax.scan(mask_grad_volkov1, jnp.ones(1), mask) # needs to be done via scan, because of memory explosion
    term3 = -1*dt*jnp.transpose(term3, (1,2,0))

    delta_S_mq = signal_t_new - signal_t.signal_t
    #Dbq = jnp.exp(-1j*position[None,:]*momentum[:,None])
    delta_S_mb = do_fft(delta_S_mq, measurement_info.sk_position_momentum, measurement_info.rn_position_momentum)
    term4 = term2 + term3

    #grad = jnp.real(jnp.einsum("mq, bq, mbk -> mk", delta_S_mq, Dbq, term4)) # here this is an issue -> wraps frequency around 
    grad = jnp.einsum("mb, mbk -> mk", delta_S_mb, term4)
    #grad = do_fft(grad, sk, rn)
    return -2*grad







def Z_gradient_DTME(signal_t, signal_t_new, tau_arr, measurement_info): # momentum domain, need to check indices, what about multichannel?
    pulse_t_euv_shifted, volkov_phase0, volkov_phase1 = signal_t.pulse_t_euv_shifted, signal_t.volkov_phase0, signal_t.volkov_phase1
    pulse_t_nir_vectorpotential = signal_t.pulse_t_nir_vectorpotential

    dt = jnp.mean(jnp.diff(measurement_info.time))
    sk_position_momentum, rn_position_momentum = measurement_info.sk_position_momentum, measurement_info.rn_position_momentum
    position = measurement_info.position

    delta_S_mq = signal_t_new - signal_t.signal_t

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

    return calculate_Z_gradient_dict[pulse_or_gate](signal_t, signal_t_new, tau_arr,  measurement_info)