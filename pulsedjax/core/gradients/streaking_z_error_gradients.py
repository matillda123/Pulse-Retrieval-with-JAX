import jax.numpy as jnp
from pulsedjax.utilities import do_fft, do_interpolation_1d
import jax
from functools import partial as Partial


def Z_gradient_EUV_pulse(signal_t, signal_t_new, tau_arr, measurement_info): # frequency domain
    dtme_shifted_and_volkov_phase0, volkov_phase1 = signal_t.dtme_shifted_and_volkov_phase0, signal_t.volkov_phase1
    dt = jnp.mean(jnp.diff(measurement_info.time))
    sk, rn = measurement_info.sk, measurement_info.rn
    omega = 2*jnp.pi*measurement_info.frequency
    exp_arr = jnp.exp(1j*tau_arr[:,None]*omega[None,:])

    delta_S_mq = signal_t_new - signal_t.signal_t
    delta_S_mb = do_fft(delta_S_mq, measurement_info.sk_position_momentum, measurement_info.rn_position_momentum)

    term1 = do_fft(jnp.conjugate(dtme_shifted_and_volkov_phase0)*jnp.exp(1j*volkov_phase1), sk, rn)
    grad = -2*1j*dt*jnp.einsum("mb, bn, mn -> mn", delta_S_mb, term1, exp_arr)

    grad = jax.vmap(Partial(do_interpolation_1d, method="linear"),
                    in_axes=(None,None,0))(measurement_info.axis_euv.frequency, measurement_info.frequency, grad)
    return grad







def _get_gradient_of_dtme_with_respect_to_vectorpotential(dtme_position, pulse_t_nir_vectorpotential, measurement_info):
    if measurement_info.dtme_momentum is None and measurement_info.retrieve_dtme==False:
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
        dtme_momentum = jnp.transpose(dtme_momentum, (0,2,1)) # -> such that output is (C,b,k)

    return dtme_momentum



def Z_gradient_vectorpotential(signal_t, signal_t_new, tau_arr, measurement_info): # frequency domain
    """ Grad contains product-rule. Thus two terms are calculcated and then added up. 
    First the part where A appears in the shift of the DTME, then the part where A appears in the volkov phase."""
    pulse_t_euv_shifted, pulse_t_nir_vectorpotential = signal_t.pulse_t_euv_shifted, signal_t.pulse_t_nir_vectorpotential
    dtme_position, dtme_shifted_and_volkov_phase0 = signal_t.dtme_position, signal_t.dtme_shifted_and_volkov_phase0
    volkov_phase0, volkov_phase1 = signal_t.volkov_phase0, signal_t.volkov_phase1
    
    gradient_dtme_Cbk = _get_gradient_of_dtme_with_respect_to_vectorpotential(dtme_position, pulse_t_nir_vectorpotential, measurement_info)

    dt = jnp.mean(jnp.diff(measurement_info.time))
    momentum = measurement_info.momentum
    
    term1 = 2*jnp.pi*dt*jnp.einsum("mk, Cbk, Cbk, bk -> mbk", jnp.conjugate(pulse_t_euv_shifted), 
                                   jnp.conjugate(gradient_dtme_Cbk), jnp.exp(1j*volkov_phase0), jnp.exp(1j*volkov_phase1))


    grad_volkov1 = (momentum[:,None] + jnp.real(pulse_t_nir_vectorpotential)[None,:])
    _term2 = jnp.conjugate(dtme_shifted_and_volkov_phase0)*jnp.exp(1j*volkov_phase1)
    _term2 = jnp.einsum("mk, bk -> mbk", jnp.conjugate(pulse_t_euv_shifted), _term2)
    #_term2 = jnp.cumsum(_term2[:,:,::-1], axis=-1)[:,:,::-1] 
    _term2 = jnp.cumsum(_term2, axis=-1) # apparently this is correct and not the flipped version
    term2 = -1*dt**2*jnp.einsum("mbj, bj -> mbj", _term2, grad_volkov1)


    delta_S_mq = signal_t_new - signal_t.signal_t
    delta_S_mb = do_fft(delta_S_mq, measurement_info.sk_position_momentum, measurement_info.rn_position_momentum)
    term3 = term1 + term2
    
    grad = jnp.real(jnp.einsum("mb, mbk -> mk", delta_S_mb, term3))
    grad = do_fft(grad, measurement_info.sk, measurement_info.rn) 
    grad = jax.vmap(Partial(do_interpolation_1d, method="cubic2"),
                    in_axes=(None,None,0))(measurement_info.axis_nir.frequency, measurement_info.frequency, grad)
    return -2*grad



# the phase doesnt seem to be retrieved correctly, maybe there is something wrong here?
# i dont think so, the implementation of the formula is fine, maybe its derivation is faulty?
def Z_gradient_DTME(signal_t, signal_t_new, tau_arr, measurement_info): # momentum domain
    pulse_t_euv_shifted, volkov_phase0, volkov_phase1 = signal_t.pulse_t_euv_shifted, signal_t.volkov_phase0, signal_t.volkov_phase1
    pulse_t_nir_vectorpotential = jnp.real(signal_t.pulse_t_nir_vectorpotential)

    dt = jnp.mean(jnp.diff(measurement_info.time))
    position, momentum = measurement_info.position, measurement_info.momentum

    delta_S_mq = signal_t_new - signal_t.signal_t
    delta_S_mb = do_fft(delta_S_mq, measurement_info.sk_position_momentum, measurement_info.rn_position_momentum)

    momentum_shift_term = jnp.exp(-1j*2*jnp.pi*position[:,None]*pulse_t_nir_vectorpotential[None,:])
    Dqb = jnp.exp(1j*2*jnp.pi*position[:,None]*momentum[None,:])
    grad_temp_q = jnp.einsum("qb, mb, mk, Cbk, bk, qk -> Cmq", Dqb, delta_S_mb, jnp.conjugate(pulse_t_euv_shifted), 
                             jnp.exp(1j*volkov_phase0), jnp.exp(1j*volkov_phase1), momentum_shift_term)

    grad = -2*1j*dt*do_fft(grad_temp_q, measurement_info.sk_position_momentum, measurement_info.rn_position_momentum)
    s = jnp.shape(grad) # C,m,b
    grad = jnp.reshape(grad, (s[0]*s[1], s[2]))
    _interpolate = Partial(do_interpolation_1d, method="cubic2")
    grad = jax.vmap(_interpolate, in_axes=(None,None,0))(measurement_info.axis_dtme.momentum, measurement_info.momentum, grad)
    grad = jnp.reshape(grad, (s[0], s[1], jnp.shape(grad)[-1]))
    return grad # with shape (C,m,b) 




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