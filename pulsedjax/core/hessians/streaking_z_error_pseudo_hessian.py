import jax.numpy as jnp
from pulsedjax.utilities import do_fft, do_interpolation_1d, calculate_newton_direction
import jax
from functools import partial as Partial


def Z_pseudo_hessian_diagonal_EUV_pulse(signal_t, signal_t_new, tau_arr, measurement_info): # frequency domain
    dtme_shifted_and_volkov_phase0, volkov_phase1 = signal_t.dtme_shifted_and_volkov_phase0, signal_t.volkov_phase1
    time = measurement_info.time
    dt = jnp.mean(jnp.diff(time))
    omega = 2*jnp.pi*measurement_info.frequency
    Dkn = jnp.exp(1j*time[:,None]*omega[None,:])

    # exp_arrs cancel on diagonal -> no explicit m dependence

    _term = jnp.einsum("kn, Nbk, Nbk -> Nbn", Dkn, dtme_shifted_and_volkov_phase0, jnp.exp(-1j*volkov_phase1))

    _interpolate = Partial(do_interpolation_1d, x_new=measurement_info.axis_euv.frequency, x=measurement_info.frequency, method="linear")
    _term = jax.vmap(jax.vmap(_interpolate))(_term)

    Uzz_diag = 0.5*dt**2*jnp.abs(_term)**2 # is the same for each m
    Uzz_diag = jnp.broadcast_to(Uzz_diag, (jnp.size(tau_arr), ) + jnp.shape(Uzz_diag))
    return Uzz_diag # has shape (N,m,n)








def _get_derivatives_of_dtme_with_respect_to_vectorpotential(dtme_position, pulse_t_nir_vectorpotential, measurement_info, gradient_or_hessian):
    if measurement_info.dtme_momentum is None and measurement_info.retrieve_dtme==False:
        dtme_momentum = jnp.zeros((jnp.shape(pulse_t_nir_vectorpotential)[0], measurement_info.no_channels, jnp.size(measurement_info.momentum), jnp.size(pulse_t_nir_vectorpotential)))
    else: 
        r = measurement_info.position
        sk, rn = measurement_info.sk_position_momentum, measurement_info.rn_position_momentum

        momentum_shift = -1*jnp.real(pulse_t_nir_vectorpotential)
        dtme_position = dtme_position[:,:,None,:]*jnp.exp(-1j*2*jnp.pi*r[None,None,None,:]*momentum_shift[:,None,:,None]) 

        if gradient_or_hessian=="gradient":
            dtme_position = 1j*2*jnp.pi*dtme_position*r[None,None,None,:]
        elif gradient_or_hessian=="hessian":
             dtme_position = -1*4*jnp.pi**2*dtme_position*(r[None,None,None,:])**2
        else:
            raise ValueError

        dr = jnp.mean(jnp.diff(r))
        phase_correction = jnp.exp(1j*2*jnp.pi*dr*momentum_shift)
        dtme_position = dtme_position*phase_correction[:,None,:,None]

        dtme_momentum = do_fft(dtme_position, sk, rn)
        dtme_momentum = jnp.transpose(dtme_momentum, (0,1,3,2)) # -> such that output is (N,C,b,k)

    return dtme_momentum





def Z_pseudo_hessian_diagonal_vectorpotential(signal_t, signal_t_new, tau_arr, measurement_info): # frequency domain
    """ Loads of product and chain rule in this one :D """
    pulse_t_euv_shifted, pulse_t_nir_vectorpotential = signal_t.pulse_t_euv_shifted, signal_t.pulse_t_nir_vectorpotential
    dtme_position, dtme_shifted_and_volkov_phase0 = signal_t.dtme_position, signal_t.dtme_shifted_and_volkov_phase0
    volkov_phase0, volkov_phase1 = signal_t.volkov_phase0, signal_t.volkov_phase1

    time, omega = measurement_info.time, 2*jnp.pi*measurement_info.frequency
    dt = jnp.mean(jnp.diff(time))
    momentum = measurement_info.momentum
    
    gradient_dtme_NCbk = _get_derivatives_of_dtme_with_respect_to_vectorpotential(dtme_position, pulse_t_nir_vectorpotential, measurement_info, "gradient")    
    term1 = jnp.einsum("Nmk, NCbk, Cbk, Nbk -> Nmbk", pulse_t_euv_shifted, gradient_dtme_NCbk, jnp.exp(-1j*volkov_phase0), jnp.exp(-1j*volkov_phase1))

    grad_volkov1 = dt*(momentum[None,:,None] + jnp.real(pulse_t_nir_vectorpotential)[:,None,:])
    _term2 = dtme_shifted_and_volkov_phase0*jnp.exp(-1j*volkov_phase1)
    _term2 = jnp.einsum("Nmk, Nbk -> Nmbk", pulse_t_euv_shifted, _term2)
    _term2 = jnp.cumsum(_term2, axis=-1)
    term2 = -1j*jnp.einsum("Nmbj, Nbj -> Nmbj", _term2, grad_volkov1)

    term_12 = do_fft(term1 + term2, measurement_info.sk, measurement_info.rn)
    Uzz_diag = dt**2*jnp.einsum("Nmbn -> Nmn", jnp.real(jnp.abs(term_12)**2)) # jnp.real is included because for the full hessian it would be necessary
    


    Dkn = jnp.exp(1j*time[:,None]*omega[None,:])

    term1 = dt*(momentum[None,:,None] + jnp.real(pulse_t_nir_vectorpotential)[:,None,:])
    term1 = jnp.einsum("Nbk, Nbj -> Nbkj", term1, term1)
    term1 = jnp.diag(jnp.ones(jnp.size(time))*dt**2) - 1j*term1

    _term1 = jnp.einsum("Nmk, Nbk, Nbk -> Nmbk", pulse_t_euv_shifted, dtme_shifted_and_volkov_phase0, jnp.exp(-1j*volkov_phase1))
    _term1 = jnp.cumsum(_term1, axis=-1)
    term1 = -1*dt*jnp.einsum("kn, nj, Nbkj, Nmbj, Nmbk -> Nmbn", Dkn, jnp.conjugate(Dkn).T, term1, _term1, _term1) # for full hessian use kn, pj, ...


    term2 = dt*(momentum[None,None,:,None,None] + jnp.real(pulse_t_nir_vectorpotential)[:,None,None,:,None])*gradient_dtme_NCbk[:,:,:,:,None]
    term2 = term2 + jnp.transpose(term2, (0,1,2,4,3))
    hessian_dtme_NCbk = _get_derivatives_of_dtme_with_respect_to_vectorpotential(dtme_position, pulse_t_nir_vectorpotential, measurement_info, "hessian")
    idx = jnp.arange(jnp.size(time))
    term2 = term2.at[..., idx,idx].add(1j*hessian_dtme_NCbk)
    term2 = -1*dt*jnp.einsum("kn, nj, Nmk, Cbk, Nbk, NCbkj -> Nmbn", Dkn, jnp.conjugate(Dkn).T, pulse_t_euv_shifted, jnp.exp(-1j*volkov_phase0), jnp.exp(-1j*volkov_phase1), term2) # for full hessian use kn, pj, ...

    delta_S_mq = signal_t_new - signal_t.signal_t
    delta_S_mb = do_fft(delta_S_mq, measurement_info.sk_position_momentum, measurement_info.rn_position_momentum)
    Vzz_diag = jnp.real(jnp.einsum("Nmbn, Nmb -> Nmn", jnp.conjugate(term1 + term2), delta_S_mb))

    hessian_diag = Uzz_diag - Vzz_diag
    _interpolate = Partial(do_interpolation_1d, x_new=measurement_info.axis_nir.frequency, x=measurement_info.frequency, method="cubic2")
    hessian_diag = jax.vmap(jax.vmap(_interpolate))(hessian_diag)
    return hessian_diag






def Z_pseudo_hessian_diagonal_DTME(signal_t, signal_t_new, tau_arr, measurement_info): # momentum domain
    dt = jnp.mean(jnp.diff(measurement_info.time))
    r, momentum = measurement_info.position, measurement_info.momentum
    pulse_t_euv_shifted, pulse_t_nir_vectorpotential = signal_t.pulse_t_euv_shifted, signal_t.pulse_t_nir_vectorpotential
    volkov_phase0, volkov_phase1 = signal_t.volkov_phase0, signal_t.volkov_phase1

    Dbq = jnp.exp(1j*2*jnp.pi*r[None,:]*momentum[:,None])
    Dqb = jnp.conjugate(Dbq).T
    momentum_shift = jnp.exp(1j*2*jnp.pi*r[None,:,None]*jnp.real(pulse_t_nir_vectorpotential)[:,None,:])

    term1 = jnp.einsum("aq, qb, Nqk -> Nkab", Dbq, Dqb, momentum_shift)
    term2 = jnp.einsum("Nmk, Cbk, Nbk, Nkab -> NCmab", pulse_t_euv_shifted, jnp.exp(-1j*volkov_phase0), jnp.exp(-1j*volkov_phase1), term1)
    Uzz_diag = 0.5*dt**2*jnp.einsum("NCmab -> NCma", jnp.abs(term2)**2)
    return Uzz_diag # has shape (N,C,m,b)







def get_pseudo_newton_direction_Z_error(grad_m, signal_t, signal_t_new, tau_arr, descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
    """
    Calculates the pseudo-newton direction for the Z-error of a FROG measurement.
    The direction is calculated in the frequency domain.

    Args:
        grad_m (jnp.array): the current Z-error gradient
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
    assert full_or_diagonal!="full", "Dont use full with streaking. Its way to expensive."

    calculate_Z_hessian_diag_dict = {"pulse": Z_pseudo_hessian_diagonal_vectorpotential,
                                     "gate": Z_pseudo_hessian_diagonal_EUV_pulse, 
                                     "dtme": Z_pseudo_hessian_diagonal_DTME}

    lambda_lm = descent_info.newton.lambda_lm
    solver = descent_info.newton.linalg_solver
    newton_direction_prev = getattr(descent_state.newton, pulse_or_gate).newton_direction_prev  

    hessian_m = calculate_Z_hessian_diag_dict[pulse_or_gate](signal_t, signal_t_new, tau_arr, measurement_info)

    # for direct phase optimization
    if pulse_or_gate!="dtme":
        if getattr(descent_info.measured_spectrum_is_provided, pulse_or_gate)==True:
            spectral_amplitude = getattr(measurement_info.spectral_amplitude, pulse_or_gate)
            hessian_m = hessian_m*spectral_amplitude[None,None,:]**2 # square because n=p

    return calculate_newton_direction(grad_m, hessian_m, lambda_lm, newton_direction_prev, solver, full_or_diagonal)

        











