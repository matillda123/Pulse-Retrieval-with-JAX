from pulsedjax.simulate_trace import MakeTrace, GaussianAmplitude, PolynomialPhase
import jax.numpy as jnp


mp = MakeTrace(N=128*20, f_max=15, df=None)

def make_nir_pulse():
    amp0 = GaussianAmplitude((1,), (0.35,), (0.075,), (1,))
    phase0 = PolynomialPhase(None, (0,0,0,0))
    time, frequency, pulse_t_nir, pulse_f_nir = mp.generate_pulse((amp0,phase0))
    return time, frequency, pulse_t_nir, pulse_f_nir


def make_euv_pulse():
    amp1 = GaussianAmplitude((1,0), (8.5,6.5), (0.95,0.5), (1,1))
    phase1 = PolynomialPhase(None, (0,0,0,0))
    time, frequency, pulse_t_euv, pulse_f_euv = mp.generate_pulse((amp1, phase1))
    return pulse_t_euv, pulse_f_euv


time_fs, frequency_PHz, pulse_t_nir, pulse_f_nir = make_nir_pulse()
pulse_t_euv, pulse_f_euv = make_euv_pulse()
delay_fs = jnp.linspace(-7,7,100)


# create a momentum axis that covers the range of the streaking trace
frequency_au = frequency_PHz*2.418884*1e-2
energy_au = frequency_au*2*jnp.pi
k = jnp.sqrt(2*jnp.abs(energy_au))*jnp.sign(energy_au)
k = jnp.linspace(jnp.min(k), jnp.max(k), jnp.size(k))

# create a fake dtme based on an analytic solution of the hydrogen atom
# give it a fake quadratic dispersion centered around k = 1.6 a.u.
k0 = 1.6
K = 25
dtme = jnp.exp(-2*jnp.arctan(k)/k)/(1-jnp.exp(-2*jnp.pi/k))*1/(1+k**2)**2 * jnp.exp(1j*2*jnp.pi*K*(k-k0)**2)


time, energy, trace, spectra = mp.generate_streaking(time_fs, frequency_PHz,
                                                     (pulse_t_nir, pulse_f_nir),
                                                     (pulse_t_euv, pulse_f_euv), 
                                                     delay_fs, Ip_eV=jnp.array([0]), 
                                                     energy_range=(25,45), N=128, 
                                                     DTME=(k, dtme))







from pulsedjax.streaking import AutoDiff
import optax
solver = optax.adam(learning_rate=0.1)


ad = AutoDiff(delay_fs, energy, trace, Ip_eV=jnp.array([0]), retrieve_dtme=True, 
              solver=solver, 
              f_range_nir_pulse=(0.1,0.6), f_range_euv_pulse=(6,11), eV_range_dtme=(28,42)
              ) # specify frequency ranges for each pulse

ad.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
ad.use_measured_spectrum(spectra.gate[0], spectra.gate[1], "gate")

population = ad.create_initial_population(2, "bsplines_5", "bsplines_5", 30, 30)

final_result = ad.run(population, 5000)
ad.plot_results(final_result)