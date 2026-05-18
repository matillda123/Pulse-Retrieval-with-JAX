from pulsedjax.simulate_trace import MakeTrace, GaussianAmplitude, PolynomialPhase
import jax.numpy as jnp

mp = MakeTrace(N=128*20, f_max=15, df=None)

def make_nir_pulse():
    amp0 = GaussianAmplitude((1,), (0.35,), (0.05,), (1,))
    phase0 = PolynomialPhase(None, (0,0,0,0))
    time, frequency, pulse_t_nir, pulse_f_nir = mp.generate_pulse((amp0,phase0))
    return time, frequency, pulse_t_nir, pulse_f_nir


def make_euv_pulse():
    amp1 = GaussianAmplitude((1,), (8.5,), (0.95,), (1,))
    phase1 = PolynomialPhase(None, (0,0,0,0))
    time, frequency, pulse_t_euv, pulse_f_euv = mp.generate_pulse((amp1, phase1))
    return pulse_t_euv, pulse_f_euv


time_fs, frequency_PHz, pulse_t_nir, pulse_f_nir = make_nir_pulse()
pulse_t_euv, pulse_f_euv = make_euv_pulse()

delay = jnp.linspace(-15,15,100)


# # this will create a fake DTME, it can be put into the retrieval algorithm via algorithm.get_DTME(momentum_au, dtme)
# frequency_au = frequency_PHz*2.418884*1e-2
# energy_au = frequency_au*2*jnp.pi
# momentum_au = jnp.sqrt(2*jnp.abs(energy_au))*jnp.sign(energy_au)
# momentum_au = jnp.linspace(jnp.min(momentum_au), jnp.max(momentum_au), jnp.size(momentum_au))

# K = 0.1
# dtme_phase_factor = jnp.exp(1j*2*jnp.pi*momentum_au/K)
# dtme_amp = jnp.exp(-2*jnp.arctan(momentum_au)/momentum_au)/(1-jnp.exp(-2*jnp.pi/momentum_au))*1/(1+momentum_au**2)**2
# dtme = dtme_amp*dtme_phase_factor


delay_fs, energy_eV, trace, spectra = mp.generate_streaking(time_fs, frequency_PHz, (pulse_t_nir, pulse_f_nir), 
                                                            (pulse_t_euv, pulse_f_euv), delay, Ip_eV=jnp.array([0]), 
                                                            energy_range=(25,50), N=128, #DTME=(momentum_au, dtme)
                                                            )






""" Streaking retrieval has to use a huge frequency axis internally. Thus it can take much longer than e.g. FROG"""

from pulsedjax.streaking import COPRA
gp = COPRA(delay_fs, energy_eV, trace, Ip_eV=jnp.array([0]), 
           f_range_nir_pulse=(0.2,0.5), f_range_euv_pulse=(6,11))  # specify frequency ranges for each pulse

population = gp.create_initial_population(1, "constant")

gp.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
gp.use_measured_spectrum(spectra.gate[0], spectra.gate[1], "gate")

gp.local_gamma = 1e4
gp.global_gamma = 4
gp.optimize_spectral_phase_directly = True

final_result = gp.run(population, 50, 150)
gp.plot_results(final_result)







from pulsedjax.streaking import AutoDiff
import optax
solver = optax.adam(learning_rate=0.5)

ad = AutoDiff(delay_fs, energy_eV, trace, Ip_eV=jnp.array([0]), solver=solver, 
              f_range_nir_pulse=(0.2,0.5), f_range_euv_pulse=(6,11)) # specify frequency ranges for each pulse

ad.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
ad.use_measured_spectrum(spectra.gate[0], spectra.gate[1], "gate")

population = ad.create_initial_population(1, "bsplines_5", "bsplines_5", 30, 30)

final_result = ad.run(population, 250)
ad.plot_results(final_result)