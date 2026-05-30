from pulsedjax.simulate_trace import MakeTrace, GaussianAmplitude, PolynomialPhase
import jax.numpy as jnp

mp = MakeTrace(N=128*20, f_max=15, df=None)

def make_nir_pulse():
    amp0 = GaussianAmplitude((1,), (0.35,), (0.05,), (1,))
    phase0 = PolynomialPhase(None, (0,0,0,0))
    time, frequency, pulse_t_nir, pulse_f_nir = mp.generate_pulse((amp0,phase0))
    return time, frequency, pulse_t_nir, pulse_f_nir


def make_euv_pulse():
    amp1 = GaussianAmplitude((1,), (8.5,), (0.95,), (1,)) # inputs are still in PHz not eV.
    phase1 = PolynomialPhase(None, (0,0,0,0))
    time, frequency, pulse_t_euv, pulse_f_euv = mp.generate_pulse((amp1, phase1))
    return pulse_t_euv, pulse_f_euv


time_fs, frequency_PHz, pulse_t_nir, pulse_f_nir = make_nir_pulse()
pulse_t_euv, pulse_f_euv = make_euv_pulse()

delay_fs = jnp.linspace(-15,15,100)


# providing multiple ionization potential, triggers usage of multiple SFA-Channels
# If a DTME is provided the number of DTMEs needs to match the number of ionization potentials
Ip_ev = jnp.array([0])

delay_fs, energy_eV, trace, spectra = mp.generate_streaking(time_fs, frequency_PHz, (pulse_t_nir, pulse_f_nir), 
                                                            (pulse_t_euv, pulse_f_euv), delay_fs, Ip_eV=Ip_ev, 
                                                            energy_range=(25,50), N=128)






""" Streaking retrieval has to use a huge frequency axis internally. Thus it can take much longer than e.g. FROG"""

from pulsedjax.streaking import COPRA
gp = COPRA(delay_fs, energy_eV, trace, Ip_eV=jnp.array([0]),
           f_range_nir_pulse=(0.2,0.5), f_range_euv_pulse=(6,11)
           )  # specify frequency ranges for each pulse

population = gp.create_initial_population(1, "constant")

gp.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
gp.use_measured_spectrum(spectra.gate[0], spectra.gate[1], "gate")

gp.local_gamma = 1e4
gp.global_gamma = 4
gp.optimize_spectral_phase_directly = True # using this option seems to enhance retrieval (in contrast to e.g. its effect in FROG)

final_result = gp.run(population, 50, 150)
gp.plot_results(final_result)







from pulsedjax.streaking import AutoDiff
import optax
solver = optax.adam(learning_rate=0.5)

ad = AutoDiff(delay_fs, energy_eV, trace, Ip_eV=jnp.array([0]),
              solver=solver, 
              f_range_nir_pulse=(0.2,0.5), f_range_euv_pulse=(6,11)
              ) # specify frequency ranges for each pulse

ad.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
ad.use_measured_spectrum(spectra.gate[0], spectra.gate[1], "gate")

population = ad.create_initial_population(1, "bsplines_5", "bsplines_5", 30, 30)

final_result = ad.run(population, 250)
ad.plot_results(final_result)