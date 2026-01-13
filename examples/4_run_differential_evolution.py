from pulsedjax.simulate_trace import MakeTrace
from pulsedjax.simulate_trace import GaussianAmplitude, PolynomialPhase
import jax.numpy as jnp

amp0 = GaussianAmplitude((1,1,1), (0.125,0.15,0.2), (0.01,0.05,0.1), (1,1,1))
phase0 = PolynomialPhase(None, (0,0,50,-25))


mp = MakeTrace(N=128*10, f_max=2)
time, pulse_t, frequency, pulse_f = mp.generate_pulse((amp0,phase0))

delay_inp = jnp.linspace(-150,150,256)
delay, frequency_trace, trace, spectra = mp.generate_frog(time, frequency, pulse_t, pulse_f, "pg", delay_inp,
                                                          interpolate_fft_conform=True)







from pulsedjax.frog import DifferentialEvolution

de = DifferentialEvolution(delay, frequency_trace, trace, "pg")

# # Incorporates spectrum by only optimizing the spectral phase. 
# frequency_spectrum, spectrum = spectra.pulse[0], spectra.pulse[1]
# de.use_measured_spectrum(frequency_spectrum, spectrum, "pulse")

de.strategy = "best1_smooth"
de.selection_mechanism = "global"

population = de.create_initial_population(150, amp_type="continuous", phase_type="continuous")

final_result = de.run(population, 100)
de.plot_results(final_result)