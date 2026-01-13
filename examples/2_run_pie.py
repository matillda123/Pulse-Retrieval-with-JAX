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





from pulsedjax.frog import PtychographicIterativeEngine

pie = PtychographicIterativeEngine(delay, frequency_trace, trace, "pg", "rPIE")

# # Incorporates Spectrum via projection onto it
# frequency_spectrum, spectrum = spectra.pulse[0], spectra.pulse[1]
# pie.use_measured_spectrum(frequency_spectrum, spectrum, "pulse")

# # Incorporates momentum into the descent.
# pie.momentum(5, 0.75)

population = pie.create_initial_population(5, "random")

pie.alpha = 0.15
pie.local_gamma = 0.1
pie.global_gamma = 0.1

final_result = pie.run(population, 100, 100)
pie.plot_results(final_result)



# is the same or very similar for all other classical solvers