from pulsedjax.simulate_trace import MakeTrace
from pulsedjax.simulate_trace import GaussianAmplitude, PolynomialPhase
import jax.numpy as jnp

amp = GaussianAmplitude((1,1), (0.25,0.3), (0.025,0.045), (1,1))
phase = PolynomialPhase(None, (0,0,50,-50))

mp = MakeTrace(N=128*10, f_max=1)
time, frequency, pulse_t, pulse_f = mp.generate_pulse((amp,phase))

delay = jnp.linspace(-120,120,128*2)
delay, frequency_trace, trace, spectra = mp.generate_frog(time, frequency, pulse_t, pulse_f, "shg", delay,
                                                          N=128*2, frequency_range=(0.1, 0.8), 
                                                          interpolate_fft_conform=True)





from pulsedjax.frog import COPRA

copra = COPRA(delay, frequency_trace, trace, "shg")
copra.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")

population = copra.create_initial_population(5, "random")

copra.global_optimize_calibration_curve = True

final_result = copra.run(population, 150, 500)
copra.plot_results(final_result)