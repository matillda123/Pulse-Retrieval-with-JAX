from pulsedjax.simulate_trace import MakeTrace
from pulsedjax.simulate_trace import GaussianAmplitude, PolynomialPhase
import jax.numpy as jnp

amp0 = GaussianAmplitude((1,0), (0.2,0.35), (0.05,0.025), (3,1))
phase0 = PolynomialPhase(None, (0,0,50,-50))

amp1 = GaussianAmplitude((1,1), (0.25,0.3), (0.025,0.045), (1,1))
phase1 = PolynomialPhase(None, (0,0,-50,50))


mp = MakeTrace(N=128*10, f_max=1)
time, frequency, pulse_t, pulse_f = mp.generate_pulse((amp0,phase1))

_, frequency_gate, _, pulse_f_gate = mp.generate_pulse((amp1,phase0))


delay = jnp.linspace(-120,120,128)
delay, frequency_trace, trace, spectra = mp.generate_frog(time, frequency, pulse_t, pulse_f, "shg", delay, real_fields=False, 
                                                          cross_correlation=True,
                                                          gate=(frequency_gate, pulse_f_gate),
                                                          N=128*2, frequency_range=(0.1,0.6), interpolate_fft_conform=True)






from pulsedjax.frog import COPRA

gp = COPRA(delay, frequency_trace, trace, "shg", cross_correlation="doubleblind")

gp.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
gp.use_measured_spectrum(spectra.gate[0], spectra.gate[1], "gate")

population = gp.create_initial_population(5, "random")

final_result = gp.run(population, 50, 500)
gp.plot_results(final_result)