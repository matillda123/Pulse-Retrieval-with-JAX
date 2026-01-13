from pulsedjax.simulate_trace import MakeTrace
from pulsedjax.simulate_trace import GaussianAmplitude, PolynomialPhase, RandomPhase
import jax.numpy as jnp

amp0 = GaussianAmplitude(1, 0.2, 0.05)
phase0 = PolynomialPhase(None, (0,0,0,0))
phase2 = RandomPhase()


mp = MakeTrace(N=128*20, f_max=2)

time, pulse_t, frequency, pulse_f = mp.generate_pulse((amp0,phase2))
_, _, frequency_gate, pulse_f_gate = mp.generate_pulse((amp0,phase0))

delay = jnp.linspace(time[0], time[-1], 256)
delay, frequency_trace, trace, spectra = mp.generate_frog(time, frequency, pulse_t, pulse_f, "shg", delay, cross_correlation=True,
                                                          gate=(frequency_gate, pulse_f_gate))