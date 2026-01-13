from pulsedjax.simulate_trace import MakeTrace, apply_noise
from pulsedjax.simulate_trace import (GaussianAmplitude, LorentzianAmplitude, 
                                PolynomialPhase, SinusoidalPhase, RandomPhase, MultiPulse)
import jax.numpy as jnp



# define pulse via predefined dataclasses
amp0 = GaussianAmplitude((1,1,1), (0.1,0.15,0.2), (0.01,0.05,0.1), (1,1,1))
amp1 = LorentzianAmplitude((1,1,1), (0.1,0.15,0.2), (0.01,0.05,0.1), (1,1,1))
phase0 = PolynomialPhase(None, (0,0,50,-25))
phase1 = SinusoidalPhase((0.1,0.1), (0.1,0.2), (20,25), (0, jnp.pi/2))
phase2 = RandomPhase()

pulse = MultiPulse(("G","G","L"), (1,1,1), (50,50), (25,25,25), (0.25,0.25,0.3), (1,2,3), (phase1, phase2, phase2))


# this class can generate pulses and traces
mp = MakeTrace(N=128*10, f_max=2)

time, pulse_t, frequency, pulse_f = mp.generate_pulse((amp1,phase1))
delay = jnp.linspace(time[0], time[-1], 256)
#time, pulse_t, frequency, pulse_f = mp.generate_pulse(pulse) # in case of MultiPulse
mp.plot_envelopes()


# make SHG-FROG
delay, frequency_trace, trace, spectra = mp.generate_frog(time, frequency, pulse_t, pulse_f, "shg", delay)


# make material based SHG-Chirp-Scan
import refractiveindex
phase_type = "material"
parameters = refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson")
z_arr = jnp.linspace(-1,1,128)

z_arr, frequency_trace, trace, spectra = mp.generate_chirpscan(time, frequency, pulse_t, pulse_f, "shg", 
                                                               z_arr, phase_type, parameters)





# generate SHG TDP
from pulsedjax import spectral_filter_funcs
spectral_filter = spectral_filter_funcs.get_filter("gaussian", frequency, (1,0.25,0.01,1))
delay, frequency_trace, trace, spectra = mp.generate_tdp(time, frequency, pulse_t, pulse_f, "shg", delay, spectral_filter)


# generate SFG 2DSI
spectral_filter1 = spectral_filter_funcs.get_filter("lorentzian", frequency, (1,0.24,0.005,1))
spectral_filter2 = spectral_filter_funcs.get_filter("lorentzian", frequency, (1,0.25,0.005,1))
delay, frequency_trace, trace, spectra = mp.generate_2dsi(time, frequency, pulse_t, pulse_f, "shg", delay, spectral_filter1, spectral_filter2,
                                                          scale_time_range=0.1)


# generate SHG VAMPIRE
delay, frequency_trace, trace, spectra = mp.generate_vampire(time, frequency, pulse_t, pulse_f, "shg", delay,
                                                             tau_interferometer=100, material_thickness=1)







# generate gaussian noise
noisy_trace = apply_noise(trace, (0.01,0.01), additive_noise=True, multiplicative_noise=True)
