from pulsedjax.simulate_trace import MakeTrace
from pulsedjax.simulate_trace import GaussianAmplitude, RandomPhase
import jax.numpy as jnp

# create broadbandspectrum with random phase
amp0 = GaussianAmplitude((1,1,1,0.9,0.5), (0.175,0.2,0.25,0.29,0.35), (0.01,0.05,0.1,0.1,0.05), (1,1,1,2,3))
phase0 = RandomPhase(number_of_points=4)


mp = MakeTrace(N=128*10, f_max=2)
time, pulse_t, frequency, pulse_f = mp.generate_pulse((amp0,phase0))


import refractiveindex
parameters = refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson")
z_arr = jnp.linspace(-2,2,128*2)
z_arr, frequency_trace, trace, spectra = mp.generate_chirpscan(time, frequency, pulse_t, pulse_f, "shg", 
                                                               z_arr, "material", parameters, 
                                                               
                                                               # shg/thg need to have the fundamental in the frequency axis
                                                               frequency_range=(0.1,0.75))




from pulsedjax.chirp_scan import COPRA
copra = COPRA(z_arr, frequency_trace, trace, "shg", phase_type="material", chirp_parameters=parameters)

# stepsizes
copra.local_gamma = 1e4 # sometimes the local step size needs to be quite large
copra.global_gamma = 0.25

# damping parameter, avoids division by zero in adaptive stepsize
copra.xi = 1e-3

population = copra.create_initial_population(5, "random")
final_result = copra.run(population, 100, 300)

copra.plot_results(final_result)