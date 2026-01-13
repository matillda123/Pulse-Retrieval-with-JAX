"""
Under specific circumstances pulse-characterization can be sensitve to the CEP and RP of pulses. 
One of these methods is named TREX (https://spj.science.org/doi/10.34133/ultrafastscience.0081)

Using the real_fields-module, such traces can be simulated and retrieved.

"""

from pulsedjax.simulate_trace import MakeTrace
from pulsedjax.simulate_trace import GaussianAmplitude, PolynomialPhase
import jax.numpy as jnp

amp0 = GaussianAmplitude((1,1), (0.15,0.175), (0.05,0.075), (2,2))
phase0 = PolynomialPhase(None, (0,0,20,0)) # setting central_frequency=None -> uses center-of-mass of spectrum

amp1 = GaussianAmplitude((1,1), (0.35,0.4), (0.05,0.075))
phase1 = PolynomialPhase(None, (0.25,0,-5,0))


mp = MakeTrace(N=128*20, f_max=2)

time, pulse_t, frequency, pulse_f = mp.generate_pulse((amp0, phase0))
time_gate, pulse_t_gate, frequency_gate, pulse_f_gate = mp.generate_pulse((amp1, phase1))

delay = jnp.linspace(-30, 30, 128*2)
delay, frequency, trace, spectra = mp.generate_frog(time, frequency, pulse_t, pulse_f, "thg", delay, 
                                                    cross_correlation=True, interferometric=True, real_fields=True,
                                                    gate=(frequency_gate, pulse_f_gate), 
                                                    
                                                    # for real_fields one needs to manually zoom into a specfic f-range
                                                    # this is because the fundamental and egative frequencies are present
                                                    frequency_range=(0.55,1.2))




from pulsedjax.real_fields import frog
import optax
import optimistix

ad = frog.AutoDiff(delay, frequency, trace, "thg", cross_correlation="doubleblind", interferometric=True, 
                   
                   # Same thing here, one needs to specify the f-range in which the pulse/gate are located
                   f_range_fields=(0.075, 0.5))


# in doubleblind providing the spectra is needed to avoid ambiguities
ad.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], pulse_or_gate="pulse")
ad.use_measured_spectrum(spectra.gate[0], spectra.gate[1], pulse_or_gate="gate")

# phase is approximated using 10 5th order B-Splines.
population = ad.create_initial_population(5, phase_type="bsplines_5", no_funcs_phase=10)

ad.solver = optax.adam(learning_rate=0.1)
#ad.solver = optimistix.LBFGS(1, 1)

final_result = ad.run(population, 500)
ad.plot_results(final_result)