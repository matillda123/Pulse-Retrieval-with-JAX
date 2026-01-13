from pulsedjax.simulate_trace import MakeTrace
from pulsedjax.simulate_trace import GaussianAmplitude, PolynomialPhase, RandomPhase
import jax.numpy as jnp


amp0 = GaussianAmplitude(1, 0.25, 0.1, 2)
phase0 = RandomPhase()


mp = MakeTrace(N=128*20, f_max=2)
time, pulse_t, frequency, pulse_f = mp.generate_pulse((amp0, phase0))


# create lorentzian spectral filters with amp=1, central frequencies of 0.245/0.255, fwhm=0.0001 and p=1
from pulsedjax import spectral_filter_funcs
spectral_filter1 = spectral_filter_funcs.get_filter("lorentzian", frequency, (1,0.255,0.0001,1))
spectral_filter2 = spectral_filter_funcs.get_filter("lorentzian", frequency, (1,0.245,0.0001,1))

delay = jnp.linspace(-30, 30, 128*2)
delay, frequency_trace, trace, spectra = mp.generate_2dsi(time, frequency, pulse_t, pulse_f, "shg", delay, 
                                                    spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2, 

                                                    # for any shg, thg based trace, the frequency axis should include the
                                                    # range of the fundamental spectrum
                                                    frequency_range=(0.1,0.6))



# the filters need to be interpolated onto frequency_trace 
from pulsedjax.utilities import do_interpolation_1d
spectral_filter1 = do_interpolation_1d(frequency_trace, frequency, spectral_filter1)
spectral_filter2 = do_interpolation_1d(frequency_trace, frequency, spectral_filter2)




from pulsedjax.twodsi import DirectReconstruction

dr = DirectReconstruction(delay, frequency_trace, trace, "shg", spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2)

# DirectReconstruction needs the fundamental spectrum
dr.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
population = dr.create_initial_population(1, "random")

# only one iteration needed
final_result = dr.run(population, 1)
dr.plot_results(final_result)