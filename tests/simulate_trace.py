from ..src.simulate_trace import MakePulse, GaussianAmplitude, MultiPulse, PolynomialPhase, SinusoidalPhase, CustomPulse

import numpy as np


GaussianAmplitude(amplitude=np.asarray([1]), central_frequency=np.asarray([0.3]), fwhm=np.asarray([0.1]))
MultiPulse(delay=, duration=, central_frequency=, amplitude=, phase=)
PolynomialPhase(central_frequency=, coefficients=)
SinusoidalPhase(amplitude=, periodicity=, phase_shift=)
CustomPulse(frequency=, amplitude=, phase=)



def generate_frog(pulse_parameters, trace_parameters):
    pulse_maker = MakePulse(N=128*4, Delta_f=1)

    time, pulse_t, frequency, pulse_f = pulse_maker.generate_pulse(pulse_parameters)

    nonlinear_method, cross_correlation, ifrog, interpolate_fft_conform, frequency_range, real_fields = trace_parameters
    simulated_measurement = pulse_maker.generate_frog(time, frequency, pulse_t, pulse_f, nonlinear_method, 
                                                      N=256, scale_time_range=1, plot_stuff=False, 
                                                      cross_correlation=cross_correlation,
                                                      gate=(frequency, pulse_f), 
                                                      ifrog=ifrog, interpolate_fft_conform=interpolate_fft_conform, 
                                                      cut_off_val=1e-6, 
                                                      frequency_range=frequency_range, real_fields=real_fields)

