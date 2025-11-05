import pytest

from src.simulate_trace import MakePulse, GaussianAmplitude, MultiPulse, PolynomialPhase, SinusoidalPhase, CustomPulse
import numpy as np


amp_g = GaussianAmplitude(amplitude=np.asarray([1]), central_frequency=np.asarray([0.3]), fwhm=np.asarray([0.1]))
phase_p = PolynomialPhase(central_frequency=np.asarray([0.3]), coefficients=np.asarray([0,0,250,-250]))
phase_s = SinusoidalPhase(amplitude=np.asarray([0.05]), periodicity=np.asarray([15]), phase_shift=np.asarray([0]))

pulse_m = MultiPulse(delay=np.asarray([-50,50]), duration=np.asarray([20,10,15]), central_frequency=np.asarray([0.2,0.25,0.35]), amplitude=np.asarray([1,2,1]), 
                     phase=[phase_s, phase_p, phase_p])
pulse_c = CustomPulse(frequency=np.linspace(0,0.5,1000), amplitude=np.ones(1000), phase=np.zeros(1000))

pulse_parameters = ((amp_g, phase_p), (amp_g, phase_s), pulse_m, pulse_c)






nonlinear_method = ("shg", "thg", "pg", "sd")
cross_correlation = (False, False, True, "doubleblind")
ifrog = (False, True, False, False)
interpolate_fft_conform = (False, False, True, False)
frequency_range = ((0, 1), (0, 1.5), (0, 1), (0.2, 0.5))
real_fields = (True, False, False, False)

parameters = []
for i in range(4):
    parameters.append((pulse_parameters[i], (nonlinear_method[i], cross_correlation[i], ifrog[i], interpolate_fft_conform[i], frequency_range[i], real_fields[i])))


@pytest.mark.parametrize("parameters", parameters)
def test_generate_frog(parameters):
    pulse_parameters, trace_parameters = parameters

    pulse_maker = MakePulse(N=128*4, Delta_f=2)
    time, pulse_t, frequency, pulse_f = pulse_maker.generate_pulse(pulse_parameters)

    nonlinear_method, cross_correlation, ifrog, interpolate_fft_conform, frequency_range, real_fields = trace_parameters
    simulated_measurement = pulse_maker.generate_frog(time, frequency, pulse_t, pulse_f, nonlinear_method, 
                                                      N=256, scale_time_range=1, plot_stuff=False, 
                                                      cross_correlation=cross_correlation,
                                                      gate=(frequency, pulse_f), 
                                                      ifrog=ifrog, interpolate_fft_conform=interpolate_fft_conform, 
                                                      cut_off_val=1e-6, 
                                                      frequency_range=frequency_range, real_fields=real_fields)









from src.chirp_scan import phase_matrix_material, phase_matrix_miips, phase_matrix_tanh
from scipy.constants import c as c0
import refractiveindex

parameters_material_scan = (refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), c0)
parameters_miips = (0.75, 5, 0.3)
parameters_tanh = (10, 0, 0.3)


z_arr = (np.linspace(-2,5,128), np.linspace(0,2*np.pi,128), np.linspace(-5,5,128))
nonlinear_method = ("shg", "thg", "pg")
phase_matrix_func = (phase_matrix_material, phase_matrix_miips, phase_matrix_tanh)
phase_parameters = (parameters_material_scan, parameters_miips, parameters_tanh)
frequency_range = ((0,1), (0,1.5), (0,0.5))
real_fields = (False, True, False)


parameters = []
for i in range(3):
    parameters.append((pulse_parameters[i], (z_arr[i], nonlinear_method[i], phase_matrix_func[i], phase_parameters[i], frequency_range[i], real_fields[i])))


@pytest.mark.parametrize("parameters", parameters)
def test_generate_chirp_scan(parameters):
    pulse_parameters, trace_parameters = parameters

    pulse_maker = MakePulse(N=128*4, Delta_f=2)
    time, pulse_t, frequency, pulse_f = pulse_maker.generate_pulse(pulse_parameters)
    
    z_arr, nonlinear_method, phase_matrix_func, parameters, frequency_range, real_fields = trace_parameters
    simulated_measurement = pulse_maker.generate_chirpscan(z_arr, time, frequency, pulse_t, pulse_f, nonlinear_method, 
                                                           phase_matrix_func=phase_matrix_func, parameters=parameters, 
                                                           N=256, plot_stuff=False, cut_off_val=1e-6, frequency_range=frequency_range, real_fields=real_fields)










nonlinear_method = ("shg", "thg", "pg", "sd")
cross_correlation = (False, False, True, "doubleblind")
frequency_range = ((0, 1), (0, 1.5), (0, 1), (0.2, 0.5))

parameters = []
for i in range(4):
    parameters.append((pulse_parameters[i], (nonlinear_method[i], cross_correlation[i], frequency_range[i])))


@pytest.mark.parametrize("parameters", parameters)
def test_generate_2dsi(parameters):
    pulse_parameters, trace_parameters = parameters

    pulse_maker = MakePulse(N=128*4, Delta_f=2)
    time, pulse_t, frequency, pulse_f = pulse_maker.generate_pulse(pulse_parameters)

    nonlinear_method, cross_correlation, frequency_range = trace_parameters
    simulated_measurement = pulse_maker.generate_2dsi(time, frequency, pulse_t, pulse_f, nonlinear_method, cross_correlation, 
                                                      anc=((frequency, pulse_f),(frequency, pulse_f)), 
                                                      N=256, scale_time_range=1, plot_stuff=False, cut_off_val=1e-6, frequency_range=frequency_range)

