from src.simulate_trace import MakePulse, GaussianAmplitude, PolynomialPhase
from src.twodsi import DifferentialEvolution, Evosax, LSF, AutoDiff
from src import spectral_filter_funcs
from src.utilities import do_interpolation_1d

import numpy as np
import optimistix
import optax
import pytest

pulse_maker = MakePulse(N=128*10, Delta_f=1)

central_f = np.array([0.3])
phase = PolynomialPhase(central_frequency = central_f, coefficients = np.array([0.5, 0, 000]))
amp = GaussianAmplitude(central_frequency = central_f, amplitude = np.array([1.0]), fwhm = np.array([0.1]))
time_inp, pulse_t_inp, frequency_inp, pulse_f_inp = pulse_maker.generate_pulse((amp, phase))

input_pulses = pulse_maker.pulses



central_f = np.array([0.4])
phase = PolynomialPhase(central_frequency=central_f, coefficients = np.zeros(3))
amp = GaussianAmplitude(central_frequency = central_f, amplitude = np.array([1.0]), fwhm = np.array([0.01]))
_, _, frequency_gate, pulse_f_gate = pulse_maker.generate_pulse((amp, phase))


spectral_filter1 = spectral_filter_funcs.get_filter("lorentzian", frequency_inp, (1, 0.3, 0.0005))
spectral_filter2 = spectral_filter_funcs.get_filter("lorentzian", frequency_inp, (1, 0.31, 0.0005))


delay, frequency, trace, spectra=pulse_maker.generate_2dsi(time_inp, frequency_inp, pulse_t_inp, pulse_f_inp, "pg", cross_correlation=True,
                                                          gate=(frequency_gate, pulse_f_gate), 
                                                          N=64, scale_time_range=0.25, plot_stuff=True, cut_off_val=0.001, frequency_range=(0, 0.5))

spectral_filter1 = do_interpolation_1d(frequency, frequency_inp, spectral_filter1)
spectral_filter2 = do_interpolation_1d(frequency, frequency_inp, spectral_filter2)




nonlinear_method = ("shg", "thg", "pg", "sd", "shg")
cross_correlation = (False, True, "doubleblind", False, True)
use_measured_spectrum = (False, True, True, False, False)
fd_grad = (False, 0, 1, False, False)
amplitude_or_intensity = ("intensity", "amplitude", 3, 0.25, 1.5)

amp_type = ("gaussian", "lorentzian", "bsplines_5", "discrete", "gaussian")
phase_type = ("polynomial", "sinusoidal", "sigmoidal", "bsplines_5", "discrete")

gate = (frequency_gate, pulse_f_gate)
parameters_measurement = (delay, frequency, trace, spectra, gate)






# DE
mutations = ("best1", "best2", "rand1", "rand2", "randtobest1", "randtobest2", "currenttorand1", "currenttorand2", "currenttobest1", "currenttobest2")

crossover = ("bin", "exp", "smooth", "bin", "exp")
selection_mechanism = ("greedy", "global", "greedy", "global", "greedy")

parameters = []
for i in range(5):
    strategy = mutations[np.random.randint(0,10)] + "_" + crossover[i]
    parameters_algorithm = (nonlinear_method[i], cross_correlation[i], strategy, selection_mechanism[i], amp_type[i], phase_type[i], fd_grad[i], amplitude_or_intensity[i])
    parameters.append((parameters_measurement, parameters_algorithm))


@pytest.mark.parametrize("parameters", parameters)
def test_DifferentialEvolution(parameters):
    parameters_measurement, parameters_algorithm = parameters
    delay, frequency, trace, spectra, gate = parameters_measurement
    nonlinear_method, cross_correlation, strategy, selection_mechanism, amp_type, phase_type, fd_grad, amplitude_or_intensity = parameters_algorithm

    de = DifferentialEvolution(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation)

    if use_measured_spectrum==True:
        de.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
        if cross_correlation=="doubleblind":
            de.use_measured_spectrum(spectra.gate[0], spectra.gate[1], "gate")

    if cross_correlation==True:
        frequency_gate, pulse_f_gate = gate
        gate = de.get_gate_pulse(frequency_gate, pulse_f_gate)

    de.fd_grad = fd_grad
    de.amplitude_or_intensity = amplitude_or_intensity

    de.strategy = strategy
    de.mutation_rate = 0.5
    de.crossover_rate = 0.5
    de.selection_mechanism = selection_mechanism
    de.temperature = 0.5

    population = de.create_initial_population(population_size=5, amp_type=amp_type, phase_type=phase_type, no_funcs_amp=5, no_funcs_phase=5)

    final_result = de.run(population, 5)






# evo
from evosax.algorithms import DifferentialEvolution as evo_de, DiffusionEvolution
solver = (evo_de, DiffusionEvolution, DiffusionEvolution, evo_de, evo_de)

parameters = []
for i in range(5):
    parameters_algorithm = (nonlinear_method[i], cross_correlation[i], solver[i], amp_type[i], phase_type[i], fd_grad[i], amplitude_or_intensity[i])
    parameters.append((parameters_measurement, parameters_algorithm))


@pytest.mark.parametrize("parameters", parameters)
def test_Evosax(parameters):
    parameters_measurement, parameters_algorithm = parameters
    delay, frequency, trace, spectra, gate = parameters_measurement
    nonlinear_method, cross_correlation, solver, amp_type, phase_type, fd_grad, amplitude_or_intensity = parameters_algorithm
    
    evo = Evosax(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation)

    if use_measured_spectrum==True:
        evo.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
        if cross_correlation=="doubleblind":
            evo.use_measured_spectrum(spectra.gate[0], spectra.gate[1], "gate")

    if cross_correlation==True:
        frequency_gate, pulse_f_gate = gate
        gate = evo.get_gate_pulse(frequency_gate, pulse_f_gate)

    evo.fd_grad = fd_grad
    evo.amplitude_or_intensity = amplitude_or_intensity
    evo.solver = solver

    population = evo.create_initial_population(population_size=5, amp_type=amp_type, phase_type=phase_type, no_funcs_amp=5, no_funcs_phase=5)
    final_result = evo.run(population, 5)








# lsf
random_direction_mode = ("random", "continuous", "random", "continuous", "random")

parameters = []
for i in range(5):
    parameters_algorithm = (nonlinear_method[i], cross_correlation[i], random_direction_mode[i], amp_type[i], phase_type[i], fd_grad[i], amplitude_or_intensity[i])
    parameters.append((parameters_measurement, parameters_algorithm))


@pytest.mark.parametrize("parameters", parameters)
def test_LSF(parameters):
    parameters_measurement, parameters_algorithm = parameters
    delay, frequency, trace, spectra, gate = parameters_measurement
    nonlinear_method, cross_correlation, random_direction_mode, amp_type, phase_type, fd_grad, amplitude_or_intensity = parameters_algorithm

    lsf = LSF(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation)

    if use_measured_spectrum==True:
        lsf.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
        if cross_correlation=="doubleblind":
            lsf.use_measured_spectrum(spectra.gate[0], spectra.gate[1], "gate")

    if cross_correlation==True:
        frequency_gate, pulse_f_gate = gate
        gate = lsf.get_gate_pulse(frequency_gate, pulse_f_gate)

    lsf.fd_grad = fd_grad
    lsf.amplitude_or_intensity = amplitude_or_intensity
    lsf.number_of_bisection_iterations = 8
    lsf.random_direction_mode = random_direction_mode
    lsf.no_points_for_continuous = 10

    population = lsf.create_initial_population(population_size=5, amp_type=amp_type, phase_type=phase_type, no_funcs_amp=5, no_funcs_phase=5)
    final_result = lsf.run(population, 5)









# ad 
solvers = (optax.adam(learning_rate=0.1), optimistix.BFGS, optimistix.LevenbergMarquardt, optimistix.BFGS, optimistix.GaussNewton)
alternating_optimization = (True, False, True, False, False)

parameters = []
for i in range(5):
    parameters_algorithm = (nonlinear_method[i], cross_correlation[i], solvers[i], alternating_optimization[i], amp_type[i], phase_type[i], fd_grad[i], amplitude_or_intensity[i])
    parameters.append((parameters_measurement, parameters_algorithm))



@pytest.mark.parametrize("parameters", parameters)
def test_AutoDiff(parameters):
    parameters_measurement, parameters_algorithm = parameters
    delay, frequency, trace, spectra, gate = parameters_measurement
    nonlinear_method, cross_correlation, solver, alternating_optimization, amp_type, phase_type, fd_grad, amplitude_or_intensity = parameters_algorithm

    ad = AutoDiff(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation)

    if use_measured_spectrum==True:
        ad.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
        if cross_correlation=="doubleblind":
            ad.use_measured_spectrum(spectra.gate[0], spectra.gate[1], "gate")

    if cross_correlation==True:
        frequency_gate, pulse_f_gate = gate
        gate = ad.get_gate_pulse(frequency_gate, pulse_f_gate)

    ad.fd_grad = fd_grad
    ad.amplitude_or_intensity = amplitude_or_intensity
    ad.solver = solver
    ad.alternating_optimization = alternating_optimization

    population = ad.create_initial_population(population_size=5, amp_type=amp_type, phase_type=phase_type, no_funcs_amp=5, no_funcs_phase=5)
    final_result = ad.run(population, 5)