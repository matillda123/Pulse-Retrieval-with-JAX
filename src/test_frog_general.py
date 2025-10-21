from src.frog import DifferentialEvolution, Evosax, LSF, AutoDiff
from src.simulate_trace import MakePulse, GaussianAmplitude, PolynomialPhase

import optax
import optimistix

import numpy as np
import pytest



amp_g = GaussianAmplitude(amplitude=np.asarray([1]), central_frequency=np.asarray([0.3]), fwhm=np.asarray([0.1]))
phase_p = PolynomialPhase(central_frequency=np.asarray([0.3]), coefficients=np.asarray([0,0,250,-2500]))

pulse_maker = MakePulse(N=128*10, Delta_f=2)
time, pulse_t, frequency, pulse_f = pulse_maker.generate_pulse((amp_g, phase_p))
gate = (frequency, pulse_f)

delay, frequency, trace, spectra = pulse_maker.generate_frog(time, frequency, pulse_t, pulse_f, nonlinear_method="pg", N=256, 
                                                             scale_time_range=1, plot_stuff=False, cross_correlation=False, 
                                                             gate=gate, ifrog=False, interpolate_fft_conform=True, 
                                                             cut_off_val=1e-1, frequency_range=(0,1), real_fields=False)



nonlinear_method = ("shg", "thg", "pg", "sd", "shg")
cross_correlation = (False, True, "doubleblind", False, True)
use_measured_spectrum = (False, True, True, False, False)

amp_type = ("gaussian", "lorentzian", "bsplines", "discrete", "gaussian")
phase_type = ("polynomial", "sinusoidal", "sigmoidal", "bsplines", "discrete")

parameters_measurement = (delay, frequency, trace, spectra, gate)





# DE
mutations = ("best1", "best2", "rand1", "rand2", "randtobest1", "randtobest2", "currenttorand1", "currenttorand2", "currenttobest1", "currenttobest2")

crossover = ("bin", "exp", "smooth", "bin", "exp")
selection_mechanism = ("greedy", "global", "greedy", "global", "greedy")

parameters = []
for i in range(5):
    strategy = mutations[np.random.randint(0,10)] + "_" + crossover[i]
    parameters_algorithm = (nonlinear_method[i], cross_correlation[i], strategy, selection_mechanism[i], amp_type[i], phase_type[i])
    parameters.append((parameters_measurement, parameters_algorithm))


@pytest.mark.parametrize("parameters", parameters)
def test_DifferentialEvolution(parameters):
    parameters_measurement, parameters_algorithm = parameters
    delay, frequency, trace, spectra, gate = parameters_measurement
    nonlinear_method, cross_correlation, strategy, selection_mechanism, amp_type, phase_type = parameters_algorithm

    de = DifferentialEvolution(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation)

    if use_measured_spectrum==True:
        de.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
        if cross_correlation=="doubleblind":
            de.use_measured_spectrum(spectra.gate[0], spectra.gate[1], "gate")

    if cross_correlation==True:
        frequency_gate, pulse_gate = gate
        gate = de.get_gate_pulse(frequency_gate, pulse_gate)

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
    parameters_algorithm = (nonlinear_method[i], cross_correlation[i], solver[i], amp_type[i], phase_type[i])
    parameters.append((parameters_measurement, parameters_algorithm))


@pytest.mark.parametrize("parameters", parameters)
def test_Evosax(parameters):
    parameters_measurement, parameters_algorithm = parameters
    delay, frequency, trace, spectra, gate = parameters_measurement
    nonlinear_method, cross_correlation, solver, amp_type, phase_type = parameters_algorithm
    
    evo = Evosax(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation)

    if use_measured_spectrum==True:
        evo.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
        if cross_correlation=="doubleblind":
            evo.use_measured_spectrum(spectra.gate[0], spectra.gate[1], "gate")

    if cross_correlation==True:
        frequency_gate, pulse_gate = gate
        gate = evo.get_gate_pulse(frequency_gate, pulse_gate)

    evo.solver = solver

    population = evo.create_initial_population(population_size=5, amp_type=amp_type, phase_type=phase_type, no_funcs_amp=5, no_funcs_phase=5)
    final_result = evo.run(population, 5)








# lsf
random_direction_mode = ("random", "continuous", "random", "continuous", "random")

parameters = []
for i in range(5):
    parameters_algorithm = (nonlinear_method[i], cross_correlation[i], random_direction_mode[i], amp_type[i], phase_type[i])
    parameters.append((parameters_measurement, parameters_algorithm))


@pytest.mark.parametrize("parameters", parameters)
def test_LSF(parameters):
    parameters_measurement, parameters_algorithm = parameters
    delay, frequency, trace, spectra, gate = parameters_measurement
    nonlinear_method, cross_correlation, random_direction_mode, amp_type, phase_type = parameters_algorithm

    lsf = LSF(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation)

    if use_measured_spectrum==True:
        lsf.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
        if cross_correlation=="doubleblind":
            lsf.use_measured_spectrum(spectra.gate[0], spectra.gate[1], "gate")

    if cross_correlation==True:
        frequency_gate, pulse_gate = gate
        gate = lsf.get_gate_pulse(frequency_gate, pulse_gate)

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
    parameters_algorithm = (nonlinear_method[i], cross_correlation[i], solvers[i], alternating_optimization[i], amp_type[i], phase_type[i])
    parameters.append((parameters_measurement, parameters_algorithm))



@pytest.mark.parametrize("parameters", parameters)
def test_AutoDiff(parameters):
    parameters_measurement, parameters_algorithm = parameters
    delay, frequency, trace, spectra, gate = parameters_measurement
    nonlinear_method, cross_correlation, solver, alternating_optimization, amp_type, phase_type = parameters_algorithm

    ad = AutoDiff(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation)

    if use_measured_spectrum==True:
        ad.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
        if cross_correlation=="doubleblind":
            ad.use_measured_spectrum(spectra.gate[0], spectra.gate[1], "gate")

    if cross_correlation==True:
        frequency_gate, pulse_gate = gate
        gate = ad.get_gate_pulse(frequency_gate, pulse_gate)

    ad.solver = solver
    ad.alternating_optimization = alternating_optimization

    population = ad.create_initial_population(population_size=5, amp_type=amp_type, phase_type=phase_type, no_funcs_amp=5, no_funcs_phase=5)
    final_result = ad.run(population, 5)