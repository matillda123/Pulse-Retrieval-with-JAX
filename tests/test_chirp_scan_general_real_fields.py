from src.real_fields import chirp_scan #DifferentialEvolution, Evosax, LSF, AutoDiff
from src.simulate_trace import MakePulse, GaussianAmplitude, PolynomialPhase

import optax
import optimistix

import numpy as np
import pytest
import jax.numpy as jnp



# only testing one phase matrix func shoud be fine. They are all tested in test_simulate_trace.py
from src.chirp_scan import phase_matrix_material
#from src.chirp_scan.phase_matrix_funcs import 
import refractiveindex
from scipy.constants import c as c0
parameters_material_scan = (refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), c0)


amp_g = GaussianAmplitude(amplitude=np.asarray([1]), central_frequency=np.asarray([0.3]), fwhm=np.asarray([0.1]))
phase_p = PolynomialPhase(central_frequency=np.asarray([0.3]), coefficients=np.asarray([0,0,250,-2500]))

pulse_maker = MakePulse(N=128*10, Delta_f=2)
time, pulse_t, frequency, pulse_f = pulse_maker.generate_pulse((amp_g, phase_p))

z_arr = jnp.linspace(-1,5,128)
z_arr, frequency, trace, spectra = pulse_maker.generate_chirpscan(z_arr, time, frequency, pulse_t, pulse_f, 
                                                                  nonlinear_method="pg", 
                                                                  phase_matrix_func=phase_matrix_material, 
                                                                  parameters=parameters_material_scan, 
                                                                  N=64, plot_stuff=False, cut_off_val=1e-6, 
                                                                  frequency_range=(0,1), real_fields=True)



nonlinear_method = ("shg", "thg", "pg", "pg", "shg")
cross_correlation = (False, True, "doubleblind", False, True)
use_measured_spectrum = (False, True, True, False, False)
fd_grad = (False, 0, 1, False, False)
amplitude_or_intensity = ("intensity", "amplitude", 3, 0.25, 1.5)

amp_type = ("gaussian", "lorentzian", "bsplines_5", "discrete", "gaussian")
phase_type = ("polynomial", "sinusoidal", "sigmoidal", "bsplines_5", "discrete")

parameters_measurement = (z_arr, frequency, trace, spectra, phase_matrix_material, parameters_material_scan)




# DE
mutations = ("best1", "best2", "rand1", "rand2", "randtobest1", "randtobest2", "currenttorand1", "currenttorand2", "currenttobest1", "currenttobest2")

crossover = ("bin", "exp", "smooth", "bin", "exp")
selection_mechanism = ("greedy", "global", "greedy", "global", "greedy")

parameters = []
for i in range(5):
    strategy = mutations[np.random.randint(0,10)] + "_" + crossover[i]
    parameters_algorithm = (nonlinear_method[i], strategy, selection_mechanism[i], amp_type[i], phase_type[i], fd_grad[i], amplitude_or_intensity[i])
    parameters.append((parameters_measurement, parameters_algorithm))



@pytest.mark.parametrize("parameters", parameters)
def test_DifferentialEvolution(parameters):
    parameters_measurement, parameters_algorithm = parameters
    z_arr, frequency, trace, spectra, phase_matrix_material, parameters_material_scan = parameters_measurement
    nonlinear_method, strategy, selection_mechanism, amp_type, phase_type, fd_grad, amplitude_or_intensity = parameters_algorithm

    de = chirp_scan.DifferentialEvolution(z_arr, frequency, trace, nonlinear_method, phase_matrix_func = phase_matrix_material, chirp_parameters=parameters_material_scan, f_range_fields=(0,1))

    if use_measured_spectrum==True:
        de.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")

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
    parameters_algorithm = (nonlinear_method[i], solver[i], amp_type[i], phase_type[i], fd_grad[i], amplitude_or_intensity[i])
    parameters.append((parameters_measurement, parameters_algorithm))


@pytest.mark.parametrize("parameters", parameters)
def test_Evosax(parameters):
    parameters_measurement, parameters_algorithm = parameters
    z_arr, frequency, trace, spectra, phase_matrix_material, parameters_material_scan = parameters_measurement
    nonlinear_method, solver, amp_type, phase_type, fd_grad, amplitude_or_intensity = parameters_algorithm
    
    evo = chirp_scan.Evosax(z_arr, frequency, trace, nonlinear_method, phase_matrix_func = phase_matrix_material, chirp_parameters=parameters_material_scan, f_range_fields=(0,1))

    if use_measured_spectrum==True:
        evo.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")

    evo.fd_grad = fd_grad
    evo.amplitude_or_intensity = amplitude_or_intensity
    evo.solver = solver

    population = evo.create_initial_population(population_size=5, amp_type=amp_type, phase_type=phase_type, no_funcs_amp=5, no_funcs_phase=5)
    final_result = evo.run(population, 5)








# lsf
random_direction_mode = ("random", "continuous", "random", "continuous", "random")

parameters = []
for i in range(5):
    parameters_algorithm = (nonlinear_method[i], random_direction_mode[i], amp_type[i], phase_type[i], fd_grad[i], amplitude_or_intensity[i])
    parameters.append((parameters_measurement, parameters_algorithm))


@pytest.mark.parametrize("parameters", parameters)
def test_LSF(parameters):
    parameters_measurement, parameters_algorithm = parameters
    z_arr, frequency, trace, spectra, phase_matrix_material, parameters_material_scan = parameters_measurement
    nonlinear_method, random_direction_mode, amp_type, phase_type, fd_grad, amplitude_or_intensity = parameters_algorithm

    lsf = chirp_scan.LSF(z_arr, frequency, trace, nonlinear_method, phase_matrix_func = phase_matrix_material, chirp_parameters=parameters_material_scan, f_range_fields=(0,1))

    if use_measured_spectrum==True:
        lsf.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")

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
    parameters_algorithm = (nonlinear_method[i], solvers[i], alternating_optimization[i], amp_type[i], phase_type[i], fd_grad[i], amplitude_or_intensity[i])
    parameters.append((parameters_measurement, parameters_algorithm))



@pytest.mark.parametrize("parameters", parameters)
def test_AutoDiff(parameters):
    parameters_measurement, parameters_algorithm = parameters
    z_arr, frequency, trace, spectra, phase_matrix_material, parameters_material_scan = parameters_measurement
    nonlinear_method, solver, alternating_optimization, amp_type, phase_type, fd_grad, amplitude_or_intensity = parameters_algorithm

    ad = chirp_scan.AutoDiff(z_arr, frequency, trace, nonlinear_method, phase_matrix_func = phase_matrix_material, chirp_parameters=parameters_material_scan, f_range_fields=(0,1))

    if use_measured_spectrum==True:
        ad.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")

    ad.fd_grad = fd_grad
    ad.amplitude_or_intensity = amplitude_or_intensity
    ad.solver = solver
    ad.alternating_optimization = alternating_optimization

    population = ad.create_initial_population(population_size=5, amp_type=amp_type, phase_type=phase_type, no_funcs_amp=5, no_funcs_phase=5)
    final_result = ad.run(population, 5)