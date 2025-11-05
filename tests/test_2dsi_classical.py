from src.simulate_trace import MakePulse, GaussianAmplitude, PolynomialPhase
from src.twodsi import DirectReconstruction, GeneralizedProjection, TimeDomainPtychography, COPRA

import numpy as np
import lineax
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
_, _, frequency_gate_1, pulse_f_gate_1 = pulse_maker.generate_pulse((amp, phase))


central_f = np.array([0.41])
phase = PolynomialPhase(central_frequency = central_f, coefficients = np.zeros(3))
amp = GaussianAmplitude(central_frequency = central_f, amplitude = np.array([1.0]), fwhm = np.array([0.01]))
_, _, frequency_gate_2, pulse_f_gate_2 = pulse_maker.generate_pulse((amp, phase))


delay, frequency, trace, spectra=pulse_maker.generate_2dsi(time_inp, frequency_inp, pulse_t_inp, pulse_f_inp, "pg", cross_correlation=True,
                                                          anc=((frequency_gate_1, pulse_f_gate_1),
                                                               (frequency_gate_2, pulse_f_gate_2)), 
                                                          N=64, scale_time_range=0.25, plot_stuff=True, cut_off_val=0.001, frequency_range=(0, 0.5))




nonlinear_method = ("shg", "thg", "sd", "pg")
guess_type = ("random", "random_phase", "constant", "constant_phase")
jit = (False, True, False, True)
windowing = (False, "hamming", "hann", False)
integration_method = ("cumsum", "euler_maclaurin_1", "euler_maclaurin_3", "euler_maclaurin_5")


gate = ((frequency_gate_1, pulse_f_gate_1),(frequency_gate_2, pulse_f_gate_2))
parameters_measurement = (delay, frequency, trace, spectra, gate)
parameters = []
for i in range(4):
    parameters_algorithm = (nonlinear_method[i], guess_type[i], jit[i], windowing[i], integration_method[i])
    parameters.append((parameters_measurement, parameters_algorithm))




@pytest.mark.parametrize("parameters", parameters)
def test_DirectReconstruction(parameters):
    parameters_measurement, parameters_algorithm = parameters
    delay, frequency, trace, spectra, gate = parameters_measurement
    nonlinear_method, guess_type, jit, windowing, integration_method = parameters_algorithm

    dr = DirectReconstruction(delay, frequency, trace, nonlinear_method, anc1_frequency=0.4, anc2_frequency=0.405)
    dr.jit = jit

    dr.windowing = windowing
    dr.integration_method = integration_method

    dr.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
    anc1, anc2 = gate
    frequency_gate_1, pulse_f_gate_1 = anc1
    frequency_gate_2, pulse_f_gate_2 = anc2
    anc1 = dr.get_anc_pulse(frequency_gate_1, pulse_f_gate_1, anc_no=1)
    anc2 = dr.get_anc_pulse(frequency_gate_2, pulse_f_gate_2, anc_no=2)

    population = dr.create_initial_population(population_size=1, guess_type=guess_type)
    final_result = dr.run(population, no_iterations=1)











pie_method = (None, "PIE", "ePIE", "rPIE", None, "PIE")
nonlinear_method = ("shg", "thg", "pg", "sd", "pg", "pg")
cross_correlation = (False, True, "doubleblind", False, True, False)
jit = (False, True, False, True, False, False)
guess_type = ("random", "random_phase", "constant", "constant_phase", "random", "random")
local_scaling = ("pade_10", "pade_20", "pade_11", "pade_01", "pade_02", "pade_10")
global_scaling = ("pade_10", "pade_20", "pade_11", "pade_01", "pade_02", "pade_10")
linesearch = (False, "backtracking", "zoom", False, False, False)
conjugate_gradients = (False, "polak_ribiere", "hestenes_stiefel", "dai_yuan", "fletcher_reeves", "average")
local_newton = (False, "diagonal", "lbfgs", False, "diagonal", "lbfgs")
global_newton = (False, "diagonal", "full", "lbfgs", "full", "full")
linalg_solver = ("scipy", "lineax", lineax.GMRES(rtol=1e-3, atol=1e-3), "scipy", "lineax", "scipy")
r_local_method = ("projection", "iteration", "iteration", "projection", "projection", "iteration")
r_global_method = ("projection", "iteration", "iteration", "iteration", "projection", "projection")
r_gradient = ("intensity", "amplitude", "intensity", "amplitude", "intensity", "amplitude")
r_newton = (False, True, False, False, True, False)
r_step_scaling = ("pade_10", "pade_20", "pade_11", "pade_01", "pade_02", "pade_10")
use_spectrum = (False, True, False, True, False, True)
use_momentum = (False, True, False, False, True, False)


gate = ((frequency_gate_1, pulse_f_gate_1),(frequency_gate_2, pulse_f_gate_2))
parameters_measurement = (delay, frequency, trace, spectra, gate)
parameters = []
for i in range(6):
    parameters_algorithm = (nonlinear_method[i], pie_method[i], cross_correlation[i], guess_type[i], use_spectrum[i], use_momentum[i], jit[i], 
                            local_scaling[i], global_scaling[i], linesearch[i], local_newton[i], global_newton[i], linalg_solver[i], r_local_method[i], 
                            r_global_method[i], r_gradient[i], r_newton[i], r_step_scaling[i], conjugate_gradients[i])
    parameters.append((parameters_measurement, parameters_algorithm))




@pytest.mark.parametrize("parameters", parameters)
def test_GeneralizedProjection(parameters):
    parameters_measurement, parameters_algorithm = parameters
    delay, frequency, trace, spectra, gate = parameters_measurement
    nonlinear_method, pie_method, cross_correlation, guess_type, use_spectrum, use_momentum, jit, local_scaling, global_scaling, linesearch, local_newton, global_newton, linalg_solver,r_local_method, r_global_method, r_gradient, r_newton, r_step_scaling, conjugate_gradients = parameters_algorithm

    gp = GeneralizedProjection(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation)

    if use_spectrum==True:
        gp.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
        if cross_correlation=="doubleblind":
            gp.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "gate")

    if use_momentum==True:
        gp.momentum(population_size=5, eta=0.5)

    if cross_correlation==True:
        anc1, anc2 = gate
        frequency_gate_1, pulse_f_gate_1 = anc1
        frequency_gate_2, pulse_f_gate_2 = anc2
        anc1 = gp.get_anc_pulse(frequency_gate_1, pulse_f_gate_1, anc_no=1)
        anc2 = gp.get_anc_pulse(frequency_gate_2, pulse_f_gate_2, anc_no=2)

    if linesearch=="zoom":
        gp.delta_gamma=1.5

    gp.jit = jit
    gp.no_steps_descent = 5

    gp.global_gamma = 0.5
    gp.global_adaptive_scaling = global_scaling

    gp.linesearch = linesearch
    gp.max_steps_linesearch = 15

    gp.conjugate_gradients = conjugate_gradients

    gp.global_newton = global_newton
    gp.linalg_solver = linalg_solver
    gp.lambda_lm = 1e-3
    gp.lbfgs_memory = 5

    gp.r_global_method = r_global_method
    gp.r_gradient = r_gradient
    gp.r_newton = r_newton
    gp.r_no_iterations = 5
    gp.r_step_scaling = r_step_scaling

    population = gp.create_initial_population(population_size=5, guess_type=guess_type)
    final_result = gp.run(population, no_iterations=5)








@pytest.mark.parametrize("parameters", parameters)
def test_TimeDomainPtychography(parameters):
    parameters_measurement, parameters_algorithm = parameters
    delay, frequency, trace, spectra, gate = parameters_measurement
    nonlinear_method, pie_method, cross_correlation, guess_type, use_spectrum, use_momentum, jit, local_scaling, global_scaling, linesearch, local_newton, global_newton, linalg_solver,r_local_method, r_global_method, r_gradient, r_newton, r_step_scaling, conjugate_gradients = parameters_algorithm

    if cross_correlation=="doubleblind":
        cross_correlation=True

    tdp = TimeDomainPtychography(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation, pie_method=pie_method)
    if use_spectrum==True:
        tdp.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
        if cross_correlation=="doubleblind":
            tdp.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "gate")

    if use_momentum==True:
        tdp.momentum(population_size=5, eta=0.5)

    if cross_correlation==True:
        anc1, anc2 = gate
        frequency_gate_1, pulse_f_gate_1 = anc1
        frequency_gate_2, pulse_f_gate_2 = anc2
        anc1 = tdp.get_anc_pulse(frequency_gate_1, pulse_f_gate_1, anc_no=1)
        anc2 = tdp.get_anc_pulse(frequency_gate_2, pulse_f_gate_2, anc_no=2)

    if linesearch=="zoom":
        tdp.delta_gamma=1.5

    tdp.jit = jit

    tdp.local_gamma = 0.5
    tdp.global_gamma = 0.5
    tdp.local_adaptive_scaling = local_scaling
    tdp.global_adaptive_scaling = global_scaling

    tdp.linesearch = linesearch
    tdp.max_steps_linesearch = 25

    tdp.conjugate_gradients = conjugate_gradients

    tdp.local_newton = local_newton
    tdp.global_newton = global_newton
    tdp.linalg_solver = linalg_solver
    tdp.lambda_lm = 1e-3
    tdp.lbfgs_memory = 5

    tdp.r_local_method = r_local_method
    tdp.r_global_method = r_global_method
    tdp.r_gradient = r_gradient
    tdp.r_newton = r_newton
    tdp.r_no_iterations = 5
    tdp.r_step_scaling = r_step_scaling

    population = tdp.create_initial_population(population_size=5, guess_type=guess_type)
    final_result = tdp.run(population, 5, 5)







@pytest.mark.parametrize("parameters", parameters)
def test_COPRA(parameters):
    parameters_measurement, parameters_algorithm = parameters
    delay, frequency, trace, spectra, gate = parameters_measurement
    nonlinear_method, pie_method, cross_correlation, guess_type, use_spectrum, use_momentum, jit, local_scaling, global_scaling, linesearch, local_newton, global_newton, linalg_solver,r_local_method, r_global_method, r_gradient, r_newton, r_step_scaling, conjugate_gradients = parameters_algorithm

        
    copra = COPRA(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation)
    if use_spectrum==True:
        copra.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
        if cross_correlation=="doubleblind":
            copra.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "gate")

    if use_momentum==True:
        copra.momentum(population_size=5, eta=0.5)

    if cross_correlation==True:
        anc1, anc2 = gate
        frequency_gate_1, pulse_f_gate_1 = anc1
        frequency_gate_2, pulse_f_gate_2 = anc2
        anc1 = copra.get_anc_pulse(frequency_gate_1, pulse_f_gate_1, anc_no=1)
        anc2 = copra.get_anc_pulse(frequency_gate_2, pulse_f_gate_2, anc_no=2)

    if linesearch=="zoom":
        copra.delta_gamma=1.5

    copra.jit = jit

    copra.local_gamma = 0.5
    copra.global_gamma = 0.5
    copra.local_adaptive_scaling = local_scaling
    copra.global_adaptive_scaling = global_scaling

    copra.linesearch = linesearch
    copra.max_steps_linesearch = 25

    copra.conjugate_gradients = conjugate_gradients

    copra.local_newton = local_newton
    copra.global_newton = global_newton
    copra.linalg_solver = linalg_solver
    copra.lambda_lm = 1e-3
    copra.lbfgs_memory = 5

    copra.r_local_method = r_local_method
    copra.r_global_method = r_global_method
    copra.r_gradient = r_gradient
    copra.r_newton = r_newton
    copra.r_no_iterations = 5
    copra.r_step_scaling = r_step_scaling

    population = copra.create_initial_population(population_size=5, guess_type=guess_type)
    final_result = copra.run(population, 5, 5)