from src.simulate_trace import MakePulse, GaussianAmplitude, PolynomialPhase
from src.chirp_scan import Basic, GeneralizedProjection, TimeDomainPtychography, COPRA

# only testing one phase matrix func shoud be fine. They are all tested in test_simulate_trace.py
from src.chirp_scan import phase_matrix_material
#from src.chirp_scan.phase_matrix_funcs import 
import refractiveindex
from scipy.constants import c as c0
parameters_material_scan = (refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), c0)

import numpy as np
import lineax
import pytest
import jax.numpy as jnp



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
                                                                  frequency_range=(0,1), real_fields=False)



pie_method = (None, "PIE", "ePIE", "rPIE", None, "PIE")
nonlinear_method = ("shg", "thg", "pg", "sd", "pg", "pg")
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


parameters_measurement = (z_arr, frequency, trace, spectra, phase_matrix_material, parameters_material_scan)
input_vals = []
for i in range(6):
    parameters_algorithm = (nonlinear_method[i], pie_method[i], guess_type[i], use_spectrum[i], use_momentum[i], jit[i], 
                            local_scaling[i], global_scaling[i], linesearch[i], local_newton[i], global_newton[i], linalg_solver[i], r_local_method[i], 
                            r_global_method[i], r_gradient[i], r_newton[i], r_step_scaling[i], conjugate_gradients[i])
    input_vals.append((parameters_measurement, parameters_algorithm))






@pytest.mark.parametrize("input_vals", input_vals)
def test_basic(input_vals):
    parameters_measurement, parameters_algorithm = input_vals
    z_arr, frequency, trace, spectra, phase_matrix_func, phase_matrix_parameters = parameters_measurement
    nonlinear_method, pie_method, guess_type, use_spectrum, use_momentum, jit, local_scaling, global_scaling, linesearch, local_newton, global_newton, linalg_solver,r_local_method, r_global_method, r_gradient, r_newton, r_step_scaling, conjugate_gradients = parameters_algorithm

    print(phase_matrix_parameters)
    basic = Basic(z_arr, frequency, trace, nonlinear_method, phase_matrix_func=phase_matrix_func, chirp_parameters=phase_matrix_parameters)
    basic.jit = jit

    if use_spectrum==True:
        basic.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
    if use_momentum==True:
        basic.momentum(population_size=5, eta=0.5)

    population = basic.create_initial_population(population_size=5, guess_type=guess_type)
    final_result = basic.run(population, no_iterations=5)








@pytest.mark.parametrize("input_vals", input_vals)
def test_GeneralizedProjection(input_vals):
    parameters_measurement, parameters_algorithm = input_vals
    z_arr, frequency, trace, spectra, phase_matrix_func, phase_matrix_parameters = parameters_measurement
    nonlinear_method, pie_method, guess_type, use_spectrum, use_momentum, jit, local_scaling, global_scaling, linesearch, local_newton, global_newton, linalg_solver,r_local_method, r_global_method, r_gradient, r_newton, r_step_scaling, conjugate_gradients = parameters_algorithm

    gp = GeneralizedProjection(z_arr, frequency, trace, nonlinear_method, phase_matrix_func=phase_matrix_func, chirp_parameters=phase_matrix_parameters)

    if use_spectrum==True:
        gp.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")
    if use_momentum==True:
        gp.momentum(population_size=5, eta=0.5)


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






@pytest.mark.parametrize("input_vals", input_vals)
def test_TimeDomainPtychography(input_vals):
    parameters_measurement, parameters_algorithm = input_vals
    z_arr, frequency, trace, spectra, phase_matrix_func, phase_matrix_parameters = parameters_measurement
    nonlinear_method, pie_method, guess_type, use_spectrum, use_momentum, jit, local_scaling, global_scaling, linesearch, local_newton, global_newton, linalg_solver,r_local_method, r_global_method, r_gradient, r_newton, r_step_scaling, conjugate_gradients = parameters_algorithm

    
    tdp = TimeDomainPtychography(z_arr, frequency, trace, nonlinear_method, pie_method, phase_matrix_func=phase_matrix_func, chirp_parameters=phase_matrix_parameters)
    if use_spectrum==True:
        tdp.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")

    if use_momentum==True:
        tdp.momentum(population_size=5, eta=0.5)


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

    if local_newton == "full":
        local_newton = "diagonal"
    if global_newton == "full":
        global_newton = "diagonal"

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







@pytest.mark.parametrize("input_vals", input_vals)
def test_COPRA(input_vals):
    parameters_measurement, parameters_algorithm = input_vals
    z_arr, frequency, trace, spectra, phase_matrix_func, phase_matrix_parameters = parameters_measurement
    nonlinear_method, pie_method, guess_type, use_spectrum, use_momentum, jit, local_scaling, global_scaling, linesearch, local_newton, global_newton, linalg_solver,r_local_method, r_global_method, r_gradient, r_newton, r_step_scaling, conjugate_gradients = parameters_algorithm

        
    copra = COPRA(z_arr, frequency, trace, nonlinear_method, phase_matrix_func=phase_matrix_func, chirp_parameters=phase_matrix_parameters)
    if use_spectrum==True:
        copra.use_measured_spectrum(spectra.pulse[0], spectra.pulse[1], "pulse")

    if use_momentum==True:
        copra.momentum(population_size=5, eta=0.5)

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