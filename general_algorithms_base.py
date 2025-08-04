import numpy as np

import jax.numpy as jnp
import jax
from jax.tree_util import Partial
from equinox import tree_at

import equinox
import optimistix

from utilities import scan_helper, get_com, optimistix_helper_loss_function, optimistix_helper_alternating_loss_function, scan_helper_equinox, MyNamespace, do_fft, do_ifft, calculate_trace, calculate_trace_error, loss_function_modifications, generate_random_continuous_function, do_interpolation_1d
from BaseClasses import AlgorithmsBASE
from create_population import create_population_general




def eval_bspline_segment(x, M, control_points):
        return control_points @ M.T @ x


def add_control_points_out_of_domain(control_points, degree, n):
    temp=jnp.zeros(n + degree)
    temp=temp.at[degree//2:n+degree//2].set(control_points)
    temp=temp.at[:degree//2].set(control_points[0])
    temp=temp.at[-degree//2:].set(control_points[-1])
    return temp


def b_spline(control_points, x):
    # M_1=jnp.array([[1, 0], [-1, 1]])
    # M_2=1/2*jnp.array([[1, 1, 0], [-2, 2, 0], [1, -2, 1]])
    # M_3=1/6*jnp.array([[1, 4, 1, 0], [-3, 0, 3, 0], [3, -6, 3, 0], [-1, 3, -3, 1]])
    
    M=1/24*jnp.array([[1, 11, 11, 1, 0], [-4, -12, 12, 4, 0], [6, -6, -6, 6, 0], [-4, 12, -12, 4, 0], [1, -4, 6, -4, 1]])
    k=5
    degree=4
    n=jnp.size(control_points)
    control_points = add_control_points_out_of_domain(control_points, degree, n)

    N=jnp.size(x)
    splines_per_point=n/N
    x=jnp.arange(0, 1, splines_per_point)
    x = x**jnp.reshape(jnp.arange(k), (-1,1))

    control_points=jax.vmap(jnp.roll, in_axes=(None, 0))(control_points, -1*jnp.arange(k))
    control_points=jnp.transpose(control_points)[:-1*degree]

    spline_curve = jax.vmap(eval_bspline_segment, in_axes=(None, None, 0))(x, M, control_points)
    spline_curve = jnp.ravel(spline_curve)

    return spline_curve









class GeneralOptimization(AlgorithmsBASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_fd_grad = False
        self.amplitude_or_intensity = "intensity"
        self.error_metric = calculate_trace_error

        

    def create_initial_population(self, population_size, phase_type="polynomial", amp_type="gaussian", no_funcs_amp=5, no_funcs_phase=6):
        self.key, subkey = jax.random.split(self.key, 2)

        population=MyNamespace(pulse=MyNamespace(amp=None, phase=None), 
                               gate=MyNamespace(amp=None, phase=None))
        
        subkey, population_pulse = create_population_general(subkey, amp_type, phase_type, population.pulse, population_size, no_funcs_amp, no_funcs_phase, 
                                                             self.descent_info.measured_spectrum_is_provided.pulse, self.measurement_info)
        population = tree_at(lambda x: x.pulse, population, population_pulse)
        
        if self.doubleblind==True:
            subkey, population_gate = create_population_general(subkey, amp_type, phase_type, population.gate, population_size, no_funcs_amp, no_funcs_phase, 
                                                                self.descent_info.measured_spectrum_is_provided.gate, self.measurement_info)
            population = tree_at(lambda x: x.gate, population, population_gate)
            
        if phase_type=="random":
            phase_type = "discrete"
            
        if amp_type=="random":
            amp_type = "discrete"
        
        self.descent_info = self.descent_info.expand(population_size=population_size, phase_type=phase_type, amp_type=amp_type)
        return population
    


    def split_population_in_amp_and_phase(self, population):
        population_amp = MyNamespace(pulse=MyNamespace(amp=population.pulse.amp, phase=None), 
                                     gate=MyNamespace(amp=population.gate.amp, phase=None))
        
        population_phase = MyNamespace(pulse=MyNamespace(amp=None, phase=population.pulse.phase), 
                                       gate=MyNamespace(amp=None, phase=population.gate.phase))

        return population_amp, population_phase
    

    def merge_population_from_amp_and_phase(self, population_amp, population_phase):
        population = MyNamespace(pulse=MyNamespace(amp=population_amp.pulse.amp, phase=population_phase.pulse.phase), 
                                 gate=MyNamespace(amp=population_amp.gate.amp, phase=population_phase.gate.phase))
        return population




    def polynomial_term(self, coefficient, order, x0, x):
        return coefficient*(x-x0)**order
    

    def polynomial_phase(self, coefficients, central_f, measurement_info):
        phase = jax.vmap(self.polynomial_term, in_axes=(0, 0, None, None))(coefficients, jnp.arange(jnp.size(coefficients))+1, central_f, measurement_info.frequency)
        phase = jnp.sum(phase, axis=0)
        return phase
    
    

    def sinusoidal_term(self, a, b, c, x):
        phase = a*jnp.sin(2*jnp.pi*b*x+c)
        return phase
    
    def sinusoidal_phase(self, coefficients, central_f, measurement_info):
        a, b, c = coefficients.a, coefficients.b, coefficients.c
        phase_arr = jax.vmap(self.sinusoidal_term, in_axes=(0, 0, 0, None))(a, b, c, measurement_info.frequency)
        phase = jnp.sum(phase_arr, axis=0)
        return phase
    

    def discrete_phase(self, coefficients, central_f, measurement_info):
        return coefficients
    


    def tanh_term(self, c, k, x):
        return 0.5*(1+jnp.tanh((x-c)/k))
    

    def tanh_phase(self, coefficients, central_f, measurement_info):
        a, c, k = coefficients.a, coefficients.c, coefficients.k
        phase_arr=jax.vmap(self.tanh_term, in_axes=(0, 0, None))(c, k, measurement_info.frequency)
        phase = jnp.sum(a[:, jnp.newaxis]*phase_arr, axis=0)
        return phase
    

    def bspline_phase(self, coefficients, central_f, measurement_info):
        phase = b_spline(coefficients.c, measurement_info.frequency)
        return phase



    def gaussian_term(self, a, b, c, frequency):
        b = b/2.355 # go from fwhm to sigma
        return a*jnp.exp(-(frequency-c)**2/(2*b**2+1e-9))
    

    def gaussian_amplitude(self, coefficients, measurement_info):
        a, b, c = coefficients.a, coefficients.b, coefficients.c
        amp_f = jax.vmap(self.gaussian_term, in_axes=(0, 0, 0, None))(a, b, c, measurement_info.frequency)
        amp_f = jnp.sum(amp_f, axis=0)
        return amp_f



    
    def lorentzian_term(self, a, b, c, frequency):
        b = b/2 # go from fwhm to sigma
        return a/((frequency-c)**2+b**2)
    

    def lorentzian_amplitude(self, coefficients, measurement_info):
        a, b, c = coefficients.a, coefficients.b, coefficients.c
        amp_f = jax.vmap(self.lorentzian_term, in_axes=(0, 0, 0, None))(a, b, c, measurement_info.frequency)
        amp_f = jnp.sum(amp_f, axis=0)
        return amp_f
    

    

    def discrete_amplitude(self, coefficients, measurement_info):
        amp_f = coefficients
        return amp_f
    

    def bspline_amplitude(self, coefficients, measurement_info):
        amp_f = b_spline(coefficients.c, measurement_info.frequency)
        return amp_f
    



    def get_phase(self, coefficients, central_f, measurement_info, descent_info):
        spectral_phase_func_dict={"polynomial": self.polynomial_phase,
                                  "sinusoidal": self.sinusoidal_phase,
                                  "sigmoidal": self.tanh_phase,
                                  "splines": self.bspline_phase,
                                  "discrete": self.discrete_phase}
        
        spectral_phase_func = spectral_phase_func_dict[descent_info.phase_type]
        return spectral_phase_func(coefficients, central_f, measurement_info)
    


    def get_amplitude(self, coefficients, measurement_info, descent_info):
        amp_func_dict={"gaussian": self.gaussian_amplitude,
                       "lorentzian": self.lorentzian_amplitude,
                       "splines": self.bspline_amplitude,
                       "discrete": self.discrete_amplitude}
            
        amp_func = amp_func_dict[descent_info.amp_type]
        amp_f = amp_func(coefficients, measurement_info)

        frequency = measurement_info.frequency
        idx_arr = jnp.arange(jnp.size(frequency))
        idx = jnp.sum(idx_arr*jnp.abs(amp_f))/jnp.sum(jnp.abs(amp_f))
        central_f = frequency[idx.astype(int)]
        return amp_f, central_f
    



    def make_pulse_f_from_individual(self, individual, measurement_info, descent_info, pulse_or_gate="pulse"):
        individual = getattr(individual, pulse_or_gate)

        if getattr(descent_info.measured_spectrum_is_provided, pulse_or_gate)==True:
            amp = getattr(measurement_info.spectral_amplitude, pulse_or_gate)
            central_f = getattr(measurement_info.central_f, pulse_or_gate)

        else:
            amp, central_f = self.get_amplitude(individual.amp, measurement_info, descent_info)
    
        phase = self.get_phase(individual.phase, central_f, measurement_info, descent_info)
        signal_f = amp*jnp.exp(1j*2*jnp.pi*phase)
        return signal_f
    


    def get_pulses_f_from_population(self, population, measurement_info, descent_info):
        make_pulse = Partial(self.make_pulse_f_from_individual, measurement_info=measurement_info, descent_info=descent_info, pulse_or_gate="pulse")
        pulse_f_arr = jax.vmap(make_pulse)(population)

        if measurement_info.doubleblind==True:
            make_gate = Partial(self.make_pulse_f_from_individual, measurement_info=measurement_info, descent_info=descent_info, pulse_or_gate="gate")
            gate_arr = jax.vmap(make_gate)(population)
        else:
            gate_arr = pulse_f_arr

        return pulse_f_arr, gate_arr
    
    

    def make_pulse_t_from_individual(self, individual, measurement_info, descent_info, pulse_or_gate="pulse"):
        signal_f = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, pulse_or_gate)
        signal = do_ifft(signal_f, measurement_info.sk, measurement_info.rn)
        return signal
    
    

    def get_pulses_t_from_population(self, population, measurement_info, descent_info):
        pulse_f_arr, gate_arr = self.get_pulses_f_from_population(population, measurement_info, descent_info)
        pulse_t_arr = do_ifft(pulse_f_arr, measurement_info.sk, measurement_info.rn)

        if measurement_info.doubleblind==True:
            gate_arr = do_ifft(gate_arr, measurement_info.sk, measurement_info.rn)
        else:
            gate_arr = pulse_t_arr

        return pulse_t_arr, gate_arr
    



    def construct_trace(self, individual, measurement_info, descent_info):
        x_arr, frequency = measurement_info.x_arr, measurement_info.frequency
        sk, rn = measurement_info.sk, measurement_info.rn
        
        signal_t = self.calculate_signal_t(individual, measurement_info.transform_arr, measurement_info)
        signal_f = do_fft(signal_t.signal_t, sk, rn)
        trace = calculate_trace(signal_f)
        return x_arr, frequency, trace
    



    def calculate_error_individual(self, individual, measurement_info, descent_info):
        measured_trace = measurement_info.measured_trace
        amplitude_or_intensity, use_fd_grad = descent_info.amplitude_or_intensity, descent_info.use_fd_grad
        error_metric = descent_info.error_metric

        x_arr, y_arr, trace = self.construct_trace(individual, measurement_info, descent_info)

        trace, measured_trace = loss_function_modifications(trace, measured_trace, x_arr, y_arr, amplitude_or_intensity, use_fd_grad)
        trace_error = error_metric(trace, measured_trace)
        return trace_error
    


    def calculate_error_population(self, population, measurement_info, descent_info):
        pulse_arr, gate_arr = self.get_pulses_from_population(population, measurement_info, descent_info)
        error_arr = jax.vmap(self.calculate_error_individual, in_axes=(0, None, None))(MyNamespace(pulse=pulse_arr, gate=gate_arr), measurement_info, descent_info)
        return error_arr
    




    def initialize_general_optimizer(self, population):
        if self.descent_info.measured_spectrum_is_provided.pulse==True:
            spectrum = self.measurement_info.spectral_amplitude.pulse
            idx = get_com(spectrum, jnp.arange(jnp.size(spectrum)))
            self.measurement_info = tree_at(lambda x: x.central_f.pulse, self.measurement_info, self.frequency[int(idx)])


        if self.descent_info.measured_spectrum_is_provided.gate==True:
            spectrum = self.measurement_info.spectral_amplitude.gate
            idx = get_com(spectrum, jnp.arange(jnp.size(spectrum)))
            self.measurement_info = tree_at(lambda x: x.central_f.gate, self.measurement_info, self.frequency[int(idx)])


        self.descent_info = self.descent_info.expand(use_fd_grad = self.use_fd_grad,
                                                     amplitude_or_intensity = self.amplitude_or_intensity,
                                                     error_metric = self.error_metric)

        self.descent_state = self.descent_state.expand(population = population)




    def get_idx_best_individual(self, descent_state):
        population = descent_state.population
        pulse_arr, gate_arr = self.get_pulses_from_population(population, self.measurement_info, self.descent_info)
        error_arr = jax.vmap(self.calculate_error_individual, in_axes=(0, None, None))(MyNamespace(pulse=pulse_arr, gate=gate_arr), 
                                                                                       self.measurement_info, self.descent_info)

        idx = jnp.argmin(error_arr)
        return idx



    def post_process_get_pulse_and_gate(self, descent_state, measurement_info, descent_info):
        idx = self.get_idx_best_individual(descent_state)

        individual = self.get_individual_from_idx(idx, descent_state.population)

        pulse_t = self.make_pulse_t_from_individual(individual, measurement_info, descent_info, "pulse")
        pulse_f = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, "pulse")

        if measurement_info.doubleblind==True:
            gate_t = self.make_pulse_t_from_individual(individual, measurement_info, descent_info, "gate")
            gate_f = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, "gate")
        else:
            gate_t, gate_f = pulse_t, pulse_f

        return pulse_t, gate_t, pulse_f, gate_f
    













class DifferentialEvolutionBASE(GeneralOptimization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "DifferentialEvolution"

        self.strategy = "best1_bin"
        self.mutation_rate = 0.5
        self.crossover_rate = 0.7
        self.selection_mechanism = "greedy"
        self.temperature = 0.1
    



    def make_bin_mask_tree(self, key, pytree, p):
        leaves, treedef = jax.tree.flatten(pytree)
        N=len(leaves)
        keys = jax.random.split(key, N)
        masks = [jax.random.choice(keys[i], jnp.array([0, 1]), jnp.shape(leaves[i]), p=jnp.array([1-p, p])) for i in range(N)]
        masks_tree = jax.tree.unflatten(treedef, masks)
        return masks_tree


    def make_exp_mask_tree(self, key, pytree, p):
        leaves, treedef = jax.tree.flatten(pytree)
        N=len(leaves)
        keys = jax.random.split(key, N)
        masks = [jax.random.choice(keys[i], jnp.tri(jnp.shape(leaves[i])[1]) , (jnp.shape(leaves[i])[0],), p=p**jnp.arange(jnp.shape(leaves[i])[1])*(1-p)) for i in range(N)]
        masks_tree = jax.tree.unflatten(treedef, masks)
        return masks_tree


    def bin_crossover(self, CR, parent_population, mutant_population, key):
        masks = self.make_bin_mask_tree(key, parent_population, CR)
        population = mutant_population*masks + parent_population*(1 + (-1)*masks)
        return population


    def exp_crossover(self, CR, parent_population, mutant_population, key):
        masks = self.make_exp_mask_tree(key, parent_population, CR)
        population = mutant_population*masks + parent_population*(1 + (-1)*masks)
        return population



    def smooth_crossover(self, CR, parent_population, mutant_population, key):
        leaves, treedef = jax.tree.flatten(parent_population)
        N=len(leaves)

        key1, key2 = jax.random.split(key, 2)
        keys1 = jax.random.split(key1, N)
        keys2 = jax.random.split(key2, N)
        
        k_vals = [jax.random.uniform(keys1[i], (jnp.shape(leaves[i])[0], 1), minval=-2, maxval=0) for i in range(N)]
        c_vals = [jax.random.choice(keys2[i], jnp.arange(jnp.shape(leaves[i])[1]), (jnp.shape(leaves[i])[0], 1), p=1/(jnp.sqrt(2*jnp.pi*CR**2))*jnp.exp(-1/(2*CR**2)*(jnp.arange(jnp.shape(leaves[i])[1])-jnp.shape(leaves[i])[1]//2)**2)) for i in range(N)]
        
        S_vals = [self.tanh_term(c_vals[i], k_vals[i], jnp.arange(jnp.shape(leaves[i])[1])) for i in range(N)]

        S_tree = jax.tree.unflatten(treedef, S_vals)

        population = mutant_population*S_tree + parent_population*(1 + (-1)*S_tree)
        return population
    


    def custom_mutation(self, F, best_individual, population, key):
        pass

    def custom_crossover(self, CR, parent_population, mutant_population, key):
        pass
    

    def do_mutation(self, mutation_strategy, F, best_individual, population, key):
        mutation_func_dict={"best1": best1,
                            "best2": best2,
                            "rand1": rand1,
                            "rand2": rand2,
                            "randtobest1": randtobest1,
                            "randtobest2": randtobest2,
                            "currenttorand1": currenttorand1,
                            "currenttorand2": currenttorand2,
                            "currenttobest1": currenttobest1,
                            "currenttobest2": currenttobest2,
                            "custom": self.custom_mutation}
        
        population = mutation_func_dict[mutation_strategy](F, best_individual, population, key)
        return population


    def do_crossover(self, crossover_strategy, CR, parent_population, mutant_population, key):
        crossover_func_dict={"bin": self.bin_crossover,
                            "exp": self.exp_crossover,
                            "smooth": self.smooth_crossover,
                            "custom": self.custom_crossover}
        
        population = crossover_func_dict[crossover_strategy](CR, parent_population, mutant_population, key)
        return population



    def select_population(self, key, error_parent, error_trial, population_parent, population_trial, descent_info):
        if descent_info.selection_mechanism=="greedy":
            trial_smaller = (jnp.sign(error_parent - error_trial)+1)//2
            leaves, treedef = jax.tree.flatten(population_parent)
            trial_smaller_leaves = [trial_smaller[:, jnp.newaxis] for _ in range(len(leaves))]
            trial_smaller_tree = jax.tree.unflatten(treedef, trial_smaller_leaves)

            error = error_trial*trial_smaller + error_parent*(1 + (-1)*trial_smaller)
            population = population_trial*trial_smaller_tree + population_parent*(1 + (-1)*trial_smaller_tree)

        elif descent_info.selection_mechanism=="global":
            N, temperature = descent_info.population_size, descent_info.temperature

            leaves_parent, treedef = jax.tree.flatten(population_parent)
            leaves_trial, treedef = jax.tree.flatten(population_trial)
            leaves_merged = [jnp.vstack((leaves_parent[i], leaves_trial[i])) for i in range(len(leaves_parent))]  # this can maybe be done with tree.map?

            error_merged = jnp.hstack((error_parent, error_trial))
            idx=jnp.argsort(error_merged)
            p_arr = 1/(jnp.exp((jnp.arange(jnp.size(idx))-N)/(temperature+1e-12))+1)
            p_arr=p_arr/jnp.sum(p_arr)
            idx_selected = jax.random.choice(key, idx, (N, ), replace=False, p=p_arr)
            error = error_merged[idx_selected]

            leaves_selected = [leaves_merged[i][idx_selected] for i in range(len(leaves_parent))]
            population = jax.tree.unflatten(treedef, leaves_selected)
            
        else:
            print("something is wrong")
        
        return population, error
    
    



    def step(self, descent_state, measurement_info, descent_info):
        mutation_strategy, mutation_rate, crossover_strategy, crossover_rate = descent_info.mutation_strategy, descent_info.mutation_rate, descent_info.crossover_strategy, descent_info.crossover_rate
        best_individual, parent_population, key = descent_state.best_individual, descent_state.population, descent_state.key

        key, subkey = jax.random.split(key, 2)
        descent_state = tree_at(lambda x: x.key, descent_state, key)
        key1, key2, key3 = jax.random.split(subkey, 3)

        mutant_population = self.do_mutation(mutation_strategy, mutation_rate, best_individual, parent_population, key1)
        trial_population = self.do_crossover(crossover_strategy, crossover_rate, parent_population, mutant_population, key2)

        trace_error_parents = self.calculate_error_population(parent_population, measurement_info, descent_info)
        trace_error_trial = self.calculate_error_population(trial_population, measurement_info, descent_info)

        population, error = self.select_population(key3, trace_error_parents, trace_error_trial, parent_population, trial_population, descent_info)

        descent_state = tree_at(lambda x: x.population, descent_state, population)
        error_mean = jnp.mean(error)
        error_min = jnp.min(error)
        error_max = jnp.max(error)

        descent_state = tree_at(lambda x: x.best_individual, descent_state, self.get_individual_from_idx(jnp.argmin(error), population))
        return descent_state, (error_mean, error_min, error_max)
    


    def initialize_run(self, population):
        self.initialize_general_optimizer(population)
        measurement_info = self.measurement_info

        mutation_strategy, crossover_strategy  = self.strategy.split("_")
        self.descent_info = self.descent_info.expand(mutation_strategy = mutation_strategy, 
                                                     crossover_strategy = crossover_strategy,
                                                     mutation_rate = self.mutation_rate,
                                                     crossover_rate = self.crossover_rate,
                                                     selection_mechanism = self.selection_mechanism,
                                                     temperature = self.temperature)
        descent_info = self.descent_info
        
        trace_error = self.calculate_error_population(population, measurement_info, descent_info)
        self.descent_state = self.descent_state.expand(best_individual = self.get_individual_from_idx(jnp.argmin(trace_error), population), 
                                                       key=self.key)
        descent_state=self.descent_state
        
        do_step=Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step=Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step




def make_key_tree(key, pytree):
    leaves, treedef = jax.tree.flatten(pytree)
    keys = jax.random.split(key, len(leaves))
    keys_tree = jax.tree.unflatten(treedef, keys)
    return keys_tree


def shuffle_pytree(key, pytree, axis):
    keys_tree = make_key_tree(key, pytree)
    pytree = jax.tree.map(Partial(jax.random.permutation, axis=axis), keys_tree, pytree)
    return pytree




def best1(F, best_individual, population, key):
    keys = jax.random.split(key, 2)
    population1, population2 = [shuffle_pytree(key, population, 0) for key in keys]
    
    population = best_individual + F*(population1 + (-1)*population2)
    return population


def best2(F, best_individual, population, key):
    keys = jax.random.split(key, 4)
    population1, population2, population3, population4 = [shuffle_pytree(key, population, 0) for key in keys]
    
    population = best_individual + F*(population1 + (-1)*population2 + population3 + (-1)*population4)
    return population



def rand1(F, best_individual, population, key):
    keys = jax.random.split(key, 3)
    population1, population2, population3 = [shuffle_pytree(key, population, 0) for key in keys]

    population = population1 + F*(population2 + (-1)*population3)
    return population


def rand2(F, best_individual, population, key):
    keys = jax.random.split(key, 5)
    population1, population2, population3, population4, population5 = [shuffle_pytree(key, population, 0) for key in keys]

    population = population1 + F*(population2 + (-1)*population3 + population4 + (-1)*population5)
    return population



def randtobest1(F, best_individual, population, key):
    keys = jax.random.split(key, 3)
    population1, population2, population3 = [shuffle_pytree(key, population, 0) for key in keys]

    population = population1 + F*(best_individual + (-1)*population + population2 + (-1)*population3)
    return population


def randtobest2(F, best_individual, population, key):
    keys = jax.random.split(key, 5)
    population1, population2, population3, population4, population5 = [shuffle_pytree(key, population, 0) for key in keys]

    population = population1 + F*(best_individual + (-1)*population + population2 + (-1)*population3 + population4 + (-1)*population5)
    return population


def currenttorand1(F, best_individual, population, key):
    keys = jax.random.split(key, 3)
    population1, population2, population3 = [shuffle_pytree(key, population, 0) for key in keys]

    population = population + F*(population1 + (-1)*population + population2 + (-1)*population3)
    return population


def currenttorand2(F, best_individual, population, key):
    keys = jax.random.split(key, 5)
    population1, population2, population3, population4, population5 = [shuffle_pytree(key, population, 0) for key in keys]

    population = population + F*(population1 + (-1)*population + population2 + (-1)*population3 + population4 + (-1)*population5)
    return population


def currenttobest1(F, best_individual, population, key):
    keys = jax.random.split(key, 2)
    population1, population2 = [shuffle_pytree(key, population, 0) for key in keys]

    population = population + F*(best_individual + (-1)*population + population1 + (-1)*population2)
    return population


def currenttobest2(F, best_individual, population, key):
    keys = jax.random.split(key, 4)
    population1, population2, population3, population4 = [shuffle_pytree(key, population, 0) for key in keys]

    population = population + F*(best_individual + (-1)*population + population1 + (-1)*population2 + population3 + (-1)*population4)
    return population

















































class EvosaxBASE(GeneralOptimization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.name = "Evosax"
        self.solver = None
        


    def step_amp_or_phase(self, population_amp_or_phase, descent_state, measurement_info, descent_info, amp_or_phase):
        solver = getattr(descent_info.solver, amp_or_phase)
        state = getattr(descent_state.evosax.state, amp_or_phase)
        params = getattr(descent_state.evosax.params, amp_or_phase)
        key = descent_state.key

        key, subkey = jax.random.split(key)
        key_ask, key_tell, key_ask_2 = jax.random.split(subkey, 3)

        population, state = solver.ask(key_ask, state, params)

        if amp_or_phase=="amp":
            population_amp, population_phase = population, population_amp_or_phase
        elif amp_or_phase=="phase":
            population_amp, population_phase = population_amp_or_phase, population

        population_eval = self.merge_population_from_amp_and_phase(population_amp, population_phase)

        fitness = self.calculate_error_population(population_eval, measurement_info, descent_info)
        state, metrics = solver.tell(key_tell, population, fitness, state, params)

        population, state = solver.ask(key_ask_2, state, params)


        descent_state = tree_at(lambda x: getattr(x.evosax.state, amp_or_phase), descent_state, state)
        descent_state = tree_at(lambda x: x.key, descent_state, key)
        return descent_state, population, fitness


        
    def step(self, descent_state, measurement_info, descent_info):
        population_amp, population_phase = self.split_population_in_amp_and_phase(descent_state.population)

        if descent_info.measured_spectrum_is_provided.pulse==False or descent_info.measured_spectrum_is_provided.gate==False:
            descent_state, population_amp, fitness = self.step_amp_or_phase(population_phase, descent_state, measurement_info, descent_info, "amp")

        descent_state, population_phase, fitness = self.step_amp_or_phase(population_amp, descent_state, measurement_info, descent_info, "phase")

        population = self.merge_population_from_amp_and_phase(population_amp, population_phase)
        descent_state = tree_at(lambda x: x.population, descent_state, population)

        errors = (jnp.mean(fitness), jnp.min(fitness), jnp.max(fitness))
        return descent_state, errors
    



    def initialize_evosax_solver(self, key, solver, population, individual, amp_or_phase):
        key, subkey = jax.random.split(key, 2)

        params = solver.default_params
        class_str=str(self.solver)
        if class_str.split(".")[2]=="population_based":
            fitness = self.calculate_error_population(population, self.measurement_info, self.descent_info)

            population_amp, population_phase = self.split_population_in_amp_and_phase(population)

            if amp_or_phase=="amp":
                population = population_amp
            elif amp_or_phase=="phase":
                population = population_phase
            else:
                print("something is wrong")

            state = solver.init(subkey, population, fitness, params)


        elif class_str.split(".")[2]=="distribution_based":
            state = solver.init(subkey, individual, params)

        else:
            print("something is wrong with", class_str)

        return state, params, key



    def initialize_run(self, population):
        self.initialize_general_optimizer(population)
        measurement_info=self.measurement_info

        population_size = self.descent_info.population_size
        individual = self.get_individual_from_idx(0, population)
        individual_amp, individual_phase = self.split_population_in_amp_and_phase(individual)

        if type(self.solver)!=tuple or type(self.solver)!=list:
            solver_amp = self.solver
            solver_phase = self.solver
        elif len(self.solver)==2:
            solver_amp, solver_phase = self.solver
        else:
            print("something is wrong")

        self.descent_info = self.descent_info.expand(solver = MyNamespace(amp=solver_amp(population_size=population_size, solution=individual_amp), 
                                                                          phase=solver_phase(population_size=population_size, solution=individual_phase)))
        descent_info=self.descent_info

        state_amp, params_amp, self.key = self.initialize_evosax_solver(self.key, descent_info.solver.amp, population, individual_amp, "amp")
        state_phase, params_phase, self.key = self.initialize_evosax_solver(self.key, descent_info.solver.phase, population, individual_phase, "phase")
        self.descent_state = self.descent_state.expand(evosax=MyNamespace(state=MyNamespace(amp=state_amp, phase=state_phase), 
                                                                          params=MyNamespace(amp=params_amp, phase=params_phase)), 
                                                       key = self.key)
        descent_state=self.descent_state
        
        do_step=Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step=Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step











class LSFBASE(GeneralOptimization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "LSF"


        self.number_of_bisection_iterations=12

        self.random_direction_mode="random"
        self.no_points_for_continuous=5

    


    def get_random_values(self, key, shape, minval, maxval, descent_info):
        mode = descent_info.random_direction_mode
        no_points_for_continuous = descent_info.no_points_for_continuous

        if mode=="random":
            values = jax.random.uniform(key, shape, minval=minval, maxval=maxval)

        elif mode=="continuous":
            x_new=jnp.linspace(0, 1, shape[0])
            N = shape[0]//no_points_for_continuous

            key1, key2 = jax.random.split(key, 2)

            x=jax.random.choice(key1, x_new, (N, ), replace=False)
            x=jnp.sort(x)
            points = jax.random.uniform(key2, (N, ), minval=minval, maxval=maxval)

            values = do_interpolation_1d(x_new, x, points)
            values = values/jnp.max(jnp.abs(values))*jnp.maximum(jnp.abs(minval), jnp.abs(maxval))

        else:
            print("something is wrong")

        return values
    


    def get_search_direction_individual(self, keys, individual, measurement_info, descent_info):
        key1, key2 = keys.pulse
        direction = MyNamespace(pulse=None, gate=None)

        pulse = individual.pulse
        shape_pulse=jnp.shape(pulse)
        d_pulse_re=self.get_random_values(key1, shape_pulse, -1, 1, descent_info)
        d_pulse_im=self.get_random_values(key2, shape_pulse, -1, 1, descent_info)
        d = d_pulse_re + 1j*d_pulse_im
        direction_pulse = d/jnp.linalg.norm(d)
        direction = tree_at(lambda x: x.pulse, direction, direction_pulse)

        if measurement_info.doubleblind==True:
            key3, key4 = keys.gate

            gate = individual.gate
            shape_gate=jnp.shape(gate)
            d_gate_re=self.get_random_values(key3, shape_gate, -1, 1, descent_info)
            d_gate_im=self.get_random_values(key4, shape_gate, -1, 1, descent_info)
            d=d_gate_re + 1j*d_gate_im
            direction_gate = d/jnp.linalg.norm(d)
            direction = tree_at(lambda x: x.gate, direction, direction_gate)

        return direction



    def get_search_direction(self, key, population, measurement_info, descent_info):
        leaves, treedef = jax.tree.flatten(population)
        keys = jax.random.split(key, len(leaves))
        keys = [jax.random.split(keys[i], jnp.shape(leaves[i])[0]*2).reshape(jnp.shape(leaves[i])[0], 2, 2) for i in range(len(leaves))]
        key_tree = jax.tree.unflatten(treedef, keys)
        return jax.vmap(self.get_search_direction_individual, in_axes=(0, 0, None, None))(key_tree, population, measurement_info, descent_info)




    def get_scalars(self, direction, signal):
        # solve jnp.abs(signal + s*direction)**2 = 1
        p = 2*jnp.real(signal*jnp.conjugate(direction))/(jnp.abs(direction)**2+1e-9)
        q = (jnp.abs(signal)**2 - 1)/(jnp.abs(direction)**2+1e-9)

        diskriminante = p**2/4 - q
        diskriminante = jnp.maximum(diskriminante, 0)
        s1 = -p/2 - jnp.sqrt(diskriminante)
        s2 = -p/2 + jnp.sqrt(diskriminante)
        return jnp.max(s1, axis=1)[:, jnp.newaxis], jnp.min(s2, axis=1)[:, jnp.newaxis]




    def bisection_step_logic_0(self, El, Em, Er, signal):
        return Em, Er

    def bisection_step_logic_1(self, El, Em, Er, signal):
        dl = jnp.linalg.norm(signal - El)
        dr = jnp.linalg.norm(signal - Er)

        x=jnp.sign(dr-dl)
        condition=(x+1)//2
        El = El*condition + Em*(1-condition)
        Er = Em*condition + Er*(1-condition)
        return El, Er

    def bisection_step_logic_2(self, El, Em, Er, signal):
        return El, Em
    

    def bisection_step(self, El, Er, population, measurement_info, descent_info, pulse_or_gate):
        Em = (El + Er)/2
        E_arr=jnp.array([El, Em, Er])

        error_arr = jax.vmap(self.calculate_error, in_axes=(0, None, None, None, None))(E_arr, population, measurement_info, descent_info, pulse_or_gate)

        idx = jnp.argmax(error_arr, axis=0)
        El, Er = jax.vmap(jax.lax.switch, in_axes=(0, None, 0, 0, 0, None))(idx, [self.bisection_step_logic_0, 
                                                                                  self.bisection_step_logic_1, 
                                                                                  self.bisection_step_logic_2], El, Em, Er, getattr(population, pulse_or_gate))
            
        return (El, Er), None
    



    def do_bisection_search(self, direction, population, measurement_info, descent_info, pulse_or_gate):
        s1, s2 = self.get_scalars(direction, getattr(population, pulse_or_gate))

        El = getattr(population, pulse_or_gate) + s1*direction
        Er = getattr(population, pulse_or_gate) + s2*direction
        
        no_iterations = descent_info.number_of_bisection_iterations
        do_bisection_step = Partial(self.bisection_step, population=population, measurement_info=measurement_info, 
                                    descent_info=descent_info, pulse_or_gate=pulse_or_gate)
        
        do_step = Partial(scan_helper, actual_function=do_bisection_step, number_of_args=2, number_of_xs=0)
        E_arr, _ = jax.lax.scan(do_step, (El, Er), length=no_iterations) 
        E_arr = jnp.array(E_arr)

        error_arr = jax.vmap(self.calculate_error, in_axes=(0, None, None, None, None))(E_arr, population, measurement_info, descent_info, pulse_or_gate)
        idx = jnp.argmin(error_arr, axis=0)
        return jax.vmap(jax.lax.switch, in_axes=(0, None, 0))(idx, [lambda x: x[0], lambda x: x[1]], jnp.transpose(E_arr, axes=(1,0,2)))




    def search_along_direction(self, direction, population, measurement_info, descent_info):
        direction_pulse=direction.pulse
        population_pulse = self.do_bisection_search(direction_pulse, population, measurement_info, descent_info, "pulse")
        population = tree_at(lambda x: x.pulse, population, population_pulse)
        
        if measurement_info.doubleblind==True:
            direction_gate=direction.gate
            population_gate = self.do_bisection_search(direction_gate, population, measurement_info, descent_info, "gate")
            population = tree_at(lambda x: x.gate, population, population_gate)

        return population
    
    


    def calculate_error(self, E_arr, population, measurement_info, descent_info, pulse_or_gate):
        if pulse_or_gate=="pulse":
            pulse_arr, gate_arr = E_arr, population.gate

        elif pulse_or_gate=="gate":
            pulse_arr, gate_arr = population.pulse, E_arr
        else:
            print("something is wrong")

        error_arr = jax.vmap(self.calculate_error_individual, 
                             in_axes=(0, None, None))(MyNamespace(pulse=pulse_arr, gate=gate_arr), measurement_info, descent_info)
        return error_arr
    




    def step(self, descent_state, measurement_info, descent_info):
        population = descent_state.population
        key, subkey = jax.random.split(descent_state.key, 2)
        descent_state = tree_at(lambda x: x.key, descent_state, key)
        
        direction = self.get_search_direction(subkey, population, measurement_info, descent_info)
        population = self.search_along_direction(direction, population, measurement_info, descent_info)

        error_arr = self.calculate_error_population(population, measurement_info, descent_info)
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        return descent_state, error_arr.reshape(-1,1)
    



    def initialize_run(self, population):
        if self.descent_info.amp_type!="discrete" or self.descent_info.phase_type!="discrete":
            print("LSF is only implemented for amp_type=discrete, phase_type=discrete and converts the populations accordingly.")

        population = self.convert_population(population, self.measurement_info, self.descent_info)
        population_pulse = jax.vmap(lambda x: x/jnp.linalg.norm(x))(population.pulse)
        population = tree_at(lambda x: x.pulse, population, population_pulse)
        self.initialize_general_optimizer(population)

        measurement_info=self.measurement_info

        
        self.descent_info = self.descent_info.expand(number_of_bisection_iterations = self.number_of_bisection_iterations,
                                                     no_points_for_continuous = self.no_points_for_continuous,
                                                     random_direction_mode = self.random_direction_mode)
        descent_info=self.descent_info


        self.descent_state = self.descent_state.expand(key = self.key)
        descent_state = self.descent_state


        do_step=Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step=Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step




    def post_process_get_pulse_and_gate(self, descent_state, measurement_info, descent_info):
        sk, rn = measurement_info.sk, measurement_info.rn
        idx = self.get_idx_best_individual(descent_state)

        individual = self.get_individual_from_idx(idx, descent_state.population)

        pulse_t = individual.pulse
        gate_t = individual.gate
        pulse_f = do_fft(pulse_t, sk, rn) 
        gate_f = do_fft(gate_t, sk, rn) 
        return pulse_t, gate_t, pulse_f, gate_f
    
















class AutoDiffBASE(GeneralOptimization):
    # Making this work for a population: (?) 
    #    vmap over solver.init and over solver.step 
    #          -> currently only equinox.filter_vmap for solver.init 
    #          -> doesnt vmap some things 
    #          -> vmap over solver.step doesnt work becasue of wrong leaf dimensions
    #   
    #    repeatedly calling optimistix.minimise does not work because of limited recursion depth for jax. 

    #        print("using jax.scipy minimize might work with vmap") -> jay.scipy is not really developed. only a prelimiary version of lbfgs

    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "AutoDiff"

        self.solver=optimistix.BFGS
        self.alternating_optimization=False

        self.optimize_individual_idx = 0


    def get_phase(self, coefficients, central_f, measurement_info, descent_info):
        phase = super().get_phase(coefficients, central_f, measurement_info, descent_info)
        if descent_info.phase_type=="polynomial" or descent_info.phase_type=="sinusoidal":
            return phase
        else:
            return jnp.cumsum(phase)*measurement_info.df



    def loss_function(self, individual, measurement_info, descent_info):
        pulse = self.make_pulse_from_individual(individual, measurement_info, descent_info, "pulse")

        if measurement_info.doubleblind==True:
            gate = self.make_pulse_from_individual(individual, measurement_info, descent_info, "gate")
        else:
            gate = pulse
            
        trace_error = self.calculate_error_individual(MyNamespace(pulse=pulse, gate=gate), measurement_info, descent_info)
        return trace_error
    


    def loss_function_amp(self, individual_amp, individual_phase, measurement_info, descent_info):
        individual = self.merge_population_from_amp_and_phase(individual_amp, individual_phase)
        trace_error = self.loss_function(individual, measurement_info, descent_info)
        return trace_error
    

    def loss_function_phase(self, individual_phase, individual_amp, measurement_info, descent_info):
        individual = self.merge_population_from_amp_and_phase(individual_amp, individual_phase)
        trace_error = self.loss_function(individual, measurement_info, descent_info)
        return trace_error
    



    def step(self, descent_state, measurement_info, descent_info):
        if descent_info.alternating_optimization==False:
            individual, optimistix_state = descent_state.individual, descent_state.optimistix_state
            individual, optimistix_state, error = self.optimistix_step(y=individual, state=optimistix_state)

            descent_state = tree_at(lambda x: x.individual, descent_state, individual)
            descent_state = tree_at(lambda x: x.optimistix_state, descent_state, optimistix_state)

        elif descent_info.alternating_optimization==True:
            individual, optimistix_state_amp, optimistix_state_phase = descent_state.individual, descent_state.optimistix_state_amp, descent_state.optimistix_state_phase

            individual_amp, individual_phase = self.split_population_in_amp_and_phase(individual)
            
            individual_amp, optimistix_state_amp, error_amp = self.optimistix_step_amp(y=individual_amp, args=individual_phase, state=optimistix_state_amp)
            individual_phase, optimistix_state_phase, error = self.optimistix_step_phase(y=individual_phase, args=individual_amp, state=optimistix_state_phase)
            
            individual = self.merge_population_from_amp_and_phase(individual_amp, individual_phase)

            descent_state = tree_at(lambda x: x.individual, descent_state, individual)
            descent_state = tree_at(lambda x: x.optimistix_state_amp, descent_state, optimistix_state_amp)
            descent_state = tree_at(lambda x: x.optimistix_state_phase, descent_state, optimistix_state_phase)
            
        else:
            print("something is wrong")

        return descent_state, error
    
    


    def initialize_alternating_optimization(self, descent_state, individual, solver, optimistix_args, measurement_info, descent_info):
        options, f_struct, aux_struct, tags = optimistix_args

        individual_amp, individual_phase = self.split_population_in_amp_and_phase(individual)

        loss_function_amp=Partial(self.loss_function_amp, measurement_info=measurement_info, descent_info=descent_info)
        loss_function_amp=Partial(optimistix_helper_loss_function, function=loss_function_amp)
        
        loss_function_phase=Partial(self.loss_function_phase, measurement_info=measurement_info, descent_info=descent_info)
        loss_function_phase=Partial(optimistix_helper_loss_function, function=loss_function_phase)


        state_amp = solver.init(loss_function_amp, individual_amp, individual_phase, options, f_struct, aux_struct, tags)
        state_phase = solver.init(loss_function_phase, individual_phase, individual_amp, options, f_struct, aux_struct, tags)
        
        self.optimistix_step_amp = Partial(solver.step, fn=loss_function_amp, options=options, tags=tags)
        self.optimistix_step_phase = Partial(solver.step, fn=loss_function_phase, options=options, tags=tags)

        descent_state = descent_state.expand(optimistix_state_amp = state_amp,
                                             optimistix_state_phase = state_phase)
        return descent_state
        


    def initialize_optimistix(self, descent_state, solver, measurement_info, descent_info):
        
        if solver.__class__.__module__.split(".")[0]=="optax":
            solver=optimistix.OptaxMinimiser(solver, rtol=1, atol=1)
        else:
            solver=solver(rtol=1, atol=1)

        loss_function = Partial(self.loss_function, measurement_info=measurement_info, descent_info=descent_info)
        loss_function = Partial(optimistix_helper_loss_function, function=loss_function)

        args = None
        options = dict()
        f_struct = jax.ShapeDtypeStruct((), jnp.float32)
        aux_struct = jax.ShapeDtypeStruct((), jnp.float32)
        tags = frozenset()

        individual = descent_state.individual
        if descent_info.alternating_optimization==False:
            state = solver.init(loss_function, individual, args, options, f_struct, aux_struct, tags)
            self.optimistix_step = Partial(solver.step, args=args, fn=loss_function, options=options, tags=tags)
            descent_state = descent_state.expand(optimistix_state = state)

        elif descent_info.alternating_optimization==True:
            optimistix_args = (options, f_struct, aux_struct, tags)
            descent_state = self.initialize_alternating_optimization(descent_state, individual, solver, optimistix_args, measurement_info, descent_info)

        else:
            print("somethong is wrong")

        descent_state, static = equinox.partition(descent_state, equinox.is_array)

        do_step = Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step = Partial(scan_helper_equinox, step=do_step, static=static)
        return descent_state, do_step

    

    
    def initialize_run(self, population):
        if self.descent_info.population_size!=1:
            print("AD will only optimize one individual in the population. This is because there are some issues when using vmap.")
            print("You can select the individual to be optimized via self.optimize_individual_idx")
            
        self.initialize_general_optimizer(population)

        measurement_info=self.measurement_info

        self.descent_info = self.descent_info.expand(alternating_optimization=self.alternating_optimization)
        descent_info=self.descent_info


        self.descent_state = self.descent_state.expand(individual = self.get_individual_from_idx(self.optimize_individual_idx, population))
        descent_state=self.descent_state

        descent_state, do_scan = self.initialize_optimistix(descent_state, self.solver, measurement_info, descent_info)
        return descent_state, do_scan
    

    

    
