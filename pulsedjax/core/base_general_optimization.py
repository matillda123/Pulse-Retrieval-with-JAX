import numpy as np

import jax.numpy as jnp
import jax
from jax.tree_util import Partial
from equinox import tree_at

import equinox
import optimistix

from pulsedjax.utilities import scan_helper, scan_helper_equinox, optimistix_helper_loss_function, MyNamespace, do_interpolation_1d
from .base_classes_algorithms import GeneralOptimizationBASE



def make_key_tree(key, pytree):
    """ For a given pytree, each leaf is replaced by a prng-key. """
    leaves, treedef = jax.tree.flatten(pytree)
    keys = jax.random.split(key, len(leaves))
    keys_tree = jax.tree.unflatten(treedef, keys)
    return keys_tree

def shuffle_pytree(key, pytree, axis):
    """ Randomizes the leafs of a given pytree along a given axis. """
    keys_tree = make_key_tree(key, pytree)
    pytree = jax.tree.map(Partial(jax.random.permutation, axis=axis), keys_tree, pytree)
    return pytree


class DifferentialEvolutionBASE(GeneralOptimizationBASE):
    """
    Implements a Differential-Evolution Algorithm. 
    Based on Qiang, J., Mitchell, C., A Unified Differential Evolution Algorithm for Global Optimization, 2014, https://www.osti.gov/servlets/purl/1163659

    Attributes:
        strategy (str): the mutation and selection strategy, analogous to scipy's differential evolution.
        mutation_rate (float): the mutation rate
        crossover_rate (float): the crossover rate
        selection_mechanism (str): the selection mechanism, can be greedy or global, defined in select_population().
        temperature (float): a temperature value for the global selection mechanism

    """

    def __init__(self, *args, strategy="best1_bin", selection_mechanism="greedy", mutation_rate=0.5, crossover_rate=0.7, **kwargs):
        super().__init__(*args, **kwargs)

        self._name = "DifferentialEvolution"

        self.strategy = strategy
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_mechanism = selection_mechanism
        self.temperature = 0.1


        
    def best1(self, F, best_individual, population, key):
        keys = jax.random.split(key, 2)
        population1, population2 = [shuffle_pytree(key, population, 0) for key in keys]
        
        population = best_individual + F*(population1 + (-1)*population2)
        return population

    def best2(self, F, best_individual, population, key):
        keys = jax.random.split(key, 4)
        population1, population2, population3, population4 = [shuffle_pytree(key, population, 0) for key in keys]
        
        population = best_individual + F*(population1 + (-1)*population2 + population3 + (-1)*population4)
        return population

    def rand1(self, F, best_individual, population, key):
        keys = jax.random.split(key, 3)
        population1, population2, population3 = [shuffle_pytree(key, population, 0) for key in keys]

        population = population1 + F*(population2 + (-1)*population3)
        return population

    def rand2(self, F, best_individual, population, key):
        keys = jax.random.split(key, 5)
        population1, population2, population3, population4, population5 = [shuffle_pytree(key, population, 0) for key in keys]

        population = population1 + F*(population2 + (-1)*population3 + population4 + (-1)*population5)
        return population

    def randtobest1(self, F, best_individual, population, key):
        keys = jax.random.split(key, 3)
        population1, population2, population3 = [shuffle_pytree(key, population, 0) for key in keys]

        population = population1 + F*(best_individual + (-1)*population + population2 + (-1)*population3)
        return population

    def randtobest2(self, F, best_individual, population, key):
        keys = jax.random.split(key, 5)
        population1, population2, population3, population4, population5 = [shuffle_pytree(key, population, 0) for key in keys]

        population = population1 + F*(best_individual + (-1)*population + population2 + (-1)*population3 + population4 + (-1)*population5)
        return population

    def currenttorand1(self, F, best_individual, population, key):
        keys = jax.random.split(key, 3)
        population1, population2, population3 = [shuffle_pytree(key, population, 0) for key in keys]

        population = population + F*(population1 + (-1)*population + population2 + (-1)*population3)
        return population

    def currenttorand2(self, F, best_individual, population, key):
        keys = jax.random.split(key, 5)
        population1, population2, population3, population4, population5 = [shuffle_pytree(key, population, 0) for key in keys]

        population = population + F*(population1 + (-1)*population + population2 + (-1)*population3 + population4 + (-1)*population5)
        return population

    def currenttobest1(self, F, best_individual, population, key):
        keys = jax.random.split(key, 2)
        population1, population2 = [shuffle_pytree(key, population, 0) for key in keys]

        population = population + F*(best_individual + (-1)*population + population1 + (-1)*population2)
        return population

    def currenttobest2(self, F, best_individual, population, key):
        keys = jax.random.split(key, 4)
        population1, population2, population3, population4 = [shuffle_pytree(key, population, 0) for key in keys]

        population = population + F*(best_individual + (-1)*population + population1 + (-1)*population2 + population3 + (-1)*population4)
        return population


    

    def make_bin_mask_tree(self, key, pytree, p):
        """ For a given pytree a random binary mask is generated on each leaf, via the probabilty p. """
        leaves, treedef = jax.tree.flatten(pytree)
        N = len(leaves)
        keys = jax.random.split(key, N)
        masks = [jax.random.choice(keys[i], jnp.array([0, 1]), jnp.shape(leaves[i]), p=jnp.array([1-p, p])) for i in range(N)]
        masks_tree = jax.tree.unflatten(treedef, masks)
        return masks_tree


    def make_exp_mask_tree(self, key, pytree, p):
        """ For a given pytree a mask (e.g. of the form 111100000) is generated via the probability p on each row of each leaf. """
        leaves, treedef = jax.tree.flatten(pytree)
        N = len(leaves)
        keys = jax.random.split(key, N)
        masks = [jax.random.choice(keys[i], 
                                   jnp.tri(jnp.shape(leaves[i])[1]) , 
                                   (jnp.shape(leaves[i])[0],), 
                                   p=p**jnp.arange(jnp.shape(leaves[i])[1])*(1-p)
                                   ) for i in range(N)]
        masks_tree = jax.tree.unflatten(treedef, masks)
        return masks_tree


    def bin_crossover(self, CR, parent_population, mutant_population, key):
        """ Peforms a binary crossover between two populations via the crossover rate CR. """
        masks = self.make_bin_mask_tree(key, parent_population, CR)
        population = mutant_population*masks + parent_population*(1 + (-1)*masks)
        return population


    def exp_crossover(self, CR, parent_population, mutant_population, key):
        """ Peforms an exponential crossover between two populations via the crossover rate CR. """
        masks = self.make_exp_mask_tree(key, parent_population, CR)
        population = mutant_population*masks + parent_population*(1 + (-1)*masks)
        return population



    def smooth_crossover(self, CR, parent_population, mutant_population, key):
        """ Similar to an exponential crossover but with a smooth nonbinary transition, which is fascilitated via a tanh function. """
        leaves, treedef = jax.tree.flatten(parent_population)
        N = len(leaves)

        key1, key2 = jax.random.split(key, 2)
        keys1 = jax.random.split(key1, N)
        keys2 = jax.random.split(key2, N)
        
        k_vals = [jax.random.uniform(keys1[i], (jnp.shape(leaves[i])[0], 1), minval=-2, maxval=0) for i in range(N)]
        c_vals = [jax.random.choice(keys2[i], 
                                    jnp.arange(jnp.shape(leaves[i])[1]), 
                                    (jnp.shape(leaves[i])[0], 1), 
                                    p = 1/(jnp.sqrt(2*jnp.pi*CR**2))*jnp.exp(-1/(2*CR**2)*(jnp.arange(jnp.shape(leaves[i])[1])-jnp.shape(leaves[i])[1]//2)**2)
                                    ) for i in range(N)]
        S_vals = [self.tanh_term(c_vals[i], k_vals[i], jnp.arange(jnp.shape(leaves[i])[1])) for i in range(N)]
        S_tree = jax.tree.unflatten(treedef, S_vals)
        population = mutant_population*S_tree + parent_population*(1 + (-1)*S_tree)
        return population
    


    def custom_mutation(self, F, best_individual, population, key):
        """ A placeholder to allow the introduction of custom mutation approaches. """
        pass

    def custom_crossover(self, CR, parent_population, mutant_population, key):
        """ A placeholder to allow the introduction of custom crossover approaches. """
        pass
    
    def do_mutation(self, mutation_strategy, F, best_individual, population, key):
        """ Creates and applies random mutations to a population. """
        mutation_func_dict={"best1": self.best1,
                            "best2": self.best2,
                            "rand1": self.rand1,
                            "rand2": self.rand2,
                            "randtobest1": self.randtobest1,
                            "randtobest2": self.randtobest2,
                            "currenttorand1": self.currenttorand1,
                            "currenttorand2": self.currenttorand2,
                            "currenttobest1": self.currenttobest1,
                            "currenttobest2": self.currenttobest2,
                            "custom": self.custom_mutation}
        
        population = mutation_func_dict[mutation_strategy](F, best_individual, population, key)
        return population


    def do_crossover(self, crossover_strategy, CR, parent_population, mutant_population, key):
        """ Creates and applies random crossover events between two populations. """
        crossover_func_dict={"bin": self.bin_crossover,
                            "exp": self.exp_crossover,
                            "smooth": self.smooth_crossover,
                            "custom": self.custom_crossover}
        
        population = crossover_func_dict[crossover_strategy](CR, parent_population, mutant_population, key)
        return population



    def select_population(self, key, selection_mechanism, error_parent, error_trial, population_parent, population_trial, descent_info):
        """
        Performs the evolutionary selection process for two populations. Currently selection_mechanism can be greedy or global.

        greedy: implements a pairwise comparison between individuals of the two population, where always the individual with the higher fitness is selected.
        global: implements a comparison/ranking between all individuals of both populations. The next generation is selected via randomized sampling based on a Fermi-Distribution. A "temperature" allows tuning of this selection process. 
        
        """

        if selection_mechanism=="greedy":
            trial_smaller = (jnp.sign(error_parent - error_trial)+1)//2
            leaves, treedef = jax.tree.flatten(population_parent)
            trial_smaller_leaves = [trial_smaller[:, jnp.newaxis] for _ in range(len(leaves))]
            trial_smaller_tree = jax.tree.unflatten(treedef, trial_smaller_leaves)

            error = error_trial*trial_smaller + error_parent*(1 + (-1)*trial_smaller)
            population = population_trial*trial_smaller_tree + population_parent*(1 + (-1)*trial_smaller_tree)

        elif selection_mechanism=="global":
            N, temperature = descent_info.population_size, descent_info.temperature

            leaves_parent, treedef = jax.tree.flatten(population_parent)
            leaves_trial, treedef = jax.tree.flatten(population_trial)
            leaves_merged = [jnp.vstack((leaves_parent[i], leaves_trial[i])) for i in range(len(leaves_parent))]  # this can maybe be done with tree.map?

            error_merged = jnp.hstack((error_parent, error_trial))
            idx = jnp.argsort(error_merged)
            p_arr = 1/(jnp.exp((jnp.arange(jnp.size(idx))-N)/(temperature+1e-12))+1)
            p_arr = p_arr/jnp.sum(p_arr)
            idx_selected = jax.random.choice(key, idx, (N, ), replace=False, p=p_arr)
            error = error_merged[idx_selected]

            leaves_selected = [leaves_merged[i][idx_selected] for i in range(len(leaves_parent))]
            population = jax.tree.unflatten(treedef, leaves_selected)
            
        else:
            raise NotImplementedError(f"Only greedy and global are available as selection_mechanism. Not {selection_mechanism}")
        
        return population, error
    
    



    def step(self, descent_state, measurement_info, descent_info):
        """ 
        Performs one step/generation of the Differential Evolution Algorithm. 

        Args:
            descent_state: Pytree, 
            measurement_info: Pytree,
            descent_info: Pytree,

        Returns:
            tuple[Pytree, jnp.array], the updated descent state, the mean, minimum and maximum error of the current population

        """

        mutation_strategy, mutation_rate, crossover_strategy, crossover_rate = descent_info.mutation_strategy, descent_info.mutation_rate, descent_info.crossover_strategy, descent_info.crossover_rate
        selection_mechanism = descent_info.selection_mechanism
        best_individual, parent_population, key = descent_state.best_individual, descent_state.population, descent_state.key

        key, subkey = jax.random.split(key, 2)
        descent_state = tree_at(lambda x: x.key, descent_state, key)
        key1, key2, key3 = jax.random.split(subkey, 3)

        mutant_population = self.do_mutation(mutation_strategy, mutation_rate, best_individual, parent_population, key1)
        trial_population = self.do_crossover(crossover_strategy, crossover_rate, parent_population, mutant_population, key2)

        trace_error_parents = self.calculate_error_population(parent_population, measurement_info, descent_info)
        trace_error_trial = self.calculate_error_population(trial_population, measurement_info, descent_info)

        population, error = self.select_population(key3, selection_mechanism, trace_error_parents, trace_error_trial, parent_population, trial_population, descent_info)

        descent_state = tree_at(lambda x: x.population, descent_state, population)
        error_mean = jnp.mean(error)
        error_min = jnp.min(error)
        error_max = jnp.max(error)

        descent_state = tree_at(lambda x: x.best_individual, descent_state, self.get_individual_from_idx(jnp.argmin(error), population))
        return descent_state, jnp.array([error_mean, error_min, error_max])
    


    def initialize_run(self, population):
        """
        Prepares all provided data and parameters for the reconstruction. 
        Here the final shape/structure of descent_state, measurement_info and descent_info are determined. 

        Args:
            population: Pytree, the initial guess as created by self.create_initial_population()
        
        Returns:
            tuple[Pytree, Callable], the initial descent state, the step-function of the algorithm.

        """

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
        descent_state = self.descent_state
        
        do_step = Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step = Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step













class EvosaxBASE(GeneralOptimizationBASE):
    """
    Employs the evosax package to perform the optimization.

    Robert Tjarko Lange, evosax: JAX-based Evolution Strategies, arXiv preprint arXiv:2212.04180 (2022)

    Attributes:
        solver (evosax-solver): any evosax-solver should work
        solver_params (any): user defined parameters for the evosax-solver, if None the default params set in evosax are used
        solver_kwargs (dict): some evosax-solver require additional input arguments. These can be supplied via this aattribute.
        
    """
    def __init__(self, *args, solver=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._name = "Evosax"
        self.solver = solver
        self.solver_params = None
        self.solver_kwargs = {}
        


    def step_amp_or_phase(self, population_amp_or_phase, descent_state, measurement_info, descent_info, amp_or_phase):
        """ 
        Evosax does not respect the structure of a pytree. Thus when optimizing parameters with vastly different properties simultaneously
        one should expose them seperately to evosax in order to avoid mixing. Thus the step function is applied once to update the amplitudes 
        and once to update the phases.
        """

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
        state, _ = solver.tell(key_tell, population, fitness, state, params)
        population, state = solver.ask(key_ask_2, state, params)

        descent_state = tree_at(lambda x: getattr(x.evosax.state, amp_or_phase), descent_state, state)
        descent_state = tree_at(lambda x: x.key, descent_state, key)
        return descent_state, population, fitness


        
    def step(self, descent_state, measurement_info, descent_info):
        """
        Performs one optimization step through an evolutionary algorithm in evosax.

        Args:
            descent_state: Pytree, 
            measurement_info: Pytree,
            descent_info: Pytree,

        Returns:
            tuple[Pytree, jnp.array], the updated descent state, the mean, minimum and maximum error of the current population

        """

        population_amp, population_phase = self.split_population_in_amp_and_phase(descent_state.population)

        if descent_info.measured_spectrum_is_provided.pulse==False or (descent_info.measured_spectrum_is_provided.gate==False and measurement_info.doubleblind==True):
            descent_state, population_amp, fitness = self.step_amp_or_phase(population_phase, descent_state, measurement_info, descent_info, "amp")

        descent_state, population_phase, fitness = self.step_amp_or_phase(population_amp, descent_state, measurement_info, descent_info, "phase")

        population = self.merge_population_from_amp_and_phase(population_amp, population_phase)
        descent_state = tree_at(lambda x: x.population, descent_state, population)

        errors = jnp.array([jnp.mean(fitness), jnp.min(fitness), jnp.max(fitness)])
        return descent_state, errors
    



    def initialize_evosax_solver(self, key, solver, population, individual, amp_or_phase):
        """ 
        Initializes the provided evosax solver. 
        The solvers need to be exposed to the entire population, one individual or the initial fitnesses.
        Since the amplitude and phase are optimized separetly a solver needs to be initialized once for each. (This done by calling this method twice.)
        """

        key, subkey = jax.random.split(key, 2)

        if self.solver_params==None:
            params = solver.default_params
        else:
            params = self.solver_params

        class_str = str(self.solver)
        if class_str.split(".")[2]=="population_based":
            fitness = self.calculate_error_population(population, self.measurement_info, self.descent_info)

            population_amp, population_phase = self.split_population_in_amp_and_phase(population)

            if amp_or_phase=="amp":
                population = population_amp
            elif amp_or_phase=="phase":
                population = population_phase
            else:
                raise ValueError(f"amp_or_phase needs to be amp or phase. Not {amp_or_phase}")

            state = solver.init(subkey, population, fitness, params)


        elif class_str.split(".")[2]=="distribution_based":
            state = solver.init(subkey, individual, params)

        else:
            raise ValueError(f"Something went wrong in the detection of evosax's submodule. Needs to be \
                             population_based or distribution_based. Found {class_str}")

        return state, params, key



    def initialize_run(self, population):
        """
        Prepares all provided data and parameters for the reconstruction. Initializes the evosax-solver appropriately.
        Here the final shape/structure of descent_state, measurement_info and descent_info are determined. 

        Args:
            population: Pytree, the initial guess as created by self.create_initial_population()
        
        Returns:
            tuple[Pytree, Callable], the initial descent state, the step-function of the algorithm.

        """

        self.initialize_general_optimizer(population)
        measurement_info=self.measurement_info

        population_size = self.descent_info.population_size
        individual = self.get_individual_from_idx(0, population)
        individual = jax.tree.map(lambda x: x[0], individual)
        individual_amp, individual_phase = self.split_population_in_amp_and_phase(individual)

        solver_amp, solver_phase = self.solver, self.solver
        solver_kwargs = self.solver_kwargs
        self.descent_info = self.descent_info.expand(solver = MyNamespace(amp=solver_amp(population_size=population_size, solution=individual_amp, **solver_kwargs), 
                                                                          phase=solver_phase(population_size=population_size, solution=individual_phase, **solver_kwargs)))
        descent_info=self.descent_info

        if descent_info.measured_spectrum_is_provided.pulse==False or (descent_info.measured_spectrum_is_provided.gate==False and measurement_info.doubleblind==True):
            state_amp, params_amp, self.key = self.initialize_evosax_solver(self.key, descent_info.solver.amp, population, individual_amp, "amp")
        else:
            state_amp, params_amp = None, None

        state_phase, params_phase, self.key = self.initialize_evosax_solver(self.key, descent_info.solver.phase, population, individual_phase, "phase")
        self.descent_state = self.descent_state.expand(evosax=MyNamespace(state=MyNamespace(amp=state_amp, phase=state_phase), 
                                                                          params=MyNamespace(amp=params_amp, phase=params_phase)), 
                                                       key = self.key)
        descent_state = self.descent_state
        
        do_step=Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step=Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step














class LSFBASE(GeneralOptimizationBASE):
    # maybe there is an issue here. 
    # In one iteration all bisection steps are done first on the pulse and only then on the gate. Maybe this should be alternating?

    """
    Implements a version of the Linesearch FROG Algorithm (LSF). Despite its name the algorithm is NOT restricted to FROG. 

    The algorithm is not implemented for the optimization of parametrized populations. Instead a population is always evaluated on the time/frequency axis
    and thus optimized in its discretized form.


    C. O. Krook and V. Pasiskevicius, Opt. Express 33, 33258-33269 (2025) 

    Attributes:
        number_of_bisection_iterations (int): as the name says
        random_direction_mode (str): can be random or continuous
        ratio_points_for_continuous (int): smaller value means more randomness/eratic

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._name = "LSF"

        self.number_of_bisection_iterations = 12

        self.random_direction_mode = "random"
        self.ratio_points_for_continuous = 0.25

        self.boundary = 1 

    

    def get_random_values(self, key, shape, minval, maxval, descent_info):
        """ LSF requires random directions. These are produced here. """
        mode = descent_info.random_direction_mode
        ratio_points_for_continuous = descent_info.ratio_points_for_continuous

        if mode=="random":
            values = jax.random.uniform(key, shape, minval=minval, maxval=maxval)

        elif mode=="continuous":
            x_new = jnp.linspace(0, 1, shape[0])
            N = int(ratio_points_for_continuous*shape[0])

            key1, key2 = jax.random.split(key, 2)
            x = jnp.sort(jax.random.choice(key1, x_new, (N, ), replace=False))
            points = jax.random.uniform(key2, (N, ), minval=minval, maxval=maxval)

            values = do_interpolation_1d(x_new, x, points)
            values = values/jnp.max(jnp.abs(values))*jnp.maximum(jnp.abs(minval), jnp.abs(maxval))
        else:
            raise NotImplementedError(f"random_direction_mode needs to be random or continuous. Not {mode}")

        return values
    


    def get_search_direction_individual(self, keys, individual, measurement_info, descent_info):
        """ Creates a pytree with random search directions for one individual. """
        key1, key2 = keys.pulse
        direction = MyNamespace(pulse=None, gate=None)

        pulse = individual.pulse
        shape_pulse = jnp.shape(pulse)
        d_pulse_re = self.get_random_values(key1, shape_pulse, -1, 1, descent_info)
        d_pulse_im = self.get_random_values(key2, shape_pulse, -1, 1, descent_info)
        d = d_pulse_re + 1j*d_pulse_im
        direction_pulse = d/jnp.linalg.norm(d)
        direction = tree_at(lambda x: x.pulse, direction, direction_pulse, is_leaf=lambda x: x is None)

        if measurement_info.doubleblind==True:
            key3, key4 = keys.gate

            gate = individual.gate
            shape_gate = jnp.shape(gate)
            d_gate_re = self.get_random_values(key3, shape_gate, -1, 1, descent_info)
            d_gate_im = self.get_random_values(key4, shape_gate, -1, 1, descent_info)
            d = d_gate_re + 1j*d_gate_im
            direction_gate = d/jnp.linalg.norm(d)
            direction = tree_at(lambda x: x.gate, direction, direction_gate, is_leaf=lambda x: x is None)

        return direction



    def get_search_direction(self, key, population, measurement_info, descent_info):
        """ Creates a pytree with random search directions for an entire population. """
        leaves, treedef = jax.tree.flatten(population)
        keys = jax.random.split(key, len(leaves))
        keys = [jax.random.split(keys[i], jnp.shape(leaves[i])[0]*2).reshape(jnp.shape(leaves[i])[0], 2, 2) for i in range(len(leaves))]
        key_tree = jax.tree.unflatten(treedef, keys)
        return jax.vmap(self.get_search_direction_individual, in_axes=(0, 0, None, None))(key_tree, population, measurement_info, descent_info)




    def get_scalars(self, direction, signal, descent_info):
        """ Calculates scalars to identify the min/max of a search direction. """
        # solve jnp.abs(signal + s*direction)**2 = jnp.abs(boundary)**2
        p = 2*jnp.real(signal*jnp.conjugate(direction))/(jnp.abs(direction)**2+1e-9)
        q = (jnp.abs(signal)**2 - jnp.abs(descent_info.boundary)**2)/(jnp.abs(direction)**2+1e-9)

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
        """ Does one bisection step of the LSF algorithm. """

        Em = (El + Er)/2
        E_arr=jnp.array([El, Em, Er])
        
        error_arr = jax.vmap(self.calculate_error, in_axes=(0, None, None, None, None))(E_arr, population, measurement_info, descent_info, pulse_or_gate)
        idx = jnp.argmax(error_arr, axis=0)

        El, Er = jax.vmap(jax.lax.switch, in_axes=(0, None, 0, 0, 0, None))(idx, [self.bisection_step_logic_0, 
                                                                                  self.bisection_step_logic_1, 
                                                                                  self.bisection_step_logic_2], El, Em, Er, getattr(population, pulse_or_gate))
        return (El, Er), None
    



    def do_bisection_search(self, direction, population, measurement_info, descent_info, pulse_or_gate):
        """ 
        Performs one bisection search to find the minimum along a given search direction. 
        The number of iterations is set through self.number_of_bisection_iterations
        """
        s1, s2 = self.get_scalars(direction, getattr(population, pulse_or_gate), descent_info)

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
        """ Performs a bisection search along one direction for pulse and the for gate. """
        direction_pulse = direction.pulse
        population_pulse = self.do_bisection_search(direction_pulse, population, measurement_info, descent_info, "pulse")
        population = tree_at(lambda x: x.pulse, population, population_pulse)
        
        if measurement_info.doubleblind==True:
            direction_gate=direction.gate
            population_gate = self.do_bisection_search(direction_gate, population, measurement_info, descent_info, "gate")
            population = tree_at(lambda x: x.gate, population, population_gate)

        return population
    
    

    def make_population_bisection_search(self, E_arr, population, measurement_info, descent_info, pulse_or_gate):
        if pulse_or_gate=="pulse":
            pulse_arr, gate_arr = E_arr, population.gate

        elif pulse_or_gate=="gate":
            pulse_arr, gate_arr = population.pulse, E_arr
        else:
            raise ValueError(f"pulse_or_gate needs to be pulse or gate. Not {pulse_or_gate}")
        return MyNamespace(pulse=pulse_arr, gate=gate_arr)


    def calculate_error(self, E_arr, population, measurement_info, descent_info, pulse_or_gate):
        """ 
        Calculates the trace error. Since pulse and gate are optimized independently the population as provided to calculate_error_individual() 
        needs to be constructed from the current population and the fields in the cureent optimization.
        """
        population = self.make_population_bisection_search(E_arr, population, measurement_info, descent_info, pulse_or_gate)
        error_arr = jax.vmap(self.calculate_error_individual, 
                             in_axes=(0, None, None))(population, measurement_info, descent_info)
        return error_arr
    




    def step(self, descent_state, measurement_info, descent_info):
        """
        Performs one step of the LSF-algorithm.

        Args:
            descent_state: Pytree, 
            measurement_info: Pytree,
            descent_info: Pytree,

        Returns:
            tuple[Pytree, jnp.array], the updated descent state, errors of the current population

        """

        population = descent_state.population
        key, subkey = jax.random.split(descent_state.key, 2)
        descent_state = tree_at(lambda x: x.key, descent_state, key)
        
        direction = self.get_search_direction(subkey, population, measurement_info, descent_info)
        population = self.search_along_direction(direction, population, measurement_info, descent_info)

        error_arr = self.calculate_error_population(population, measurement_info, descent_info)
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        return descent_state, error_arr.reshape(-1,1)
    



    def initialize_run(self, population):
        """
        Prepares all provided data and parameters for the reconstruction.
        Here the final shape/structure of descent_state, measurement_info and descent_info are determined. 

        Args:
            population: Pytree, the initial guess as created by self.create_initial_population()
        
        Returns:
            tuple[Pytree, Callable], the initial descent state, the step-function of the algorithm.

        """

        if self.descent_info.amp_type!="continuous" or self.descent_info.phase_type!="continuous":
            print("LSF is only implemented for amp_type=discrete, phase_type=discrete and converts the populations accordingly.")

        self.initialize_general_optimizer(population)
        population = self.convert_population(population, self.measurement_info, self.descent_info)
        population_pulse = jax.vmap(lambda x: x/jnp.linalg.norm(x))(population.pulse)
        population = tree_at(lambda x: x.pulse, population, population_pulse)

        assert (0 < self.ratio_points_for_continuous < 1) | (self.random_direction_mode!="continuous")

        measurement_info = self.measurement_info
        self.descent_info = self.descent_info.expand(number_of_bisection_iterations = self.number_of_bisection_iterations,
                                                     ratio_points_for_continuous = self.ratio_points_for_continuous,
                                                     random_direction_mode = self.random_direction_mode,
                                                     boundary = self.boundary)
        descent_info = self.descent_info

        self.descent_state = tree_at(lambda x: x.population, self.descent_state, population)
        self.descent_state = self.descent_state.expand(key = self.key)
        descent_state = self.descent_state

        do_step = Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step = Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step





    def post_process_get_pulse_and_gate(self, descent_state, measurement_info, descent_info, idx=None):
        """ Some custom post-processing since the optimization is done in the explicit discretized form. """
        sk, rn = measurement_info.sk, measurement_info.rn
        
        if idx==None:
            idx = self.get_idx_best_individual(descent_state)

        individual = self.get_individual_from_idx(idx, descent_state.population)

        pulse_t = individual.pulse
        gate_t = individual.gate
        pulse_f = self.fft(pulse_t, sk, rn) 
        gate_f = self.fft(gate_t, sk, rn) 
        return pulse_t, gate_t, pulse_f, gate_f
    












class AutoDiffBASE(GeneralOptimizationBASE):
    """
    Employs the optimistix package to perform the optimization via Automatic-Differentiation.

    J. Rader, T. Lyons and P.Kidger, Optimistix: modular optimisation in JAX and Equinox, arXiv:2402.09983 (2024)
    DeepMind et al., The DeepMind JAX Ecosystem, http://github.com/google-deepmind (2020)

    Attributes:
        solver (optimistix-solver, optax-solver): solvers need to be initialized
        alternating_optimization (bool): if true, the optimizer alternates between amplitude and phase
        
    """

    
    def __init__(self, *args, solver=None, **kwargs):
        super().__init__(*args, **kwargs)

        self._name = "AutoDiff"

        self.solver = solver
        self.alternating_optimization = False


    def get_phase(self, coefficients, central_f, measurement_info, descent_info):
        """ Wraps around GeneralOptimizationBASE.get_phase() in order to fascilliate the optimization of the group delay. """
        phase = super().get_phase(coefficients, central_f, measurement_info, descent_info)
        if descent_info.phase_type=="polynomial" or descent_info.phase_type=="sinusoidal":
            return phase
        else:
            return jnp.cumsum(phase)*measurement_info.df



    def loss_function(self, individual, measurement_info, descent_info):
        """ Wraps around self.calculate_error_individual() to return the error of the current guess. """
        pulse = self.make_pulse_from_individual(individual, measurement_info, descent_info, "pulse")

        if measurement_info.doubleblind==True:
            gate = self.make_pulse_from_individual(individual, measurement_info, descent_info, "gate")
        else:
            gate = pulse
            
        trace_error = self.calculate_error_individual(MyNamespace(pulse=pulse, gate=gate), measurement_info, descent_info)
        return trace_error
    


    def loss_function_amp(self, individual_amp, individual_phase, measurement_info, descent_info):
        """ Helper to fascilliate optimization with respect to the amplitude only. """
        individual = self.merge_population_from_amp_and_phase(individual_amp, individual_phase)
        trace_error = self.loss_function(individual, measurement_info, descent_info)
        return trace_error
    

    def loss_function_phase(self, individual_phase, individual_amp, measurement_info, descent_info):
        """ Helper to fascilliate optimization with respect to the phase only. """
        individual = self.merge_population_from_amp_and_phase(individual_amp, individual_phase)
        trace_error = self.loss_function(individual, measurement_info, descent_info)
        return trace_error
    



    def step(self, descent_state, measurement_info, descent_info):
        """
        Performs one optimization step.

        Args:
            descent_state: Pytree, 
            measurement_info: Pytree,
            descent_info: Pytree,

        Returns:
            tuple[Pytree, jnp.array], the updated descent state, errors of the current population

        """

        if descent_info.alternating_optimization==False:
            population, optimistix_state = descent_state.population, descent_state.optimistix_state
            population, optimistix_state, error = self.optimistix_step(population, optimistix_state)

            descent_state = tree_at(lambda x: x.population, descent_state, population)
            descent_state = tree_at(lambda x: x.optimistix_state, descent_state, optimistix_state)

        elif descent_info.alternating_optimization==True:
            population, optimistix_state_amp, optimistix_state_phase = descent_state.population, descent_state.optimistix_state_amp, descent_state.optimistix_state_phase

            population_amp, population_phase = self.split_population_in_amp_and_phase(population)
            
            population_amp, optimistix_state_amp, error_amp = self.optimistix_step_amp(population_amp, population_phase, optimistix_state_amp)
            population_phase, optimistix_state_phase, error = self.optimistix_step_phase(population_phase, population_amp, optimistix_state_phase)
            
            population = self.merge_population_from_amp_and_phase(population_amp, population_phase)

            descent_state = tree_at(lambda x: x.population, descent_state, population)
            descent_state = tree_at(lambda x: x.optimistix_state_amp, descent_state, optimistix_state_amp)
            descent_state = tree_at(lambda x: x.optimistix_state_phase, descent_state, optimistix_state_phase)
            
        else:
            raise ValueError(f"alternating_optimization needs to be True/False. Not {descent_info.alternating_optimization}")

        return descent_state, error
    
    


    def initialize_alternating_optimization(self, descent_state, population, solver, optimistix_args, measurement_info, descent_info):
        """ If the amplitude and phase are supposed to be optimized in an alternating fashion instead of simultaneously, different methods have to be initialized. """
        options, f_struct, aux_struct, tags = optimistix_args

        population_amp, population_phase = self.split_population_in_amp_and_phase(population)
        
        loss_function_amp = Partial(self.loss_function_amp, measurement_info=measurement_info, descent_info=descent_info)
        loss_function_amp = Partial(optimistix_helper_loss_function, function=loss_function_amp, no_of_args=1)

        loss_function_phase = Partial(self.loss_function_phase, measurement_info=measurement_info, descent_info=descent_info)
        loss_function_phase = Partial(optimistix_helper_loss_function, function=loss_function_phase, no_of_args=1)
        

        @equinox.filter_vmap
        def solver_init_amp(individual_amp, individual_phase):
            return solver.init(loss_function_amp, individual_amp, individual_phase, options, f_struct, aux_struct, tags)
        
        @equinox.filter_vmap
        def solver_init_phase(individual_phase, individual_amp):
            return solver.init(loss_function_phase, individual_phase, individual_amp, options, f_struct, aux_struct, tags)
        

        @equinox.filter_vmap
        def solver_step_amp(individual_amp, individual_phase, state_amp):
            return solver.step(loss_function_amp, individual_amp, individual_phase, options, state_amp, tags)
        
        @equinox.filter_vmap
        def solver_step_phase(individual_phase, individual_amp, state_phase):
            return solver.step(loss_function_phase, individual_phase, individual_amp, options, state_phase, tags)
            

        state_amp = solver_init_amp(population_amp, population_phase)
        state_phase = solver_init_phase(population_phase, population_amp)
        
        self.optimistix_step_amp = solver_step_amp
        self.optimistix_step_phase = solver_step_phase

        descent_state = descent_state.expand(optimistix_state_amp = state_amp,
                                             optimistix_state_phase = state_phase)
        return descent_state
        


    def initialize_optimistix(self, descent_state, solver, measurement_info, descent_info):
        """
        Initializes the optimistix-solver.
        """
        
        if solver.__class__.__module__.split(".")[0]=="optax":
            solver=optimistix.OptaxMinimiser(solver, rtol=1, atol=1)
        # else:
        #     solver=solver(rtol=1, atol=1)


        args = None
        options = dict(jac="bwd")
        f_struct = jax.ShapeDtypeStruct((), jnp.float32)
        aux_struct = jax.ShapeDtypeStruct((), jnp.float32)
        tags = frozenset()

        population = descent_state.population
        if descent_info.alternating_optimization==False:

            loss_function = Partial(self.loss_function, measurement_info=measurement_info, descent_info=descent_info)
            loss_function = Partial(optimistix_helper_loss_function, function=loss_function, no_of_args=0)

            @equinox.filter_vmap
            def solver_init(individual):
                return solver.init(loss_function, individual, args, options, f_struct, aux_struct, tags)
            
            @equinox.filter_vmap
            def solver_step(individual, state):
                return solver.step(loss_function, individual, args, options, state, tags)

            state = solver_init(population)
            descent_state = descent_state.expand(optimistix_state = state)
            self.optimistix_step = solver_step

        elif descent_info.alternating_optimization==True:
            optimistix_args = (options, f_struct, aux_struct, tags)
            descent_state = self.initialize_alternating_optimization(descent_state, population, solver, optimistix_args, measurement_info, descent_info)

        else:
            raise ValueError(f"alternating_optimization needs to be True/False. Not {descent_info.alternating_optimization}")
            

        descent_state, static = equinox.partition(descent_state, equinox.is_array)

        do_step = Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)        
        do_step = Partial(scan_helper_equinox, step=do_step, static=static)
        return descent_state, do_step

    

    
    def initialize_run(self, population):
        """
        Prepares all provided data and parameters for the reconstruction.
        Here the final shape/structure of descent_state, measurement_info and descent_info are determined. 

        Args:
            population: Pytree, the initial guess as created by self.create_initial_population()
        
        Returns:
            tuple[Pytree, Callable], the initial descent state, the step-function of the algorithm.

        """

        assert self.solver!=optimistix.IndirectLevenbergMarquardt, f"{self.solver} cannot be used here, because of a jax/xla issue involving the memory layout for FFTs."


        self.initialize_general_optimizer(population)
        
        measurement_info = self.measurement_info

        self.descent_info = self.descent_info.expand(alternating_optimization = self.alternating_optimization)
        descent_info = self.descent_info


        descent_state = self.descent_state

        descent_state, do_scan = self.initialize_optimistix(descent_state, self.solver, measurement_info, descent_info)
        return descent_state, do_scan
    

    

    
