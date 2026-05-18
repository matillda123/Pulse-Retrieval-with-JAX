from functools import partial as Partial
import jax
import jax.numpy as jnp
from equinox import tree_at
from pulsedjax.core.create_population import create_population_classic
from pulsedjax.core.base_classes_algorithms import ClassicAlgorithmsBASE, GeneralOptimizationBASE
from pulsedjax.core.base_classic_algorithms import GeneralizedProjectionBASE, PtychographicIterativeEngineBASE, COPRABASE
from pulsedjax.core.base_general_optimization import DifferentialEvolutionBASE, EvosaxBASE, AutoDiffBASE

from pulsedjax.utilities import MyNamespace, calculate_Z_error, calculate_trace, calculate_trace_error, get_com



def estimate_vectorpotential_max_scale(tau_arr, momentum_au, measured_trace):
    """ Approximates the max scale of the vectorpotential in the f-domain, based on the wiggles in the streaaking trace. """
    trace_com = get_com(jnp.sum(measured_trace, axis=0), jnp.arange(jnp.size(momentum_au))).astype(int)
    kmax0, kmax1 = jnp.abs(momentum_au[trace_com] - jnp.max(momentum_au)), jnp.abs(momentum_au[trace_com] - jnp.min(momentum_au))
    delta_k = jnp.min(jnp.asarray([kmax0, kmax1]))
    delta_tau = jnp.max(tau_arr) - jnp.min(tau_arr)
    scale = delta_tau*delta_k # max in f domain
    return 2*jnp.sqrt(scale)



def normalize_population(population, measurement_info, descent_info, pulse_or_gate):
    """ 
    Forstreaking the absolute scale of the vectorpotential is crucial. 
    Thus it is not normalized but just rescaled if the scale grows out of bounds.
    """
    if descent_info.normalize==True:
        if pulse_or_gate=="pulse":

            a = estimate_vectorpotential_max_scale(measurement_info.tau_arr, measurement_info.momentum, measurement_info.measured_trace)
            out_of_range = (a < jnp.max(jnp.abs(population.pulse), axis=-1))
            a = a*0.025
            pulse_corrected = a*population.pulse/jnp.max(jnp.abs(population.pulse), axis=-1)[:,None]
            population_pulse = out_of_range*pulse_corrected + (1-out_of_range)*population.pulse
            population_gate = population.gate

        elif pulse_or_gate=="gate":
            population_pulse = population.pulse
            if measurement_info.interferometric==False:
                population_gate = population.gate/jnp.linalg.norm(population.gate,axis=-1)[:,jnp.newaxis]
            else:
                population_gate = population.gate
    else:
        population_pulse = population.pulse
        population_gate = population.gate

    return population.expand(pulse=population_pulse, gate=population_gate)





class ClassicalAlgorithmsBASESTREAKING(ClassicAlgorithmsBASE):
    __doc__ = ClassicAlgorithmsBASE.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def create_initial_population(self, population_size=1, guess_type="random"):
        """ 
        Creates an initial population of pulses, parametrized as complex values on a grid.

        Args:
            population_size (int): the number of guesses to be optimized
            guess_type (str): Has to be one of random, random_phase, constant or constant_phase.
        
        Returns:
            tuple[jnp.array, jnp.array, jnp.array or None, jnp.array or None], initial populations for the pulse and possibly the gate-pulse in time domain or frequency domain for ChirpScans

        """
        population = super().create_initial_population(population_size=population_size, guess_type=guess_type)
        a = estimate_vectorpotential_max_scale(self.measurement_info.tau_arr, self.measurement_info.momentum, self.measurement_info.measured_trace)
        a = a/4
        population_pulse = population.pulse/jnp.max(jnp.abs(population.pulse), axis=1)[:,None]*a
        
        if self.measurement_info.retrieve_dtme==True:
            self.key, subkey = jax.random.split(self.key, 2)
            shape = (population_size, self.measurement_info.no_channels, jnp.size(self.measurement_info.momentum))
            population_dtme = create_population_classic(subkey, shape, guess_type, self.measurement_info, "dtme")
        else:
            population_dtme = None
        
        return population.expand(pulse=population_pulse, dtme=population_dtme)
    



    




class GeneralizedProjectionBASESTREAKING(ClassicalAlgorithmsBASESTREAKING, GeneralizedProjectionBASE):
    __doc__ = GeneralizedProjectionBASE.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _optimize_spectral_phase_factor(self, grad, measurement_info, descent_info, pulse_or_gate):
        if getattr(descent_info.measured_spectrum_is_provided, pulse_or_gate)==True and descent_info.optimize_spectral_phase_directly==True:
          
            if pulse_or_gate=="pulse":
                grad = grad*getattr(measurement_info.spectral_amplitude, pulse_or_gate)
            elif pulse_or_gate=="gate":
                grad = grad*getattr(measurement_info.spectral_amplitude, pulse_or_gate)
            else:
                pass
                    
        return grad
    

    
    def do_descent_Z_error_step(self, descent_state, signal_t_new, measurement_info, descent_info):
        """ Does one Z-error descent step. Calls descent_Z_error_step for pulse and or gate. """

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        Z_error = jax.vmap(calculate_Z_error, in_axes=(0,0))(signal_t, signal_t_new)

        population = self.descent_Z_error_step(signal_t, signal_t_new, Z_error, descent_state, measurement_info, descent_info, "pulse")
        population = normalize_population(population, measurement_info, descent_info, "pulse")
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population.pulse)
        
        if measurement_info.doubleblind==True:
            population = self.descent_Z_error_step(signal_t, signal_t_new, Z_error, descent_state, 
                                                            measurement_info, descent_info, "gate")
            population = normalize_population(population, measurement_info, descent_info, "gate")
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population.gate)

        if measurement_info.retrieve_dtme==True:
            population = self.descent_Z_error_step(signal_t, signal_t_new, Z_error, descent_state, 
                                                            measurement_info, descent_info, "dtme")
            descent_state = tree_at(lambda x: x.population.dtme, descent_state, population.dtme)

        return descent_state, None





class PtychographicIterativeEngineBASESTREAKING(ClassicalAlgorithmsBASESTREAKING, PtychographicIterativeEngineBASE):
    __doc__ = PtychographicIterativeEngineBASE.__doc__
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)





class COPRABASESTREAKING(ClassicalAlgorithmsBASESTREAKING, COPRABASE):
    __doc__ = COPRABASE.__doc__
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _optimize_spectral_phase_factor(self, grad, measurement_info, descent_info, pulse_or_gate):
        if getattr(descent_info.measured_spectrum_is_provided, pulse_or_gate)==True and descent_info.optimize_spectral_phase_directly==True:
          
            if pulse_or_gate=="pulse":
                grad = grad*getattr(measurement_info.spectral_amplitude, pulse_or_gate)
            elif pulse_or_gate=="gate":
                grad = grad*getattr(measurement_info.spectral_amplitude, pulse_or_gate)
            else:
                pass
                    
        return grad



    def local_iteration(self, descent_state, transform_arr_m, trace_line, measurement_info, descent_info):
        """ Peforms one local iteration. Calls do_iteration() with the appropriate (randomized) signal fields. """
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
        signal_t_new = self.calculate_S_prime_population(signal_t, trace_line, descent_state._local.mu, 
                                                         measurement_info, descent_info, "_local", 
                                                         axes=(0,0,0,None,None,None))
        
        local_state = descent_state._local
        local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, 
                                                    descent_state.population, local_state, 
                                                    measurement_info, descent_info, 
                                                   "pulse", "_local")
        population = normalize_population(population, measurement_info, descent_info, "pulse")
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population.pulse)


        if measurement_info.doubleblind==True:
            signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
            signal_t_new = self.calculate_S_prime_population(signal_t, trace_line, descent_state._local.mu, 
                                                            measurement_info, descent_info, "_local", 
                                                            axes=(0,0,0,None,None,None))
            local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, 
                                                        descent_state.population, local_state, 
                                                        measurement_info, descent_info, 
                                                        "gate", "_local")
            population = normalize_population(population, measurement_info, descent_info, "gate")
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population.gate)



        if measurement_info.retrieve_dtme==True:
            signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
            signal_t_new = self.calculate_S_prime_population(signal_t, trace_line, descent_state._local.mu, 
                                                            measurement_info, descent_info, "_local", 
                                                            axes=(0,0,0,None,None,None))
            local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, 
                                                        descent_state.population, local_state, 
                                                        measurement_info, descent_info, 
                                                        "dtme", "_local")
            descent_state = tree_at(lambda x: x.population.dtme, descent_state, population.dtme)

        descent_state = tree_at(lambda x: x._local, descent_state, local_state)
        return descent_state, None
    






    

    def global_step(self, descent_state, measurement_info, descent_info):
        """
        Performs one global iteration of the Common Pulse Retrieval Algorithm. 
        This means the method updates the population once using all measured data at once.
        
        Args:
            descent_state (Pytree):
            measurement_info (Pytree):
            descent_info (Pytree):

        Returns:
            tuple[Pytree, jnp.array], the updated descent state and the current trace errors of the population.
        """

        measured_trace = measurement_info.measured_trace
        _calculate_trace = Partial(calculate_trace, measured_trace=measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_global")
        
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        signal_t_new = self.calculate_S_prime_population(signal_t, measured_trace, mu, 
                                                         measurement_info, descent_info, "_global", 
                                                         axes=(0,None,0,None,None,None))

        global_state = descent_state._global
        global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, 
                                                     descent_state.population, global_state, measurement_info, 
                                                     descent_info, "pulse", "_global")
        population = normalize_population(population, measurement_info, descent_info, "pulse")
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population.pulse)

        if measurement_info.doubleblind==True:
            signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
            trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
            signal_t_new = self.calculate_S_prime_population(signal_t, measured_trace, mu, 
                                                            measurement_info, descent_info, "_global", 
                                                            axes=(0,None,0,None,None,None))
        
            global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, 
                                                         descent_state.population, global_state, measurement_info, 
                                                         descent_info, "gate", "_global")
            
            population = normalize_population(population, measurement_info, descent_info, "gate")
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population.gate)




        if measurement_info.retrieve_dtme==True:
            signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
            trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
            signal_t_new = self.calculate_S_prime_population(signal_t, measured_trace, mu, 
                                                            measurement_info, descent_info, "_global", 
                                                            axes=(0,None,0,None,None,None))
        
            global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, 
                                                         descent_state.population, global_state, measurement_info, 
                                                         descent_info, "dtme", "_global")
            descent_state = tree_at(lambda x: x.population.dtme, descent_state, population.dtme)

            
        descent_state = tree_at(lambda x: x._global, descent_state, global_state)

        #signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        #trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,0,None))(mu, trace, measured_trace)

        descent_state = tree_at(lambda x: x._global.mu, descent_state, mu)
        return descent_state, trace_error.reshape(-1,1)
    



















class GeneralOptimizationBASESTREAKING(GeneralOptimizationBASE):
    __doc__ = GeneralOptimizationBASE.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def create_initial_population(self, population_size, amp_type="gaussian", phase_type="polynomial", no_funcs_amp=5, no_funcs_phase=6):
        population = super().create_initial_population(population_size, amp_type=amp_type, phase_type=phase_type, no_funcs_amp=no_funcs_amp, no_funcs_phase=no_funcs_phase)
        a = estimate_vectorpotential_max_scale(self.measurement_info.tau_arr, self.measurement_info.momentum, self.measurement_info.measured_trace)
        a = a/4
        amp_type_list = ["gaussian", "lorentzian"]

        if (any([amp_type == _amp_type for _amp_type in amp_type_list])==True and self.descent_info.measured_spectrum_is_provided.pulse==True):
            population = tree_at(lambda x: x.pulse.amp, population, population.pulse.amp.a*0 + a)
        else:
            population = tree_at(lambda x: x.pulse.amp, population, population.pulse.amp*0 + a)
        
        population = population.expand(dtme = MyNamespace(amp=None, phase=None))
        
        bsplines_Nx = self.descent_info.bsplines_Nx
        amp, phase = bsplines_Nx.amp.expand(dtme=None), bsplines_Nx.phase.expand(dtme=None)
        self.descent_info = self.descent_info.expand(bsplines_Nx=MyNamespace(amp=amp, phase=phase))

        if self.measurement_info.retrieve_dtme == True:
            population_size_comb = population_size*self.measurement_info.no_channels
            population = self._create_inital_population(population, population_size_comb, amp_type, phase_type, no_funcs_amp, no_funcs_phase, "dtme")
        
            do_reshape = lambda x: jnp.reshape(x, (population_size, self.measurement_info.no_channels) + jnp.shape(x)[1:])
            population_dtme = jax.tree.map(do_reshape, population.dtme)
            population = population.expand(dtme = population_dtme)
            
        return population
            


    def split_population_in_amp_and_phase(self, population):
        """ Splits a population into an amplitude and phase population. """
        population_amp = MyNamespace(pulse=MyNamespace(amp=population.pulse.amp, phase=None), 
                                     gate=MyNamespace(amp=population.gate.amp, phase=None),
                                     dtme=MyNamespace(amp=population.dtme.amp, phase=None))
        
        population_phase = MyNamespace(pulse=MyNamespace(amp=None, phase=population.pulse.phase), 
                                        gate=MyNamespace(amp=None, phase=population.gate.phase),
                                        dtme=MyNamespace(amp=None, phase=population.dtme.phase))

        return population_amp, population_phase
        


    def merge_population_from_amp_and_phase(self, population_amp, population_phase):
        """ Undoes split_population_in_amp_and_phase() """
        population = MyNamespace(pulse=MyNamespace(amp=population_amp.pulse.amp, phase=population_phase.pulse.phase), 
                                    gate=MyNamespace(amp=population_amp.gate.amp, phase=population_phase.gate.phase),
                                    dtme=MyNamespace(amp=population_amp.dtme.amp, phase=population_phase.dtme.phase))
        return population
    


    
    def get_pulses_f_from_population(self, population, measurement_info, descent_info):
        """ Evaluates a parametrized population onto the frequency axis. """
        make_pulse = Partial(self.make_pulse_f_from_individual, pulse_or_gate="pulse")
        pulse_f_arr = jax.vmap(make_pulse, in_axes=(0,None,None))(population, measurement_info, descent_info)

        if measurement_info.doubleblind==True:
            make_gate = Partial(self.make_pulse_f_from_individual, pulse_or_gate="gate")
            gate_f_arr = jax.vmap(make_gate, in_axes=(0,None,None))(population, measurement_info, descent_info)
        else:
            gate_f_arr = None

        if measurement_info.retrieve_dtme==True: # here one needs to do a extra vmap-for the multichannel sfa
            make_dtme = Partial(self.make_pulse_f_from_individual, measurement_info=measurement_info, descent_info=descent_info, pulse_or_gate="dtme")
            dtme_b_arr = jax.vmap(jax.vmap(make_dtme))(MyNamespace(pulse=None, gate=None, dtme=population.dtme))
        else:
            dtme_b_arr = None

        return MyNamespace(pulse=pulse_f_arr, gate=gate_f_arr, dtme=dtme_b_arr)
    




class DifferentialEvolutionBASESTREAKING(GeneralOptimizationBASESTREAKING, DifferentialEvolutionBASE):
    __doc__ = DifferentialEvolutionBASE.__doc__
    def __init__(self, *args, strategy="best1_bin", selection_mechanism="greedy", mutation_rate=0.5, crossover_rate=0.7, **kwargs):
        super().__init__(*args, strategy=strategy, selection_mechanism=selection_mechanism, mutation_rate=mutation_rate, crossover_rate=crossover_rate, **kwargs)



class EvosaxBASESTREAKING(GeneralOptimizationBASESTREAKING, EvosaxBASE):
    __doc__ = EvosaxBASE.__doc__
    def __init__(self, *args, solver=None, **kwargs):
        super().__init__(*args, solver=solver, **kwargs)



class AutoDiffBASESTREAKING(GeneralOptimizationBASESTREAKING, AutoDiffBASE):
    __doc__ = AutoDiffBASE.__doc__

    def __init__(self, *args, solver=None, **kwargs):
        super().__init__(*args, solver=solver, **kwargs)

        self.optimize_group_delay_dtme = True

    
    def loss_function(self, individual, measurement_info, descent_info):
        """ Wraps around self.calculate_error_individual() to return the error of the current guess. """
        pulse_f = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, "pulse")

        if measurement_info.doubleblind==True:
            gate_f = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, "gate")
        else:
            gate_f = None
            
        if measurement_info.retrieve_dtme==True: # for multichannel one needs to vmap here 
            _make_dtme = Partial(self.make_pulse_f_from_individual, measurement_info=measurement_info, descent_info=descent_info, pulse_or_gate="dtme")
            # remove pulse and gate because of scaling factor in pulse -> messes with vmap
            dtme_k = jax.vmap(_make_dtme)(MyNamespace(pulse=None, gate=None, dtme=individual.dtme)) 
        else:
            dtme_k = None
            
        trace_error, mu = self.calculate_error_individual(MyNamespace(pulse=pulse_f, gate=gate_f, dtme=dtme_k), measurement_info, descent_info)
        return trace_error, mu