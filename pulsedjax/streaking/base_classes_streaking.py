from functools import partial as Partial
import jax
import jax.numpy as jnp
import equinox
from pulsedjax.core.create_population import create_population_classic, create_population_general
from pulsedjax.core.base_classes_algorithms import ClassicAlgorithmsBASE, GeneralOptimizationBASE
from pulsedjax.core.base_classic_algorithms import GeneralizedProjectionBASE, PtychographicIterativeEngineBASE, COPRABASE
from pulsedjax.core.base_general_optimization import DifferentialEvolutionBASE, EvosaxBASE, AutoDiffBASE

from pulsedjax.utilities import MyNamespace







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
        
        if self.measurement_info.retrieve_dtme==True:
            self.key, subkey = jax.random.split(self.key, 2)
            shape = (population_size, self.measurement_info.no_channels, jnp.size(self.measurement_info.momentum))
            population_dtme = create_population_classic(subkey, shape, guess_type, self.measurement_info)
        else:
            population_dtme = None
        
        return population.expand(dtme=population_dtme)




class GeneralizedProjectionBASESTREAKING(ClassicalAlgorithmsBASESTREAKING, GeneralizedProjectionBASE):
    __doc__ = GeneralizedProjectionBASE.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




class PtychographicIterativeEngineBASESTREAKING(ClassicalAlgorithmsBASESTREAKING, PtychographicIterativeEngineBASE):
    __doc__ = PtychographicIterativeEngineBASE.__doc__
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



class COPRABASESTREAKING(ClassicalAlgorithmsBASESTREAKING, COPRABASE):
    __doc__ = COPRABASE.__doc__
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


























class GeneralOptimizationBASESTREAKING(GeneralOptimizationBASE):
    __doc__ = GeneralOptimizationBASE.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def create_initial_population(self, population_size, amp_type="gaussian", phase_type="polynomial", no_funcs_amp=5, no_funcs_phase=6):
        population = super().create_initial_population(population_size, amp_type=amp_type, phase_type=phase_type, no_funcs_amp=no_funcs_amp, no_funcs_phase=no_funcs_phase)
        
        population_dtme = MyNamespace(amp=None, phase=None)
        self.key, subkey = jax.random.split(self.key, 2)


        if any([amp_type==i for i in self._classical_guess_types])==True:
            no_funcs_amp = jnp.size(self.measurement_info.momentum)
        
        if any([phase_type==i for i in self._classical_guess_types])==True:
            no_funcs_phase = jnp.size(self.measurement_info.momentum)


        if self.measurement_info.retrieve_dtme == True:
            measured_spectrum_is_provided_dtme = False
            population_size_comb = population_size*self.measurement_info.no_channels
            subkey, population_dtme = create_population_general(subkey, amp_type, phase_type, population_dtme, population_size_comb, no_funcs_amp, no_funcs_phase, 
                                                                measured_spectrum_is_provided_dtme, self.measurement_info, "dtme")
            
            do_reshape = lambda x: jnp.reshape(x, (population_size, self.measurement_info.no_channels) + jnp.shape(x)[1:])
            population_dtme = jax.tree.map(do_reshape, population_dtme)
            
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

    
    def loss_function(self, individual, measurement_info, descent_info):
        """ Wraps around self.calculate_error_individual() to return the error of the current guess. """
        pulse = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, "pulse")

        if measurement_info.doubleblind==True:
            gate = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, "gate")
        else:
            #gate = pulse # why like this?, why not None?, maybe because of alternating optimization of amp and phase?
            gate = None
            
        if measurement_info.retrieve_dtme==True: # for multichannel one needs to vmap here 
            _make_dtme = Partial(self.make_pulse_f_from_individual, measurement_info=measurement_info, descent_info=descent_info, pulse_or_gate="dtme")
            # remove pulse and gate because of scaling factor in pulse -> messes with vmap
            dtme = equinox.filter_vmap(_make_dtme)(MyNamespace(pulse=None, gate=None, dtme=individual.dtme)) 
        else:
            dtme = None
            
        trace_error, mu = self.calculate_error_individual(MyNamespace(pulse=pulse, gate=gate, dtme=dtme), measurement_info, descent_info)
        return trace_error, mu