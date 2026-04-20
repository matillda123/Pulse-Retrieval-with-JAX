from functools import partial as Partial
import jax
from pulsedjax.core.create_population import create_population_general
from pulsedjax.core.base_classes_algorithms import GeneralOptimizationBASE
from pulsedjax.core.base_general_optimization import DifferentialEvolutionBASE, EvosaxBASE, AutoDiffBASE

from pulsedjax.utilities import MyNamespace


class GeneralOptimizationBASESTREAKING(GeneralOptimizationBASE):
    __doc__ = GeneralOptimizationBASE.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def create_initial_population(self, population_size, amp_type="gaussian", phase_type="polynomial", no_funcs_amp=5, no_funcs_phase=6):
        population = super().create_initial_population(population_size, amp_type=amp_type, phase_type=phase_type, no_funcs_amp=no_funcs_amp, no_funcs_phase=no_funcs_phase)
        
        population_dtme = MyNamespace(amp=None, phase=None)

        self.key, subkey = jax.random.split(self.key, 2)

        if self.measurement_info.retrieve_dtme == True:
            measured_spectrum_is_provided_dtme = False
            subkey, population_dtme = create_population_general(subkey, amp_type, phase_type, population_dtme, population_size, no_funcs_amp, no_funcs_phase, 
                                                                measured_spectrum_is_provided_dtme, self.measurement_info)
            
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

    
    def get_pulses_f_from_population(self, population, measurement_info, descent_info):
        """ Evaluates a parametrized population onto the frequency axis. """
        make_pulse = Partial(self.make_pulse_f_from_individual, pulse_or_gate="pulse")
        pulse_f_arr = jax.vmap(make_pulse, in_axes=(0,None,None))(population, measurement_info, descent_info)

        if measurement_info.doubleblind==True:
            make_gate = Partial(self.make_pulse_f_from_individual, pulse_or_gate="gate")
            gate_f_arr = jax.vmap(make_gate, in_axes=(0,None,None))(population, measurement_info, descent_info)
        else:
            gate_f_arr = None

        if measurement_info.retrieve_dtme==True:
            make_gate = Partial(self.make_pulse_f_from_individual, pulse_or_gate="dtme")
            dtme_b_arr = jax.vmap(make_gate, in_axes=(0,None,None))(population, measurement_info, descent_info)
        else:
            dtme_b_arr = None

        return MyNamespace(pulse=pulse_f_arr, gate=gate_f_arr, dtme=dtme_b_arr)