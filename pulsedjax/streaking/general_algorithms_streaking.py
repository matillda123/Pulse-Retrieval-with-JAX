from pulsedjax.core.base_classes_methods import RetrievePulsesSTREAKING
from pulsedjax.streaking.base_classes_streaking import DifferentialEvolutionBASESTREAKING, EvosaxBASESTREAKING, AutoDiffBASESTREAKING
import jax.numpy as jnp




class DifferentialEvolution(DifferentialEvolutionBASESTREAKING, RetrievePulsesSTREAKING):
    __doc__ = DifferentialEvolutionBASESTREAKING.__doc__
    
    def __init__(self, delay_fs, energy_eV, measured_trace, Ip_eV=jnp.array([0]), retrieve_dtme=False, cross_correlation="doubleblind", interferometric=False, 
                 strategy="best1_bin", selection_mechanism="greedy", mutation_rate=0.5, crossover_rate=0.7, **kwargs):
        super().__init__(delay_fs, energy_eV, measured_trace, Ip_eV=Ip_eV, retrieve_dtme=retrieve_dtme, cross_correlation=cross_correlation, interferometric=interferometric, 
                         strategy=strategy, selection_mechanism=selection_mechanism, mutation_rate=mutation_rate, crossover_rate=crossover_rate, **kwargs)



class Evosax(EvosaxBASESTREAKING, RetrievePulsesSTREAKING):
    __doc__ = EvosaxBASESTREAKING.__doc__

    def __init__(self, delay_fs, energy_eV, measured_trace, Ip_eV=jnp.array([0]), retrieve_dtme=False, cross_correlation="doubleblind", interferometric=False, solver=None, **kwargs):
        super().__init__(delay_fs, energy_eV, measured_trace, Ip_eV=Ip_eV, retrieve_dtme=retrieve_dtme, cross_correlation=cross_correlation, interferometric=interferometric, solver=solver, **kwargs)




class AutoDiff(AutoDiffBASESTREAKING, RetrievePulsesSTREAKING):
    __doc__ = AutoDiffBASESTREAKING.__doc__

    def __init__(self, delay_fs, energy_eV, measured_trace, Ip_eV=jnp.array([0]), retrieve_dtme=False, cross_correlation="doubleblind", interferometric=False, solver=None, **kwargs):
        super().__init__(delay_fs, energy_eV, measured_trace, Ip_eV=Ip_eV, retrieve_dtme=retrieve_dtme, cross_correlation=cross_correlation, interferometric=interferometric, solver=solver, **kwargs)
