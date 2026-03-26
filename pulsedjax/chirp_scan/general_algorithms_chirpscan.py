from pulsedjax.core.base_classes_methods import RetrievePulsesCHIRPSCAN
from pulsedjax.core.base_general_optimization import DifferentialEvolutionBASE, EvosaxBASE, AutoDiffBASE


class DifferentialEvolution(DifferentialEvolutionBASE, RetrievePulsesCHIRPSCAN):
    __doc__ = DifferentialEvolutionBASE.__doc__

    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, 
                 strategy="best1_bin", selection_mechanism="greedy", mutation_rate=0.5, crossover_rate=0.7, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters,
                         strategy=strategy, selection_mechanism=selection_mechanism, mutation_rate=mutation_rate, crossover_rate=crossover_rate, **kwargs)
    



class Evosax(EvosaxBASE, RetrievePulsesCHIRPSCAN):
    __doc__ = EvosaxBASE.__doc__

    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, solver=None, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters, solver=solver, **kwargs)




class AutoDiff(AutoDiffBASE, RetrievePulsesCHIRPSCAN):
    __doc__ = AutoDiffBASE.__doc__
    
    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, solver=None, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters, solver=solver, **kwargs)