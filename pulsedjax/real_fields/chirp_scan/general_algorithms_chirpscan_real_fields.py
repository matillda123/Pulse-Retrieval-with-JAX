from pulsedjax.real_fields.core.base_classes_methods import RetrievePulsesCHIRPSCANwithRealFields
from pulsedjax.core.base_general_optimization import DifferentialEvolutionBASE, EvosaxBASE, AutoDiffBASE



class DifferentialEvolution(DifferentialEvolutionBASE, RetrievePulsesCHIRPSCANwithRealFields):
    __doc__ = DifferentialEvolutionBASE.__doc__

    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, f_range_fields=(None,None), f_range_pulse=(None, None), f_max_all_fields=None,
                 strategy="best1_bin", selection_mechanism="greedy", mutation_rate=0.5, crossover_rate=0.7, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, f_range_fields=f_range_fields, f_range_pulse=f_range_pulse, f_max_all_fields=f_max_all_fields, phase_type=phase_type, chirp_parameters=chirp_parameters,
                         strategy=strategy, selection_mechanism=selection_mechanism, mutation_rate=mutation_rate, crossover_rate=crossover_rate, **kwargs)
        self._post_init()



class Evosax(EvosaxBASE, RetrievePulsesCHIRPSCANwithRealFields):
    __doc__ = EvosaxBASE.__doc__

    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, f_range_fields=(None,None), f_range_pulse=(None, None), f_max_all_fields=None, solver=None, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, f_range_fields=f_range_fields, f_range_pulse=f_range_pulse, f_max_all_fields=f_max_all_fields, phase_type=phase_type, chirp_parameters=chirp_parameters, solver=solver, **kwargs)
        self._post_init()



    

class AutoDiff(AutoDiffBASE, RetrievePulsesCHIRPSCANwithRealFields):
    __doc__ = AutoDiffBASE.__doc__

    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, f_range_fields=(None,None), f_range_pulse=(None, None), f_max_all_fields=None, solver=None, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, f_range_fields=f_range_fields, f_range_pulse=f_range_pulse, f_max_all_fields=f_max_all_fields, phase_type=phase_type, chirp_parameters=chirp_parameters, solver=solver, **kwargs)
        self._post_init()

