from pulsedjax.core.base_classes_methods import RetrievePulses2DSI
from pulsedjax.core.base_general_optimization import DifferentialEvolutionBASE, EvosaxBASE, AutoDiffBASE




class DifferentialEvolution(DifferentialEvolutionBASE, RetrievePulses2DSI):
    __doc__ = DifferentialEvolutionBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter1, spectral_filter2, cross_correlation=False, 
                 strategy="best1_bin", selection_mechanism="greedy", mutation_rate=0.5, crossover_rate=0.7, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2, cross_correlation=cross_correlation,
                         strategy=strategy, selection_mechanism=selection_mechanism, mutation_rate=mutation_rate, crossover_rate=crossover_rate, **kwargs)





class Evosax(EvosaxBASE, RetrievePulses2DSI):
    __doc__ = EvosaxBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter1, spectral_filter2, cross_correlation=False, solver=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2, cross_correlation=cross_correlation, solver=solver, **kwargs)


    


class AutoDiff(AutoDiffBASE, RetrievePulses2DSI):
    __doc__ = AutoDiffBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter1, spectral_filter2, cross_correlation=False, solver=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2, cross_correlation=cross_correlation, solver=solver, **kwargs)
