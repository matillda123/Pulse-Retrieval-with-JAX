from BaseClasses import RetrievePulsesFROGwithRealFields, RetrievePulsesRealFields
from general_algorithms_frog import DifferentialEvolution as DifferentialEvolutionFROG, Evosax as EvosaxFROG, LSF as LSFFROG, AutoDiff as AutoDiffFROG




class DifferentialEvolution(RetrievePulsesRealFields, DifferentialEvolutionFROG, RetrievePulsesFROGwithRealFields):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)





class Evosax(RetrievePulsesRealFields, EvosaxFROG, RetrievePulsesFROGwithRealFields):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)





class LSF(RetrievePulsesRealFields, LSFFROG, RetrievePulsesFROGwithRealFields):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)





class AutoDiff(RetrievePulsesRealFields, AutoDiffFROG, RetrievePulsesFROGwithRealFields):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)
