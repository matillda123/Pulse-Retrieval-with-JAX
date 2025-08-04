from BaseClasses import RetrievePulsesFROGwithRealFields
from general_algorithms_frog import DifferentialEvolution as DifferentialEvolutionFROG, Evosax as EvosaxFROG, LSF as LSFFROG, AutoDiff as AutoDiffFROG




class DifferentialEvolution(DifferentialEvolutionFROG, RetrievePulsesFROGwithRealFields):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)







class Evosax(EvosaxFROG, RetrievePulsesFROGwithRealFields):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)





class LSF(LSFFROG, RetrievePulsesFROGwithRealFields):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)




class AutoDiff(AutoDiffFROG, RetrievePulsesFROGwithRealFields):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)
