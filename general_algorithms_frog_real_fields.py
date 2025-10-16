from BaseClasses import RetrievePulsesFROGwithRealFields, RetrievePulsesRealFields
from general_algorithms_frog import DifferentialEvolution as DifferentialEvolutionFROG, Evosax as EvosaxFROG, LSF as LSFFROG, AutoDiff as AutoDiffFROG




class DifferentialEvolution(RetrievePulsesRealFields, DifferentialEvolutionFROG, RetrievePulsesFROGwithRealFields):
    """ The Differential Evolution Algorithm applied to FROG with real fields. 
    Inherits from  RetrievePulsesRealFields, DifferentialEvolutionFROG and RetrievePulsesFROGwithRealFields"""
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)



class Evosax(RetrievePulsesRealFields, EvosaxFROG, RetrievePulsesFROGwithRealFields):
    """ The Evosax package applied to FROG with real fields. 
    Inherits from  RetrievePulsesRealFields, EvosaxFROG and RetrievePulsesFROGwithRealFields"""
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)



class LSF(RetrievePulsesRealFields, LSFFROG, RetrievePulsesFROGwithRealFields):
    """ The LSF Algorithm applied to FROG with real fields. 
    Inherits from  RetrievePulsesRealFields, LSFFROG and RetrievePulsesFROGwithRealFields"""
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)



class AutoDiff(RetrievePulsesRealFields, AutoDiffFROG, RetrievePulsesFROGwithRealFields):
    """ The Optimistix package applied to FROG with real fields. 
    Inherits from  RetrievePulsesRealFields, AutoDiffFROG and RetrievePulsesFROGwithRealFields"""
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)
