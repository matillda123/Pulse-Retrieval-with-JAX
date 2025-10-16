from src.core.BaseClasses import RetrievePulsesCHIRPSCANwithRealFields, RetrievePulsesRealFields
from .general_algorithms_chirpscan import (DifferentialEvolution as DifferentialEvolutionCHIRPSCAN, Evosax as EvosaxCHIRPSCAN, 
                                          LSF as LSFCHIRPSCAN, AutoDiff as AutoDiffCHIRPSCAN)




class DifferentialEvolution(RetrievePulsesRealFields, DifferentialEvolutionCHIRPSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    """ The Differential Evolution Algorithm applied to Chirp-Scans with real fields. 
    Inherits from  RetrievePulsesRealFields, DifferentialEvolutionCHIRPSCAN and RetrievePulsesCHIRPSCANwithRealFields"""
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)



class Evosax(RetrievePulsesRealFields, EvosaxCHIRPSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    """ The Evosax package applied to Chirp-Scans with real fields. 
    Inherits from  RetrievePulsesRealFields, EvosaxCHIRPSCAN and RetrievePulsesCHIRPSCANwithRealFields"""
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)



class LSF(RetrievePulsesRealFields, LSFCHIRPSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    """ The LSF Algorithm applied to Chirp-Scans with real fields. 
    Inherits from  RetrievePulsesRealFields, LSFCHIRPSCAN and RetrievePulsesCHIRPSCANwithRealFields"""
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)



class AutoDiff(RetrievePulsesRealFields, AutoDiffCHIRPSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    """ The Optimistix package applied to Chirp-Scans with real fields. 
    Inherits from  RetrievePulsesRealFields, AutoDiffCHIRPSCAN and RetrievePulsesCHIRPSCANwithRealFields"""
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)
