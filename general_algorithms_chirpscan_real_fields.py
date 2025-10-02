from BaseClasses import RetrievePulsesCHIRPSCANwithRealFields, RetrievePulsesRealFields
from general_algorithms_chirpscan import DifferentialEvolution as DifferentialEvolutionDSCAN, Evosax as EvosaxDSCAN, LSF as LSFDSCAN, AutoDiff as AutoDiffDSCAN




class DifferentialEvolution(RetrievePulsesRealFields, DifferentialEvolutionDSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)







class Evosax(RetrievePulsesRealFields, EvosaxDSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)





class LSF(RetrievePulsesRealFields, LSFDSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)




class AutoDiff(RetrievePulsesRealFields, AutoDiffDSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)
