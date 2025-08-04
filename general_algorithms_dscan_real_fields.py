from BaseClasses import RetrievePulsesDSCANwithRealFields
from general_algorithms_dscan import DifferentialEvolution as DifferentialEvolutionDSCAN, Evosax as EvosaxDSCAN, LSF as LSFDSCAN, AutoDiff as AutoDiffDSCAN




class DifferentialEvolution(DifferentialEvolutionDSCAN, RetrievePulsesDSCANwithRealFields):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)







class Evosax(EvosaxDSCAN, RetrievePulsesDSCANwithRealFields):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)





class LSF(LSFDSCAN, RetrievePulsesDSCANwithRealFields):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)




class AutoDiff(AutoDiffDSCAN, RetrievePulsesDSCANwithRealFields):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)
