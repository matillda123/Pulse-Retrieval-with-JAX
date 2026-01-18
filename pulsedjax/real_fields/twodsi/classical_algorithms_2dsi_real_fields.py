from pulsedjax.real_fields.base_classes_methods import RetrievePulses2DSIwithRealFields
from pulsedjax.real_fields.base_classic_algorithms import GeneralizedProjectionBASE, PtychographicIterativeEngineBASE, COPRABASE



class GeneralizedProjection(RetrievePulses2DSIwithRealFields, GeneralizedProjectionBASE):
    __doc__ = GeneralizedProjectionBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter1, spectral_filter2, cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2, cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)
        
        



class PtychographicIterativeEngine(RetrievePulses2DSIwithRealFields, PtychographicIterativeEngineBASE):
    __doc__ = PtychographicIterativeEngineBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter1, spectral_filter2, cross_correlation=False, f_range_fields=(None, None), **kwargs): 
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2, cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)




class COPRA(RetrievePulses2DSIwithRealFields, COPRABASE):
    __doc__ = COPRABASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter1, spectral_filter2, cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2, cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)
        
        