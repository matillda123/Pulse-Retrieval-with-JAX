from pulsedjax.real_fields.base_classes_methods import RetrievePulses2DSIwithRealFields
from pulsedjax.real_fields.frog import (GeneralizedProjection as GeneralizedProjectionFROG,
                                  PtychographicIterativeEngine as PtychographicIterativeEngineFROG,
                                  COPRA as COPRAFROG)




class GeneralizedProjection(RetrievePulses2DSIwithRealFields, GeneralizedProjectionFROG):
    __doc__ = GeneralizedProjectionFROG.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter1, spectral_filter2, cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2, cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)





class PtychographicIterativeEngine(RetrievePulses2DSIwithRealFields, PtychographicIterativeEngineFROG):
    __doc__ = PtychographicIterativeEngineFROG.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter1, spectral_filter2, cross_correlation=False, f_range_fields=(None, None), **kwargs): 
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2, cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)







class COPRA(RetrievePulses2DSIwithRealFields, COPRAFROG):
    __doc__ = COPRAFROG.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter1, spectral_filter2, cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2, cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)

