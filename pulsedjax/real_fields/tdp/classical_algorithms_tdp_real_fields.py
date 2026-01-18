from pulsedjax.real_fields.base_classes_methods import RetrievePulsesTDPwithRealFields
from pulsedjax.real_fields.base_classic_algorithms import GeneralizedProjectionBASE, PtychographicIterativeEngineBASE, COPRABASE




class GeneralizedProjection(RetrievePulsesTDPwithRealFields, GeneralizedProjectionBASE):
    __doc__ = GeneralizedProjectionBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter, cross_correlation=False, interferometric=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, interferometric=interferometric, f_range_fields=f_range_fields, **kwargs)






class PtychographicIterativeEngine(RetrievePulsesTDPwithRealFields, PtychographicIterativeEngineBASE):
    __doc__ = PtychographicIterativeEngineBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter, cross_correlation=False, interferometric=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, interferometric=interferometric, f_range_fields=f_range_fields, **kwargs)






class COPRA(RetrievePulsesTDPwithRealFields, COPRABASE):
    __doc__ = COPRABASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter, cross_correlation=False, interferometric=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter=spectral_filter, cross_correlation=cross_correlation, interferometric=interferometric, f_range_fields=f_range_fields, **kwargs)

