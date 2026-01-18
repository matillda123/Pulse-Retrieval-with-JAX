from pulsedjax.real_fields.base_classes_methods import RetrievePulsesFROGwithRealFields
from pulsedjax.real_fields.base_classic_algorithms import GeneralizedProjectionBASE, PtychographicIterativeEngineBASE, COPRABASE




class GeneralizedProjection(RetrievePulsesFROGwithRealFields, GeneralizedProjectionBASE):
    __doc__ = GeneralizedProjectionBASE.__doc__


    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, interferometric=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, interferometric=interferometric, f_range_fields=f_range_fields, **kwargs)
        



class PtychographicIterativeEngine(RetrievePulsesFROGwithRealFields, PtychographicIterativeEngineBASE):
    __doc__ = PtychographicIterativeEngineBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, interferometric=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, interferometric=interferometric, f_range_fields=f_range_fields, **kwargs)


    


class COPRA(RetrievePulsesFROGwithRealFields, COPRABASE):
    __doc__ = COPRABASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, interferometric=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, interferometric=interferometric, f_range_fields=f_range_fields, **kwargs)

