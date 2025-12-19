from src.real_fields.base_classes_methods import RetrievePulsesTDPwithRealFields
from src.real_fields.frog import (GeneralizedProjection as GeneralizedProjectionFROG, 
                                  PtychographicIterativeEngine as PtychgraphicIterativeEngineFROG,
                                  COPRA as COPRAFROG)





class GeneralizedProjection(RetrievePulsesTDPwithRealFields, GeneralizedProjectionFROG):
    """
    The Generalized Projection Algorithm for FROG.
    
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)






class PtychographicIterativeEngine(RetrievePulsesTDPwithRealFields, PtychgraphicIterativeEngineFROG):
    """
    The Ptychographic Iterative Engine (PIE) for FROG.

    Attributes:
        pie_method (None, str): specifies the PIE variant. Can be one of None, PIE, ePIE, rPIE. Where None indicates that the pure gradient is used.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, pie_method="rPIE", cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)

        self.pie_method=pie_method





class COPRA(RetrievePulsesTDPwithRealFields, COPRAFROG):
    """
    The Common Pulse Retrieval Algorithm (COPRA) for FROG.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)

