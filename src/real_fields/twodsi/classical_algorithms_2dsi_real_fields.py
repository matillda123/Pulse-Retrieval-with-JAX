from src.real_fields.base_classes_methods import RetrievePulses2DSIwithRealFields
from src.real_fields.frog import (GeneralizedProjection as GeneralizedProjectionFROG,
                                  PtychographicIterativeEngine as PtychographicIterativeEngineFROG,
                                  COPRA as COPRAFROG)




class GeneralizedProjection(RetrievePulses2DSIwithRealFields, GeneralizedProjectionFROG):
    """
    The Generalized Projection Algorithm for 2DSI.

    """

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)





class PtychographicIterativeEngine(RetrievePulses2DSIwithRealFields, PtychographicIterativeEngineFROG):
    """
    The Ptychographic Iterative Engine (PIE) for 2DSI.
    Is not set up to be used for doubleblind. The PIE was not invented for reconstruction of interferometric signals.

    Attributes:
        pie_method (None, str): specifies the PIE variant. Can be one of None, PIE, ePIE, rPIE. Where None indicates that the pure gradient is used.

    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, pie_method="rPIE", **kwargs): 
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)

        self.pie_method = pie_method






class COPRA(RetrievePulses2DSIwithRealFields, COPRAFROG):
    """
    The Common Pulse Retrieval Algorithm (COPRA) for 2DSI.
    
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)

