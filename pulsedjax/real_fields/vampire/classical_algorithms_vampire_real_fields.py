import refractiveindex

from pulsedjax.real_fields.base_classes_methods import RetrievePulsesVAMPIREwithRealFields
from pulsedjax.real_fields.frog import (GeneralizedProjection as GeneralizedProjectionFROG,
                                  PtychographicIterativeEngine as PtychographicIterativeEngineFROG,
                                  COPRA as COPRAFROG)





class GeneralizedProjection(RetrievePulsesVAMPIREwithRealFields, GeneralizedProjectionFROG):
    __doc__ = GeneralizedProjectionFROG.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)







class PtychographicIterativeEngine(RetrievePulsesVAMPIREwithRealFields, PtychographicIterativeEngineFROG):
    __doc__ = PtychographicIterativeEngineFROG.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)










class COPRA(RetrievePulsesVAMPIREwithRealFields, COPRAFROG):
    __doc__ = COPRAFROG.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, f_range_fields=(None, None), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, f_range_fields=f_range_fields, **kwargs)
