import refractiveindex

from pulsedjax.real_fields.core.base_classes_methods import RetrievePulsesVAMPIREwithRealFields
from pulsedjax.core.base_general_optimization import DifferentialEvolutionBASE, EvosaxBASE, AutoDiffBASE




class DifferentialEvolution(DifferentialEvolutionBASE, RetrievePulsesVAMPIREwithRealFields):
    __doc__ = DifferentialEvolutionBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, f_range_fields=(None, None), f_range_pulse=(None, None), f_max_all_fields=None, 
                 strategy="best1_bin", selection_mechanism="greedy", mutation_rate=0.5, crossover_rate=0.7, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, f_range_fields=f_range_fields, f_range_pulse=f_range_pulse, f_max_all_fields=f_max_all_fields,
                         strategy=strategy, selection_mechanism=selection_mechanism, mutation_rate=mutation_rate, crossover_rate=crossover_rate, **kwargs)
        self._post_init()



class Evosax(EvosaxBASE, RetrievePulsesVAMPIREwithRealFields):
    __doc__ = EvosaxBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, f_range_fields=(None, None), f_range_pulse=(None, None), f_max_all_fields=None, solver=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, f_range_fields=f_range_fields, f_range_pulse=f_range_pulse, f_max_all_fields=f_max_all_fields, solver=solver, **kwargs)
        self._post_init()





class AutoDiff(AutoDiffBASE, RetrievePulsesVAMPIREwithRealFields):
    __doc__ = AutoDiffBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, f_range_fields=(None, None), f_range_pulse=(None, None), f_max_all_fields=None, solver=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, f_range_fields=f_range_fields, f_range_pulse=f_range_pulse, f_max_all_fields=f_max_all_fields, solver=solver, **kwargs)
        self._post_init()
