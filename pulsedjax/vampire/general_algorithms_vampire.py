import refractiveindex

from pulsedjax.core.base_classes_methods import RetrievePulsesVAMPIRE
from pulsedjax.core.base_general_optimization import DifferentialEvolutionBASE, EvosaxBASE, AutoDiffBASE





class DifferentialEvolution(DifferentialEvolutionBASE, RetrievePulsesVAMPIRE):
    __doc__ = DifferentialEvolutionBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, 
                 strategy="best1_bin", selection_mechanism="greedy", mutation_rate=0.5, crossover_rate=0.7, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation,
                         strategy=strategy, selection_mechanism=selection_mechanism, mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                         **kwargs)






class Evosax(EvosaxBASE, RetrievePulsesVAMPIRE):
    __doc__ = EvosaxBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, solver=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, solver=solver, **kwargs)




class AutoDiff(AutoDiffBASE, RetrievePulsesVAMPIRE):
    __doc__ = AutoDiffBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, solver=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, solver=solver, **kwargs)
