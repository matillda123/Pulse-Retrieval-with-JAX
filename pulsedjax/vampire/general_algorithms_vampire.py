import refractiveindex

from pulsedjax.core.base_classes_methods import RetrievePulsesVAMPIRE
from pulsedjax.core.base_general_optimization import DifferentialEvolutionBASE, EvosaxBASE, LSFBASE, AutoDiffBASE

from pulsedjax.utilities import MyNamespace




class DifferentialEvolution(DifferentialEvolutionBASE, RetrievePulsesVAMPIRE):
    __doc__ = DifferentialEvolutionBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, **kwargs)


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """ Calls get_pulses_t_from_population() """
        return self.get_pulses_t_from_population(population, measurement_info, descent_info)
    





class Evosax(EvosaxBASE, RetrievePulsesVAMPIRE):
    __doc__ = EvosaxBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, solver=None, 
                 strategy="best1_bin", selection_mechanism="greedy", mutation_rate=0.5, crossover_rate=0.7,**kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, solver=solver,
                         strategy=strategy, selection_mechanism=selection_mechanism, mutation_rate=mutation_rate, crossover_rate=crossover_rate, **kwargs)


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """ Calls get_pulses_t_from_population() """
        return self.get_pulses_t_from_population(population, measurement_info, descent_info)
    





class LSF(LSFBASE, RetrievePulsesVAMPIRE):
    __doc__ = LSFBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, **kwargs)



    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """ Returns the pulse and gate population. Does not need to call get_pulses_t_from_population() since LSF works with discetized fields only. """
        return population.pulse, population.gate
    

    def convert_population(self, population, measurement_info, descent_info):
        """ Converts any population into a discretized one. """
        pulse_arr, gate_arr = self.get_pulses_t_from_population(population, measurement_info, descent_info)
        return MyNamespace(pulse=pulse_arr, gate=gate_arr)
    


    # #These Two Methods should be used when attempting to do the frog optimization with the pulses in the frequency domain
    # #-> the optimization doesnt really work if these are used
    
    # def get_pulses_from_population(self, population, measurement_info, descent_info):
    #     sk, rn = measurement_info.sk, measurement_info.rn
    #     pulse_f_arr, gate_f_arr = population.pulse, population.gate
    #     pulse_t_arr = do_ifft(pulse_f_arr, sk, rn)

    #     if measurement_info.doubleblind==True:
    #         gate_t_arr = do_ifft(gate_f_arr, sk, rn)
    #     else:
    #         gate_t_arr = None
    
    #     return pulse_t_arr, gate_t_arr
    

    # def convert_population(self, population, measurement_info, descent_info):
    #     pulse_arr, gate_arr = self.get_pulses_f_from_population(population, measurement_info, descent_info)
    #     return MyNamespace(pulse=pulse_arr, gate=gate_arr)
        

    






class AutoDiff(AutoDiffBASE, RetrievePulsesVAMPIRE):
    __doc__ = AutoDiffBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                 cross_correlation=False, solver=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, tau_interferometer=tau_interferometer, 
                         material_thickness=material_thickness, refractive_index=refractive_index, 
                         cross_correlation=cross_correlation, solver=solver, **kwargs)



    def get_pulses_from_population(self, population, measurement_info, descent_info):
        return self.get_pulses_t_from_population(population, measurement_info, descent_info)
    


    def make_pulse_from_individual(self, individual, measurement_info, descent_info, pulse_or_gate):
        """ Evaluates a pulse/gate for an individual. """
        signal = self.make_pulse_t_from_individual(individual, measurement_info, descent_info, pulse_or_gate)
        return signal