from pulsedjax.real_fields.base_classes_methods import RetrievePulsesCHIRPSCANwithRealFields
from pulsedjax.core.base_general_optimization import DifferentialEvolutionBASE, EvosaxBASE, LSFBASE, AutoDiffBASE

from pulsedjax.utilities import MyNamespace


class DifferentialEvolution(DifferentialEvolutionBASE, RetrievePulsesCHIRPSCANwithRealFields):
    __doc__ = DifferentialEvolutionBASE.__doc__

    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, f_range_fields=(None,None), 
                 strategy="best1_bin", selection_mechanism="greedy", mutation_rate=0.5, crossover_rate=0.7, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, f_range_fields=f_range_fields, phase_type=phase_type, chirp_parameters=chirp_parameters,
                         strategy=strategy, selection_mechanism=selection_mechanism, mutation_rate=mutation_rate, crossover_rate=crossover_rate, **kwargs)
        self._post_init()


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """ Calls get_pulses_f_from_population() """
        return self.get_pulses_f_from_population(population, measurement_info, descent_info)



class Evosax(EvosaxBASE, RetrievePulsesCHIRPSCANwithRealFields):
    __doc__ = EvosaxBASE.__doc__

    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, f_range_fields=(None,None), solver=None, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, f_range_fields=f_range_fields, phase_type=phase_type, chirp_parameters=chirp_parameters, solver=solver, **kwargs)
        self._post_init()

    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """ Calls get_pulses_f_from_population() """
        return self.get_pulses_f_from_population(population, measurement_info, descent_info)



class LSF(LSFBASE, RetrievePulsesCHIRPSCANwithRealFields):
    __doc__ = LSFBASE.__doc__
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, f_range_fields=(None,None), **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, f_range_fields=f_range_fields, phase_type=phase_type, chirp_parameters=chirp_parameters, **kwargs)
        self._post_init()


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """ Returns the pulse and gate population. Does not need to call get_pulses_f_from_population() since LSF works with discetized fields only. """
        return population.pulse, population.gate
    

    def convert_population(self, population, measurement_info, descent_info):
        """ Converts any population into a discretized one. """
        pulse_arr, gate_arr = self.get_pulses_f_from_population(population, measurement_info, descent_info)
        return MyNamespace(pulse=pulse_arr, gate=gate_arr)
    



class AutoDiff(AutoDiffBASE, RetrievePulsesCHIRPSCANwithRealFields):
    __doc__ = AutoDiffBASE.__doc__

    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, f_range_fields=(None,None), solver=None, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, f_range_fields=f_range_fields, phase_type=phase_type, chirp_parameters=chirp_parameters, solver=solver, **kwargs)
        self._post_init()


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        return self.get_pulses_f_from_population(population, measurement_info, descent_info)
    


    def make_pulse_from_individual(self, individual, measurement_info, descent_info, pulse_or_gate):
        """ Evaluates a pulse/gate for an individual. """
        signal = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, pulse_or_gate)
        return signal
