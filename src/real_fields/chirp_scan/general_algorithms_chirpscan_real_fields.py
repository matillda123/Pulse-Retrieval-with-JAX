from src.real_fields.base_classes_methods import RetrievePulsesCHIRPSCANwithRealFields
from src.core.base_general_optimization import DifferentialEvolutionBASE, EvosaxBASE, LSFBASE, AutoDiffBASE

from src.utilities import MyNamespace


class DifferentialEvolution(DifferentialEvolutionBASE, RetrievePulsesCHIRPSCANwithRealFields):
    """ The Differential Evolution Algorithm applied to Chirp-Scans with real fields."""
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, f_range_fields=(None, None), phase_type=None, chirp_parameters=None, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, f_range_fields=f_range_fields, phase_type=phase_type, chirp_parameters=chirp_parameters, **kwargs)
        self._post_init()


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """ Calls get_pulses_f_from_population() """
        return self.get_pulses_f_from_population(population, measurement_info, descent_info)



class Evosax(EvosaxBASE, RetrievePulsesCHIRPSCANwithRealFields):
    """ The Evosax package applied to Chirp-Scans with real fields."""
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, f_range_fields=(None, None), phase_type=None, chirp_parameters=None, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, f_range_fields=f_range_fields, phase_type=phase_type, chirp_parameters=chirp_parameters, **kwargs)
        self._post_init()

    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """ Calls get_pulses_f_from_population() """
        return self.get_pulses_f_from_population(population, measurement_info, descent_info)



class LSF(LSFBASE, RetrievePulsesCHIRPSCANwithRealFields):
    """ The LSF Algorithm applied to Chirp-Scans with real fields. """
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, f_range_fields=(None, None), phase_type=None, chirp_parameters=None, **kwargs):
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
    """ The Optimistix package applied to Chirp-Scans with real fields. """
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, f_range_fields=(None, None), phase_type=None, chirp_parameters=None, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, f_range_fields=f_range_fields, phase_type=phase_type, chirp_parameters=chirp_parameters, **kwargs)
        self._post_init()


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        return self.get_pulses_f_from_population(population, measurement_info, descent_info)
    


    def make_pulse_from_individual(self, individual, measurement_info, descent_info, pulse_or_gate):
        """ Evaluates a pulse/gate for an individual. """
        signal = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, pulse_or_gate)
        return signal
