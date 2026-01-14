from pulsedjax.core.base_classes_methods import RetrievePulsesCHIRPSCAN
from pulsedjax.core.base_general_optimization import DifferentialEvolutionBASE, EvosaxBASE, LSFBASE, AutoDiffBASE

from pulsedjax.utilities import MyNamespace



class DifferentialEvolution(DifferentialEvolutionBASE, RetrievePulsesCHIRPSCAN):
    __doc__ = DifferentialEvolutionBASE.__doc__

    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, 
                 strategy="best1_bin", selection_mechanism="greedy", mutation_rate=0.5, crossover_rate=0.7, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters,
                         strategy=strategy, selection_mechanism=selection_mechanism, mutation_rate=mutation_rate, crossover_rate=crossover_rate, **kwargs)


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """ Calls get_pulses_f_from_population() """
        return self.get_pulses_f_from_population(population, measurement_info, descent_info)
    





class Evosax(EvosaxBASE, RetrievePulsesCHIRPSCAN):
    __doc__ = EvosaxBASE.__doc__

    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, solver=None, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters, solver=solver, **kwargs)


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """ Calls get_pulses_f_from_population() """
        return self.get_pulses_f_from_population(population, measurement_info, descent_info)
    





class LSF(LSFBASE, RetrievePulsesCHIRPSCAN):
    __doc__ = LSFBASE.__doc__

    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters, **kwargs)


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """ Returns the pulse and gate population. Does not need to call get_pulses_f_from_population() since LSF works with discetized fields only. """
        return population.pulse, population.gate
    

    def convert_population(self, population, measurement_info, descent_info):
        """ Converts any population into a discretized one. """
        pulse_arr, gate_arr = self.get_pulses_f_from_population(population, measurement_info, descent_info)
        return MyNamespace(pulse=pulse_arr, gate=gate_arr)
    

    # These Two Methods should be used when attempting to do the dscan optimization with the pulses in the time domain
    # -> weirdly it works even worse than doing it in the frequency domain
    #
    # def get_pulses_from_population(self, population, measurement_info, descent_info):
    #     sk, rn = measurement_info.sk, measurement_info.rn
    #     pulse_t_arr, gate_t_arr = population.pulse, population.gate
    #     pulse_f_arr = do_fft(pulse_t_arr, sk, rn)

    #     if measurement_info.doubleblind==True:
    #         gate_f_arr = do_fft(gate_t_arr, sk, rn)
    #     else:
    #         gate_f_arr = None

    #     return pulse_f_arr, gate_f_arr
    

    # def convert_population(self, population, measurement_info, descent_info):
    #     pulse_arr, gate_arr = self.get_pulses_t_from_population(population, measurement_info, descent_info)
    #     return MyNamespace(pulse=pulse_arr, gate=gate_arr)
        

    






class AutoDiff(AutoDiffBASE, RetrievePulsesCHIRPSCAN):
    __doc__ = AutoDiffBASE.__doc__
    
    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, solver=None, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters, solver=solver, **kwargs)


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        return self.get_pulses_f_from_population(population, measurement_info, descent_info)
    


    def make_pulse_from_individual(self, individual, measurement_info, descent_info, pulse_or_gate):
        """ Evaluates a pulse/gate for an individual. """
        signal = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, pulse_or_gate)
        return signal
    