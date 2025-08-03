from BaseClasses import RetrievePulsesDSCAN, RetrievePulsesDSCANwithRealFields
from general_algorithms_base import DifferentialEvolutionBASE, EvosaxBASE, LSFBASE, AutoDiffBASE

from utilities import MyNamespace, do_fft, do_ifft


# RetrievePulsesDSCAN = RetrievePulsesDSCANwithRealFields




class DifferentialEvolution(DifferentialEvolutionBASE, RetrievePulsesDSCAN):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        return self.get_pulses_f_from_population(population, measurement_info, descent_info)
    





class Evosax(EvosaxBASE, RetrievePulsesDSCAN):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        return self.get_pulses_f_from_population(population, measurement_info, descent_info)
    





class LSF(LSFBASE, RetrievePulsesDSCAN):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        return population.pulse, population.gate
    

    def convert_population(self, population, measurement_info, descent_info):
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
        

    






class AutoDiff(AutoDiffBASE, RetrievePulsesDSCAN):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)


    # def get_pulses_from_population(self, population, measurement_info, descent_info):
    #     return self.get_pulses_t_from_population(population, measurement_info, descent_info)
    


    def make_pulse_from_individual(self, individual, measurement_info, descent_info, pulse_or_gate):
        signal = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, pulse_or_gate)
        return signal
    


    def post_process_get_pulse_and_gate(self, descent_state, measurement_info, descent_info):
        # needs to be overwritten because the original function works on a population

        pulse_f = self.make_pulse_from_individual(descent_state.individual, measurement_info, descent_info, "pulse")
        pulse_t = do_ifft(pulse_f, measurement_info.sk, measurement_info.rn)

        if measurement_info.doubleblind==True:
            gate_f = self.make_pulse_from_individual(descent_state.individual, measurement_info, descent_info, "gate")
            gate_t = do_ifft(gate_f, measurement_info.sk, measurement_info.rn)
        else:
            gate_t, gate_f = pulse_t, pulse_f

        return pulse_t, gate_t, pulse_f, gate_f