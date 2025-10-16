from src.core.BaseClasses import RetrievePulses2DSI
from src.core.general_algorithms_base import DifferentialEvolutionBASE, EvosaxBASE, LSFBASE, AutoDiffBASE

from src.utilities import MyNamespace




class DifferentialEvolution(DifferentialEvolutionBASE, RetrievePulses2DSI):
    """ 
    The Differential Evolution Algorithm applied to 2DSI. Inherits from DifferentialEvolutionBASE and RetrievePulses2DSI.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """ Calls get_pulses_t_from_population() """
        return self.get_pulses_t_from_population(population, measurement_info, descent_info)
    





class Evosax(EvosaxBASE, RetrievePulses2DSI):
    """
    The Evosax package utilized for pulse reconstruction from 2DSI. Inherits from EvosaxBASE and RetrievePulses2DSI.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)


    def get_pulses_from_population(self, population, measurement_info, descent_info):
        """ Calls get_pulses_t_from_population() """
        return self.get_pulses_t_from_population(population, measurement_info, descent_info)
    





class LSF(LSFBASE, RetrievePulses2DSI):
    """
    The LSF Algorithm applied to 2DSI. Inherits from LSFBASE and RetrievePulses2DSI.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)



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
        

    






class AutoDiff(AutoDiffBASE, RetrievePulses2DSI):
    """
    The Optimistix package utilized for pulse reconstruction from 2DSI. Inherits from AutoDiffBASE and RetrievePulses2DSI.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)



    # def get_pulses_from_population(self, population, measurement_info, descent_info):
    #     return self.get_pulses_t_from_population(population, measurement_info, descent_info)
    


    def make_pulse_from_individual(self, individual, measurement_info, descent_info, pulse_or_gate):
        """ Evaluates a pulse/gate for an individual. """
        signal = self.make_pulse_t_from_individual(individual, measurement_info, descent_info, pulse_or_gate)
        return signal
    


    def post_process_get_pulse_and_gate(self, descent_state, measurement_info, descent_info):
        """ AD specific post processing because AD only works with an individual while the original post processing works on a population. """

        pulse_t = self.make_pulse_from_individual(descent_state.individual, measurement_info, descent_info, "pulse")
        pulse_f = self.fft(pulse_t, measurement_info.sk, measurement_info.rn)

        if measurement_info.doubleblind==True:
            gate_t = self.make_pulse_from_individual(descent_state.individual, measurement_info, descent_info, "gate")
            gate_f = self.fft(gate_t, measurement_info.sk, measurement_info.rn)
        else:
            gate_t, gate_f = pulse_t, pulse_f

        return pulse_t, gate_t, pulse_f, gate_f