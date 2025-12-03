import jax.numpy as jnp
import jax

from equinox import tree_at

from src.utilities import MyNamespace, get_sk_rn, do_interpolation_1d, calculate_gate_with_Real_Fields, calculate_trace, center_signal
from src.core.base_classes_methods import RetrievePulsesFROG, RetrievePulsesTDP, RetrievePulsesCHIRPSCAN, RetrievePulses2DSI, RetrievePulsesVAMPIRE






# this is meant to be a parent to general optimization for real fields
# needs to come in first position to yield the correct mro
class RetrievePulsesRealFields:
    """  
    A Base-Class for reconstruction via real fields. Real fields need to be considered if multiple nonlinear signals are present in the same trace.
    A complex signal does not inherently express difference frequency generation. Because complex signals do not possess negative frequencies.
    This can only be used with general solvers, because for classical solvers analytic gradients/hessians are required. 
    Does not inherit from any class. But is supposed to be used via composition of its child classes with solver classes.

    Attributes:
        frequency_exp: jnp.array, the frequencies corresponding to the measured trace
        frequency: jnp.array, the frequencies correpsonding to pulse/gate-pulse
        frequency_big: jnp.array, a large frequency axis needed for the signal field due to negative frequencies
        time_big: jnp.array, the corresponding time axis to frequency_big
        sk_big: jnp.array, correction values for FFT->DFT
        rn_big: jnp.array, correction values for FFT->DFT

    """

    def __init__(self, *args, f_range_fields=(None, None), **kwargs):
        self._fmin, self._fmax = f_range_fields
        super().__init__(*args, **kwargs)

        self.measurement_info = self.measurement_info.expand(frequency_exp = self.frequency_exp, 
                                                             real_fields = True)
        

        f = jnp.abs(self.measurement_info.frequency_exp)
        df = jnp.mean(jnp.diff(self.measurement_info.frequency_exp))
        self.frequency_big = jnp.arange(-1*jnp.max(f), jnp.max(f)+df, df)
        self.time_big = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(self.frequency_big), jnp.mean(jnp.diff(self.frequency_big))))
        self.sk_big, self.rn_big = get_sk_rn(self.time_big, self.frequency_big)
        self.measurement_info = self.measurement_info.expand(time_big=self.time_big, frequency_big=self.frequency_big, sk_big=self.sk_big, rn_big=self.rn_big)



    def get_data(self, x_arr, frequency, measured_trace):
        """ Prepare/Convert data. """
        measured_trace = measured_trace/jnp.linalg.norm(measured_trace)

        self.x_arr = jnp.asarray(x_arr)

        self.frequency_exp = jnp.asarray(frequency)
        df = jnp.mean(jnp.diff(jnp.asarray(frequency)))
        self.frequency = jnp.arange(self._fmin, self._fmax+df, df)
        N = jnp.size(self.frequency)
        
        self.time = jnp.fft.fftshift(jnp.fft.fftfreq(N, df))
        self.measured_trace = jnp.asarray(measured_trace)

        return self.x_arr, self.time, self.frequency, self.measured_trace
    



    def construct_trace(self, individual, measurement_info, descent_info):
        """ Generates a trace for a given individual. Calls the method specific function for calculating the nonlinear signal fields. """
        x_arr = measurement_info.x_arr
        frequency_exp = measurement_info.frequency_exp
        frequency_big = measurement_info.frequency_big
        sk_big, rn_big = measurement_info.sk_big, measurement_info.rn_big

        
        signal_t = self.calculate_signal_t(individual, measurement_info.transform_arr, measurement_info)
        signal_f = self.fft(signal_t.signal_t, sk_big, rn_big)
        trace = calculate_trace(signal_f)

        trace = do_interpolation_1d(frequency_exp, frequency_big, trace.T, method="linear").T
        return x_arr, frequency_exp, trace
    



    def post_process_center_pulse_and_gate(self, pulse_t, gate_t):
        """ This essentially removes the linear phase. But only approximately since no fits are done. """
        sk_big, rn_big = self.measurement_info.sk_big, self.measurement_info.rn_big

        pulse_t = center_signal(pulse_t)
        gate_t = center_signal(gate_t)

        pulse_f = self.fft(pulse_t, sk_big, rn_big)
        gate_f = self.fft(gate_t, sk_big, rn_big)

        return pulse_t, gate_t, pulse_f, gate_f
    

    

    def post_process_create_trace(self, individual):
        """ Post processing to get the final trace """
        sk_big, rn_big = self.measurement_info.sk_big, self.measurement_info.rn_big
        transform_arr = self.measurement_info.transform_arr
    
        signal_t = self.calculate_signal_t(individual, transform_arr, self.measurement_info)
        signal_f = self.fft(signal_t.signal_t, sk_big, rn_big)
        trace = calculate_trace(signal_f)
        return trace


    def post_process(self, descent_state, error_arr):
        """ Creates the final_result object from the final descent_state. """
        final_result = super().post_process(descent_state, error_arr)

        frequency_exp, frequency, frequency_big = self.measurement_info.frequency_exp, self.measurement_info.frequency, self.measurement_info.frequency_big
        trace = final_result.trace
        trace = do_interpolation_1d(frequency_exp, frequency_big, trace.T, method="linear").T
        trace = trace/jnp.linalg.norm(trace)

        final_result = final_result.expand(frequency_exp = self.measurement_info.frequency_exp, 
                                           trace = trace)


        pulse_f = do_interpolation_1d(frequency, frequency_big, final_result.pulse_f)
        pulse_t = self.ifft(pulse_f, self.measurement_info.sk, self.measurement_info.rn)
        final_result = final_result.expand(pulse_t=pulse_t, pulse_f=pulse_f)

        if self.measurement_info.doubleblind==True:
            gate_f = do_interpolation_1d(frequency, frequency_big, final_result.gate_f)
            gate_t = self.ifft(gate_f, self.measurement_info.sk, self.measurement_info.rn)
            final_result = final_result.expand(gate_t=gate_t, gate_f=gate_f)

        return final_result
    



    def make_pulse_f_from_individual(self, individual, measurement_info, descent_info, pulse_or_gate="pulse"):
        """ Evaluates an individual onto the frequency domain. Interpolates it onto frequency_big. """
        signal_f = super().make_pulse_f_from_individual(individual, measurement_info, descent_info, pulse_or_gate=pulse_or_gate)
        
        frequency_big, frequency = measurement_info.frequency_big, measurement_info.frequency
        signal_f = do_interpolation_1d(frequency_big, frequency, signal_f)
        return signal_f
    

    def make_pulse_t_from_individual(self, individual, measurement_info, descent_info, pulse_or_gate="pulse"):
        """ Evaluates an individual onto the time domain. Onto time_big. """
        signal_f = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, pulse_or_gate)
        signal = self.ifft(signal_f, measurement_info.sk_big, measurement_info.rn_big)
        return signal
    






class RetrievePulsesFROGwithRealFields(RetrievePulsesFROG):
    """ 
    Inherits from RetrievePulsesFROG. Has the same purpose. It overwrites the generation of the 
    signal field in order to use real fields instead of complex ones.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




    def get_gate_pulse(self, frequency, gate_f):
        """ For crosscorrelation=True the actual gate pulse has to be provided. """
        gate_f = do_interpolation_1d(self.measurement_info.frequency_big, frequency, gate_f)
        self.gate = self.ifft(gate_f, self.measurement_info.sk_big, self.measurement_info.rn_big)
        self.measurement_info = self.measurement_info.expand(gate = self.gate)
        return self.gate


        
    def calculate_signal_t(self, individual, tau_arr, measurement_info):
        """
        Calculates the signal field of a FROG in the time domain. Does so by using real fields instead of complex ones.

        Args:
            individual: Pytree, a population containing only one member. (jax.vmap over whole population)
            tau_arr: jnp.array, the delays
            measurement_info: Pytree, contains the measurement parameters (e.g. nonlinear method, interferometric, ... )

        Returns:
            Pytree, contains the signal field in the time domain as well as the fields used to calculate it.
        """

        time_big, frequency_big = measurement_info.time_big, measurement_info.frequency_big
        cross_correlation, doubleblind, ifrog = measurement_info.cross_correlation, measurement_info.doubleblind, measurement_info.ifrog
        frogmethod = measurement_info.nonlinear_method

        pulse, gate = individual.pulse, individual.gate


        pulse_t_shifted = self.calculate_shifted_signal(pulse, frequency_big, tau_arr, time_big)

        if cross_correlation==True:
            gate_pulse_shifted = self.calculate_shifted_signal(measurement_info.gate, frequency_big, tau_arr, time_big)
            gate_shifted = calculate_gate_with_Real_Fields(gate_pulse_shifted, frogmethod)

        elif doubleblind==True:
            gate_pulse_shifted = self.calculate_shifted_signal(gate, frequency_big, tau_arr, time_big)
            gate_shifted = calculate_gate_with_Real_Fields(gate_pulse_shifted, frogmethod)

        else:
            gate_pulse_shifted = None
            gate_shifted = calculate_gate_with_Real_Fields(pulse_t_shifted, frogmethod)


        if ifrog==True and cross_correlation==False and doubleblind==False:
            signal_t = jnp.real(pulse + pulse_t_shifted)*calculate_gate_with_Real_Fields(pulse + pulse_t_shifted, frogmethod)
        elif ifrog==True:
            signal_t = jnp.real(pulse + gate_pulse_shifted)*calculate_gate_with_Real_Fields(pulse + gate_pulse_shifted, frogmethod)
        else:
            signal_t = jnp.real(pulse)*gate_shifted
            

        signal_t = MyNamespace(signal_t = signal_t, 
                               pulse_t_shifted = pulse_t_shifted, 
                               gate_shifted = gate_shifted, 
                               gate_pulse_shifted = gate_pulse_shifted)
        return signal_t
    


    



class RetrievePulsesTDPwithRealFields(RetrievePulsesTDP):
    """ 
    Inherits from RetrievePulsesFROG. Has the same purpose. It overwrites the generation of the 
    signal field in order to use real fields instead of complex ones.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _post_init(self):
        """ 
        The phase-matrix needs to be interpolated onto frequency_big. 
        Overwriting its creation would be possible but a bit cumbersome. 
        """
        frequency, frequency_big = self.frequency, self.measurement_info.frequency_big
        self.spectral_filter = do_interpolation_1d(frequency_big, frequency, self.spectral_filter)
        self.measurement_info = tree_at(lambda x: x.spectral_filter, self.measurement_info, self.spectral_filter)



    def get_gate_pulse(self, frequency, gate_f):
        """ For crosscorrelation=True the actual gate pulse has to be provided. """
        gate_f = do_interpolation_1d(self.measurement_info.frequency_big, frequency, gate_f)
        self.gate = self.ifft(gate_f, self.measurement_info.sk_big, self.measurement_info.rn_big)
        self.measurement_info = self.measurement_info.expand(gate = self.gate)
        return self.gate


    def calculate_signal_t(self, individual, tau_arr, measurement_info):
        """
        Calculates the signal field of a FROG in the time domain. Does so by using real fields instead of complex ones.

        Args:
            individual: Pytree, a population containing only one member. (jax.vmap over whole population)
            tau_arr: jnp.array, the delays
            measurement_info: Pytree, contains the measurement parameters (e.g. nonlinear method, interferometric, ... )

        Returns:
            Pytree, contains the signal field in the time domain as well as the fields used to calculate it.
        """

        time_big, frequency_big = measurement_info.time_big, measurement_info.frequency_big
        cross_correlation, doubleblind, ifrog = measurement_info.cross_correlation, measurement_info.doubleblind, measurement_info.ifrog
        frogmethod = measurement_info.nonlinear_method
        sk_big, rn_big = measurement_info.sk_big, measurement_info.rn_big

        pulse, gate = individual.pulse, individual.gate

        pulse_t_shifted = self.calculate_shifted_signal(pulse, frequency_big, tau_arr, time_big)

        if cross_correlation==True:
            gate = self.apply_spectral_filter(measurement_info.gate, measurement_info.spectral_filter, sk_big, rn_big)
            gate_pulse_shifted = self.calculate_shifted_signal(gate, frequency_big, tau_arr, time_big)
            gate_shifted = calculate_gate_with_Real_Fields(gate_pulse_shifted, frogmethod)

        elif doubleblind==True:
            gate = self.apply_spectral_filter(gate, measurement_info.spectral_filter, sk_big, rn_big)
            gate_pulse_shifted = self.calculate_shifted_signal(gate, frequency_big, tau_arr, time_big)
            gate_shifted = calculate_gate_with_Real_Fields(gate_pulse_shifted, frogmethod)

        else:
            gate_pulse_shifted = None
            pulse_t_shifted = self.apply_spectral_filter(pulse_t_shifted, measurement_info.spectral_filter, sk_big, rn_big)
            gate_shifted = calculate_gate_with_Real_Fields(pulse_t_shifted, frogmethod)


        if ifrog==True and cross_correlation==False and doubleblind==False:
            signal_t = jnp.real(pulse + pulse_t_shifted)*calculate_gate_with_Real_Fields(pulse + pulse_t_shifted, frogmethod)
        elif ifrog==True:
            signal_t = jnp.real(pulse + gate_pulse_shifted)*calculate_gate_with_Real_Fields(pulse + gate_pulse_shifted, frogmethod)
        else:
            signal_t = jnp.real(pulse)*gate_shifted
            

        signal_t = MyNamespace(signal_t = signal_t, 
                               pulse_t_shifted = pulse_t_shifted, 
                               gate_shifted = gate_shifted, 
                               gate_pulse_shifted = gate_pulse_shifted)
        return signal_t
    






class RetrievePulsesCHIRPSCANwithRealFields(RetrievePulsesCHIRPSCAN):
    """ 
    Inherits from RetrievePulsesCHIRPSCAN. Has the same purpose. It overwrites the generation of the 
    signal field in order to use real fields instead of complex ones.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def _post_init(self):
        """ 
        The phase-matrix needs to be interpolated onto frequency_big. 
        Overwriting its creation would be possible but a bit cumbersome. 
        """
        frequency, frequency_big = self.measurement_info.frequency, self.measurement_info.frequency_big
        self.phase_matrix = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(frequency_big, frequency, self.phase_matrix)
        self.transform_arr = self.phase_matrix
        self.measurement_info = tree_at(lambda x: x.phase_matrix, self.measurement_info, self.phase_matrix)
        self.measurement_info = tree_at(lambda x: x.transform_arr, self.measurement_info, self.transform_arr)
    

    def calculate_signal_t(self, individual, phase_matrix, measurement_info):
        """
        Calculates the signal field of a Chirp-Scan in the time domain. Does so by using real fields instead of complex ones.

        Args:
            individual: Pytree, a population containing only one member. (jax.vmap over whole population)
            phase_matrix: jnp.array, the applied phases
            measurement_info: Pytree, contains the measurement parameters (e.g. nonlinear method, ... )

        Returns:
            Pytree, contains the signal field in the time domain as well as the fields used to calculate it.
        """

        pulse = individual.pulse

        pulse_t_disp, phase_matrix = self.get_dispersed_pulse_t(pulse, phase_matrix, measurement_info.sk_big, measurement_info.rn_big)
        gate_disp = calculate_gate_with_Real_Fields(pulse_t_disp, measurement_info.nonlinear_method)
        signal_t = jnp.real(pulse_t_disp)*gate_disp

        signal_t = MyNamespace(signal_t = signal_t, 
                               pulse_t_disp = pulse_t_disp, 
                               gate_disp = gate_disp)
        return signal_t







class RetrievePulses2DSIwithRealFields(RetrievePulses2DSI):
    """ 
    Inherits from RetrievePulses2DSI. Has the same purpose. It overwrites the generation of the 
    signal field in order to use real fields instead of complex ones.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _post_init(self):
        """ 
        The phase-matrix needs to be interpolated onto frequency_big. 
        Overwriting its creation would be possible but a bit cumbersome. 
        """
        frequency_exp, frequency_big = self.measurement_info.frequency_exp, self.measurement_info.frequency_big
        self.phase_matrix = do_interpolation_1d(frequency_big, frequency_exp, self.phase_matrix)
        self.measurement_info = tree_at(lambda x: x.phase_matrix, self.measurement_info, self.phase_matrix)

        self.spectral_filter1 = do_interpolation_1d(frequency_big, frequency_exp, self.spectral_filter1)
        self.spectral_filter2 = do_interpolation_1d(frequency_big, frequency_exp, self.spectral_filter2)
        self.measurement_info = tree_at(lambda x: x.spectral_filter1, self.measurement_info, self.spectral_filter1)
        self.measurement_info = tree_at(lambda x: x.spectral_filter2, self.measurement_info, self.spectral_filter2)




    def get_gate_pulse(self, frequency, gate_f):
        """ For crosscorrelation=True the actual gate pulse has to be provided. """
        gate_f = do_interpolation_1d(self.measurement_info.frequency_big, frequency, gate_f)
        self.gate = self.ifft(gate_f, self.measurement_info.sk_big, self.measurement_info.rn_big)
        self.measurement_info = self.measurement_info.expand(gate = self.gate)
        return self.gate


    def calculate_signal_t(self, individual, tau_arr, measurement_info):
        """
        Calculates the signal field of 2DSI in the time domain. Does so by using real fields instead of complex ones.

        Args:
            individual: Pytree, a population containing only one member. (jax.vmap over whole population)
            tau_arr: jnp.array, the delays
            measurement_info: Pytree, contains the measurement parameters (e.g. nonlinear method, ... )

        Returns:
            Pytree, contains the signal field in the time domain as well as the fields used to calculate it.
        """

        time_big, frequency_big = measurement_info.time_big, measurement_info.frequency_big
        sk_big, rn_big = measurement_info.sk_big, measurement_info.rn_big
        nonlinear_method = measurement_info.nonlinear_method

        pulse_t = individual.pulse

        if measurement_info.cross_correlation==True:
            gate = measurement_info.gate

        elif measurement_info.doubleblind==True:
            gate = individual.gate

        else:
            gate = pulse_t
        
        # shift in time is solved, by jnp.roll -> isnt exact
        gate, delay = self.apply_phase(gate, measurement_info, sk_big, rn_big) 

        gate1 = self.apply_spectral_filter(gate, measurement_info.spectral_filter1, sk_big, rn_big)
        gate2 = self.apply_spectral_filter(gate, measurement_info.spectral_filter2, sk_big, rn_big)
            
        gate2_shifted = self.calculate_shifted_signal(gate2, frequency_big, tau_arr, time_big)
        tau = measurement_info.tau_pulse_anc1
        gate1 = self.calculate_shifted_signal(gate1, frequency_big, jnp.asarray([tau]), time_big)
        gate_pulses = jnp.squeeze(gate1) + gate2_shifted
        gate = calculate_gate_with_Real_Fields(gate_pulses, nonlinear_method)

        signal_t = jnp.real(pulse_t)*gate

        signal_t = MyNamespace(signal_t=signal_t, gate_pulses=gate_pulses, gate=gate, delay=delay)
        return signal_t





























class RetrievePulsesVAMPIREwithRealFields(RetrievePulsesVAMPIRE):
    """ 
    Inherits from RetrievePulses2DSI. Has the same purpose. It overwrites the generation of the 
    signal field in order to use real fields instead of complex ones.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _post_init(self):
        """ 
        The phase-matrix needs to be interpolated onto frequency_big. 
        Overwriting its creation would be possible but a bit cumbersome. 
        """
        frequency_exp, frequency_big = self.measurement_info.frequency_exp, self.measurement_info.frequency_big
        self.phase_matrix = do_interpolation_1d(frequency_big, frequency_exp, self.phase_matrix)
        self.measurement_info = tree_at(lambda x: x.phase_matrix, self.measurement_info, self.phase_matrix)





    def calculate_signal_t(self, individual, tau_arr, measurement_info):
        """
        Calculates the signal field in the time domain. 

        Args:
            individual: Pytree, a population containing only one member. (jax.vmap over whole population)
            tau_arr: jnp.array, the delays
            measurement_info: Pytree, contains the measurement parameters (e.g. nonlinear method, ... )

        Returns:
            Pytree, contains the signal field in the time domain as well as the fields used to calculate it.
        """

        time_big, frequency_big = measurement_info.time_big, measurement_info.frequency_big
        sk_big, rn_big = measurement_info.sk_big, measurement_info.rn_big
        nonlinear_method = measurement_info.nonlinear_method

        pulse_t = individual.pulse

        if measurement_info.cross_correlation==True:
            gate_pulse = measurement_info.gate

        elif measurement_info.doubleblind==True:
            gate_pulse = individual.gate

        else:
            gate_pulse = pulse_t

        gate_disp, delay = self.apply_phase(gate_pulse, measurement_info, sk_big, rn_big) 

        tau = measurement_info.tau_interferometer
        gate_pulse = self.calculate_shifted_signal(gate_pulse, frequency_big, jnp.asarray([tau]), time_big)

        gate_pulses = jnp.squeeze(gate_pulse) + gate_disp
        gate_pulses = self.calculate_shifted_signal(gate_pulses, frequency_big, tau_arr, time_big)
        gate = calculate_gate_with_Real_Fields(gate_pulses, nonlinear_method)

        signal_t = jnp.real(pulse_t)*gate

        signal_t = MyNamespace(signal_t=signal_t, gate_pulses=gate_pulses, gate=gate, delay=delay)
        return signal_t