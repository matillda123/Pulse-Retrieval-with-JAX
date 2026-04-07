import jax.numpy as jnp
import jax
from jax.tree_util import Partial

from equinox import tree_at

from pulsedjax.utilities import MyNamespace, get_sk_rn, do_interpolation_1d, calculate_gate_with_Real_Fields
from pulsedjax.core.base_classes_methods import RetrievePulses, RetrievePulsesFROG, RetrievePulsesTDP, RetrievePulsesCHIRPSCAN, RetrievePulses2DSI, RetrievePulsesVAMPIRE





class RetrievePulsesRealFields(RetrievePulses):
    """  
    A Base-Class for reconstruction via real fields. Real fields need to be considered if multiple nonlinear signals are present in the same trace.
    A complex signal does not inherently express difference frequency generation. Because complex signals do not possess negative frequencies.
    
    Attributes:
        frequency (jnp.array): the frequencies correpsonding to pulse/gate-pulse
        frequency_big (jnp.array): a large frequency axis needed for the signal field due to negative frequencies
        time_big (jnp.array): the corresponding time axis to frequency_big
        sk_big (jnp.array): correction values for FFT->DFT
        rn_big (jnp.array): correction values for FFT->DFT

    """


    def __init__(self, *args, f_range_fields=(None, None), f_range_pulse=(None,None), f_max_all_fields=None, **kwargs):
        self._fmin_fields, self._fmax_fields = f_range_fields
        self._fmin_pulse, self._fmax_pulse = f_range_pulse
        assert self._fmin_fields!=None and self._fmax_fields!=None, "f_range_fields needs to be provided"
        assert self._fmin_pulse!=None and self._fmax_pulse!=None, "f_range_pulse needs to be provided"

        self.f_max_all_fields = f_max_all_fields
        if self.f_max_all_fields==None: 
            print(f"The maximum nonlinear signal is assumed to be the maximum of the provided frequency_range.")
        
        super().__init__(*args, **kwargs)

        self.measurement_info = self.measurement_info.expand(real_fields = True)
        

    def get_data(self, theta, frequency_exp, measured_trace):
        """ Prepare/Convert data. """
        self.measured_trace = measured_trace/jnp.linalg.norm(measured_trace)

        self.theta = jnp.asarray(theta)
        df = jnp.mean(jnp.diff(jnp.asarray(frequency_exp)))

        self.frequency_exp = jnp.copy(frequency_exp)
        self.time_exp = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(self.frequency_exp), df))
        self.dt_exp = jnp.mean(jnp.diff(self.time_exp))
        self.df_exp = jnp.mean(jnp.diff(self.frequency_exp))
        self.sk_exp, self.rn_exp = get_sk_rn(self.time_exp, self.frequency_exp)

        self.frequency = jnp.linspace(self._fmin_pulse, self._fmax_pulse, jnp.size(self.frequency_exp))
        self.time = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(self.frequency), jnp.mean(jnp.diff(self.frequency))))
        self.dt = jnp.mean(jnp.diff(self.time))
        self.df = jnp.mean(jnp.diff(self.frequency))
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)


        if self.f_max_all_fields!=None:
            max_f = self.f_max_all_fields
        else:
            max_f = jnp.max(self.frequency_exp)
        self.frequency_big = jnp.arange(-1*max_f, max_f+df, df)
        self.time_big = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(self.frequency_big), jnp.mean(jnp.diff(self.frequency_big))))
        self.dt_big = jnp.mean(jnp.diff(self.time_big))
        self.df_big = jnp.mean(jnp.diff(self.frequency_big))
        self.sk_big, self.rn_big = get_sk_rn(self.time_big, self.frequency_big)
        

        mask = jnp.zeros(jnp.size(self.frequency_big))
        idx0, idx1 = (jnp.argmin(jnp.abs(self.frequency_big - self._fmin_fields)), 
                      jnp.argmin(jnp.abs(self.frequency_big - self._fmax_fields)))
        self.mask = mask.at[idx0:idx1+1].set(1)

        self.measurement_info = self.measurement_info.expand(time_exp=self.time_exp, frequency_exp=self.frequency_exp, 
                                                             sk_exp=self.sk_exp, rn_exp=self.rn_exp, 
                                                             dt_exp=self.dt_exp, df_exp=self.df_exp,

                                                             time=self.time, frequency=self.frequency, 
                                                             sk=self.sk, rn=self.rn, 
                                                             dt=self.dt, df=self.df,

                                                             time_big=self.time_big, frequency_big=self.frequency_big, 
                                                             sk_big=self.sk_big, rn_big=self.rn_big, 
                                                             dt_big=self.dt_big, df_big=self.df_big,
                                                             
                                                             mask = self.mask)
        
        self.measured_trace = self.interpolate_signal_f(self.measured_trace, self.measurement_info, "exp", "big")
        return self.theta, self.time, self.frequency, self.measured_trace
    


    def get_gate_pulse(self, frequency, gate_f):
        """ For crosscorrelation=True the actual gate pulse has to be provided. """
        gate_f = do_interpolation_1d(self.measurement_info.frequency_big, frequency, gate_f)
        self.gate = self.ifft(gate_f, self.measurement_info.sk_big, self.measurement_info.rn_big)
        self.measurement_info = self.measurement_info.expand(gate = self.gate)
        return self.gate
    


    def interpolate_signal_f(self, signal_f, measurement_info, axis_in, axis_out, batch_axes=-2):
        axis_dict = {"main": measurement_info.frequency,
                     "exp": measurement_info.frequency_exp,
                     "big": measurement_info.frequency_big}
        
        frequency_1 = axis_dict[axis_in]
        frequency_2 = axis_dict[axis_out]
        
        interpolate = Partial(do_interpolation_1d, method="linear")
        if signal_f.ndim==0:
            raise ValueError
        elif signal_f.ndim==1:
            signal_f = interpolate(frequency_2, frequency_1, signal_f)
        else:
            signal_f = jax.vmap(interpolate, 
                                in_axes = (None,None,batch_axes), 
                                out_axes = batch_axes)(frequency_2, frequency_1, signal_f)
        
        return signal_f


    def interpolate_signal_t(self, signal_t, measurement_info, axis_in, axis_out, batch_axes=-2):
        axis_dict = {"main": (measurement_info.sk, measurement_info.rn),
                     "exp": (measurement_info.sk_exp, measurement_info.rn_exp),
                     "big": (measurement_info.sk_big, measurement_info.rn_big)}
        
        sk_1, rn_1 = axis_dict[axis_in]
        sk_2, rn_2 = axis_dict[axis_out]

        signal_f = self.fft(signal_t, sk_1, rn_1)
        signal_f = self.interpolate_signal_f(signal_f, measurement_info, axis_in, axis_out, batch_axes=batch_axes)
        signal_t = self.ifft(signal_f, sk_2, rn_2)
        return signal_t, signal_f
    

    def apply_mask(self, signal_t, measurement_info):
        signal_f = self.fft(signal_t, measurement_info.sk_big, measurement_info.rn_big)
        signal_f = signal_f*measurement_info.mask
        signal_t = self.ifft(signal_f, measurement_info.sk_big, measurement_info.rn_big)
        return signal_t, signal_f
    



    

    

    

    






class RetrievePulsesFROGwithRealFields(RetrievePulsesRealFields, RetrievePulsesFROG):
    """ 
    Overwrites the generation of the signal field in order to use real fields instead of complex ones.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        
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
        cross_correlation, doubleblind, interferometric = measurement_info.cross_correlation, measurement_info.doubleblind, measurement_info.interferometric
        frogmethod = measurement_info.nonlinear_method

        pulse_f, gate_f = individual.pulse, individual.gate

        pulse_f = self.interpolate_signal_f(pulse_f, measurement_info, "main", "big")
        pulse_t = self.ifft(pulse_f, measurement_info.sk_big, measurement_info.rn_big)
        

        pulse_t_shifted = self.calculate_shifted_signal(pulse_t, frequency_big, tau_arr, time_big)

        if cross_correlation==True:
            gate_t = measurement_info.gate
            gate_pulse_shifted = self.calculate_shifted_signal(gate_t, frequency_big, tau_arr, time_big)
            gate_shifted = calculate_gate_with_Real_Fields(gate_pulse_shifted, frogmethod)

        elif doubleblind==True:
            gate_f = self.interpolate_signal_f(gate_f, measurement_info, "main", "big")
            gate_t = self.ifft(gate_f, measurement_info.sk_big, measurement_info.rn_big)

            gate_pulse_shifted = self.calculate_shifted_signal(gate_t, frequency_big, tau_arr, time_big)
            gate_shifted = calculate_gate_with_Real_Fields(gate_pulse_shifted, frogmethod)

        else:
            gate_pulse_shifted, gate_t = None, None
            gate_shifted = calculate_gate_with_Real_Fields(pulse_t_shifted, frogmethod)


        if interferometric==True and cross_correlation==False and doubleblind==False:
            signal_t = jnp.real(pulse_t + pulse_t_shifted)*calculate_gate_with_Real_Fields(pulse_t + pulse_t_shifted, frogmethod)
        elif interferometric==True:
            signal_t = jnp.real(pulse_t + gate_pulse_shifted)*calculate_gate_with_Real_Fields(pulse_t + gate_pulse_shifted, frogmethod)
        else:
            signal_t = jnp.real(pulse_t)*gate_shifted

        signal_t, signal_f = self.apply_mask(signal_t, measurement_info)
        pulse_t_shifted = jnp.real(pulse_t_shifted)

        if doubleblind==True or cross_correlation==True:
            gate_pulse_shifted = jnp.real(gate_pulse_shifted)

        signal_t = MyNamespace(signal_t = signal_t, 
                               signal_f = signal_f,
                               pulse_t_shifted = pulse_t_shifted, 
                               gate_shifted = gate_shifted, 
                               gate_pulse_shifted = gate_pulse_shifted,
                               pulse_t = jnp.real(pulse_t))
        return signal_t
    


    



class RetrievePulsesTDPwithRealFields(RetrievePulsesRealFields, RetrievePulsesTDP):
    """ 
    Overwrites the generation of the signal field in order to use real fields instead of complex ones.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _post_init(self):
        """ 
        The phase-matrix needs to be interpolated onto frequency_big. 
        Overwriting its creation would be possible but a bit cumbersome. 
        """
        frequency, frequency_big = self.measurement_info.frequency, self.measurement_info.frequency_big
        self.spectral_filter = do_interpolation_1d(frequency_big, frequency, self.spectral_filter)
        self.measurement_info = tree_at(lambda x: x.spectral_filter, self.measurement_info, self.spectral_filter)



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
        cross_correlation, doubleblind, interferometric = measurement_info.cross_correlation, measurement_info.doubleblind, measurement_info.interferometric
        frogmethod = measurement_info.nonlinear_method
        sk_big, rn_big = measurement_info.sk_big, measurement_info.rn_big

        pulse_f, gate_f = individual.pulse, individual.gate
        pulse_f = self.interpolate_signal_f(pulse_f, measurement_info, "main", "big")
        pulse_t = self.ifft(pulse_f, sk_big, rn_big)


        pulse_t_shifted = self.calculate_shifted_signal(pulse_t, frequency_big, tau_arr, time_big)

        if cross_correlation==True:
            # might break
            gate_t = self.apply_spectral_filter(measurement_info.gate, measurement_info.spectral_filter, sk_big, rn_big)
            
            gate_pulse_shifted = self.calculate_shifted_signal(gate_t, frequency_big, tau_arr, time_big)
            gate_shifted = calculate_gate_with_Real_Fields(gate_pulse_shifted, frogmethod)

        elif doubleblind==True:
            gate_f = self.interpolate_signal_f(gate_f, measurement_info, "main", "big")
            gate_t = self.ifft(gate_f, sk_big, rn_big)
            gate_t = self.apply_spectral_filter(gate_t, measurement_info.spectral_filter, sk_big, rn_big)
            gate_pulse_shifted = self.calculate_shifted_signal(gate_t, frequency_big, tau_arr, time_big)
            gate_shifted = calculate_gate_with_Real_Fields(gate_pulse_shifted, frogmethod)

        else:
            gate_pulse_shifted, gate_t = None, None
            pulse_t_shifted = self.apply_spectral_filter(pulse_t_shifted, measurement_info.spectral_filter, sk_big, rn_big)
            gate_shifted = calculate_gate_with_Real_Fields(pulse_t_shifted, frogmethod)


        if interferometric==True and cross_correlation==False and doubleblind==False:
            signal_t = jnp.real(pulse_t + pulse_t_shifted)*calculate_gate_with_Real_Fields(pulse_t + pulse_t_shifted, frogmethod)
        elif interferometric==True:
            signal_t = jnp.real(pulse_t + gate_pulse_shifted)*calculate_gate_with_Real_Fields(pulse_t + gate_pulse_shifted, frogmethod)
        else:
            signal_t = jnp.real(pulse_t)*gate_shifted

        signal_t, signal_f = self.apply_mask(signal_t, measurement_info)
        pulse_t_shifted = jnp.real(pulse_t_shifted)

        if doubleblind==True or cross_correlation==True:
            gate_pulse_shifted = jnp.real(gate_pulse_shifted)

        signal_t = MyNamespace(signal_t = signal_t, 
                               signal_f = signal_f,
                               pulse_t_shifted = pulse_t_shifted, 
                               gate_shifted = gate_shifted, 
                               gate_pulse_shifted = gate_pulse_shifted,
                               pulse_t = jnp.real(pulse_t))
        return signal_t
    






class RetrievePulsesCHIRPSCANwithRealFields(RetrievePulsesRealFields, RetrievePulsesCHIRPSCAN):
    """ 
    Overwrites the generation of the signal field in order to use real fields instead of complex ones.
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

        pulse_f = individual.pulse
        pulse_f = self.interpolate_signal_f(pulse_f, measurement_info, "main", "big")

        pulse_t_disp, phase_matrix = self.get_dispersed_pulse_t(pulse_f, phase_matrix, measurement_info.sk_big, measurement_info.rn_big)
        gate_disp = calculate_gate_with_Real_Fields(pulse_t_disp, measurement_info.nonlinear_method)
        signal_t = jnp.real(pulse_t_disp)*gate_disp

        signal_t, signal_f = self.apply_mask(signal_t, measurement_info)
        signal_t = MyNamespace(signal_t = signal_t, 
                               signal_f = signal_f,
                               pulse_t_disp = jnp.real(pulse_t_disp), 
                               gate_disp = gate_disp)
        return signal_t







class RetrievePulses2DSIwithRealFields(RetrievePulsesRealFields, RetrievePulses2DSI):
    """ 
    Overwrites the generation of the signal field in order to use real fields instead of complex ones.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _post_init(self):
        """ 
        The phase-matrix needs to be interpolated onto frequency_big. 
        Overwriting its creation would be possible but a bit cumbersome. 
        """
        frequency, frequency_big = self.measurement_info.frequency, self.measurement_info.frequency_big
        self.spectral_filter1 = do_interpolation_1d(frequency_big, frequency, self.spectral_filter1)
        self.spectral_filter2 = do_interpolation_1d(frequency_big, frequency, self.spectral_filter2)
        self.measurement_info = tree_at(lambda x: x.spectral_filter1, self.measurement_info, self.spectral_filter1)
        self.measurement_info = tree_at(lambda x: x.spectral_filter2, self.measurement_info, self.spectral_filter2)



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

        pulse_f = individual.pulse
        pulse_f = self.interpolate_signal_f(pulse_f, measurement_info, "main", "big")
        pulse_t = self.ifft(pulse_f, sk_big, rn_big)

        if measurement_info.cross_correlation==True:
            gate_t = measurement_info.gate

        elif measurement_info.doubleblind==True:
            gate_f = individual.gate
            gate_f = self.interpolate_signal_f(gate_f, measurement_info, "main", "big")
            gate_t = self.ifft(gate_f, sk_big, rn_big)

        else:
            gate_t = pulse_t

        gate1, gate2 = self.apply_spectral_filter(gate_t, measurement_info.spectral_filter1, 
                                                  measurement_info.spectral_filter2, sk_big, rn_big)
            
        gate2_shifted = self.calculate_shifted_signal(gate2, frequency_big, tau_arr, time_big)
        tau = measurement_info.tau_pulse_anc1
        gate1 = self.calculate_shifted_signal(gate1, frequency_big, jnp.asarray([tau]), time_big)
        gate_pulses = jnp.squeeze(gate1) + gate2_shifted
        gate = calculate_gate_with_Real_Fields(gate_pulses, nonlinear_method)

        signal_t = jnp.real(pulse_t)*gate

        signal_t, signal_f = self.apply_mask(signal_t, measurement_info)
        signal_t = MyNamespace(signal_t=signal_t, 
                               signal_f=signal_f, 
                               gate_pulses=jnp.real(gate_pulses), 
                               gate_shifted=gate,
                               pulse_t = jnp.real(pulse_t))
        return signal_t








class RetrievePulsesVAMPIREwithRealFields(RetrievePulsesRealFields, RetrievePulsesVAMPIRE):
    """ 
    Overwrites the generation of the signal field in order to use real fields instead of complex ones.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _post_init(self):
        """ 
        The phase-matrix needs to be interpolated onto frequency_big. 
        Overwriting its creation would be possible but a bit cumbersome. 
        """
        frequency, frequency_big = self.measurement_info.frequency, self.measurement_info.frequency_big
        self.phase_matrix = do_interpolation_1d(frequency_big, frequency, self.phase_matrix)
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

        pulse_f = individual.pulse
        pulse_f = self.interpolate_signal_f(pulse_f, measurement_info, "main", "big")
        pulse_t = self.ifft(pulse_f, sk_big, rn_big)

        if measurement_info.cross_correlation==True:
            gate_t = measurement_info.gate

        elif measurement_info.doubleblind==True:
            gate_f = individual.gate
            gate_f = self.interpolate_signal_f(gate_f, measurement_info, "main", "big")
            gate_t = self.ifft(gate_f, sk_big, rn_big)
        else:
            gate_t = pulse_t

        gate_disp = self.apply_phase(gate_t, measurement_info, sk_big, rn_big) 

        tau = measurement_info.tau_interferometer
        gate_t_shifted = self.calculate_shifted_signal(gate_t, frequency_big, jnp.asarray([tau]), time_big)

        gate_pulses = jnp.squeeze(gate_t_shifted) + gate_disp
        gate_pulses = self.calculate_shifted_signal(gate_pulses, frequency_big, tau_arr, time_big)
        gate = calculate_gate_with_Real_Fields(gate_pulses, nonlinear_method)

        signal_t = jnp.real(pulse_t)*gate

        signal_t, signal_f = self.apply_mask(signal_t, measurement_info)
        signal_t = MyNamespace(signal_t=signal_t, 
                               signal_f=signal_f, 
                               gate_pulses=jnp.real(gate_pulses), 
                               gate_shifted=gate,
                               pulse_t = jnp.real(pulse_t))
        return signal_t