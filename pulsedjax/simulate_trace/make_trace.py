import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c0

import refractiveindex

import jax.numpy as jnp
import jax

from pulsedjax.utilities import MyNamespace, do_fft, do_ifft, get_sk_rn, do_interpolation_1d, center_signal_to_max
from pulsedjax.core.base_classes_methods import RetrievePulsesFROG, RetrievePulsesCHIRPSCAN, RetrievePulses2DSI, RetrievePulsesTDP, RetrievePulsesVAMPIRE
from pulsedjax.real_fields.base_classes_methods import RetrievePulsesFROGwithRealFields, RetrievePulsesCHIRPSCANwithRealFields, RetrievePulses2DSIwithRealFields, RetrievePulsesTDPwithRealFields, RetrievePulsesVAMPIREwithRealFields
from .make_pulse import MakePulse as MakePulseBase



def apply_noise(trace, scale_val=0.01, additive_noise=False, multiplicative_noise=False):
    """ Applies additive and/or multiplicative gaussian noise to a trace. """
    trace = trace/np.max(trace)
    shape = np.shape(trace)

    if additive_noise==True and multiplicative_noise==True:
        assert len(scale_val)==2, "scale_val needs to have len=2 when using both additive and multiplicative"

        noise_additive = np.random.normal(0, scale_val[0], size=shape)
        noise_multiplicative = np.random.normal(1, scale_val[1], size=shape)
        trace = trace*np.abs(noise_multiplicative) + noise_additive

    elif multiplicative_noise==True:
        noise = np.random.normal(1, scale_val, size=shape)
        trace = trace*np.abs(noise)

    elif additive_noise==True:
        noise = np.random.normal(0, scale_val, size=shape)
        trace = trace + noise

    else:
        raise ValueError("One of additive_noise or multiplicative_noise must be True.")

    return trace







class MakeTrace(MakePulseBase):
    """ 
    Simulates measurement traces based in input pulses.
    Inherits from make_pulse.MakePulse.

    Attributes:
        maketrace (MakeTraceFROG, MakeTraceCHIRPSCAN, MakeTrace2DSI, MakeTraceTDP, or MakeTraceVAMPIRE): defined via the respective generate_ ... () method

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maketrace = None

    
    def generate_frog(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, cross_correlation=False, interferometric=False, gate=(None, None), 
                      real_fields=False, frequency_range=None, N=256, cut_off_val=0.001, interpolate_fft_conform=False, plot_stuff=True):
        """
        Generates a FROG trace using the provide pulse/gate. 

        Args:
            time (jnp.array): the time axis of pulse_t
            frequency (jnp.array): the frequency axis of pulse_f
            pulse_t (jnp.array): the input pulse in the time domain
            pulse_f (jnp.array): the input pulse in the frequency domain
            nonlinear_method (str): the nonlinear method
            delay (jnp.array): the delays
            cross_correlation (bool): whether cross_correlation should be used
            interferometric (bool): whether interferometric setup should be used
            gate (tuple[jnp.array, jnp.array]): a tuple containing the frequency axis and the gate-pulse in the frequency domain. Is used as gate if cross_correlation=True
            real_fields (bool): whether the nonlinear signal should be generated using real fields
            frequency_range (tuple[Scalar,Scalar]): defines the frequenyc range of the trace
            N (int): defines the number of points along the frequency axis of the trace
            cut_off_val (float): defines how far the trace is zoomed in. Should be between zero and one.
            interpolate_fft_conform (bool): whether the time axis of the trace is interpolated to conform to the fft requirements.
            plot_stuff (bool): whether the trace and pulse should be plotted

        Returns:
            tuple[jnp.array, jnp.array, jnp.array, Pytree], the time and frequency axis, the trace, the spectra

        """
        
        if real_fields==True:
            maketrace = MakeTraceFROGReal
        else:
            maketrace = MakeTraceFROG
        
        self.maketrace = maketrace(time, frequency, pulse_t, pulse_f, nonlinear_method, delay, cross_correlation, interferometric, 
                                   frequency_range, N, cut_off_val, interpolate_fft_conform)


        if cross_correlation==True:
            frequency_gate, gate_f = gate
            gate = self.maketrace.get_gate_pulse(frequency_gate, gate_f)

        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()

        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, time_trace, frequency_trace, trace, spectra)

        return time_trace, frequency_trace, trace, spectra
    



    def generate_tdp(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, spectral_filter, cross_correlation=False, interferometric=False, gate=(None, None), 
                     real_fields=False, frequency_range=None, N=256, cut_off_val=0.001, interpolate_fft_conform=False, plot_stuff=True):
        
        """
        Generates a TDP trace using the provide pulse/gate. 

        Args:
            time (jnp.array): the time axis of pulse_t
            frequency (jnp.array): the frequency axis of pulse_f
            pulse_t (jnp.array): the input pulse in the time domain
            pulse_f (jnp.array): the input pulse in the frequency domain
            nonlinear_method (str): the nonlinear method
            delay (jnp.array): the delays
            spectral_filter (jnp.array): the spectral filter used in the setup
            cross_correlation (bool): whether cross_correlation should be used
            interferometric (bool): whether interferometric setup should be used
            gate (tuple[jnp.array, jnp.array]): a tuple containing the frequency axis and the gate-pulse in the frequency domain. Is used as gate if cross_correlation=True
            real_fields (bool): whether the nonlinear signal should be generated using real fields
            frequency_range (tuple[Scalar,Scalar]): defines the frequenyc range of the trace
            N (int): defines the number of points along the frequency axis of the trace
            cut_off_val (float): defines how far the trace is zoomed in. Should be between zero and one.
            interpolate_fft_conform (bool): whether the time axis of the trace is interpolated to conform to the fft requirements.
            plot_stuff (bool): whether the trace and pulse should be plotted

        Returns:
            tuple[jnp.array, jnp.array, jnp.array, Pytree], the time and frequency axis, the trace, the spectra

        """
        
        if real_fields==True:
            maketrace = MakeTraceTDPReal
        else:
            maketrace = MakeTraceTDP
        
        self.maketrace = maketrace(time, frequency, pulse_t, pulse_f, nonlinear_method, delay, spectral_filter, cross_correlation, interferometric, 
                                   frequency_range, N, cut_off_val, interpolate_fft_conform)


        if cross_correlation==True:
            frequency_gate, gate_f = gate
            gate = self.maketrace.get_gate_pulse(frequency_gate, gate_f)

        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()

        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, time_trace, frequency_trace, trace, spectra)

        return time_trace, frequency_trace, trace, spectra
    




    def generate_chirpscan(self, time, frequency, pulse_t, pulse_f, nonlinear_method, z_arr, phase_type, parameters, real_fields=False, 
                           frequency_range=None, N=256, cut_off_val=0.001, plot_stuff=True):
        
        """
        Generates a Chirp-Scan trace using the provide pulse/gate. 

        Args:
            time (jnp.array): the time axis of pulse_t
            frequency (jnp.array): the frequency axis of pulse_f
            pulse_t (jnp.array): the input pulse in the time domain
            pulse_f (jnp.array): the input pulse in the frequency domain
            nonlinear_method (str): the nonlinear method
            z_arr (jnp.array): defines the shift arr of the chirp scan. (e.g. material thickness, phase_shift in MIIPS, ...)
            phase_type (str, Callable): defines how the applied phase is created, (e.g. material, MIIPS, ... )
            parameters (tuple): defines further necessary input parameters to the function that calculates phase_matrix
            real_fields (bool): whether the nonlinear signal should be generated using real fields
            frequency_range (tuple[Scalar,Scalar]): defines the frequenyc range of the trace
            N (int): defines the number of points along the frequency axis of the trace
            cut_off_val (float): defines how far the trace is zoomed in. Should be between zero and one.
            plot_stuff (bool): whether the trace and pulse should be plotted

        Returns:
            tuple[jnp.array, jnp.array, jnp.array, Pytree], the shift and frequency axis, the trace, the spectra

        """

        if real_fields==True:
            maketrace = MakeTraceCHIRPSCANReal
        else:
            maketrace = MakeTraceCHIRPSCAN

        self.maketrace = maketrace(time, frequency, pulse_t, pulse_f, nonlinear_method, z_arr, phase_type, parameters, 
                                frequency_range, N, cut_off_val)
        
        
        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()
            
        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, z_arr, frequency_trace, trace, spectra)

        return z_arr, frequency_trace, trace, spectra
    




    def generate_2dsi(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, spectral_filter1=None, spectral_filter2=None, tau_pulse_anc1=0, 
                      material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                      cross_correlation=False, gate=(None, None), real_fields=False, frequency_range=None, N=256, cut_off_val=0.001, 
                      interpolate_fft_conform=False, plot_stuff=True):
        
        """
        Generates a 2DSI trace using the provide pulse/gate. 

        Args:
            time (jnp.array): the time axis of pulse_t
            frequency (jnp.array): the frequency axis of pulse_f
            pulse_t (jnp.array): the input pulse in the time domain
            pulse_f (jnp.array): the input pulse in the frequency domain
            nonlinear_method (str): the nonlinear method
            delay (jnp.array): the delays
            spectral_filter1 (jnp.array): the first spectral filter in the interferometer
            spectral_filter2 (jnp.array): the second spectral filter in the interferometer
            tau_pulse_anc1 (int, float): the delay of the fixed interferometer arm and the external pulse
            material_thickness (int, float): material thickness in the interferometer
            refractive_index (refractiveindex.RefractiveIndexMaterial): refractive index of the material in the interferometer
            cross_correlation (bool): whether cross_correlation should be used
            gate (tuple[jnp.array, jnp.array]): a tuple containing the frequency axis and the gate-pulse in the frequency domain. Is used as gate if cross_correlation=True
            real_fields (bool): whether the nonlinear signal should be generated using real fields
            frequency_range (tuple[Scalar,Scalar]): defines the frequenyc range of the trace
            N (int): defines the number of points along the frequency axis of the trace
            cut_off_val (float): defines how far the trace is zoomed in. Should be between zero and one.
            interpolate_fft_conform (bool): whether the time axis of the trace is interpolated to conform to the fft requirements.
            plot_stuff (bool): whether the trace and pulse should be plotted

        Returns:
            tuple[jnp.array, jnp.array, jnp.array, Pytree], the time and frequency axis, the trace, the spectra

        """

        if real_fields==True:
            maketrace = MakeTrace2DSIReal
        else:
            maketrace = MakeTrace2DSI

        self.maketrace = maketrace(time, frequency, pulse_t, pulse_f, nonlinear_method, delay, spectral_filter1, spectral_filter2, tau_pulse_anc1, 
                                   material_thickness, refractive_index, cross_correlation, frequency_range, N, cut_off_val, 
                                   interpolate_fft_conform)
        
        if self.maketrace.cross_correlation==True:
            gate = self.maketrace.get_gate_pulse(gate[0], gate[1])

        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()
            
        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, time_trace, frequency_trace, trace, spectra)

        return time_trace, frequency_trace, trace, spectra
    



    

    def generate_vampire(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, tau_interferometer=0, material_thickness=0, 
                         refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                         cross_correlation=False, gate=(None, None), real_fields=False, frequency_range=None, N=256, 
                         cut_off_val=0.001, interpolate_fft_conform=False, plot_stuff=True):
        

        """
        Generates a VAMPIRE trace using the provide pulse/gate. 

        Args:
            time (jnp.array): the time axis of pulse_t
            frequency (jnp.array): the frequency axis of pulse_f
            pulse_t (jnp.array): the input pulse in the time domain
            pulse_f (jnp.array): the input pulse in the frequency domain
            nonlinear_method (str): the nonlinear method
            delay (jnp.array): the delays
            tau_interferometer (int, float): the delay inside the interferometer
            material_thickness (int, float): material thickness in the interferometer
            refractive_index (refractiveindex.RefractiveIndexMaterial): refractive index of the material in the interferometer
            cross_correlation (bool): whether cross_correlation should be used
            gate (tuple[jnp.array, jnp.array]): a tuple containing the frequency axis and the gate-pulse in the frequency domain. Is used as gate if cross_correlation=True
            real_fields (bool): whether the nonlinear signal should be generated using real fields
            frequency_range (tuple[Scalar,Scalar]): defines the frequenyc range of the trace
            N (int): defines the number of points along the frequency axis of the trace
            cut_off_val (float): defines how far the trace is zoomed in. Should be between zero and one.
            interpolate_fft_conform (bool): whether the time axis of the trace is interpolated to conform to the fft requirements.
            plot_stuff (bool): whether the trace and pulse should be plotted

        Returns:
            tuple[jnp.array, jnp.array, jnp.array, Pytree], the time and frequency axis, the trace, the spectra

        """
        

        if real_fields==True:
            maketrace = MakeTraceVAMPIREReal
        else:
            maketrace = MakeTraceVAMPIRE

        self.maketrace = maketrace(time, frequency, pulse_t, pulse_f, nonlinear_method, delay, tau_interferometer, material_thickness, 
                                   refractive_index, cross_correlation, frequency_range, N, cut_off_val, interpolate_fft_conform)

        if self.maketrace.cross_correlation==True:
            gate = self.maketrace.get_gate_pulse(gate[0], gate[1])

        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()
            
        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, time_trace, frequency_trace, trace, spectra)

        return time_trace, frequency_trace, trace, spectra
    




















def interpolate_spectrum(frequency, pulse_f, N):
    spectrum = jnp.abs(pulse_f)**2

    idx=np.where(spectrum/jnp.max(spectrum)>1e-5)
    idx_1 = np.sort(idx)[0]
    idx_1_min, idx_1_max = idx_1[0], idx_1[-1]+1
    
    frequency_zoom = frequency[idx_1_min:idx_1_max]
    frequency_interpolate_spectrum = np.linspace(frequency_zoom[0], frequency_zoom[-1], N)
    
    spectrum = do_interpolation_1d(frequency_interpolate_spectrum, frequency, spectrum, method="linear")
    return frequency_interpolate_spectrum, spectrum



class MakeTraceBASE:
    def __init__(self, *args, **kwargs):
        self.cross_correlation = False

        self.fft = do_fft
        self.ifft = do_ifft


    def generate_trace(self):
        if self.nonlinear_method=="shg":
            self.factor=2
        elif self.nonlinear_method=="thg":
            self.factor=3
        else:
            self.factor=1

        individual, measurement_info, transform_arr = self.get_parameters_to_make_signal_t()

        self.signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        self.trace = jnp.abs(self.signal_t.signal_f)**2

        time, frequency, trace, spectra = self.interpolate_trace()

        self.trace = trace/np.max(trace)
        return time, frequency, self.trace, spectra



    
    def interpolate_trace(self, is_delay_based=True):
        max_val = np.max(self.trace)

        idx = np.where(self.trace>max_val*self.cut_off_val)
        idx_0, idx_1 = np.sort(idx)

        idx_0_min, idx_0_max = idx_0[0], idx_0[-1]+1
        idx_1_min, idx_1_max = idx_1[0], idx_1[-1]+1


        time_zoom = self.x_arr[idx_0_min:idx_0_max]
        frequency_zoom = self.frequency[idx_1_min:idx_1_max]

        if self.frequency_range!=None:
            fmin, fmax = self.frequency_range
            if self.nonlinear_method=="sd":
                fmin, fmax = np.sort([-1*fmin, -1*fmax])
        else:
            fmin, fmax = np.min(frequency_zoom), np.max(frequency_zoom)

        if is_delay_based==True:
            if self.interpolate_fft_conform==True:
                central_t = (self.x_arr[0] + self.x_arr[-1])/2
                dt = 1/np.abs((fmax-fmin))

                tmin = central_t - dt*self.N/2
                tmax = central_t + dt*self.N/2
                time_interpolate = np.linspace(tmin, tmax, self.N)
            else:
                time_interpolate = self.x_arr
        else:		
            time_interpolate = self.x_arr

        frequency_interpolate = np.linspace(fmin, fmax, self.N)
        trace_interpolate = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(frequency_interpolate, self.frequency, self.trace)

        if is_delay_based==True:
            trace_interpolate = jax.vmap(do_interpolation_1d, in_axes=(None,None,1))(time_interpolate, self.x_arr, trace_interpolate)
            trace_interpolate = np.abs(trace_interpolate).T
        else:
            trace_interpolate = np.abs(trace_interpolate)


        if self.nonlinear_method=="sd":
            frequency_interpolate = -1*np.flip(frequency_interpolate)
            trace_interpolate = np.flip(trace_interpolate, axis=0)


        frequency_pulse_spectrum, spectrum_pulse = interpolate_spectrum(self.frequency, self.pulse_f, self.N)
        if self.cross_correlation==True:
            self.gate_f = self.fft(self.gate, self.sk, self.rn)
            frequency_gate_spectrum, spectrum_gate = interpolate_spectrum(self.frequency, self.gate_f, self.N)
        else:
            frequency_gate_spectrum, spectrum_gate = None, None
            
        spectra = MyNamespace(pulse = (frequency_pulse_spectrum, spectrum_pulse), 
                              gate = (frequency_gate_spectrum, spectrum_gate))

        return time_interpolate, frequency_interpolate, trace_interpolate, spectra



    def plot_trace(self, time, pulse_t, frequency, pulse_f, x_arr, frequency_trace, trace, spectra):
        
        fig=plt.figure(figsize=(18,8))
        ax1=plt.subplot(2,2,1)
        ax1.plot(time, np.abs(pulse_t), label="Amplitude")
        ax1.set_xlabel("Time [fs]")
        ax1.set_ylabel("Amplitude [arb. u.]")
        ax1.legend(loc=2)

        ax2 = ax1.twinx()
        ax2.plot(time, np.unwrap(np.angle(pulse_t))*1/np.pi, c="tab:orange", label="Phase")
        ax2.set_ylabel(r"Phase [$\pi$]")
        ax2.legend(loc=1)

        if self.cross_correlation==True:
            ax1.plot(time, np.abs(self.gate), label="Gate-Pulse", c="tab:red")
            ax2.plot(time, np.unwrap(np.angle(self.gate))*1/np.pi, label="Gate-Pulse", c="tab:green")

        ax1=plt.subplot(2,2,2)
        ax1.plot(frequency, np.abs(pulse_f), label="Amplitude")
        ax1.set_xlabel("Frequency [PHz]")
        ax1.set_ylabel("Amplitude [arb. u.]")
        ax1.legend(loc=2)

        ax2 = ax1.twinx()
        ax2.plot(frequency, np.unwrap(np.angle(pulse_f))*1/np.pi, c="tab:orange", label="Phase")
        ax2.set_ylabel(r"Phase [$\pi$]")
        ax2.legend(loc=1)

        if self.cross_correlation==True:
            ax1.plot(frequency, np.abs(self.gate_f), label="Gate-Pulse", c="tab:red")
            ax2.plot(frequency, np.unwrap(np.angle(self.gate_f))*1/np.pi, label="Gate-Pulse", c="tab:green")


        plt.subplot(2,2,3)
        plt.plot(spectra.pulse[0], spectra.pulse[1], label="Pulse Spectrum")

        if self.cross_correlation==True:
            plt.plot(spectra.gate[0], spectra.gate[1], label="Gate Spectrum")

        plt.xlabel("Frequency [PHz]")
        plt.ylabel("Amplitude [arb. u.]")
        plt.legend()

        plt.subplot(2,2,4)
        plt.pcolormesh(x_arr, frequency_trace, trace.T, cmap="nipy_spectral")
        plt.xlabel("Shift [arb. u.]")
        plt.ylabel("Frequency [PHz]")
        plt.colorbar()

        plt.show()






class MakeTraceFROG(MakeTraceBASE, RetrievePulsesFROG):
    def __init__(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, cross_correlation, interferometric, 
                 frequency_range, N, cut_off_val, interpolate_fft_conform):
        super().__init__()
        
        self.time=time
        self.frequency=frequency
        self.pulse_t=pulse_t
        self.pulse_f=pulse_f
        self.nonlinear_method=nonlinear_method
        self.N=N
        self.interpolate_fft_conform=interpolate_fft_conform
        self.cut_off_val = cut_off_val
        self.frequency_range = frequency_range
        self.cross_correlation=cross_correlation
        self.interferometric=interferometric
        self.gate = None

        self.x_arr = delay

        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        self.central_frequency = jnp.sum(jnp.abs(pulse_f)*frequency)/jnp.sum(jnp.abs(pulse_f))


    def get_gate_pulse(self, frequency_gate, gate_f):
        gate_f = do_interpolation_1d(self.frequency, frequency_gate, gate_f)
        self.gate = self.ifft(gate_f, self.sk, self.rn)
        return self.gate



    def get_parameters_to_make_signal_t(self):
        measurement_info = MyNamespace(gate=self.gate, time=self.time, frequency=self.frequency, frequency_exp=self.frequency, 
                                       time_big=self.time, frequency_big=self.frequency, sk_big=self.sk, rn_big=self.rn, sk=self.sk, rn=self.rn,
                                       sk_exp=self.sk, rn_exp=self.rn,
                                       cross_correlation=self.cross_correlation, interferometric=self.interferometric, 
                                       nonlinear_method=self.nonlinear_method, doubleblind=False, central_frequency = self.central_frequency)
        individual = MyNamespace(pulse=self.pulse_t, gate=self.gate)
        return individual, measurement_info, self.x_arr




class MakeTraceTDP(MakeTraceBASE, RetrievePulsesTDP):
    def __init__(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, spectral_filter, cross_correlation, interferometric, 
                                   frequency_range, N, cut_off_val, interpolate_fft_conform):
        super().__init__()
        
        self.time=time
        self.frequency=frequency
        self.pulse_t=pulse_t
        self.pulse_f=pulse_f
        self.nonlinear_method=nonlinear_method
        self.N=N
        self.interpolate_fft_conform=interpolate_fft_conform
        self.cut_off_val = cut_off_val
        self.frequency_range = frequency_range
        self.cross_correlation=cross_correlation
        self.interferometric=interferometric
        self.gate = None

        self.x_arr = delay

        self.sk, self.rn = get_sk_rn(self.time, self.frequency)
        if spectral_filter==None:
            self.spectral_filter = jnp.ones(jnp.size(frequency))
        else:
            self.spectral_filter = spectral_filter

        self.central_frequency = jnp.sum(jnp.abs(pulse_f)*frequency)/jnp.sum(jnp.abs(pulse_f))
        
        

    def get_gate_pulse(self, frequency_gate, gate_f):
        return MakeTraceFROG.get_gate_pulse(self, frequency_gate, gate_f)


    def get_parameters_to_make_signal_t(self):
        individual, measurement_info, x_arr = MakeTraceFROG.get_parameters_to_make_signal_t(self)
        measurement_info = measurement_info.expand(spectral_filter=self.spectral_filter)
        return individual, measurement_info, x_arr
    

    def interpolate_trace(self):
        return MakeTraceFROG.interpolate_trace(self)
    









class MakeTraceCHIRPSCAN(MakeTraceBASE, RetrievePulsesCHIRPSCAN):
    def __init__(self, time, frequency, pulse_t, pulse_f, nonlinear_method, z_arr, phase_type, parameters, frequency_range, N, cut_off_val):
        super().__init__()

        self.z_arr = z_arr
        self.time = time
        self.frequency = frequency
        self.pulse_t = pulse_t
        self.pulse_f = pulse_f
        self.nonlinear_method = nonlinear_method
        self.N=N
        self.cut_off_val = cut_off_val
        self.frequency_range = frequency_range

        self.x_arr = z_arr

        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        self.central_frequency = jnp.sum(jnp.abs(pulse_f)*frequency)/jnp.sum(jnp.abs(pulse_f))

        self.phase_type = phase_type
        self.parameters = parameters


    def get_parameters_to_make_signal_t(self):
        self.measurement_info = MyNamespace(z_arr=self.z_arr, frequency=self.frequency, 
                                            frequency_exp=self.frequency, time_big=self.time, frequency_big=self.frequency, 
                                            sk_big=self.sk, rn_big=self.rn, sk=self.sk, rn=self.rn, sk_exp=self.sk, rn_exp=self.rn, 
                                            nonlinear_method=self.nonlinear_method, doubleblind=False, central_frequency = self.central_frequency)
        individual = MyNamespace(pulse=self.pulse_f, gate=None)

        self.phase_matrix = self.get_phase_matrix(self.parameters)
        return individual, self.measurement_info, self.phase_matrix
    
    
    def interpolate_trace(self):
        return super().interpolate_trace(is_delay_based=False)




class MakeTrace2DSI(MakeTraceBASE, RetrievePulses2DSI):
    def __init__(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, spectral_filter1, spectral_filter2, tau_pulse_anc1, 
                 material_thickness, refractive_index, cross_correlation, frequency_range, N, cut_off_val, 
                 interpolate_fft_conform):
        super().__init__()

        self.interpolate_fft_conform=interpolate_fft_conform


        self.c0 = c0
        
        self.time=time
        self.frequency=frequency
        self.pulse_t=pulse_t
        self.pulse_f=pulse_f
        self.nonlinear_method=nonlinear_method
        self.N=N
        self.cut_off_val = cut_off_val
        self.frequency_range = frequency_range
        self.cross_correlation = cross_correlation
        self.gate = None
        self.tau_pulse_anc1 = tau_pulse_anc1

        self.x_arr = delay

        self.sk, self.rn = get_sk_rn(self.time, self.frequency)
        self.refractive_index, self.material_thickness = refractive_index, material_thickness

        self.central_frequency = jnp.sum(jnp.abs(pulse_f)*frequency)/jnp.sum(jnp.abs(pulse_f))

        if spectral_filter1==None:
            self.spectral_filter1 = jnp.ones(jnp.size(self.frequency))
        else:
            self.spectral_filter1 = spectral_filter1

        if spectral_filter2==None:
            self.spectral_filter2 = jnp.ones(jnp.size(self.frequency))
        else:
            self.spectral_filter2 = spectral_filter2



    def get_gate_pulse(self, frequency_gate, gate_f):
        gate_f = do_interpolation_1d(self.frequency, frequency_gate, gate_f)
        self.gate = self.ifft(gate_f, self.sk, self.rn)
        return self.gate
                

    def get_parameters_to_make_signal_t(self):
        measurement_info = MyNamespace(time=self.time, frequency=self.frequency, frequency_exp=self.frequency, tau_pulse_anc1 = self.tau_pulse_anc1,
                                       time_big=self.time, frequency_big=self.frequency, sk_big=self.sk, rn_big=self.rn, sk=self.sk, rn=self.rn, 
                                       cross_correlation=self.cross_correlation, gate=self.gate,
                                       nonlinear_method=self.nonlinear_method, doubleblind=False, c0=self.c0, 
                                       spectral_filter1=self.spectral_filter1, spectral_filter2=self.spectral_filter2, central_frequency = self.central_frequency)
        
        self.phase_matrix = self.get_phase_matrix(self.refractive_index, self.material_thickness, measurement_info)
        measurement_info = measurement_info.expand(phase_matrix = self.phase_matrix)

        individual = MyNamespace(pulse=self.pulse_t, gate=self.gate)
        return individual, measurement_info, self.x_arr


    def interpolate_trace(self):
        return MakeTraceFROG.interpolate_trace(self)
    






class MakeTraceVAMPIRE(MakeTraceBASE, RetrievePulsesVAMPIRE):
    def __init__(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, tau_interferometer, material_thickness, 
                 refractive_index, cross_correlation, frequency_range, N, cut_off_val, interpolate_fft_conform):
        super().__init__()

        self.c0=c0
        self.refractive_index=refractive_index
        self.material_thickness = material_thickness
        
        self.time=time
        self.frequency=frequency
        self.pulse_t=pulse_t
        self.pulse_f=pulse_f
        self.nonlinear_method=nonlinear_method
        self.N=N
        self.interpolate_fft_conform=interpolate_fft_conform
        self.cut_off_val = cut_off_val
        self.frequency_range = frequency_range
        self.cross_correlation=cross_correlation
        self.interferometric=False
        self.gate = None
        self.tau_interferometer = tau_interferometer

        self.central_frequency = jnp.sum(jnp.abs(pulse_f)*frequency)/jnp.sum(jnp.abs(pulse_f))

        self.x_arr = delay
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)
        

    def get_gate_pulse(self, frequency_gate, gate_f):
        return MakeTraceFROG.get_gate_pulse(self, frequency_gate, gate_f)


    def get_parameters_to_make_signal_t(self):
        individual, measurement_info, x_arr = MakeTraceFROG.get_parameters_to_make_signal_t(self)
        measurement_info = measurement_info.expand(c0=self.c0)
        self.phase_matrix = self.get_phase_matrix(self.refractive_index, self.material_thickness, measurement_info)
        measurement_info = measurement_info.expand(tau_interferometer=self.tau_interferometer, phase_matrix = self.phase_matrix)
        return individual, measurement_info, x_arr
    

    def interpolate_trace(self):
        return MakeTraceFROG.interpolate_trace(self)







class MakeTraceFROGReal(MakeTraceFROG, RetrievePulsesFROGwithRealFields):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_gate_pulse(self, frequency_gate, gate_f):
        return MakeTraceFROG.get_gate_pulse(self, frequency_gate, gate_f)
    




class MakeTraceTDPReal(MakeTraceTDP, RetrievePulsesTDPwithRealFields):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_gate_pulse(self, frequency_gate, gate_f):
        return MakeTraceFROG.get_gate_pulse(self, frequency_gate, gate_f)




class MakeTraceCHIRPSCANReal(MakeTraceCHIRPSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




class MakeTrace2DSIReal(MakeTrace2DSI, RetrievePulses2DSIwithRealFields):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_gate_pulse(self, frequency_gate, gate_f):
        return MakeTraceFROG.get_gate_pulse(self, frequency_gate, gate_f)




class MakeTraceVAMPIREReal(MakeTraceVAMPIRE, RetrievePulsesVAMPIREwithRealFields):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_gate_pulse(self, frequency_gate, gate_f):
        return MakeTraceFROG.get_gate_pulse(self, frequency_gate, gate_f)
