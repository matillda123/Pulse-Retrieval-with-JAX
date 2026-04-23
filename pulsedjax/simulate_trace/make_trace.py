import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c0
from functools import partial as Partial

import refractiveindex

import jax.numpy as jnp
import jax

from pulsedjax.utilities import MyNamespace, do_fft, do_ifft, get_sk_rn, do_interpolation_1d, center_signal_to_max, integrate_signal_1D
from pulsedjax.core.base_classes_methods import RetrievePulsesFROG, RetrievePulsesCHIRPSCAN, RetrievePulses2DSI, RetrievePulsesTDP, RetrievePulsesVAMPIRE, RetrievePulsesSTREAKING
from pulsedjax.real_fields.core.base_classes_methods import RetrievePulsesFROGwithRealFields, RetrievePulsesCHIRPSCANwithRealFields, RetrievePulses2DSIwithRealFields, RetrievePulsesTDPwithRealFields, RetrievePulsesVAMPIREwithRealFields
from pulsedjax.simulate_trace.make_pulse import MakePulse as MakePulseBase



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
        maketrace (MakeTraceFROG, MakeTraceCHIRPSCAN, MakeTrace2DSI, MakeTraceTDP, or MakeTraceVAMPIRE): default is None, defined via respective method

    """
    def __init__(self, N=256, f_max=1, df=None):
        super().__init__(N=N, f_max=f_max, df=df)
        self.maketrace = None

    
    def generate_frog(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, cross_correlation=False, interferometric=False, gate=(None, None), 
                      real_fields=False, frequency_range=None, f_range_fields=None, N=256, cut_off_val=0.001, interpolate_fft_conform=False, plot_stuff=True):
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
            f_range_fields (tuple[Scalar,Scalar]): defines the frequency range of the nonlinear signals to consider
            N (int): defines the number of points along the frequency axis of the trace
            cut_off_val (float): defines how far the trace is zoomed in. Should be between zero and one.
            interpolate_fft_conform (bool): whether the time axis of the trace is interpolated to conform to the fft requirements.
            plot_stuff (bool): whether the trace and pulse should be plotted

        Returns:
            tuple[jnp.array, jnp.array, jnp.array, Pytree], the time and frequency axis, the trace, the spectra

        """
        if cross_correlation=="doubleblind":
            cross_correlation=True

        if real_fields==True:
            maketrace = MakeTraceFROGReal
        else:
            maketrace = MakeTraceFROG
        
        self.maketrace = maketrace(time, frequency, pulse_t, pulse_f, nonlinear_method, delay, cross_correlation, interferometric, 
                                   frequency_range, f_range_fields, frequency_range, N, cut_off_val, interpolate_fft_conform)


        if cross_correlation==True:
            frequency_gate, gate_f = gate
            gate = self.maketrace.get_gate_pulse(frequency_gate, gate_f)

        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()

        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, time_trace, frequency_trace, trace, spectra)

        return time_trace, frequency_trace, trace, spectra
    



    def generate_tdp(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, spectral_filter, cross_correlation=False, interferometric=False, gate=(None, None), 
                     real_fields=False, frequency_range=None, f_range_fields=None, N=256, cut_off_val=0.001, interpolate_fft_conform=False, plot_stuff=True):
        
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
            f_range_fields (tuple[Scalar,Scalar]): defines the frequency range of the nonlinear signals to consider
            N (int): defines the number of points along the frequency axis of the trace
            cut_off_val (float): defines how far the trace is zoomed in. Should be between zero and one.
            interpolate_fft_conform (bool): whether the time axis of the trace is interpolated to conform to the fft requirements.
            plot_stuff (bool): whether the trace and pulse should be plotted

        Returns:
            tuple[jnp.array, jnp.array, jnp.array, Pytree], the time and frequency axis, the trace, the spectra

        """
        if cross_correlation=="doubleblind":
            cross_correlation=True
            
        if real_fields==True:
            maketrace = MakeTraceTDPReal
        else:
            maketrace = MakeTraceTDP
        
        self.maketrace = maketrace(time, frequency, pulse_t, pulse_f, nonlinear_method, delay, spectral_filter, cross_correlation, interferometric, 
                                   frequency_range, f_range_fields, frequency_range, N, cut_off_val, interpolate_fft_conform)


        if cross_correlation==True:
            frequency_gate, gate_f = gate
            gate = self.maketrace.get_gate_pulse(frequency_gate, gate_f)

        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()

        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, time_trace, frequency_trace, trace, spectra)

        return time_trace, frequency_trace, trace, spectra
    




    def generate_chirpscan(self, time, frequency, pulse_t, pulse_f, nonlinear_method, theta, phase_type, chirp_parameters, real_fields=False, 
                           frequency_range=None, f_range_fields=None, N=256, cut_off_val=0.001, plot_stuff=True):
        
        """
        Generates a Chirp-Scan trace using the provide pulse/gate. 

        Args:
            time (jnp.array): the time axis of pulse_t
            frequency (jnp.array): the frequency axis of pulse_f
            pulse_t (jnp.array): the input pulse in the time domain
            pulse_f (jnp.array): the input pulse in the frequency domain
            nonlinear_method (str): the nonlinear method
            theta (jnp.array): defines the shift arr of the chirp scan. (e.g. material thickness, phase_shift in MIIPS, ...)
            phase_type (str, Callable): defines how the applied phase is created, (e.g. material, MIIPS, ... )
            parameters (tuple): defines further necessary input parameters to the function that calculates phase_matrix
            real_fields (bool): whether the nonlinear signal should be generated using real fields
            frequency_range (tuple[Scalar,Scalar]): defines the frequenyc range of the trace
            f_range_fields (tuple[Scalar,Scalar]): defines the frequency range of the nonlinear signals to consider
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

        self.maketrace = maketrace(time, frequency, pulse_t, pulse_f, nonlinear_method, theta, phase_type, chirp_parameters, 
                                frequency_range, f_range_fields, frequency_range, N, cut_off_val)
        
        
        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()
            
        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, theta, frequency_trace, trace, spectra)

        return theta, frequency_trace, trace, spectra
    




    def generate_2dsi(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, spectral_filter1=None, spectral_filter2=None, tau_pulse_anc1=0, 
                      material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                      cross_correlation=False, gate=(None, None), real_fields=False, frequency_range=None, f_range_fields=None, N=256, cut_off_val=0.001, 
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
            f_range_fields (tuple[Scalar,Scalar]): defines the frequency range of the nonlinear signals to consider
            N (int): defines the number of points along the frequency axis of the trace
            cut_off_val (float): defines how far the trace is zoomed in. Should be between zero and one.
            interpolate_fft_conform (bool): whether the time axis of the trace is interpolated to conform to the fft requirements.
            plot_stuff (bool): whether the trace and pulse should be plotted

        Returns:
            tuple[jnp.array, jnp.array, jnp.array, Pytree], the time and frequency axis, the trace, the spectra

        """

        if cross_correlation=="doubleblind":
            cross_correlation=True

        if real_fields==True:
            maketrace = MakeTrace2DSIReal
        else:
            maketrace = MakeTrace2DSI

        self.maketrace = maketrace(time, frequency, pulse_t, pulse_f, nonlinear_method, delay, spectral_filter1, spectral_filter2, tau_pulse_anc1, 
                                   material_thickness, refractive_index, cross_correlation, frequency_range, f_range_fields, frequency_range, N, cut_off_val, 
                                   interpolate_fft_conform)
        
        if self.maketrace.cross_correlation==True:
            gate = self.maketrace.get_gate_pulse(gate[0], gate[1])

        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()
            
        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, time_trace, frequency_trace, trace, spectra)

        return time_trace, frequency_trace, trace, spectra
    



    

    def generate_vampire(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, tau_interferometer=0, material_thickness=0, 
                         refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), 
                         cross_correlation=False, gate=(None, None), real_fields=False, frequency_range=None, f_range_fields=None, N=256, 
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
            f_range_fields (tuple[Scalar,Scalar]): defines the frequency range of the nonlinear signals to consider
            N (int): defines the number of points along the frequency axis of the trace
            cut_off_val (float): defines how far the trace is zoomed in. Should be between zero and one.
            interpolate_fft_conform (bool): whether the time axis of the trace is interpolated to conform to the fft requirements.
            plot_stuff (bool): whether the trace and pulse should be plotted

        Returns:
            tuple[jnp.array, jnp.array, jnp.array, Pytree], the time and frequency axis, the trace, the spectra

        """

        if cross_correlation=="doubleblind":
            cross_correlation=True
        

        if real_fields==True:
            maketrace = MakeTraceVAMPIREReal
        else:
            maketrace = MakeTraceVAMPIRE

        self.maketrace = maketrace(time, frequency, pulse_t, pulse_f, nonlinear_method, delay, tau_interferometer, material_thickness, 
                                   refractive_index, cross_correlation, frequency_range, f_range_fields, frequency_range, N, cut_off_val, interpolate_fft_conform)

        if self.maketrace.cross_correlation==True:
            gate = self.maketrace.get_gate_pulse(gate[0], gate[1])

        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()
            
        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, time_trace, frequency_trace, trace, spectra)

        return time_trace, frequency_trace, trace, spectra
    






    def generate_streaking(self, time, frequency, nir_pulse, euv_pulse, delay, Ip_eV=jnp.array([[0]]), DTME=(None, None), energy_range=None, N=2048, 
                         cut_off_val=0.001, interpolate_fft_conform=False, plot_stuff=True):
        
        """
        Generates an Attosecond-Streaking trace using the provide pulse and gate. 
        Assumes the pulse to be the femtosecond pulse and gate the EUV-pulse.

        Args:
            time (jnp.array): the time axis of pulse_t
            frequency (jnp.array): the frequency axis of pulse_f
            nir_pulse (tuple[jnp.array, jnp.array]): a tuple containing the nir pulse in the time and frequency domain
            euv_pulse (tuple[jnp.array, jnp.array]): a tuple containing the euv pulse in the time and frequency domain
            delay (jnp.array): the delays
            Ip_eV (jnp.array): the ionization potential in eV
            DTME (tuple[jnp.array, jnp.array]): a tuple containing the momentum axis in atomic units and the Dipole-Transition-Matrix-Elements
            energy_range (tuple[Scalar,Scalar]): defines the energy range of the trace (in eV)
            N (int): defines the number of points along the frequency axis of the trace
            cut_off_val (float): defines how far the trace is zoomed in. Should be between zero and one.
            interpolate_fft_conform (bool): whether the time axis of the trace is interpolated to conform to the fft requirements.
            plot_stuff (bool): whether the trace and pulse should be plotted

        Returns:
            tuple[jnp.array, jnp.array, jnp.array, Pytree], the time and frequency axis, the trace, the spectra

        """
        pulse_t, pulse_f = nir_pulse
        gate_t, gate_f = euv_pulse
        
        self.maketrace = MakeTraceSTREAKING(time, frequency, pulse_t, pulse_f, delay, Ip_eV,
                                            energy_range, N, cut_off_val, interpolate_fft_conform)
        gate = self.maketrace.get_gate_pulse(frequency, gate_f)

        if DTME!=(None,None):
            dtme = self.maketrace.get_DTME(DTME[0], DTME[1])


        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()
            
        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, time_trace, frequency_trace, trace, spectra)

        return time_trace, frequency_trace, trace, spectra
    




















def interpolate_spectrum(frequency, pulse_f, N, do_interpolation):
    spectrum = jnp.abs(pulse_f)**2

    idx = np.where(spectrum/jnp.max(spectrum)>1e-6)
    idx_1 = np.sort(idx)[0]
    idx_1_min, idx_1_max = idx_1[0], idx_1[-1]+1
    
    frequency_zoom = frequency[idx_1_min:idx_1_max]
    frequency_interpolate_spectrum = np.linspace(frequency_zoom[0], frequency_zoom[-1], N)
    
    spectrum = do_interpolation(frequency_interpolate_spectrum, frequency, spectrum)
    return frequency_interpolate_spectrum, spectrum



class MakeTraceBASE:
    def __init__(self, time, frequency, pulse_t, pulse_f, nonlinear_method, 
                 N, cut_off_val, interpolate_fft_conform, frequency_range, f_range_fields, f_range_pulse, 
                 cross_correlation, interferometric, *args, **kwargs):
        
        if nonlinear_method is None:
            self.factor=1
        elif nonlinear_method=="shg":
            self.factor=2
        elif nonlinear_method=="thg":
            self.factor=3
        elif nonlinear_method[-2:]=="hg":
            n = int(nonlinear_method[0])
            self.factor=n
        else:
            self.factor=1

        self.fft = do_fft
        self.ifft = do_ifft
        self.do_interpolation = Partial(do_interpolation_1d, method="cubic")

        self.time = time
        self.frequency = frequency
        self.pulse_t = pulse_t
        self.pulse_f = pulse_f
        self.gate_t = None
        self.gate_f = None
        self.nonlinear_method = nonlinear_method
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        central_frequency_pulse = jnp.sum(jnp.abs(pulse_f)*frequency)/jnp.sum(jnp.abs(pulse_f))
        self.central_frequency = MyNamespace(pulse=central_frequency_pulse, gate=None)

        self.N = N
        self.cut_off_val = cut_off_val
        self.interpolate_fft_conform = interpolate_fft_conform
        self.frequency_range = frequency_range
        self.f_range_fields = f_range_fields
        self.f_range_pulse = f_range_pulse
        
        self.cross_correlation = cross_correlation
        self.interferometric = interferometric

        self.c0 = c0

        if self.f_range_fields!=None:
            _fmin, _fmax = self.f_range_fields
            mask = jnp.zeros(jnp.size(self.frequency))
            idx0, idx1 = jnp.argmin(jnp.abs(self.frequency - _fmin)), jnp.argmin(jnp.abs(self.frequency - _fmax))
            self.mask = mask.at[idx0:idx1+1].set(1)
        else:
            self.mask = 1


        self.measurement_info = MyNamespace(mask=self.mask,
                                       time=self.time, frequency=self.frequency, sk=self.sk, rn=self.rn,
                                       time_big=self.time, frequency_big=self.frequency, sk_big=self.sk, rn_big=self.rn,
                                       time_exp=self.time, frequency_exp=self.frequency, sk_exp=self.sk, rn_exp=self.rn,
                                       cross_correlation=self.cross_correlation, 
                                       interferometric=self.interferometric, 
                                       doubleblind=False, 
                                       nonlinear_method=self.nonlinear_method,
                                       central_frequency = self.central_frequency, c0 = self.c0)


    def _generate_trace(self):
        individual, measurement_info, transform_arr = self.get_parameters_to_make_signal_t()
        self.signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        self.trace = jnp.abs(self.signal_t.signal_f)**2

    def generate_trace(self):
        self._generate_trace()
        time, frequency, trace, spectra = self.interpolate_trace()

        self.trace = trace/np.max(trace)
        return time, frequency, self.trace, spectra



    
    def interpolate_trace(self, is_delay_based=True): # this is way to complicated
        max_val = np.max(self.trace)

        idx = np.where(self.trace>max_val*self.cut_off_val)
        idx_0, idx_1 = np.sort(idx)

        idx_0_min, idx_0_max = idx_0[0], idx_0[-1]+1
        idx_1_min, idx_1_max = idx_1[0], idx_1[-1]+1


        time_zoom = self.theta[idx_0_min:idx_0_max]
        frequency_zoom = self.frequency[idx_1_min:idx_1_max]

        if self.frequency_range!=None:
            fmin, fmax = self.frequency_range
            if self.nonlinear_method=="sd":
                fmin, fmax = np.sort([-1*fmin, -1*fmax])
        else:
            fmin, fmax = np.min(frequency_zoom), np.max(frequency_zoom)

        if is_delay_based==True:
            if self.interpolate_fft_conform==True:
                central_f = (fmin + fmax)/2
                df = 1/np.abs((self.theta[-1] - self.theta[0]))

                fmin = central_f - df*self.N/2
                fmax = central_f + df*self.N/2
                time_interpolate = self.theta
                frequency_interpolate = np.linspace(fmin, fmax, self.N)
            else:
                time_interpolate = self.theta
                frequency_interpolate = np.linspace(fmin, fmax, self.N)
        else:		
            time_interpolate = self.theta
            frequency_interpolate = np.linspace(fmin, fmax, self.N)


        trace_interpolate = jax.vmap(self.do_interpolation, in_axes=(None,None,0))(frequency_interpolate, self.frequency, self.trace)

        if is_delay_based==True:
            trace_interpolate = jax.vmap(self.do_interpolation, in_axes=(None,None,1))(time_interpolate, self.theta, trace_interpolate)
            trace_interpolate = np.abs(trace_interpolate).T
        else:
            trace_interpolate = np.abs(trace_interpolate)


        if self.nonlinear_method=="sd":
            frequency_interpolate = -1*np.flip(frequency_interpolate)
            trace_interpolate = np.flip(trace_interpolate, axis=0)


        frequency_pulse_spectrum, spectrum_pulse = interpolate_spectrum(self.frequency, self.pulse_f, self.N, self.do_interpolation)
        if self.cross_correlation==True:
            frequency_gate_spectrum, spectrum_gate = interpolate_spectrum(self.frequency, self.gate_f, self.N, self.do_interpolation)
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
            ax1.plot(time, np.abs(self.gate_t), label="Gate-Pulse", c="tab:red")
            ax2.plot(time, np.unwrap(np.angle(self.gate_t))*1/np.pi, label="Gate-Pulse", c="tab:green")

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


    def get_gate_pulse(self, frequency_gate, gate_f):
        gate_f = self.do_interpolation(self.frequency, frequency_gate, gate_f)
        self.gate_f = gate_f
        self.gate_t = self.ifft(gate_f, self.sk, self.rn)

        central_frequency_gate = jnp.sum(jnp.abs(gate_f)*self.frequency)/jnp.sum(jnp.abs(gate_f))
        self.central_frequency = self.central_frequency.expand(gate=central_frequency_gate)

        self.measurement_info = self.measurement_info.expand(gate=self.gate_t, 
                                                             central_frequency=self.central_frequency)
        return self.gate_t




class MakeTraceFROG(MakeTraceBASE, RetrievePulsesFROG):
    def __init__(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, cross_correlation, interferometric, 
                 frequency_range, f_range_fields, f_range_pulse, N, cut_off_val, interpolate_fft_conform):
        
        super().__init__(time, frequency, pulse_t, pulse_f, nonlinear_method, 
                         N, cut_off_val, interpolate_fft_conform, frequency_range, f_range_fields, f_range_pulse, 
                         cross_correlation, interferometric)
        
        self.theta = delay
       

    def get_parameters_to_make_signal_t(self):
        individual = MyNamespace(pulse=self.pulse_f, gate=self.gate_f)
        return individual, self.measurement_info, self.theta




class MakeTraceTDP(MakeTraceBASE, RetrievePulsesTDP):
    def __init__(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, spectral_filter, cross_correlation, interferometric, 
                                   frequency_range, f_range_fields, f_range_pulse, N, cut_off_val, interpolate_fft_conform):
        
        super().__init__(time, frequency, pulse_t, pulse_f, nonlinear_method, 
                         N, cut_off_val, interpolate_fft_conform, frequency_range, f_range_fields, f_range_pulse, 
                         cross_correlation, interferometric)
        
        self.theta = delay

        if spectral_filter==None:
            self.spectral_filter = jnp.ones(jnp.size(frequency))
        else:
            self.spectral_filter = spectral_filter


    def get_parameters_to_make_signal_t(self):
        individual = MyNamespace(pulse=self.pulse_f, gate=self.gate_f)
        self.measurement_info = self.measurement_info.expand(spectral_filter=self.spectral_filter)
        return individual, self.measurement_info, self.theta

    









class MakeTraceCHIRPSCAN(MakeTraceBASE, RetrievePulsesCHIRPSCAN):
    def __init__(self, time, frequency, pulse_t, pulse_f, nonlinear_method, theta, phase_type, chirp_parameters, frequency_range, f_range_fields, f_range_pulse, N, cut_off_val):
        
        super().__init__(time, frequency, pulse_t, pulse_f, nonlinear_method, 
                         N, cut_off_val, False, frequency_range, f_range_fields, f_range_pulse, 
                         False, False)

        self.theta = theta
        self.theta = theta
        self.phase_type = phase_type
        self.chirp_parameters = chirp_parameters


    def get_parameters_to_make_signal_t(self):
        self.measurement_info = self.measurement_info.expand(theta = self.theta)
        self.phase_matrix = self.get_phase_matrix(self.chirp_parameters)
        individual = MyNamespace(pulse=self.pulse_f, gate=None)
        return individual, self.measurement_info, self.phase_matrix
    
    
    def interpolate_trace(self):
        return super().interpolate_trace(is_delay_based=False)




class MakeTrace2DSI(MakeTraceBASE, RetrievePulses2DSI):
    def __init__(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, spectral_filter1, spectral_filter2, tau_pulse_anc1, 
                 material_thickness, refractive_index, cross_correlation, frequency_range, f_range_fields, f_range_pulse, N, cut_off_val, 
                 interpolate_fft_conform):
        super().__init__(time, frequency, pulse_t, pulse_f, nonlinear_method, 
                         N, cut_off_val, interpolate_fft_conform, frequency_range, f_range_fields, f_range_pulse, 
                         cross_correlation, False)

        self.theta = delay
        self.tau_pulse_anc1 = tau_pulse_anc1
        self.refractive_index, self.material_thickness = refractive_index, material_thickness

        if spectral_filter1==None:
            self.spectral_filter1 = jnp.ones(jnp.size(self.frequency))
        else:
            self.spectral_filter1 = spectral_filter1

        if spectral_filter2==None:
            self.spectral_filter2 = jnp.ones(jnp.size(self.frequency))
        else:
            self.spectral_filter2 = spectral_filter2



    def get_parameters_to_make_signal_t(self):
        self.measurement_info = self.measurement_info.expand(tau_pulse_anc1 = self.tau_pulse_anc1, 
                                                             spectral_filter1=self.spectral_filter1, 
                                                             spectral_filter2=self.spectral_filter2)
        individual = MyNamespace(pulse=self.pulse_f, gate=self.gate_f)
        return individual, self.measurement_info, self.theta
    




class MakeTraceVAMPIRE(MakeTraceBASE, RetrievePulsesVAMPIRE):
    def __init__(self, time, frequency, pulse_t, pulse_f, nonlinear_method, delay, tau_interferometer, material_thickness, 
                 refractive_index, cross_correlation, frequency_range, f_range_fields, f_range_pulse, N, cut_off_val, interpolate_fft_conform):
        
        super().__init__(time, frequency, pulse_t, pulse_f, nonlinear_method, 
                         N, cut_off_val, interpolate_fft_conform, frequency_range, f_range_fields, f_range_pulse, 
                         cross_correlation, False)
        self.theta = delay
        self.tau_interferometer = tau_interferometer
        self.refractive_index, self.material_thickness = refractive_index, material_thickness
        


    def get_parameters_to_make_signal_t(self):
        self.phase_matrix = self.get_phase_matrix(self.refractive_index, self.material_thickness, self.measurement_info)
        self.measurement_info = self.measurement_info.expand(tau_interferometer = self.tau_interferometer, 
                                                             phase_matrix = self.phase_matrix)
        individual = MyNamespace(pulse=self.pulse_f, gate=self.gate_f)
        return individual, self.measurement_info, self.theta
    






class MakeTraceSTREAKING(MakeTraceBASE, RetrievePulsesSTREAKING):
    def __init__(self, time, frequency, pulse_t, pulse_f, delay, Ip_eV,
                 energy_range, N, cut_off_val, interpolate_fft_conform):
        
        nonlinear_method = None
        f_range_fields = f_range_pulse = None
        interferometric = False
        cross_correlation = True

        self.dtme_momentum = None
        self.ionization_potential = RetrievePulsesSTREAKING.convert_energy_eV_au(RetrievePulsesSTREAKING, Ip_eV, "eV", "au")
        self.theta = RetrievePulsesSTREAKING.convert_time_fs_au(RetrievePulsesSTREAKING, delay, "fs", "au")

        time = RetrievePulsesSTREAKING.convert_time_fs_au(RetrievePulsesSTREAKING, time, "fs", "au")
        frequency = RetrievePulsesSTREAKING.convert_frequency_PHz_au(RetrievePulsesSTREAKING, frequency, "PHz", "au")
        self.energy_au = frequency*2*jnp.pi

        #self.energy_au = jnp.linspace(0, jnp.max(energy_au), jnp.size(energy_au)//2)

        momentum_au = jnp.sqrt(2*jnp.abs(self.energy_au))*jnp.sign(self.energy_au)
        self.momentum_au = jnp.linspace(jnp.min(momentum_au), jnp.max(momentum_au), jnp.size(momentum_au))
        self.position_au = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(self.momentum_au), jnp.mean(jnp.diff(self.momentum_au))))
        self.sk_position_momentum, self.rn_position_momentum = get_sk_rn(self.position_au, self.momentum_au)

        
        if energy_range is not None:
            emin, emax = energy_range
            emin = RetrievePulsesSTREAKING.convert_energy_eV_au(RetrievePulsesSTREAKING, emin, "eV", "au")
            emax = RetrievePulsesSTREAKING.convert_energy_eV_au(RetrievePulsesSTREAKING, emax, "eV", "au")
            fmin, fmax = emin/(2*jnp.pi), emax/(2*jnp.pi)
            frequency_range = (fmin, fmax)
        else:
            frequency_range = None


        super().__init__(time, frequency, pulse_t, pulse_f, nonlinear_method, 
                         N, cut_off_val, interpolate_fft_conform, frequency_range, f_range_fields, f_range_pulse, 
                         cross_correlation, interferometric)


    def get_gate_pulse(self, frequency_gate, gate_f):
        frequency_gate = RetrievePulsesSTREAKING.convert_frequency_PHz_au(RetrievePulsesSTREAKING, frequency_gate, "PHz", "au")
        return super().get_gate_pulse(frequency_gate, gate_f)
    

    def get_DTME(self, momentum_au, dtme_momentum):
        if dtme_momentum.ndims==1:
            dtme_momentum = jnp.asarray([self.do_interpolation(self.momentum_au, momentum_au, dtme_momentum)])
        else:
            dtme_momentum = jax.vmap(self.do_interpolation, in_axes=(None,None,0))(self.momentum_au, momentum_au, dtme_momentum)
        self.dtme_momentum = dtme_momentum
        return self.dtme_momentum
    
    
    def get_parameters_to_make_signal_t(self):
        pulse_t_nir = self.ifft(self.pulse_f, self.sk, self.rn)
        pulse_t_nir_vectorpotential = -1 * integrate_signal_1D(pulse_t_nir, self.time, integration_method="cumsum", integration_order=None)
        pulse_f_nir_vectorpotential = self.fft(pulse_t_nir_vectorpotential, self.sk, self.rn)

        if self.dtme_momentum is None:
            dtme = jnp.ones((jnp.size(self.ionization_potential), jnp.size(self.momentum_au)))
        else:
            dtme = self.dtme_momentum

        self.measurement_info = self.measurement_info.expand(momentum = self.momentum_au,
                                                             position = self.position_au,
                                                             sk_position_momentum = self.sk_position_momentum,
                                                             rn_position_momentum = self.rn_position_momentum, 
                                                             ionization_potential=self.ionization_potential, 
                                                             dtme_momentum = dtme,
                                                             retrieve_dtme = False)
        

        individual = MyNamespace(pulse=pulse_f_nir_vectorpotential, gate=self.gate_f, dtme=None)
        return individual, self.measurement_info, self.theta


    def _generate_trace(self):
        super()._generate_trace()
        momentum_au_nonuniform = jnp.sqrt(2*jnp.abs(self.energy_au))*jnp.sign(self.energy_au)
        self.trace = jax.vmap(self.do_interpolation, 
                              in_axes=(None,None,0))(momentum_au_nonuniform, self.momentum_au, self.trace)


    def generate_trace(self):
        time, frequency, trace, spectra = super().generate_trace()

        time = RetrievePulsesSTREAKING.convert_time_fs_au(RetrievePulsesSTREAKING, time, "au", "fs")
        frequency = RetrievePulsesSTREAKING.convert_frequency_PHz_au(RetrievePulsesSTREAKING, frequency, "au", "PHz")

        fp, p = spectra.pulse
        fp = RetrievePulsesSTREAKING.convert_frequency_PHz_au(RetrievePulsesSTREAKING, fp, "au", "PHz")
        fg, g = spectra.gate
        fg = RetrievePulsesSTREAKING.convert_frequency_PHz_au(RetrievePulsesSTREAKING, fg, "au", "PHz")
        spectra = MyNamespace(pulse=(fp,p), gate=(fg,g))
        
        return time, frequency, trace, spectra







class MakeTraceFROGReal(MakeTraceFROG, RetrievePulsesFROGwithRealFields):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



class MakeTraceTDPReal(MakeTraceTDP, RetrievePulsesTDPwithRealFields):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



class MakeTraceCHIRPSCANReal(MakeTraceCHIRPSCAN, RetrievePulsesCHIRPSCANwithRealFields):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




class MakeTrace2DSIReal(MakeTrace2DSI, RetrievePulses2DSIwithRealFields):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    


class MakeTraceVAMPIREReal(MakeTraceVAMPIRE, RetrievePulsesVAMPIREwithRealFields):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# this is unnecessary and useless
# class MakeTraceSTREAKINGReal(MakeTraceSTREAKING, RetrievePulsesSTREAKINGwithRealFields):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
