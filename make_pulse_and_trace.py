import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

import refractiveindex


import jax.numpy as jnp
import jax

from utilities import MyNamespace, do_fft, do_ifft, get_sk_rn, generate_random_continuous_function, do_interpolation_1d
from BaseClasses import RetrievePulsesFROG, RetrievePulsesDSCAN, RetrievePulsesFROGwithRealFields, RetrievePulsesDSCANwithRealFields





def apply_noise(trace, scale_val=0.01, additive_noise=False, multiplicative_noise=False):
    trace=trace/np.max(trace)
    shape=np.shape(trace)


    if additive_noise==True and multiplicative_noise:
        assert len(scale_val)==2, "scale_val needs to have len=2 when using both additive and multiplicative"

        noise_additive=np.random.normal(0, scale_val[0], size=shape)
        noise_multiplicative=np.random.normal(1, scale_val[1], size=shape)
        trace = trace*np.abs(noise_multiplicative) + noise_additive

    elif multiplicative_noise==True:
        noise=np.random.normal(1, scale_val, size=shape)
        trace=trace*np.abs(noise)

    elif additive_noise==True:
        noise=np.random.normal(0, scale_val, size=shape)
        trace=trace+noise

    else:
        print("something is wrong")

    return trace



class MakePulse:
    def __init__(self, N=256, Deltaf=1):
        self.N=N
        self.Deltaf=Deltaf

        self.df=self.Deltaf/self.N

        self.no_points_random_phase=10
        self.multi_pulse_time_domain_length_multiplier=8


        self.minval_rand_phase, self.maxval_rand_phase = -4*np.pi, 4*np.pi



    def gaussian(self, x, amp, fwhm, shift):
        b=fwhm/2.355 # fwhm to sigma
        return amp*np.exp(-(x-shift)**2/(2*b**2))
    

    def generate_polynomial_phase(self, frequency, central_frequency, parameters):
        n=len(parameters)
        phase=0
        for i in range(n):
            phase=phase+parameters[i]*(frequency-central_frequency)**i
    
        return 2*np.pi*phase
    
    def generate_sinusoidal_phase(self, frequency, parameters):
        n=len(parameters)
        phase=0
        for i in range(n):
            a,b,c=parameters[i]
            phase=phase+a*np.sin(2*np.pi*b*(frequency-c))
        
        return 2*np.pi*phase
    
    def interpolate_custom_phase(self, frequency, custom_frequency, custom_phase):
        return do_interpolation_1d(frequency, custom_frequency, custom_phase)


    def get_spectral_phase(self, frequency, central_frequency, phase_type="flat", parameters_phase=None, amp_f=None):
        if phase_type=="flat":
            phase=np.zeros(self.N)
        elif phase_type=="polynomial":
            assert parameters_phase!=None, "You need to provide parameters for the polynomial phase."
            phase=self.generate_polynomial_phase(frequency, central_frequency, parameters_phase)
        elif phase_type=="sinusoidal":
            assert parameters_phase!=None, "You need to provide parameters for the polynomial phase."
            phase=self.generate_sinusoidal_phase(frequency, parameters_phase)
        elif phase_type=="random":
            phase=generate_random_continuous_function(jax.random.PRNGKey(np.random.randint(0,1e9)), self.no_points_random_phase, frequency, 
                                                      self.minval_rand_phase, self.maxval_rand_phase, gaussian_filter1d(amp_f, sigma=10))
        elif phase_type=="custom":
            phase=self.interpolate_custom_phase(frequency, parameters_phase[0], parameters_phase[1])
        else:
            print("something went wrong?")

        return phase
    

    def get_multi_pulse(self, shift_arr, duration_arr, central_frequency_arr, amplitude_arr, phase_type_arr, parameters_phase_arr):
        DeltaT=(np.max(shift_arr)-np.min(shift_arr))*self.multi_pulse_time_domain_length_multiplier
        time=np.linspace(-DeltaT/2, DeltaT/2, self.N)
        cf=np.mean(central_frequency_arr)
        frequency=np.fft.fftshift(np.fft.fftfreq(self.N, np.mean(np.diff(time))))+cf

        sk, rn = get_sk_rn(time, frequency)

        pulse_total=0
        for i in range(len(shift_arr)):
            pulse_t=self.gaussian(time, amplitude_arr[i], duration_arr[i], shift_arr[i])*np.exp(1j*2*np.pi*central_frequency_arr[i]*time)
            pulse_f=do_fft(pulse_t, sk, rn)
            phase_f=self.get_spectral_phase(frequency, central_frequency_arr[i], phase_type=phase_type_arr[i], 
                                            parameters_phase=parameters_phase_arr[i], amp_f=np.abs(pulse_f))

            pulse_total=pulse_total+pulse_f*np.exp(1j*phase_f)
            
        pulse_f=pulse_total

        return time, do_ifft(pulse_f, sk, rn), frequency, pulse_f
        

    def generate_pulse_t(self, spectral_amp_parameters=[0.5, 0.1], type="flat", parameters_phase=None, multi_pulse_parameters=None):
        if np.size(spectral_amp_parameters)==2:
            self.central_f=spectral_amp_parameters[0]
            fwhm_f=spectral_amp_parameters[1]
            broadband_pulse=False
        else:
            self.central_f=np.mean(spectral_amp_parameters,axis=0)[0]
            broadband_pulse=True


        frequency=np.linspace(self.central_f-self.df*self.N/2, self.central_f+self.df*self.N/2, self.N)
        time=np.fft.fftshift(np.fft.fftfreq(self.N, self.df))
        sk, rn = get_sk_rn(time, frequency)

        if multi_pulse_parameters==None:
            if broadband_pulse==True:
                shape=np.shape(spectral_amp_parameters)
                amp=0
                for i in range(shape[0]):
                    central_f=spectral_amp_parameters[i][0]
                    fwhm_f=spectral_amp_parameters[i][1]
                    a=spectral_amp_parameters[i][2]
                    amp=amp+self.gaussian(frequency, a, fwhm_f, central_f)
            else:
                amp=self.gaussian(frequency, 1, fwhm_f, self.central_f)
            phase=self.get_spectral_phase(frequency, self.central_f, phase_type=type, parameters_phase=parameters_phase, amp_f=amp)
            pulse_f=amp*np.exp(1j*phase)
            pulse_t=do_ifft(pulse_f, sk, rn)

        else:
            delay_arr=multi_pulse_parameters[0]
            duration_arr=multi_pulse_parameters[1]
            central_frequency_arr=multi_pulse_parameters[2]
            amplitude_arr=multi_pulse_parameters[3]
            phase_type_arr=multi_pulse_parameters[4]
            parameters_phase_arr=multi_pulse_parameters[5]

            delay_arr=list(delay_arr)
            delay_arr.insert(0,0)
            shift_arr=np.cumsum(delay_arr)-np.mean(np.cumsum(delay_arr))

            time, pulse_t, frequency, pulse_f = self.get_multi_pulse(shift_arr, duration_arr, central_frequency_arr, amplitude_arr, phase_type_arr, parameters_phase_arr)


        self.input_pulses=MyNamespace(time=time, frequency=frequency, pulse_t=pulse_t, pulse_f=pulse_f)
        return time, pulse_t, frequency, pulse_f
    








    def generate_frog_trace_and_spectrum(self, time, frequency, pulse_t, pulse_f, nonlinear_method, N=256, scale_time_range=1, plot_stuff=True, 
                                    xfrog=False, gate=(None, None), ifrog=False, interpolate_fft_conform=True, cut_off_val=0.001, frequency_range=None, 
                                    real_fields=False):
        
        if real_fields==True:
            self.maketrace = MakeFROGTraceReal(time, frequency, pulse_t, pulse_f, nonlinear_method, N, scale_time_range, xfrog, ifrog, 
                                             interpolate_fft_conform, cut_off_val, frequency_range)
        else:
            self.maketrace = MakeFROGTrace(time, frequency, pulse_t, pulse_f, nonlinear_method, N, scale_time_range, xfrog, ifrog, 
                                         interpolate_fft_conform, cut_off_val, frequency_range)


        if xfrog==True:
            frequency_gate, gate_f = gate
            gate = self.maketrace.get_gate_pulse(frequency_gate, gate_f, time, frequency)

        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()

        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, time_trace, frequency_trace, trace, spectra)

        return time_trace, frequency_trace, trace, spectra
    




    def generate_dscan_trace_and_spectrum(self, z_arr, time, frequency, pulse_t, pulse_f, nonlinear_method, N=256, plot_stuff=True, 
                                          cut_off_val=0.001, frequency_range=None, real_fields=False):

        if real_fields==True:
            self.maketrace = MakeDScanTraceReal(z_arr, time, frequency, pulse_t, pulse_f, nonlinear_method, N, cut_off_val, frequency_range)
        else:
            self.maketrace = MakeDScanTrace(z_arr, time, frequency, pulse_t, pulse_f, nonlinear_method, N, cut_off_val, frequency_range)

        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()
            
        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, z_arr, frequency_trace, trace, spectra)

        return z_arr, frequency_trace, trace, spectra
    



























def interpolate_spectrum(frequency, pulse_f, N):
    spectrum=jnp.abs(pulse_f)**2
    spectrum=spectrum/jnp.max(spectrum)

    idx=np.where(spectrum>1e-5)
    idx_1 = np.sort(idx)[0]
    idx_1_min, idx_1_max = idx_1[0], idx_1[-1]+1
    
    frequency_zoom = frequency[idx_1_min:idx_1_max]
    frequency_interpolate_spectrum = np.linspace(frequency_zoom[0], frequency_zoom[-1], N)
    
    spectrum = do_interpolation_1d(frequency_interpolate_spectrum, frequency, spectrum)
    spectrum = spectrum/np.max(spectrum)
    return frequency_interpolate_spectrum, spectrum



class MakeTrace:
    def __init__(self, *args, **kwargs):
        pass


    def generate_trace(self):
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        if self.nonlinear_method=="shg":
            self.factor=2
        elif self.nonlinear_method=="thg":
            self.factor=3
        else:
            self.factor=1

        individual, measurement_info, transform_arr = self.get_parameters_to_make_signal_t()
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        signal_f = do_fft(signal_t.signal_t, self.sk, self.rn)
        self.trace = jnp.abs(signal_f)**2


        time, frequency, trace, spectra = self.interpolate_trace()

        self.trace=trace/np.max(trace)
        return time, frequency, self.trace, spectra


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

        ax1=plt.subplot(2,2,2)
        ax1.plot(frequency, np.abs(pulse_f), label="Amplitude")
        ax1.set_xlabel("Frequency [PHz]")
        ax1.set_ylabel("Amplitude [arb. u.]")
        ax1.legend(loc=2)

        ax2 = ax1.twinx()
        ax2.plot(frequency, np.unwrap(np.angle(pulse_f))*1/np.pi, c="tab:orange", label="Phase")
        ax2.set_ylabel(r"Phase [$\pi$]")
        ax2.legend(loc=1)


        plt.subplot(2,2,3)
        plt.plot(spectra.pulse[0], spectra.pulse[1], label="Pulse Spectrum")
        plt.xlabel("Frequency [PHz]")
        plt.ylabel("Amplitude [arb. u.]")
        plt.legend()

        plt.subplot(2,2,4)
        plt.pcolormesh(x_arr, frequency_trace, trace.T, cmap="nipy_spectral")
        plt.xlabel("Shift [arb. u.]")
        plt.ylabel("Frequency [PHz]")
        plt.colorbar()






class MakeFROGTrace(MakeTrace, RetrievePulsesFROG):
    def __init__(self, time, frequency, pulse_t, pulse_f, nonlinear_method, N, scale_time_range, xfrog, ifrog, 
                                             interpolate_fft_conform, cut_off_val, frequency_range):
        

        self.time=time
        self.frequency=frequency
        self.pulse_t=pulse_t
        self.pulse_f=pulse_f
        self.nonlinear_method=nonlinear_method
        self.N=N
        self.scale_time_range=scale_time_range
        self.interpolate_fft_conform=interpolate_fft_conform
        self.cut_off_val = cut_off_val
        self.frequency_range = frequency_range
        self.xfrog=xfrog
        self.ifrog=ifrog
        self.gate = None

        self.x_arr = time


    def get_gate_pulse(self, frequency_gate, gate_f, time, frequency):
        gate_f=do_interpolation_1d(frequency, frequency_gate, gate_f)

        self.sk, self.rn = get_sk_rn(time, frequency)
        self.gate=do_ifft(gate_f, self.sk, self.rn)
        return self.gate



    def get_parameters_to_make_signal_t(self):
        measurement_info = MyNamespace(xfrog_gate=self.gate, time=self.time, frequency=self.frequency, xfrog=self.xfrog, ifrog=self.ifrog, 
                                       nonlinear_method=self.nonlinear_method, doubleblind=False)
        individual = MyNamespace(pulse=self.pulse_t, gate=self.gate)
        return individual, measurement_info, self.time
        
    
    def interpolate_trace(self):
        max_val=np.max(self.trace)

        idx=np.where(self.trace>max_val*self.cut_off_val)
        idx_0, idx_1 = np.sort(idx)

        idx_0_min, idx_0_max = idx_0[0], idx_0[-1]+1
        idx_1_min, idx_1_max = idx_1[0], idx_1[-1]+1


        time_zoom=self.time[idx_0_min:idx_0_max]
        frequency_zoom=self.frequency[idx_1_min:idx_1_max]

        if self.frequency_range!=None:
            fmin, fmax = self.frequency_range
        else:
            fmin, fmax = np.min(frequency_zoom), np.max(frequency_zoom)
            deltaf = fmax - fmin
            fmin = fmin - deltaf/2
            fmax = fmax + deltaf/2
            

        if self.interpolate_fft_conform==True:
            central_f=(fmin+fmax)/2
            df=1/np.abs((time_zoom[-1]-time_zoom[0])*self.scale_time_range)

            frequency_min=central_f-df*self.N/2
            frequency_max=central_f+df*self.N/2

            frequency_interpolate=np.linspace(frequency_min, frequency_max, self.N)
            time_interpolate=np.fft.fftshift(np.fft.fftfreq(len(frequency_interpolate), df))

        else:
            frequency_interpolate = np.linspace(fmin, fmax, self.N)

            t_central = (time_zoom[0]+time_zoom[-1])/2
            Delta_t = np.abs(time_zoom[-1]-time_zoom[0])*self.scale_time_range
            time_interpolate = np.linspace(t_central-Delta_t/2, t_central+Delta_t/2, self.N)
        

        trace_interpolate_freq = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(frequency_interpolate, self.frequency, self.trace)
        trace_interpolate = jax.vmap(do_interpolation_1d, in_axes=(None,None,1))(time_interpolate, self.time, trace_interpolate_freq)

        if self.nonlinear_method=="sd":
            frequency_interpolate = -1*frequency_interpolate
            trace_interpolate = np.flip(trace_interpolate, axis=0)
        

        frequency_pulse_spectrum, spectrum_pulse = interpolate_spectrum(self.frequency, self.pulse_f, self.N)
        if self.xfrog==True:
            gate_f=do_fft(self.gate, self.sk, self.rn)
            frequency_gate_spectrum, spectrum_gate = interpolate_spectrum(self.frequency, gate_f, self.N)
        else:
            frequency_gate_spectrum, spectrum_gate = None, None
        spectra = MyNamespace(pulse=(frequency_pulse_spectrum, spectrum_pulse), 
                              gate=(frequency_gate_spectrum, spectrum_gate))

        return time_interpolate, frequency_interpolate, np.abs(trace_interpolate).T, spectra
    






class MakeDScanTrace(MakeTrace, RetrievePulsesDSCAN):
    def __init__(self, z_arr, time, frequency, pulse_t, pulse_f, nonlinear_method, N, cut_off_val, frequency_range, 
                 refractive_index = refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson")):
        self.refractive_index=refractive_index

        from scipy.constants import c as c0
        self.c0 = c0

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



    def get_parameters_to_make_signal_t(self):
        measurement_info = MyNamespace(frequency=self.frequency, c0=self.c0, sk=self.sk, rn=self.rn, nonlinear_method=self.nonlinear_method, doubleblind=False)
        individual = MyNamespace(pulse=self.pulse_f, gate=None)

        phase_matrix = self.get_phase_matrix(self.refractive_index, self.z_arr, measurement_info)
        return individual, measurement_info, phase_matrix


    
    def interpolate_trace(self):
        max_val=np.max(self.trace)
        idx=np.where(self.trace>max_val*self.cut_off_val)
        idx_0, idx_1 = np.sort(idx)

        idx_0_min, idx_0_max = idx_0[0], idx_0[-1]+1
        idx_1_min, idx_1_max = idx_1[0], idx_1[-1]+1

        frequency_zoom=self.frequency[idx_1_min:idx_1_max]

        if self.frequency_range!=None:
            fmin, fmax = self.frequency_range
        else:
            fmin, fmax = np.min(frequency_zoom), np.max(frequency_zoom)
            deltaf=fmax-fmin
            fmin=fmin-deltaf/2
            fmax=fmax+deltaf/2

        frequency_interpolate=np.linspace(fmin, fmax, self.N)

        trace_interpolate=jax.vmap(do_interpolation_1d, in_axes=(None, None, 0))(frequency_interpolate, self.frequency, self.trace)

        if self.nonlinear_method=="sd":
            frequency_interpolate = -1*frequency_interpolate
            trace_interpolate = np.flip(trace_interpolate, axis=1)

        frequency_interpolate_spectrum, spectrum = interpolate_spectrum(self.frequency, self.pulse_f, self.N)
        spectra = MyNamespace(pulse=(frequency_interpolate_spectrum, spectrum), gate=None)
        
        return self.time, frequency_interpolate, np.abs(trace_interpolate), spectra




class MakeFROGTraceReal(RetrievePulsesFROGwithRealFields, MakeFROGTrace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




class MakeDScanTraceReal(RetrievePulsesDSCANwithRealFields, MakeDScanTrace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
