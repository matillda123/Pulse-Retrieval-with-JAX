import numpy as np
import matplotlib.pyplot as plt

import refractiveindex

import jax.numpy as jnp
import jax

from utilities import MyNamespace, do_fft, do_ifft, get_sk_rn, do_interpolation_1d, center_signal_to_max
from BaseClasses import RetrievePulsesFROG, RetrievePulsesCHIRPSCAN, RetrievePulsesFROGwithRealFields, RetrievePulsesCHIRPSCANwithRealFields, RetrievePulses2DSI
from make_pulse import MakePulse as MakePulseBase



def apply_noise(trace, scale_val=0.01, additive_noise=False, multiplicative_noise=False):
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







class MakePulse(MakePulseBase):
    """ 
    Simulates measurement traces based in input pulses.
    Inherits from make_pulse.MakePulse.

    Attributes:
        maketrace: None, MakeTraceFROG, MakeTraceCHIRPSCAN or MakeTrace2DSI,

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maketrace = None

    
    def generate_frog(self, time, frequency, pulse_t, pulse_f, nonlinear_method, N=256, scale_time_range=1, plot_stuff=True, 
                                    cross_correlation=False, gate=(None, None), ifrog=False, interpolate_fft_conform=True, cut_off_val=0.001, frequency_range=None, 
                                    real_fields=False):
        
        if real_fields==True:
            self.maketrace = MakeTraceFROGReal(time, frequency, pulse_t, pulse_f, nonlinear_method, N, scale_time_range, cross_correlation, ifrog, 
                                             interpolate_fft_conform, cut_off_val, frequency_range)
        else:
            self.maketrace = MakeTraceFROG(time, frequency, pulse_t, pulse_f, nonlinear_method, N, scale_time_range, cross_correlation, ifrog, 
                                         interpolate_fft_conform, cut_off_val, frequency_range)


        if cross_correlation==True:
            frequency_gate, gate_f = gate
            gate = self.maketrace.get_gate_pulse(frequency_gate, gate_f, time, frequency)

        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()

        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, time_trace, frequency_trace, trace, spectra)

        return time_trace, frequency_trace, trace, spectra
    




    def generate_chirpscan(self, z_arr, time, frequency, pulse_t, pulse_f, nonlinear_method, phase_matrix_func, parameters, N=256, plot_stuff=True, 
                                          cut_off_val=0.001, frequency_range=None, real_fields=False):

        if real_fields==True:
            self.maketrace = MakeTraceCHIRPSCANReal(z_arr, time, frequency, pulse_t, pulse_f, nonlinear_method, N, cut_off_val, frequency_range, phase_matrix_func, parameters)
        else:
            self.maketrace = MakeTraceCHIRPSCAN(z_arr, time, frequency, pulse_t, pulse_f, nonlinear_method, N, cut_off_val, frequency_range, phase_matrix_func, parameters)

        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()
            
        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, z_arr, frequency_trace, trace, spectra)

        return z_arr, frequency_trace, trace, spectra
    




    def generate_2dsi(self, time, frequency, pulse_t, pulse_f, nonlinear_method, cross_correlation=True, anc=((None,None),(None,None)), N=256, scale_time_range=1, 
                                         plot_stuff=True, 
                                         cut_off_val=0.001, frequency_range=None):

        
        self.maketrace = MakeTrace2DSI(time, frequency, pulse_t, pulse_f, nonlinear_method, cross_correlation, N, scale_time_range, cut_off_val, frequency_range)

        if self.maketrace.cross_correlation==True:
            anc_1, anc_2 = anc
            frequency_anc_1, anc_1_f = anc_1
            frequency_anc_2, anc_2_f = anc_2
            anc1 = self.maketrace.get_gate_pulse(frequency_anc_1, anc_1_f, anc_no=1)
            anc2 = self.maketrace.get_gate_pulse(frequency_anc_2, anc_2_f, anc_no=2)

        time_trace, frequency_trace, trace, spectra = self.maketrace.generate_trace()
            
        if plot_stuff==True:
            self.maketrace.plot_trace(time, pulse_t, frequency, pulse_f, time_trace, frequency_trace, trace, spectra)

        return time_trace, frequency_trace, trace, spectra
    




















def interpolate_spectrum(frequency, pulse_f, N):
    spectrum=jnp.abs(pulse_f)**2
    spectrum=spectrum/jnp.max(spectrum)

    idx=np.where(spectrum>1e-5)
    idx_1 = np.sort(idx)[0]
    idx_1_min, idx_1_max = idx_1[0], idx_1[-1]+1
    
    frequency_zoom = frequency[idx_1_min:idx_1_max]
    frequency_interpolate_spectrum = np.linspace(frequency_zoom[0], frequency_zoom[-1], N)
    
    spectrum = do_interpolation_1d(frequency_interpolate_spectrum, frequency, spectrum, method="linear")
    spectrum = spectrum/np.max(spectrum)
    return frequency_interpolate_spectrum, spectrum



class MakeTraceBASE:
    def __init__(self, *args, **kwargs):
        self.cross_correlation = False


    def generate_trace(self):

        if self.nonlinear_method=="shg":
            self.factor=2
        elif self.nonlinear_method=="thg":
            self.factor=3
        else:
            self.factor=1

        individual, measurement_info, transform_arr = self.get_parameters_to_make_signal_t()

        self.signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        signal_f = do_fft(self.signal_t.signal_t, self.sk, self.rn)
        self.trace = jnp.abs(signal_f)**2

        time, frequency, trace, spectra = self.interpolate_trace()

        self.trace = trace/np.max(trace)
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

        if self.cross_correlation==True and not isinstance(self, MakeTrace2DSI):
            plt.plot(spectra.gate[0], spectra.gate[1], label="Gate Spectrum")

        elif self.cross_correlation==True and isinstance(self, MakeTrace2DSI):
            plt.plot(spectra.anc_1[0], spectra.anc_1[1], label="Anc 1 Spectrum")
            plt.plot(spectra.anc_2[0], spectra.anc_2[1], label="Anc 2 Spectrum")

        plt.xlabel("Frequency [PHz]")
        plt.ylabel("Amplitude [arb. u.]")
        plt.legend()

        plt.subplot(2,2,4)
        plt.pcolormesh(x_arr, frequency_trace, trace.T, cmap="nipy_spectral")
        plt.xlabel("Shift [arb. u.]")
        plt.ylabel("Frequency [PHz]")
        plt.colorbar()






class MakeTraceFROG(MakeTraceBASE, RetrievePulsesFROG):
    def __init__(self, time, frequency, pulse_t, pulse_f, nonlinear_method, N, scale_time_range, cross_correlation, ifrog, 
                                             interpolate_fft_conform, cut_off_val, frequency_range):
        super().__init__()
        
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
        self.cross_correlation=cross_correlation
        self.ifrog=ifrog
        self.gate = None

        self.x_arr = time

        self.sk, self.rn = get_sk_rn(self.time, self.frequency)


    def get_gate_pulse(self, frequency_gate, gate_f, time, frequency):
        gate_f=do_interpolation_1d(frequency, frequency_gate, gate_f)

        self.sk, self.rn = get_sk_rn(time, frequency)
        self.gate=do_ifft(gate_f, self.sk, self.rn)
        return self.gate



    def get_parameters_to_make_signal_t(self):
        measurement_info = MyNamespace(cross_correlation_gate=self.gate, time=self.time, frequency=self.frequency, cross_correlation=self.cross_correlation, ifrog=self.ifrog, 
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
            if self.nonlinear_method=="sd":
                fmin, fmax = np.sort([-1*fmin, -1*fmax])
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
            frequency_interpolate = -1*np.flip(frequency_interpolate)
            trace_interpolate = np.flip(trace_interpolate, axis=0)
        

        frequency_pulse_spectrum, spectrum_pulse = interpolate_spectrum(self.frequency, self.pulse_f, self.N)
        if self.cross_correlation==True:
            gate_f=do_fft(self.gate, self.sk, self.rn)
            frequency_gate_spectrum, spectrum_gate = interpolate_spectrum(self.frequency, gate_f, self.N)
        else:
            frequency_gate_spectrum, spectrum_gate = None, None
        spectra = MyNamespace(pulse=(frequency_pulse_spectrum, spectrum_pulse), 
                              gate=(frequency_gate_spectrum, spectrum_gate))

        return time_interpolate, frequency_interpolate, np.abs(trace_interpolate).T, spectra
    






class MakeTraceCHIRPSCAN(MakeTraceBASE, RetrievePulsesCHIRPSCAN):
    def __init__(self, z_arr, time, frequency, pulse_t, pulse_f, nonlinear_method, N, cut_off_val, frequency_range, phase_matrix_func, parameters):
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

        self.calculate_phase_matrix = phase_matrix_func
        self.parameters = parameters



    def get_dispersed_pulse_t(self, pulse_f, phase_matrix, measurement_info):
        pulse_t_disp, phase_matrix = super().get_dispersed_pulse_t(pulse_f, phase_matrix, measurement_info)
        pulse_t_disp = jax.vmap(center_signal_to_max)(pulse_t_disp)   # THIS FUCKS the retrieval. Only use in generation of traces
        return pulse_t_disp, phase_matrix



    def get_parameters_to_make_signal_t(self):
        self.measurement_info = MyNamespace(z_arr=self.z_arr, frequency=self.frequency, sk=self.sk, rn=self.rn, nonlinear_method=self.nonlinear_method, doubleblind=False)
        individual = MyNamespace(pulse=self.pulse_f, gate=None)

        self.phase_matrix = self.get_phase_matrix(self.parameters)
        return individual, self.measurement_info, self.phase_matrix


    
    def interpolate_trace(self):
        max_val=np.max(self.trace)
        idx=np.where(self.trace>max_val*self.cut_off_val)
        idx_0, idx_1 = np.sort(idx)

        idx_0_min, idx_0_max = idx_0[0], idx_0[-1]+1
        idx_1_min, idx_1_max = idx_1[0], idx_1[-1]+1

        frequency_zoom=self.frequency[idx_1_min:idx_1_max]

        if self.frequency_range!=None:
            fmin, fmax = self.frequency_range
            if self.nonlinear_method=="sd":
                fmin, fmax = np.sort([-1*fmin, -1*fmax])
        else:
            fmin, fmax = np.min(frequency_zoom), np.max(frequency_zoom)
            deltaf=fmax-fmin
            fmin=fmin-deltaf/2
            fmax=fmax+deltaf/2

        frequency_interpolate=np.linspace(fmin, fmax, self.N)

        trace_interpolate=jax.vmap(do_interpolation_1d, in_axes=(None, None, 0))(frequency_interpolate, self.frequency, self.trace)

        if self.nonlinear_method=="sd":
            frequency_interpolate = -1*np.flip(frequency_interpolate)
            trace_interpolate = np.flip(trace_interpolate, axis=1)

        frequency_interpolate_spectrum, spectrum = interpolate_spectrum(self.frequency, self.pulse_f, self.N)
        spectra = MyNamespace(pulse=(frequency_interpolate_spectrum, spectrum), gate=None)
        
        return self.time, frequency_interpolate, np.abs(trace_interpolate), spectra





class MakeTraceFROGReal(RetrievePulsesFROGwithRealFields, MakeTraceFROG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




class MakeTraceCHIRPSCANReal(RetrievePulsesCHIRPSCANwithRealFields, MakeTraceCHIRPSCAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)








class MakeTrace2DSI(MakeTraceBASE, RetrievePulses2DSI):
    def __init__(self, time, frequency, pulse_t, pulse_f, nonlinear_method, cross_correlation, N, scale_time_range, cut_off_val, frequency_range,
                 material_thickness = 0,
                 refractive_index = refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson")):
        super().__init__()

        from scipy.constants import c as c0
        self.c0 = c0
        
        self.time=time
        self.frequency=frequency
        self.pulse_t=pulse_t
        self.pulse_f=pulse_f
        self.nonlinear_method=nonlinear_method
        self.N=N
        self.scale_time_range=scale_time_range
        self.cut_off_val = cut_off_val
        self.frequency_range = frequency_range
        self.cross_correlation = cross_correlation
        self.gate = None
        self.anc_1 = self.anc_2 = None

        self.x_arr = time

        self.sk, self.rn = get_sk_rn(self.time, self.frequency)
        self.refractive_index, self.material_thickness = refractive_index, material_thickness




    def get_gate_pulse(self, frequency, gate_f, anc_no=1):
        gate_f = do_interpolation_1d(self.frequency, frequency, gate_f)
        gate = do_ifft(gate_f, self.sk, self.rn)

        anc = {1: "anc_1", 
               2: "anc_2"}
        setattr(self, anc[anc_no], gate)
        return gate
    
    

    def get_parameters_to_make_signal_t(self):
        measurement_info = MyNamespace(anc_1=self.anc_1, anc_2=self.anc_2, time=self.time, frequency=self.frequency, cross_correlation=self.cross_correlation, 
                                       nonlinear_method=self.nonlinear_method, doubleblind=False, sk=self.sk, rn=self.rn, c0=self.c0)
        
        self.phase_matrix = self.get_phase_matrix(self.refractive_index, self.material_thickness, measurement_info)
        measurement_info = measurement_info.expand(phase_matrix = self.phase_matrix)

        individual = MyNamespace(pulse=self.pulse_t, gate=None)
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
            if self.nonlinear_method=="sd":
                fmin, fmax = np.sort([-1*fmin, -1*fmax])
        else:
            fmin, fmax = np.min(frequency_zoom), np.max(frequency_zoom)
            deltaf = fmax - fmin
            fmin = fmin - deltaf/2
            fmax = fmax + deltaf/2


        frequency_interpolate = np.linspace(fmin, fmax, self.N)
        t_central = (time_zoom[0]+time_zoom[-1])/2
        Delta_t = np.abs(time_zoom[-1]-time_zoom[0])*self.scale_time_range
        time_interpolate = np.linspace(t_central-Delta_t/2, t_central+Delta_t/2, self.N)
        

        trace_interpolate_freq = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(frequency_interpolate, self.frequency, self.trace)
        trace_interpolate = jax.vmap(do_interpolation_1d, in_axes=(None,None,1))(time_interpolate, self.time, trace_interpolate_freq)

        if self.nonlinear_method=="sd":
            frequency_interpolate = -1*np.flip(frequency_interpolate)
            trace_interpolate = np.flip(trace_interpolate, axis=0)
        

        frequency_pulse_spectrum, spectrum_pulse = interpolate_spectrum(self.frequency, self.pulse_f, self.N)

        if self.cross_correlation==True:
            anc1_f=do_fft(self.anc_1, self.sk, self.rn)
            anc2_f=do_fft(self.anc_2, self.sk, self.rn)
            frequency_gate_spectrum_1, spectrum_anc_1 = interpolate_spectrum(self.frequency, anc1_f, self.N)
            frequency_gate_spectrum_2, spectrum_anc_2 = interpolate_spectrum(self.frequency, anc2_f, self.N)

            spectra = MyNamespace(pulse=(frequency_pulse_spectrum, spectrum_pulse), 
                                anc_1=(frequency_gate_spectrum_1, spectrum_anc_1),
                                anc_2=(frequency_gate_spectrum_2, spectrum_anc_2))
        else:
            spectra = MyNamespace(pulse=(frequency_pulse_spectrum, spectrum_pulse))

        return time_interpolate, frequency_interpolate, np.abs(trace_interpolate).T, spectra
    