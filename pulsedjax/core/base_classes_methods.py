import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c0
import refractiveindex
from functools import partial as Partial

import jax
import jax.numpy as jnp

from pulsedjax.utilities import MyNamespace, get_com, center_signal, get_sk_rn, do_interpolation_1d, calculate_gate, calculate_trace, center_signal, project_onto_amplitude
from pulsedjax.core.initial_guess_doublepulse import make_population_doublepulse
from pulsedjax.core.phase_matrix_funcs import phase_func_dict, calculate_phase_matrix, calculate_phase_matrix_material, calc_group_delay_phase, _eval_refractive_index




class RetrievePulses:
    """
    The Base-Class for all reconstruction methods. Defines general initialization, preprocessing and postprocessing.

    Attributes:
        nonlinear_method (str): SHG, THG, PG or SD
        f0 (float): rarely some solvers need the central frequency to be zero. This saves the original central frequency.
        doubleblind (bool): whether the reconstruction is supposed to yield the gate in addition to the pulse.
        spectrum_is_being_used (bool):
        momentum_is_being_used (bool):
        measurement_info (Pytree): a container of variable (but static) structure. Holds measurement data and parameters.
        descent_info (Pytree): a container of variable (but static) structure. Holds parameters of the reconstruction algorithm.
        descent_state (Pytree): a container of variable (but static) structure. Contains the current state of the solver.
        prng_seed (int): seed for the key
        key (jnp.array): a jax.random.PRNGKey
        factor (int): for SHG/THG the a correction factor of 2/3 needs to applied occasionally.

        x_arr (jnp.array): an alias for the shifts/delays, internally indexed via m
        time (jnp.array): the time axis, internally indexed via k
        frequency (jnp.array): the frequency axis, internally indexed via n
        measured_trace (jnp.array): 2D-array with the measured data. axis=0 corresponds to shift/delay (index m), axis=1 correpsonds to the frequencies (index n)

    """

    def __init__(self, nonlinear_method, *args, cross_correlation=False, interferometric=False, seed=None, 
                 central_frequency=(None,None), **kwargs):
        super().__init__(*args, **kwargs)

        assert len(central_frequency) == 2

        self.nonlinear_method = nonlinear_method
        self.f0 = 0
        self.cross_correlation = cross_correlation
        self.interferometric = interferometric

        if self.cross_correlation=="doubleblind":
            self.doubleblind = True
            self.cross_correlation = False
        elif self.cross_correlation==True or self.cross_correlation==False:
            self.doubleblind = False
        else: 
            raise ValueError(f"cross_correlation can only take one of doubleblind, True or False. Got {self.cross_correlation}") 

        self.calibration_curve_is_provided = False
        self.calibration_curve = None

        # central frequency will be overwritten if spectra are provided
        central_frequency_pulse, central_frequency_gate = central_frequency
        self.central_frequency = MyNamespace(pulse=central_frequency_pulse, gate=central_frequency_gate)

        # central frequency is needed in order to remove group delay from material dispersion
        # this could/should be moved into checks in run -> because central_f is evaluated from optionally provided spectra
        if isinstance(self, (RetrievePulsesCHIRPSCAN, RetrievePulsesVAMPIRE)):
            if (self.central_frequency.pulse==None and self.cross_correlation==False and self.doubleblind==False):
                raise ValueError("You need to provide a central_frequency for the pulse.")
            elif (self.central_frequency.gate==None and (self.cross_correlation==True or self.doubleblind==True)):
                raise ValueError("You need to provide a central_frequency for the gate.")


        if nonlinear_method=="shg":
            self.factor = 2
        elif nonlinear_method=="thg":
            self.factor = 3
        elif nonlinear_method[-2:]=="hg":
            self.factor = int(nonlinear_method[0])
        else:
            self.factor = 1


        self.measurement_info = MyNamespace(nonlinear_method = self.nonlinear_method, 
                                            spectral_amplitude = MyNamespace(pulse=None, gate=None), 
                                            central_frequency = self.central_frequency,
                                            real_fields = False,
                                            interferometric = self.interferometric,
                                            cross_correlation = self.cross_correlation,
                                            doubleblind = self.doubleblind,
                                            calibration_curve = self.calibration_curve)
        self.descent_info = MyNamespace(measured_spectrum_is_provided = MyNamespace(pulse=False, gate=False),
                                        calibration_curve_is_provided = self.calibration_curve_is_provided)
        self.descent_state = MyNamespace()

        self.key = None
        if seed==None:
            self.prng_seed = np.random.randint(0, 1e9)
        else:
            self.prng_seed = int(seed)

        self.update_PRNG_key(self.prng_seed)

            


    def update_PRNG_key(self, seed):
        self.prng_seed = seed
        self.key = jax.random.PRNGKey(seed)


    def get_data(self, x_arr, frequency, measured_trace):
        """ Prepare/Convert data. """

        self.x_arr = jnp.asarray(x_arr)
        self.frequency = jnp.asarray(frequency)
        self.time = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(self.frequency), jnp.mean(jnp.diff(self.frequency))))
        self.measured_trace = jnp.asarray(measured_trace/jnp.linalg.norm(measured_trace))

        self.dt = jnp.mean(jnp.diff(self.time))
        self.df = jnp.mean(jnp.diff(self.frequency))
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        self.measurement_info = self.measurement_info.expand(time=self.time, frequency=self.frequency, 
                                                        sk=self.sk, rn=self.rn, 
                                                        dt=self.dt, df=self.df)
        return self.x_arr, self.time, self.frequency, self.measured_trace




    def get_spectral_amplitude(self, measured_frequency, measured_spectrum, pulse_or_gate):
        """ Used to provide a measured pulse spectrum. A spectrum for the gate pulse can also be provided. """
        frequency = self.frequency

        spectral_intensity = do_interpolation_1d(frequency, measured_frequency-self.f0/self.factor, measured_spectrum)
        spectral_amplitude = jnp.sqrt(jnp.abs(spectral_intensity))*jnp.sign(spectral_intensity)
        
        if pulse_or_gate=="pulse":
            self.measurement_info.spectral_amplitude.pulse = spectral_amplitude
            self.descent_info.measured_spectrum_is_provided.pulse = True
            central_f = jnp.sum(jnp.abs(spectral_amplitude)*frequency)/jnp.sum(jnp.abs(spectral_amplitude))
            self.central_frequency = self.central_frequency.expand(pulse=central_f)

        elif pulse_or_gate=="gate":
            self.measurement_info.spectral_amplitude.gate = spectral_amplitude
            self.descent_info.measured_spectrum_is_provided.gate = True
            central_f = jnp.sum(jnp.abs(spectral_amplitude)*frequency)/jnp.sum(jnp.abs(spectral_amplitude))
            self.central_frequency = self.central_frequency.expand(gate=central_f)

        else:
            raise ValueError(f"pulse_or_gate needs to be pulse or gate. Not {pulse_or_gate}")
        
        self.measurement_info = self.measurement_info.expand(central_frequency = self.central_frequency)
        return spectral_amplitude
    


    def get_gate_pulse(self, frequency, gate_f):
        """ For crosscorrelation=True the actual gate pulse has to be provided. """
        gate_f = do_interpolation_1d(self.frequency, frequency, gate_f)
        self.gate = self.ifft(gate_f, self.sk, self.rn)
        self.measurement_info = self.measurement_info.expand(gate = self.gate)
        return self.gate
    


    def get_calibration_curve(self, frequency, calibration_curve):
        self.calibration_curve = do_interpolation_1d(self.frequency, frequency, calibration_curve)
        self.measurement_info = self.measurement_info.expand(calibration_curve = self.calibration_curve)
        self.calibration_curve_is_provided = True
        self.descent_info = self.descent_info.expand(calibration_curve_is_provided = self.calibration_curve_is_provided)
        return self.calibration_curve
        


    def plot_results(self, final_result, exact_pulse=None):
        pulse_t, pulse_f, trace = final_result.pulse_t, final_result.pulse_f, final_result.trace
        gate_t, gate_f = final_result.gate_t, final_result.gate_f
        error_arr = final_result.error_arr

        x_arr, time, frequency, measured_trace = final_result.x_arr, final_result.time, final_result.frequency, final_result.measured_trace
        frequency_exp = final_result.frequency_exp

        trace = trace/jnp.max(trace)
        measured_trace = measured_trace/jnp.max(measured_trace)
        trace_difference = (measured_trace - trace)

        fig=plt.figure(figsize=(22,16))
        ax1=plt.subplot(2,3,1)
        ax1.plot(time, np.abs(pulse_t), label="Amplitude")
        ax1.set_xlabel(r"Time [arb. u.]")
        ax1.legend(loc=2)
        ax1.set_title("Pulse Time-Domain")

        ax2 = ax1.twinx()
        ax2.plot(time, np.unwrap(np.angle(pulse_t))*1/np.pi, c="tab:orange", label="Phase")
        ax2.set_ylabel(r"Phase [$\pi$]")
        ax2.legend(loc=1)

        if self.measurement_info.doubleblind==True:
            ax1.plot(time, np.abs(gate_t), label="Gate-Pulse", c="tab:red")
            ax2.plot(time, np.unwrap(np.angle(gate_t))*1/np.pi, label="Gate-Pulse", c="tab:green")

        if exact_pulse!=None:
            ax1.plot(exact_pulse.time, np.abs(exact_pulse.pulse_t)*np.max(np.abs(pulse_t))/np.max(np.abs(exact_pulse.pulse_t)), 
                     "--", c="black", label="Exact Amplitude")
            ax2.plot(exact_pulse.time, np.unwrap(np.angle(exact_pulse.pulse_t)), "--", c="black", label="Exact Phase", alpha=0.5)

        ax1=plt.subplot(2,3,2)
        ax1.plot(frequency,jnp.abs(pulse_f), label="Amplitude")
        ax1.set_xlabel(r"Frequency [arb. u.]")
        ax1.legend(loc=2)
        ax1.set_title("Pulse Frequency-Domain")

        ax2 = ax1.twinx()
        ax2.plot(frequency, jnp.unwrap(jnp.angle(pulse_f))*1/np.pi, c="tab:orange", label="Phase")
        ax2.set_ylabel(r"Phase [$\pi$]")
        ax2.legend(loc=1)

        if self.measurement_info.doubleblind==True:
            ax1.plot(frequency, np.abs(gate_f), label="Gate-Pulse", c="tab:red")
            ax2.plot(frequency, np.unwrap(np.angle(gate_f))*1/np.pi, label="Gate-Pulse", c="tab:green")

        if exact_pulse!=None:
            ax1.plot(exact_pulse.frequency, np.abs(exact_pulse.pulse_f)*np.max(np.abs(pulse_f))/np.max(np.abs(exact_pulse.pulse_f)),
                     "--", c="black", label="Exact Amplitude")
            ax2.plot(exact_pulse.frequency, np.unwrap(np.angle(exact_pulse.pulse_f)), "--", c="black", label="Exact Phase", alpha=0.5)

        plt.subplot(2,3,3)
        plt.plot(error_arr)
        plt.yscale("log")
        plt.title("Trace Error")
        plt.xlabel("Iteration No.")

        plt.subplot(2,3,4)
        plt.pcolormesh(x_arr, frequency_exp, measured_trace.T)
        plt.xlabel("Shift [arb. u.]")
        plt.ylabel("Frequency [arb. u.]")
        plt.title("Measured Trace")

        plt.subplot(2,3,5)
        plt.pcolormesh(x_arr, frequency_exp, trace.T)
        plt.xlabel("Shift [arb. u.]")
        plt.ylabel("Frequency [arb. u.]")
        plt.title("Retrieved Trace")

        plt.subplot(2,3,6)
        plt.pcolormesh(x_arr, frequency_exp, trace_difference.T)
        plt.xlabel("Shift [arb. u.]")
        plt.ylabel("Frequency [arb. u.]")
        plt.colorbar()
        plt.title("Normalized Difference Traces")

        plt.tight_layout()
        plt.show()



        local_or_global=None

        if self.descent_info.optimize_calibration_curve._local==True:
            local_or_global="_local"

        if self.descent_info.optimize_calibration_curve._global==True:
            local_or_global="_global"

        if local_or_global!=None:
            fig=plt.figure()
            plt.plot(frequency_exp, getattr(final_result.mu, local_or_global))
            plt.xlabel("Frequency [PHz]")
            plt.ylabel("Scaling Factor [arb.u.]")
            plt.title("Calibration Curve")
            plt.tight_layout()
            plt.show()



    def get_individual_from_idx(self, idx, population):
        individual = jax.tree.map(lambda x: x[jnp.newaxis, idx], population)
        return individual


    def _get_idx_individual(self, population, idx_func):
        """ Calculates the error for a population. Returns the index of the worst individual. And the population """
        error_arr, mu, population = self.calculate_error_population(population, self.measurement_info, self.descent_info)
        idx = idx_func(error_arr)
        return idx, population
    

    def get_idx_best_individual(self, population):
        """ Calculates the error for a population. Returns the index of the fittest individual. And the population"""
        return self._get_idx_individual(population, jnp.nanargmin)
    

    def get_idx_worst_individual(self, population):
        """ Calculates the error for a population. Returns the index of the worst individual. And the population """
        return self._get_idx_individual(population, jnp.nanargmax)


    def get_idx_average_individual(self, population):
        """ Calculates the error for a population. Returns the index of an average individual. And the population """
        get_mean_idx = lambda x: jnp.nanargmin(jnp.abs(x - jnp.nanmean(x)))
        return self._get_idx_individual(population, get_mean_idx)
    




    

    def post_process_get_pulse_and_gate(self, descent_state, measurement_info, descent_info, idx=None):
        """ FROG specific post processing to get the final pulse/gate """
        sk, rn = measurement_info.sk, measurement_info.rn

        if idx==None:
            idx, population = self.get_idx_best_individual(descent_state.population)
        else:
            population = descent_state.population

        individual = self.get_individual_from_idx(idx, population)
        pulse_f = individual.pulse[0]
        pulse_t = self.ifft(pulse_f, sk, rn)

        if measurement_info.doubleblind==True:
            gate_f = individual.gate[0]
            gate_t = self.ifft(gate_f, sk, rn)
        else:
            gate_t, gate_f = pulse_t, pulse_f

        return pulse_t, gate_t, pulse_f, gate_f, idx
    




    def post_process_center_pulse_and_gate(self, pulse_t, gate_t, pulse_f, gate_f):
        """ This essentially removes the linear phase. But only approximately since no fits are done. """

        sk, rn = self.measurement_info.sk, self.measurement_info.rn
        pulse_t = center_signal(pulse_t)
        gate_t = center_signal(gate_t)
        pulse_f_new, gate_f_new = self.fft(pulse_t, sk, rn), self.fft(gate_t, sk, rn)

        if self.descent_info.measured_spectrum_is_provided.pulse==True and self.eta_spectral_amplitude==1:
            amp_f_pulse = self.measurement_info.spectral_amplitude.pulse
        else:
            amp_f_pulse = jnp.abs(pulse_f)

        if self.descent_info.measured_spectrum_is_provided.gate==True and self.eta_spectral_amplitude==1:
            amp_f_gate = self.measurement_info.spectral_amplitude.gate
        else:
            amp_f_gate = jnp.abs(gate_f)

        pulse_f = project_onto_amplitude(pulse_f_new, amp_f_pulse)
        gate_f = project_onto_amplitude(gate_f_new, amp_f_gate)

        pulse_t, gate_t = self.ifft(pulse_f, sk, rn), self.ifft(gate_f, sk, rn)
        return pulse_t, gate_t, pulse_f, gate_f
    



    def post_process(self, descent_state, error_arr):
        """ Creates the final_result object from the final descent_state. """
        error_arr = jnp.squeeze(error_arr)
        self.descent_state = descent_state

        pulse_t, gate_t, pulse_f, gate_f, idx = self.post_process_get_pulse_and_gate(descent_state, self.measurement_info, self.descent_info)
        pulse_t, gate_t, pulse_f, gate_f = self.post_process_center_pulse_and_gate(pulse_t, gate_t, pulse_f, gate_f)
        
        trace = self.post_process_create_trace(descent_state, self.measurement_info, self.descent_info, idx)
        measured_trace = self.measurement_info.measured_trace

        if self._name=="PtychographicIterativeEngine" or self._name=="COPRA":
            local_mu, global_mu = descent_state._local.mu[idx], descent_state._global.mu[idx]
        else:
            local_mu, global_mu = None, descent_state.mu[idx]


        x_arr = self.measurement_info.x_arr
        time, frequency = self.measurement_info.time, self.measurement_info.frequency + self.f0

        if self.measurement_info.real_fields==True:
            frequency_exp = self.measurement_info.frequency_big
        else:
            frequency_exp = frequency

        final_result = MyNamespace(x_arr=x_arr, time=time, frequency=frequency, frequency_exp = frequency_exp,
                                 pulse_t=pulse_t, pulse_f=pulse_f, gate_t=gate_t, gate_f=gate_f,
                                 trace=trace, measured_trace=measured_trace,
                                 error_arr=error_arr, mu=MyNamespace(_local=local_mu, _global=global_mu))
        return final_result
    
    
    































class RetrievePulsesFROG(RetrievePulses):
    """
    The reconstruction class for FROG. Inherits from RetrievePulses.

    R. Trebino, "Frequency-Resolved Optical Gating: The Measurement of Ultrashort Laser Pulses", 10.1007/978-1-4615-1181-6 (2000)

    Attributes:
        tau_arr (jnp.array): the delays
        gate (jnp.array): the gate-pulse (if its known).
        transform_arr (jnp.array): an alias for tau_arr
        idx_arr (jnp.array): an array with indices for tau_arr
        dt (float):
        df (float):
        sk (jnp.array): correction values for FFT->DFT
        rn (jnp.array): correction values for FFT->DFT
        cross_correlation (bool):
        interferometric (bool): 

    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, interferometric=False, **kwargs):
        
        super().__init__(nonlinear_method, cross_correlation=cross_correlation, interferometric=interferometric, **kwargs)

        self.tau_arr, self.time, self.frequency, self.measured_trace = self.get_data(delay, frequency, measured_trace)
        self.gate = jnp.zeros(jnp.size(self.time))

        self.transform_arr = self.tau_arr
        
        self.measurement_info = self.measurement_info.expand(tau_arr = self.tau_arr,
                                                             measured_trace = self.measured_trace,
                                                             gate = self.gate,
                                                             transform_arr = self.transform_arr,
                                                             x_arr = self.x_arr)
        


    def create_initial_population_doublepulse(self, population_size, **kwargs):
        """ 
        Calls initial_guess_doublepulse.make_population_doublepulse to create an initial guess.
        The guess is in the time domain. Assumes an autocorrelation FROG.
        
        Args:
            population_size (int):
            **kwargs: passed to make_population_doublepulse()

        Returns:
            Pytree, the initial population

        """
        measurement_info = self.measurement_info
        assert measurement_info.doubleblind==False, "Only implemented for doubleblind=False"
        
        self.key, subkey = jax.random.split(self.key, 2)

        tau_arr, frequency, measured_trace = measurement_info.tau_arr, measurement_info.frequency, measurement_info.measured_trace
        nonlinear_method = measurement_info.nonlinear_method
        pulse_f_arr = make_population_doublepulse(subkey, population_size, tau_arr, frequency, measured_trace, nonlinear_method, **kwargs)

        population = MyNamespace(pulse=pulse_f_arr, gate=None)
        return population
        
    


    def shift_signal_in_time(self, signal, tau, frequency, sk, rn): # change this to a precomuped and the applied phase matrix
        """ The Fourier-Shift theorem. """
        signal_f = self.fft(signal, sk, rn)
        signal_f = signal_f*jnp.exp(-1j*2*jnp.pi*frequency*tau)
        signal = self.ifft(signal_f, sk, rn)
        return signal


    def calculate_shifted_signal(self, signal, frequency, tau_arr, time, in_axes=(None, 0, None, None, None)):
        """ The Fourier-Shift theorem applied to a list of signals. """

        # im really unhappy with this, but this re-definition/calculation of sk, rn is necessary(?)
        # in the original case a global phase shift dependent on tau and f[0] occured, which i couldnt figure out
        frequency = frequency - (frequency[-1] + frequency[0])/2

        N = jnp.size(frequency)
        pad_arr = [(0,0)]*(signal.ndim-1) + [(0,N)]
        signal = jnp.pad(signal, pad_arr)
        
        frequency = jnp.linspace(jnp.min(frequency), jnp.max(frequency), 2*N)
        time = jnp.concatenate([time, time+(jnp.size(time)+1)*jnp.mean(jnp.diff(time))])
        sk, rn = get_sk_rn(time, frequency)

        signal_shifted = jax.vmap(self.shift_signal_in_time, in_axes=in_axes)(signal, tau_arr, frequency, sk, rn)
        return signal_shifted[ ... , :N] 




    def calculate_signal_t(self, individual, tau_arr, measurement_info):
        """
        Calculates the signal field of a FROG in the time domain. 

        Args:
            individual (Pytree): a population containing only one member. (jax.vmap over whole population)
            tau_arr (jnp.array): the delays
            measurement_info (Pytree): contains the measurement parameters (e.g. nonlinear method, interferometric, ... )

        Returns:
            Pytree, contains the signal field in the time domain as well as the fields used to calculate it.
        """

        time, frequency = measurement_info.time, measurement_info.frequency
        cross_correlation, doubleblind, interferometric = measurement_info.cross_correlation, measurement_info.doubleblind, measurement_info.interferometric
        frogmethod = measurement_info.nonlinear_method

        pulse_f, gate = individual.pulse, individual.gate
        pulse_t = self.ifft(pulse_f, measurement_info.sk, measurement_info.rn)
        pulse_t_shifted = self.calculate_shifted_signal(pulse_t, frequency, tau_arr, time)

        if cross_correlation==True:
            gate_t = measurement_info.gate
            gate_pulse_shifted = self.calculate_shifted_signal(gate_t, frequency, tau_arr, time)
            gate_shifted = calculate_gate(gate_pulse_shifted, frogmethod)

        elif doubleblind==True:
            gate_t = self.ifft(gate, measurement_info.sk, measurement_info.rn)
            gate_pulse_shifted = self.calculate_shifted_signal(gate_t, frequency, tau_arr, time)
            gate_shifted = calculate_gate(gate_pulse_shifted, frogmethod)

        else:
            gate_pulse_shifted, gate_t = None, None
            gate_shifted = calculate_gate(pulse_t_shifted, frogmethod)


        if interferometric==True and cross_correlation==False and doubleblind==False:
            signal_t = (pulse_t + pulse_t_shifted)*calculate_gate(pulse_t + pulse_t_shifted, frogmethod)
        elif interferometric==True:
            signal_t = (pulse_t + gate_pulse_shifted)*calculate_gate(pulse_t + gate_pulse_shifted, frogmethod)
        else:
            signal_t = pulse_t*gate_shifted
            
        signal_f = self.fft(signal_t, measurement_info.sk, measurement_info.rn)
        signal_t = MyNamespace(signal_t = signal_t, 
                               signal_f = signal_f, 
                               pulse_t_shifted = pulse_t_shifted, 
                               gate_shifted = gate_shifted, 
                               gate_pulse_shifted = gate_pulse_shifted,
                               pulse_t = pulse_t)
        return signal_t









class RetrievePulsesTDP(RetrievePulsesFROG):
    """
    The reconstruction class for Time-Domain-Ptychography.

    D. Spangenberg et al., Phys. Rev. A 91, 021803(R), 10.1103/PhysRevA.91.021803 (2015)

    Attributes:
        spectral_filter (jnp.array): the spectral filter in the gate arm.

    """

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter=None, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, **kwargs)

        if spectral_filter==None:
            self.spectral_filter = jnp.ones(jnp.size(self.frequency))
            print("If spectral_filter=None, then TDP is the same as FROG.")
        else:
            self.spectral_filter = spectral_filter

        self.measurement_info = self.measurement_info.expand(spectral_filter=self.spectral_filter)
        


    def apply_spectral_filter(self, signal, spectral_filter, sk, rn): # here the ffts are probably also unnecessary
        """ Apply a spectral filter to a signal. """
        signal_f = self.fft(signal, sk, rn)
        signal_f = signal_f*spectral_filter
        signal = self.ifft(signal_f, sk, rn)
        return signal
    


    def calculate_signal_t(self, individual, tau_arr, measurement_info):
        """
        Calculates the signal field of TDP in the time domain. 

        Args:
            individual (Pytree): a population containing only one member. (jax.vmap over whole population)
            tau_arr (jnp.array): the delays
            measurement_info (Pytree): contains the measurement parameters (e.g. nonlinear method, interferometric, ... )

        Returns:
            Pytree, contains the signal field in the time domain as well as the fields used to calculate it.
        """

        time, frequency = measurement_info.time, measurement_info.frequency
        cross_correlation, doubleblind, interferometric = measurement_info.cross_correlation, measurement_info.doubleblind, measurement_info.interferometric
        frogmethod = measurement_info.nonlinear_method

        spectral_filter, sk, rn = measurement_info.spectral_filter, measurement_info.sk, measurement_info.rn

        pulse_f, gate_f = individual.pulse, individual.gate
        pulse_t = self.ifft(pulse_f, measurement_info.sk, measurement_info.rn)
        pulse_t_shifted = self.calculate_shifted_signal(pulse_t, frequency, tau_arr, time)

        if cross_correlation==True:
            gate_t = measurement_info.gate
            gate_pulse = self.apply_spectral_filter(gate_t, spectral_filter, sk, rn)
            gate_pulse_shifted = self.calculate_shifted_signal(gate_pulse, frequency, tau_arr, time)
            gate_shifted = calculate_gate(gate_pulse_shifted, frogmethod)

        elif doubleblind==True:
            gate_t = self.ifft(gate_f, measurement_info.sk, measurement_info.rn)
            gate_pulse = self.apply_spectral_filter(gate_t, spectral_filter, sk, rn)
            gate_pulse_shifted = self.calculate_shifted_signal(gate_pulse, frequency, tau_arr, time)
            gate_shifted = calculate_gate(gate_pulse_shifted, frogmethod)

        else:
            gate_pulse_shifted, gate_t = None, None
            pulse_t_shifted = self.apply_spectral_filter(pulse_t_shifted, spectral_filter, sk, rn)
            gate_shifted = calculate_gate(pulse_t_shifted, frogmethod)


        if interferometric==True and cross_correlation==False and doubleblind==False:
            signal_t = (pulse_t + pulse_t_shifted)*calculate_gate(pulse_t + pulse_t_shifted, frogmethod)
        elif interferometric==True:
            signal_t = (pulse_t + gate_pulse_shifted)*calculate_gate(pulse_t + gate_pulse_shifted, frogmethod)
        else:
            signal_t = pulse_t*gate_shifted
            
        signal_f = self.fft(signal_t, sk, rn)
        signal_t = MyNamespace(signal_t = signal_t, 
                               signal_f = signal_f, 
                               pulse_t_shifted = pulse_t_shifted, 
                               gate_shifted = gate_shifted, 
                               gate_pulse_shifted = gate_pulse_shifted,
                               pulse_t = pulse_t)
        return signal_t













class RetrievePulsesCHIRPSCAN(RetrievePulses):
    """
    The reconstruction class for Chirp-Scan methods.

    V. V. Lozovoy et al., Optics Letters 29, 775-777 (2004)
    M. Miranda et al., Opt. Express 20, 18732-18743 (2012) 

    Attributes:
        theta (jnp.array): the shifts
        dt (float):
        df (float):
        sk (jnp.array): correction values for FFT->DFT
        rn (jnp.array): correction values for FFT->DFT
        phase_matrix (jnp.array): a 2D-array with the phase values applied to pulse
        parameters (tuple): parameters for the chirp function
        transform_arr (jnp.array): an alias for phase_matrix
        idx_arr (jnp.array): indices for theta

    """
    
    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, **kwargs):
        super().__init__(nonlinear_method, **kwargs)

        self.theta, self.time, self.frequency, self.measured_trace = self.get_data(theta, frequency, measured_trace)
        self.measurement_info = self.measurement_info.expand(theta = self.theta,
                                                             measured_trace = self.measured_trace,
                                                             x_arr = self.x_arr)
        

        self.phase_type = phase_type
        self.phase_matrix = self.get_phase_matrix(chirp_parameters)
        



    def get_phase_matrix(self, parameters):
        """ Calls phase_matrix_func in order to calculate the phase matrix. """
        self.parameters = parameters

        if self.phase_type=="material":
            self.phase_matrix = calculate_phase_matrix_material(self.measurement_info, parameters, 
                                                                self.measurement_info.central_frequency.pulse)

        elif type(self.phase_type)==str:
            self.phase_matrix = calculate_phase_matrix(self.measurement_info, parameters, phase_func=phase_func_dict[self.phase_type])

        elif callable(self.phase_type)==True:
            self.phase_matrix = self.phase_type(self.measurement_info, parameters)

        else:
            raise ValueError(f"phase_type needs to be a string or a callable. Not {self.phase_type}")


        self.transform_arr = self.phase_matrix
        self.measurement_info = self.measurement_info.expand(phase_matrix = self.phase_matrix,
                                                             transform_arr = self.transform_arr)
        return self.phase_matrix
        
    



    def get_dispersed_pulse_t(self, pulse_f, phase_matrix, sk, rn):
        """ Applies phase-matrix to a signal. """
        
        pulse_f = pulse_f*jnp.exp(1j*phase_matrix)
        pulse_t_disp = self.ifft(pulse_f, sk, rn)
        return pulse_t_disp, phase_matrix
    



    def calculate_signal_t(self, individual, phase_matrix, measurement_info):
        """
        Calculates the signal field of a Chirp-Scan in the time domain. 

        Args:
            individual (Pytree): a population containing only one member. (jax.vmap over whole population)
            phase_matrix (jnp.array): the applied phases
            measurement_info (Pytree): contains the measurement parameters (e.g. nonlinear method, ... )

        Returns:
            Pytree, contains the signal field in the time domain as well as the fields used to calculate it.
        """

        pulse = individual.pulse

        pulse_t_disp, phase_matrix = self.get_dispersed_pulse_t(pulse, phase_matrix, measurement_info.sk, measurement_info.rn)
        gate_disp = calculate_gate(pulse_t_disp, measurement_info.nonlinear_method)
        signal_t = pulse_t_disp*gate_disp

        signal_f = self.fft(signal_t, measurement_info.sk, measurement_info.rn)
        signal_t = MyNamespace(signal_t=signal_t, signal_f=signal_f, pulse_t_disp=pulse_t_disp, gate_disp=gate_disp)
        return signal_t
    



    







class RetrievePulses2DSI(RetrievePulsesFROG):
    """
    The reconstruction class for 2DSI.

    [1] J. R. Birge et al., Opt. Lett. 31, 2063-2065 (2006) 

    Attributes:
        spectral_filter1 (jnp.array): 1st filter in interferometer 
        spectral_filter2 (jnp.array): 2nd filter in interferometer
        tau_pulse_anc1 (float): delay between 1st interferometer arm and exterior pulse
        anc_frequency1 (float): central frequency of pulse in 1st arm (calculated from max of spectral_filter1)
        anc_frequency2 (float): central frequency of pulse in 2nd arm (calculated from max of spectral_filter2)
        c0 (float): the speed of light
        refractive_index (refractiveindex.RefractiveIndexMaterial, Callable): returns the refractive index for a material given a wavelength in um
        phase_matrix (jnp.array): a 2D-array with phase values that could potentially have been applied to a pulse

    """

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, 
                 spectral_filter1=None, spectral_filter2=None, tau_pulse_anc1=0, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, interferometric=False, **kwargs)

        self.tau_pulse_anc1 = tau_pulse_anc1

        if spectral_filter1==None:
            self.spectral_filter1 = jnp.ones(jnp.size(self.frequency))
        else:
            self.spectral_filter1 = spectral_filter1
        if spectral_filter2==None:
            self.spectral_filter2 = jnp.ones(jnp.size(self.frequency))
        else:
            self.spectral_filter2 = spectral_filter2

        self.anc1_frequency = self.frequency[jnp.argmax(self.spectral_filter1)]
        self.anc2_frequency = self.frequency[jnp.argmax(self.spectral_filter2)]

        self.measurement_info = self.measurement_info.expand(anc1_frequency=self.anc1_frequency, anc2_frequency=self.anc2_frequency, 
                                                             spectral_filter1=self.spectral_filter1,
                                                             spectral_filter2=self.spectral_filter2,
                                                             tau_pulse_anc1 = self.tau_pulse_anc1)
    


    def apply_spectral_filter(self, signal, spectral_filter1, spectral_filter2, sk, rn): # same here as above
        """ Apply a spectral filter to a signal. """
        signal_f = self.fft(signal, sk, rn)
        signal_f1 = signal_f*spectral_filter1
        signal_f2 = signal_f*spectral_filter2
        signal1 = self.ifft(signal_f1, sk, rn)
        signal2 = self.ifft(signal_f2, sk, rn)
        return signal1, signal2


    def calculate_signal_t(self, individual, tau_arr, measurement_info):
        """
        Calculates the signal field of 2DSI in the time domain. 

        Args:
            individual (Pytree): a population containing only one member. (jax.vmap over whole population)
            tau_arr (jnp.array): the delays
            measurement_info (Pytree): contains the measurement parameters (e.g. nonlinear method, ... )

        Returns:
            Pytree, contains the signal field in the time domain as well as the fields used to calculate it.
        """

        time, frequency, nonlinear_method = measurement_info.time, measurement_info.frequency, measurement_info.nonlinear_method
        sk, rn = measurement_info.sk, measurement_info.rn
        
        pulse_f = individual.pulse
        pulse_t = self.ifft(pulse_f, sk, rn)

        if measurement_info.cross_correlation==True:
            gate_t = measurement_info.gate
        elif measurement_info.doubleblind==True:
            gate_f = individual.gate
            gate_t = self.ifft(gate_f, sk, rn)
        else:
            gate_t = pulse_t
        
        gate1, gate2 = self.apply_spectral_filter(gate_t, measurement_info.spectral_filter1, 
                                                  measurement_info.spectral_filter2, sk, rn)
        
        gate2_shifted = self.calculate_shifted_signal(gate2, frequency, tau_arr, time)
        tau = measurement_info.tau_pulse_anc1
        gate1 = self.calculate_shifted_signal(gate1, frequency, jnp.asarray([tau]), time)
        gate_pulses = jnp.squeeze(gate1) + gate2_shifted
        gate = calculate_gate(gate_pulses, nonlinear_method)

        signal_t = pulse_t*gate

        signal_f = self.fft(signal_t, sk, rn)
        signal_t = MyNamespace(signal_t=signal_t, signal_f=signal_f, gate_pulses=gate_pulses, gate_shifted=gate, pulse_t=pulse_t)
        return signal_t








class RetrievePulsesVAMPIRE(RetrievePulsesFROG):
    """
    The reconstruction class for VAMPIRE.

    [1] B. Seifert and H. Stolz, Meas. Sci. Technol. 20 (2009) 015303 (7pp), 10.1088/0957-0233/20/1/015303 (2008)

    Attributes:
        tau_interferometer (float): delay of the interferometer arms
        c0 (float): the speed of light
        refractive_index (refractiveindex.RefractiveIndexMaterial, Callable): returns the refractive index for a material given a wavelength in um
        phase_matrix (jnp.array): a 2D-array with phase values that could potentially have been applied to a pulse

    """

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, tau_interferometer=0,
                 material_thickness=0, refractive_index=refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, interferometric=False, **kwargs)

        self.tau_interferometer = tau_interferometer
        self.c0 = c0
        self.refractive_index = refractive_index

        self.measurement_info = self.measurement_info.expand(c0=self.c0)
        self.phase_matrix = self.get_phase_matrix(self.refractive_index, material_thickness, self.measurement_info)
        self.measurement_info = self.measurement_info.expand(phase_matrix=self.phase_matrix,
                                                             tau_interferometer=self.tau_interferometer)
        



    def get_phase_matrix(self, refractive_index, material_thickness, measurement_info):
        """ 
        Calculates the phase matrix that is applied of a pulse passes through a material.
        """
        frequency, c0 = measurement_info.frequency, measurement_info.c0

        if measurement_info.cross_correlation==True or measurement_info.doubleblind==True:
           central_f = measurement_info.central_frequency.gate
        else:
            central_f = measurement_info.central_frequency.pulse

        wavelength = c0/frequency*1e-6 # wavelength in nm
        n_arr = _eval_refractive_index(refractive_index, jnp.abs(wavelength)) # wavelength needs to be in nm
        n_arr = jnp.where(jnp.isnan(n_arr)==False, n_arr, 1.0)
        k0_arr = 2*jnp.pi/(wavelength*1e-6 + 1e-9) #wavelength is needed in mm
        k_arr = k0_arr*n_arr

        wavelength_0 = c0/(central_f + 1e-9)*1e-6 
        Tg_phase = calc_group_delay_phase(refractive_index, n_arr, k0_arr, wavelength_0, wavelength)
        phase_matrix = material_thickness*(k_arr-Tg_phase)
        return phase_matrix



    def apply_phase(self, pulse_t, measurement_info, sk, rn): # smae here as above
        """
        For an VAMPIRE reconstruction one may need to consider effects of material dispersion in the interferometer.
        This applies a dispersion based on phase_matrix in order to achieve this. 
        """

        pulse_f = self.fft(pulse_t, sk, rn)
        pulse_f = pulse_f*jnp.exp(1j*measurement_info.phase_matrix)
        pulse_t_disp = self.ifft(pulse_f, sk, rn)

        return pulse_t_disp
    


    def calculate_signal_t(self, individual, tau_arr, measurement_info):
        """
        Calculates the signal field in the time domain. 

        Args:
            individual (Pytree): a population containing only one member. (jax.vmap over whole population)
            tau_arr (jnp.array): the delays
            measurement_info (Pytree): contains the measurement parameters (e.g. nonlinear method, ... )

        Returns:
            Pytree, contains the signal field in the time domain as well as the fields used to calculate it.
        """

        time, frequency, nonlinear_method = measurement_info.time, measurement_info.frequency, measurement_info.nonlinear_method
        sk, rn = measurement_info.sk, measurement_info.rn

        pulse_f = individual.pulse
        pulse_t = self.ifft(pulse_f, sk, rn)

        if measurement_info.cross_correlation==True:
            gate_t = measurement_info.gate
        elif measurement_info.doubleblind==True:
            gate_f = individual.gate
            gate_t = self.ifft(gate_f, sk, rn)
        else:
            gate_t = pulse_t

        gate_disp = self.apply_phase(gate_t, measurement_info, sk, rn) 

        tau = measurement_info.tau_interferometer
        gate_t_shifted = self.calculate_shifted_signal(gate_t, frequency, jnp.asarray([tau]), time)

        gate_pulses = jnp.squeeze(gate_t_shifted) + gate_disp
        gate_pulses = self.calculate_shifted_signal(gate_pulses, frequency, tau_arr, time)
        gate = calculate_gate(gate_pulses, nonlinear_method)

        signal_t = pulse_t*gate

        signal_f = self.fft(signal_t, sk, rn)
        signal_t = MyNamespace(signal_t=signal_t, signal_f=signal_f, gate_pulses=gate_pulses, gate_shifted=gate, pulse_t=pulse_t)
        return signal_t








    