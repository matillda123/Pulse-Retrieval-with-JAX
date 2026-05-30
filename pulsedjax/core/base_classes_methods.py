import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c0
import refractiveindex
from functools import partial as Partial

import jax
import jax.numpy as jnp

from equinox import tree_at

from pulsedjax.utilities import MyNamespace, get_com, center_signal, get_sk_rn, do_interpolation_1d, calculate_gate, calculate_trace, center_signal, project_onto_amplitude, integrate_signal_1D
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

        theta (jnp.array): an alias for the shifts/delays, internally indexed via m
        time (jnp.array): the time axis, internally indexed via k
        frequency (jnp.array): the frequency axis, internally indexed via n
        measured_trace (jnp.array): 2D-array with the measured data. axis=0 corresponds to shift/delay (index m), axis=1 correpsonds to the frequencies (index n)

    """

    def __init__(self, nonlinear_method, *args, cross_correlation=False, interferometric=False, key=None, seed=None, 
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


        if nonlinear_method is None:
            self.factor = 1
        elif nonlinear_method=="shg":
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

        if key is None:
            if seed is None:
                self.prng_seed = np.random.randint(0, 1e9)
            else:
                self.prng_seed = int(seed)

            self.update_PRNG_key(self.prng_seed)
        else:
            self.key = key

            


    def update_PRNG_key(self, seed):
        self.prng_seed = seed
        self.key = jax.random.PRNGKey(seed)


    def get_data(self, theta, frequency, measured_trace):
        """ Prepare/Convert data. """

        self.theta = jnp.asarray(theta)
        self.frequency = jnp.asarray(frequency)
        self.time = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(self.frequency), jnp.mean(jnp.diff(self.frequency))))
        self.measured_trace = jnp.asarray(measured_trace/jnp.linalg.norm(measured_trace))

        self.dt = jnp.mean(jnp.diff(self.time))
        self.df = jnp.mean(jnp.diff(self.frequency))
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        self.measurement_info = self.measurement_info.expand(time=self.time, frequency=self.frequency, 
                                                        sk=self.sk, rn=self.rn, 
                                                        dt=self.dt, df=self.df)
        return self.theta, self.time, self.frequency, self.measured_trace




    def get_spectral_amplitude(self, measured_frequency, measured_spectrum, pulse_or_gate, frequency=None, **kwargs):
        """ Used to provide a measured pulse spectrum. A spectrum for the gate pulse can also be provided. """

        if frequency is None:
            frequency = self.frequency

        spectral_intensity = do_interpolation_1d(frequency, measured_frequency-self.f0/self.factor, measured_spectrum, **kwargs)
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
        pulse_t, pulse_f, trace = final_result.pulses.pulse_t, final_result.pulses.pulse_f, final_result.trace
        gate_t, gate_f = final_result.pulses.gate_t, final_result.pulses.gate_f
        error_arr = final_result.error_arr

        theta, time, frequency, measured_trace = final_result.theta, final_result.time, final_result.frequency, final_result.measured_trace
        frequency_exp = final_result.frequency_exp

        trace = trace/jnp.max(trace)
        measured_trace = measured_trace/jnp.max(measured_trace)
        trace_difference = (measured_trace - trace)

        fig=plt.figure(figsize=(22,16))
        ax1=plt.subplot(2,3,1)
        ax1.plot(time, np.abs(pulse_t), label="Pulse (Amp)")
        ax1.set_xlabel(r"Time [arb. u.]")
        ax1.set_title("Pulse Time-Domain")

        ax2 = ax1.twinx()
        ax2.plot(time, np.unwrap(np.angle(pulse_t))*1/np.pi, c="tab:orange", label="Pulse (Phase)")
        ax2.set_ylabel(r"Phase [$\pi$]")

        if self.measurement_info.doubleblind==True:
            ax1.plot(time, np.abs(gate_t), label="Gate-Pulse (Amp)", c="tab:red")
            ax2.plot(time, np.unwrap(np.angle(gate_t))*1/np.pi, label="Gate-Pulse (Phase)", c="tab:green")

        if exact_pulse!=None:
            ax1.plot(exact_pulse.time, np.abs(exact_pulse.pulse_t)*np.max(np.abs(pulse_t))/np.max(np.abs(exact_pulse.pulse_t)), 
                     "--", c="black", label="Exact Amplitude")
            ax2.plot(exact_pulse.time, np.unwrap(np.angle(exact_pulse.pulse_t)), "--", c="black", label="Exact Phase", alpha=0.5)
        
        ax1.legend(loc=2)
        ax2.legend(loc=1)


        ax1=plt.subplot(2,3,2)
        ax1.plot(frequency,jnp.abs(pulse_f), label="Pulse (Amp)")
        ax1.set_xlabel(r"Frequency [arb. u.]")
        ax1.set_title("Pulse Frequency-Domain")

        ax2 = ax1.twinx()
        ax2.plot(frequency, jnp.unwrap(jnp.angle(pulse_f))*1/np.pi, c="tab:orange", label="Pulse (Phase)")
        ax2.set_ylabel(r"Phase [$\pi$]")

        if self.measurement_info.doubleblind==True:
            ax1.plot(frequency, np.abs(gate_f), label="Gate-Pulse (Amp)", c="tab:red")
            ax2.plot(frequency, np.unwrap(np.angle(gate_f))*1/np.pi, label="Gate-Pulse (Phase)", c="tab:green")

        if exact_pulse!=None:
            ax1.plot(exact_pulse.frequency, np.abs(exact_pulse.pulse_f)*np.max(np.abs(pulse_f))/np.max(np.abs(exact_pulse.pulse_f)),
                     "--", c="black", label="Exact Amplitude")
            ax2.plot(exact_pulse.frequency, np.unwrap(np.angle(exact_pulse.pulse_f)), "--", c="black", label="Exact Phase", alpha=0.5)

        ax1.legend(loc=2)
        ax2.legend(loc=1)


        plt.subplot(2,3,3)
        plt.plot(error_arr)
        plt.yscale("log")
        plt.title("Trace Error")
        plt.xlabel("Iteration No.")

        plt.subplot(2,3,4)
        plt.pcolormesh(theta, frequency_exp, measured_trace.T)
        plt.xlabel("Shift [arb. u.]")
        plt.ylabel("Frequency [arb. u.]")
        plt.title("Measured Trace")

        plt.subplot(2,3,5)
        plt.pcolormesh(theta, frequency_exp, trace.T)
        plt.xlabel("Shift [arb. u.]")
        plt.ylabel("Frequency [arb. u.]")
        plt.title("Retrieved Trace")

        plt.subplot(2,3,6)
        plt.pcolormesh(theta, frequency_exp, trace_difference.T)
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
        """ post processing to get the final pulse/gate """
        sk, rn = measurement_info.sk, measurement_info.rn

        # the returned population is evaluated in the case of parametrized retrievals
        idx, population = self.get_idx_best_individual(descent_state.population)

        individual = self.get_individual_from_idx(idx, population)
        pulse_f = individual.pulse[0]
        pulse_t = self.ifft(pulse_f, sk, rn)

        if measurement_info.doubleblind==True:
            gate_f = individual.gate[0]
            gate_t = self.ifft(gate_f, sk, rn)
        else:
            gate_t, gate_f = None, None

        return MyNamespace(pulse_t=pulse_t, gate_t=gate_t, pulse_f=pulse_f, gate_f=gate_f, idx=idx)
    




    def post_process_center_pulse_and_gate(self, pulses):
        """ This essentially removes the linear phase. But only approximately since no fits are done. """
        sk, rn = self.measurement_info.sk, self.measurement_info.rn

        pulse_t = center_signal(pulses.pulse_t)
        pulse_f_new = self.fft(pulse_t, sk, rn)

        if self.descent_info.measured_spectrum_is_provided.pulse==True and self.eta_spectral_amplitude==1:
            amp_f_pulse = self.measurement_info.spectral_amplitude.pulse
        else:
            amp_f_pulse = jnp.abs(pulses.pulse_f)

        pulse_f = project_onto_amplitude(pulse_f_new, amp_f_pulse)
        pulse_t = self.ifft(pulse_f, sk, rn)
        pulses = tree_at(lambda x: x.pulse_t, pulses, pulse_t)
        pulses = tree_at(lambda x: x.pulse_f, pulses, pulse_f)

        if pulses.gate_t is not None:
            gate_t = center_signal(pulses.gate_t)
            gate_f_new = self.fft(gate_t, sk, rn)
            if self.descent_info.measured_spectrum_is_provided.gate==True and self.eta_spectral_amplitude==1:
                amp_f_gate = self.measurement_info.spectral_amplitude.gate
            else:
                amp_f_gate = jnp.abs(pulses.gate_f)

            gate_f = project_onto_amplitude(gate_f_new, amp_f_gate)
            gate_t = self.ifft(gate_f, sk, rn)
            pulses = tree_at(lambda x: x.gate_t, pulses, gate_t)
            pulses = tree_at(lambda x: x.gate_f, pulses, gate_f)

        return pulses
    



    def post_process(self, descent_state, error_arr):
        """ Creates the final_result object from the final descent_state. """
        error_arr = jnp.squeeze(error_arr)
        self.descent_state = descent_state

        pulses = self.post_process_get_pulse_and_gate(descent_state, self.measurement_info, self.descent_info)
        pulses = self.post_process_center_pulse_and_gate(pulses)
        idx = pulses.idx
        
        trace = self.post_process_create_trace(descent_state, self.measurement_info, self.descent_info, idx)
        measured_trace = self.measurement_info.measured_trace

        if self._name=="PtychographicIterativeEngine" or self._name=="COPRA":
            local_mu, global_mu = descent_state._local.mu[idx], descent_state._global.mu[idx]
        else:
            local_mu, global_mu = None, descent_state.mu[idx]


        theta = self.measurement_info.theta
        time, frequency = self.measurement_info.time, self.measurement_info.frequency + self.f0

        if self.measurement_info.real_fields==True:
            frequency_exp = self.measurement_info.frequency_big
        else:
            frequency_exp = frequency

        final_result = MyNamespace(theta=theta, time=time, frequency=frequency, frequency_exp = frequency_exp,
                                   pulses=pulses,
                                   trace=trace, measured_trace=measured_trace,
                                   error_arr=error_arr, mu=MyNamespace(_local=local_mu, _global=global_mu), idx_best_individual=idx)
        return final_result
    
    
    































class RetrievePulsesFROG(RetrievePulses):
    """
    The reconstruction class for FROG. Inherits from RetrievePulses.

    R. Trebino, "Frequency-Resolved Optical Gating: The Measurement of Ultrashort Laser Pulses", 10.1007/978-1-4615-1181-6 (2000)

    Attributes:
        tau_arr (jnp.array): the delays
        gate (jnp.array): the gate-pulse (if its known).
        transform_arr (jnp.array): an alias for tau_arr
        idtheta (jnp.array): an array with indices for tau_arr
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
                                                             theta = self.theta)
        


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

    
    # time is an uncesseray input -> not used but redefined -> can be removed
    def calculate_shifted_signal(self, signal, frequency, tau_arr, time, in_axes=(None, 0, None, None, None)):
        """ The Fourier-Shift theorem applied to a list of signals. """

        N = jnp.size(frequency)
        # pads the axis=-1 with N values 
        # padding on the right side is technically not necessary, but makes things numerically more stable.
        # padding on left side is definitely needed. Dont remove
        pad_arr = [(0,0)]*(signal.ndim-1) + [(N,N)]
        signal = jnp.pad(signal, pad_arr)
            
        # due to padding f, t, sk and rn need to be redefined 
        frequency = jnp.linspace(jnp.min(frequency), jnp.max(frequency), 3*N)
        time = jnp.fft.fftshift(jnp.fft.fftfreq(3*N, jnp.mean(jnp.diff(frequency))))
        sk, rn = get_sk_rn(time, frequency)

        # the frequency axis is not centered around zero, this causes a global phase in the fourier shift 
        # this phase-factor compensates this global phase
        df = jnp.mean(jnp.diff(frequency))
        phase_correction = jnp.exp(1j*2*jnp.pi*df*tau_arr)

        signal_shifted = jax.vmap(self.shift_signal_in_time, in_axes=in_axes)(signal, tau_arr, frequency, sk, rn)
        signal_shifted = signal_shifted*phase_correction[... , :, None]
        return signal_shifted[ ... , N:2*N] 




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

        # can only happen in collinear setup -> what leaks? pulse, shifted_pulse, both?
        # needs to be incorporated into gradient -> but is nonlinear method independent -> nice :)
        # the leakage, factor could maybe be obtained like the calibration curve?
        # if has_leakage==True:
        #     signal_f = signal_f + measurement_info.leakage_factor*pulse_f
        #     signal_t = self.ifft(signal_f, measurement_info.sk, measurement_info.rn)


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
        idtheta (jnp.array): indices for theta

    """
    
    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, **kwargs):
        super().__init__(nonlinear_method, **kwargs)

        self.theta, self.time, self.frequency, self.measured_trace = self.get_data(theta, frequency, measured_trace)
        self.measurement_info = self.measurement_info.expand(theta = self.theta,
                                                             measured_trace = self.measured_trace)
        

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










class RetrievePulsesSTREAKING(RetrievePulsesFROG):
    """
    The reconstruction class for Attosecond Streaking. Implements the single- and multi-channel Strong-Field-Approximation.
    All axis (time, frequency, ... ) are converted to atomic units for convenience and because that standard with the Strong-Field-Approximation. 

    [1] R. Kienberger, E. Goulielmakis, M. Uiberacker, et al., Atomic transient recorder. Nature 427, 817-821 (2004), 10.1038/nature02277
    [2] P. D. Keathley et al., (2016), New J. Phys. 18 073009, 10.1088/1367-2630/18/7/073009    

    Attributes:
        energy_au (jnp.array): the energy axis in atomic units
        momentum_au (jnp.array): the photoelectron momenta in atomic units
        position_au (jnp.array): the conjugate axis to momentum_au
        sk_position_momentum (jnp.array): same function as sk, but for the position-momentum fourier pair
        rn_position_momentum (jnp.array): same function as rn, but for the position-momentum fourier pair
        Ip_au (jnp.array): the ionization potential in atomic units, the number of channels is infered from the number of Ips
        retrieve_dtme (bool): if true, the dipole-transition-matrix-elements will be retrieved
        dtme_momentum (None, jnp.array): the optionally provided dipole-transition-matrix-elements
        
    """
    
    def __init__(self, delay_fs, energy_eV, measured_trace, Ip_eV=jnp.array([0]), retrieve_dtme=False, 
                 cross_correlation="doubleblind", interferometric=False, 
                 f_range_nir_pulse=(None,None), f_range_euv_pulse=(None,None), eV_range_dtme=(None,None), **kwargs):

        if interferometric!=False:
            print("Interferometric doesnt do anything. Its just kept around because why not.")
        
        delay = self.convert_time_fs_au(delay_fs, "fs", "au")
        self.energy_au = self.convert_energy_eV_au(energy_eV, "eV", "au")
        self.Ip_au = self.convert_energy_eV_au(Ip_eV, "eV", "au")

        # construct frequency grid 
        df_PHz = 0.5/(jnp.max(delay_fs)-jnp.min(delay_fs)) # 0.5 makes sure that the time-axis is bigger than the scanned delay
        df = self.convert_frequency_PHz_au(df_PHz, "PHz", "au")
        axis_nir_pulse, axis_euv_pulse = self.make_axis(f_range_nir_pulse, f_range_euv_pulse, df)

        if retrieve_dtme==True:
            axis_dtme = self.make_axis_dtme(eV_range_dtme, jnp.mean(jnp.diff(energy_eV)))
        else:
            axis_dtme = self.make_axis_dtme((jnp.min(energy_eV), jnp.max(energy_eV)), jnp.mean(jnp.diff(energy_eV)))

        # making the main axis this big is necessary, otherwise aliasing issues prevent convergence
        frequency = jnp.arange(-1*jnp.max(axis_euv_pulse.frequency), jnp.max(axis_euv_pulse.frequency), df)

        momentum_au = jnp.sqrt(2*jnp.abs(self.energy_au))*jnp.sign(self.energy_au)
        # I(k)*dk = I(E)*dE
        # I(k) = I(E)*dE/dk
        # -> multiply by momentum_au

        measured_trace = measured_trace * momentum_au
        self.momentum_au = jnp.linspace(jnp.min(momentum_au), jnp.max(momentum_au), jnp.size(momentum_au)) # this requires interpolation of trace
        measured_trace = do_interpolation_1d(0.5*self.momentum_au**2, self.energy_au, measured_trace, method="cubic2")

        self.position_au = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(self.momentum_au), jnp.mean(jnp.diff(self.momentum_au))))
        self.sk_position_momentum, self.rn_position_momentum = get_sk_rn(self.position_au, self.momentum_au)

        assert cross_correlation!=False, "Streaking is a cross-correlation-like method."
        super().__init__(delay, frequency, measured_trace, None, cross_correlation=cross_correlation, interferometric=interferometric, **kwargs)

        self.no_channels = jnp.size(Ip_eV)
        self.retrieve_dtme = retrieve_dtme
        self.dtme_momentum = None
        self.measurement_info = self.measurement_info.expand(momentum = self.momentum_au,
                                                             position = self.position_au,
                                                             sk_position_momentum = self.sk_position_momentum,
                                                             rn_position_momentum = self.rn_position_momentum,
                                                             ionization_potential = self.Ip_au,
                                                             dtme_momentum = self.dtme_momentum,
                                                             retrieve_dtme = self.retrieve_dtme,
                                                             no_channels = self.no_channels,
                                                             axis_nir = axis_nir_pulse,
                                                             axis_euv = axis_euv_pulse,
                                                             axis_dtme = axis_dtme)
        
        self.descent_info = self.descent_info.expand(measured_spectrum_is_provided = MyNamespace(pulse=False, gate=False, dtme=False))



    def make_axis(self, f_range_nir_pulse, f_range_euv_pulse, df_au):
        fmin_nir, fmax_nir = f_range_nir_pulse
        fmin_euv, fmax_euv = f_range_euv_pulse

        assert fmin_nir is not None and fmin_euv is not None and fmax_nir is not None and fmax_euv is not None, "Frequency ranges for pulses needed."

        fmin_nir = self.convert_frequency_PHz_au(fmin_nir, "PHz", "au")
        fmax_nir = self.convert_frequency_PHz_au(fmax_nir, "PHz", "au")
        fmin_euv = self.convert_frequency_PHz_au(fmin_euv, "PHz", "au")
        fmax_euv = self.convert_frequency_PHz_au(fmax_euv, "PHz", "au")

        frequency_euv = jnp.arange(fmin_euv, fmax_euv, df_au)
        frequency_nir = jnp.arange(fmin_nir, fmax_nir, df_au)
        
        time_nir = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(frequency_nir), jnp.mean(jnp.diff(frequency_nir))))
        time_euv = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(frequency_euv), jnp.mean(jnp.diff(frequency_euv))))

        sk_nir, rn_nir = get_sk_rn(time_nir, frequency_nir)
        sk_euv, rn_euv = get_sk_rn(time_euv, frequency_euv)

        axis_nir = MyNamespace(time=time_nir, frequency=frequency_nir, sk=sk_nir, rn=rn_nir)
        axis_euv = MyNamespace(time=time_euv, frequency=frequency_euv, sk=sk_euv, rn=rn_euv)
            
        return axis_nir, axis_euv


    def make_axis_dtme(self, eV_range_dtme, dE_eV):
        emin, emax = eV_range_dtme
        assert emin is not None and emax is not None, "provide energy range for dtme"

        N = ((emax-emin)/dE_eV).astype(int)
        emin = self.convert_energy_eV_au(emin, "eV", "au")
        emax = self.convert_energy_eV_au(emax, "eV", "au")
        kmin, kmax = jnp.sqrt(2*jnp.abs(emin))*jnp.sign(emin), jnp.sqrt(2*jnp.abs(emax))*jnp.sign(emax)

        momentum = jnp.linspace(kmin, kmax, N)
        position = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(momentum), jnp.mean(jnp.diff(momentum))))
        sk, rn = get_sk_rn(position, momentum)
        return MyNamespace(momentum=momentum, position=position, sk=sk, rn=rn)



    def convert_time_fs_au(self, time, unit_in, unit_out):
        assert (unit_in=="fs" and unit_out=="au") or (unit_out=="fs" and unit_in=="au")
        factor = 2.418884*1e-2
        factor_dict = {"fs_au": 1/factor,
                       "au_fs": factor}
        
        conversion = unit_in + "_" + unit_out
        return time*factor_dict[conversion]


    def convert_frequency_PHz_au(self, frequency, unit_in, unit_out):
        assert (unit_in=="PHz" and unit_out=="au") or (unit_out=="PHz" and unit_in=="au")
        factor = 2.418884*1e-2
        factor_dict = {"PHz_au": factor,
                       "au_PHz": 1/factor}
        
        conversion = unit_in + "_" + unit_out
        return frequency*factor_dict[conversion]
    

    def convert_energy_eV_au(self, energy, unit_in, unit_out):
        assert (unit_in=="eV" and unit_out=="au") or (unit_out=="eV" and unit_in=="au")
        factor = 27.211386
        factor_dict = {"eV_au": 1/factor,
                       "au_eV": factor}
        
        conversion = unit_in + "_" + unit_out
        return energy*factor_dict[conversion]
    


    def get_DTME(self, momentum_au, dtme_momentum):
        dtme_momentum = jnp.atleast_2d(dtme_momentum)
        dtme_momentum = do_interpolation_1d(self.momentum_au, momentum_au, dtme_momentum)
        self.dtme_momentum = dtme_momentum
        self.measurement_info = self.measurement_info.expand(dtme_momentum = self.dtme_momentum)
        return self.dtme_momentum

    

    def get_spectral_amplitude(self, measured_frequency_PHz, measured_spectrum, pulse_or_gate, cutoff_frequency_PHz=0.1, **kwargs):
        """ 
        Used to provide a measured pulse spectrum. A spectrum for the gate pulse can also be provided. 
        The cutoff_frequency_PHz is used if pulse_or_gate="pulse", for the vectorpotential the spectrum is divided 
        by the frequency. In order to avoid divergences the spectrum is set to zero below this cutoff. 
        """

        # pulse is the vectorpotential -> E(w) = iw*A(w)
        if pulse_or_gate=="pulse":
            measured_spectrum = jnp.asarray(measured_spectrum)
            idx0 = jnp.argmin(jnp.abs(measured_frequency_PHz + cutoff_frequency_PHz))
            idx1 = jnp.argmin(jnp.abs(measured_frequency_PHz - cutoff_frequency_PHz))
            measured_spectrum = measured_spectrum.at[idx0:idx1].set(0)
            measured_spectrum = measured_spectrum/((2*jnp.pi*measured_frequency_PHz)**2 + 1e-15)

            frequency = self.measurement_info.axis_nir.frequency
        else:
            frequency = self.measurement_info.axis_euv.frequency

        measured_frequency = self.convert_frequency_PHz_au(measured_frequency_PHz, "PHz", "au")
        return super().get_spectral_amplitude(measured_frequency, measured_spectrum, pulse_or_gate, frequency=frequency, **kwargs)
    
    
    def make_volkov_phase_0(self, Ip, measurement_info):
        time, momentum = measurement_info.time, measurement_info.momentum
        momentum = momentum[:,None]
        return (Ip + 0.5*momentum**2)*time[None,:]


    def make_volkov_phase_1(self, pulse_t_nir_vectorpotential, measurement_info):
        _integrate = Partial(integrate_signal_1D, integration_method="cumsum", integration_order=None)
        time, momentum = measurement_info.time, measurement_info.momentum
        
        pulse_t_nir_vectorpotential = jnp.real(pulse_t_nir_vectorpotential)
        A2_int = _integrate((pulse_t_nir_vectorpotential**2)[::-1], time)[::-1]
        A_int = _integrate(pulse_t_nir_vectorpotential[::-1], time)[::-1]

        momentum = momentum[:,None]
        return momentum*A_int[None,:] + A2_int[None,:]/2


    def make_dressed_DTME(self, dtme_position, pulse_t_nir_vectorpotential, measurement_info):
        if measurement_info.retrieve_dtme==True or measurement_info.dtme_momentum is not None: 
            # this shifting should be done like shifts in time -> with padding and redefinition of axis
            # but its probably fine because shift is much smaller than total axis
            r = measurement_info.position
            sk, rn = measurement_info.sk_position_momentum, measurement_info.rn_position_momentum

            # -1 because one wants to shift to positive values
            momentum_shift = -1*jnp.real(pulse_t_nir_vectorpotential)
            dtme_position = dtme_position*jnp.exp(-1j*2*jnp.pi*r[None,:]*momentum_shift[:,None]) 
            dtme_momentum = self.fft(dtme_position, sk, rn)

            # the momentum axis is not centered around zero, this causes a global phase in the fourier shift 
            # this phase-factor compensates this global phase
            dr = jnp.mean(jnp.diff(r))
            phase_correction = jnp.exp(1j*2*jnp.pi*dr*momentum_shift)
            dtme_momentum = dtme_momentum*phase_correction[:,None]
        else:
            if measurement_info.dtme_momentum is None:
                dtme_momentum = jnp.ones((jnp.size(pulse_t_nir_vectorpotential), jnp.size(dtme_position)))
            else:
                raise ValueError("this shouldnt be reachable")

        return dtme_momentum.T # transpose to go from (k,b) to (b,k)


    def make_streaking_amplitude(self, dtme_position, pulse_t_nir_vectorpotential, pulse_t_euv, measurement_info):
        phase0 = jax.vmap(self.make_volkov_phase_0, in_axes=(0,None))(measurement_info.ionization_potential, measurement_info)
        phase1 = self.make_volkov_phase_1(pulse_t_nir_vectorpotential, measurement_info)
        dtme_momentum = jax.vmap(self.make_dressed_DTME, in_axes=(0,None,None))(dtme_position, pulse_t_nir_vectorpotential, measurement_info)
        dtme_momentum_and_phase0 = jnp.einsum("Cbk, Cbk -> bk", dtme_momentum, jnp.exp(-1j*phase0))

        # naming stuff signal_t, signal_f is technically wrong by convention, but potentially necessary for consistency
        dt = jnp.mean(jnp.diff(measurement_info.time))
        sk, rn = measurement_info.sk_position_momentum, measurement_info.rn_position_momentum
        signal_f = -1j*dt*jnp.einsum("mk, bk, bk -> mb", pulse_t_euv, dtme_momentum_and_phase0, jnp.exp(-1j*phase1))
        signal_t = self.ifft(signal_f, sk, rn)
        return signal_t, signal_f, phase0, phase1, dtme_momentum_and_phase0



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

        pulse_f_nir_vectorpotential = individual.pulse 
        pulse_f_nir_vectorpotential = do_interpolation_1d(measurement_info.frequency, measurement_info.axis_nir.frequency, 
                                                          pulse_f_nir_vectorpotential, method="cubic2")
        pulse_t_nir_vectorpotential = self.ifft(pulse_f_nir_vectorpotential, 
                                               measurement_info.sk, measurement_info.rn)
        


        if measurement_info.doubleblind==True:
            pulse_f_euv = individual.gate
        elif measurement_info.cross_correlation==True:
            pulse_f_euv = measurement_info.gate
        else:
            raise ValueError

        pulse_f_euv = do_interpolation_1d(measurement_info.frequency, measurement_info.axis_euv.frequency, 
                                          pulse_f_euv, method="linear")
        pulse_t_euv = self.ifft(pulse_f_euv, measurement_info.sk, measurement_info.rn)


        if measurement_info.retrieve_dtme == True: # maybe one can optimize for this if nir and euv spectra are provided?
            dtme = individual.dtme
            dtme = do_interpolation_1d(measurement_info.momentum, measurement_info.axis_dtme.momentum, dtme, method="cubic2")
            dtme_position = self.ifft(dtme, measurement_info.sk_position_momentum, measurement_info.rn_position_momentum)
        else: 
            if measurement_info.dtme_momentum is None:
                dtme_position = jnp.ones(measurement_info.no_channels) # a dummy for vmap to work
            else:
                dtme_position = self.ifft(measurement_info.dtme_momentum, measurement_info.sk_position_momentum, measurement_info.rn_position_momentum)


        pulse_t_euv_shifted = self.calculate_shifted_signal(pulse_t_euv, measurement_info.frequency, tau_arr, measurement_info.time)
        signal_t, signal_f, volkov_phase0, volkov_phase1, dtme_shifted_and_volkov_phase0 = self.make_streaking_amplitude(dtme_position, pulse_t_nir_vectorpotential, pulse_t_euv_shifted, measurement_info)        

        signal_t = MyNamespace(signal_t = signal_t,
                               signal_f = signal_f,
                               pulse_t_euv_shifted = pulse_t_euv_shifted,
                               pulse_t_nir_vectorpotential = pulse_t_nir_vectorpotential,
                               volkov_phase0 = volkov_phase0,
                               volkov_phase1 = volkov_phase1,
                               dtme_shifted_and_volkov_phase0 = dtme_shifted_and_volkov_phase0,
                               dtme_position = dtme_position)
        return signal_t
    




    def post_process_get_pulse_and_gate(self, descent_state, measurement_info, descent_info, idx=None):
        """ post processing to get the final pulse/gate """

        idx, population = self.get_idx_best_individual(descent_state.population)

        individual = self.get_individual_from_idx(idx, population)
        pulse_f = individual.pulse[0]
        pulse_t = self.ifft(pulse_f, measurement_info.axis_nir.sk, measurement_info.axis_nir.rn)

        if measurement_info.doubleblind==True:
            gate_f = individual.gate[0] 
            gate_t = self.ifft(gate_f, measurement_info.axis_euv.sk, measurement_info.axis_euv.rn)
        else:
            gate_t, gate_f = pulse_t, pulse_f

        if measurement_info.retrieve_dtme==True:
            dtme = individual.dtme[0]
        else:
            dtme = None

        return MyNamespace(pulse_t=pulse_t, gate_t=gate_t, pulse_f=pulse_f, gate_f=gate_f, dtme_momentum=dtme, idx=idx)



    def post_process_center_pulse_and_gate(self, pulses):
        # dont center in streaking
        """ This essentially removes the linear phase. But only approximately since no fits are done. """
        return pulses


    def post_process(self, descent_state, error_arr):
        fr = super().post_process(descent_state, error_arr)

        delay_fs = self.convert_time_fs_au(fr.theta, "au", "fs")
        time_nir = self.convert_time_fs_au(self.measurement_info.axis_nir.time, "au", "fs")
        time_euv = self.convert_time_fs_au(self.measurement_info.axis_euv.time, "au", "fs")
        frequency_nir = self.convert_frequency_PHz_au(self.measurement_info.axis_nir.frequency, "au", "PHz")
        frequency_euv = self.convert_frequency_PHz_au(self.measurement_info.axis_euv.frequency, "au", "PHz")

        pulses = MyNamespace(vectorpotential_t=fr.pulses.pulse_t, pulse_t_euv=fr.pulses.gate_t, 
                             vectorpotential_f=fr.pulses.pulse_f, pulse_f_euv=fr.pulses.gate_f, 
                             idx=fr.pulses.idx)

        final_result = MyNamespace(theta=delay_fs, time_nir=time_nir, time_euv=time_euv, 
                                   frequency_nir=frequency_nir, frequency_euv=frequency_euv, 
                                   pulses = pulses,
                                   momentum_dtme = self.measurement_info.axis_dtme.momentum,
                                   dtme = fr.pulses.dtme_momentum, 
                                   momentum_au=self.measurement_info.momentum, trace=fr.trace, 
                                   measured_trace=fr.measured_trace, error_arr=fr.error_arr, mu=fr.mu, 
                                   idx_best_individual=fr.idx_best_individual)

        return final_result
    



    
    def plot_results(self, final_result):
        time_nir, time_euv, frequency_nir, frequency_euv = final_result.time_nir, final_result.time_euv, final_result.frequency_nir, final_result.frequency_euv
        delay_fs, momentum_au = final_result.theta, final_result.momentum_au
        trace, measured_trace = final_result.trace, final_result.measured_trace
        error_arr = final_result.error_arr

        vectorpotential_t, vectorpotential_f = final_result.pulses.vectorpotential_t, final_result.pulses.vectorpotential_f
        pulse_t_euv, pulse_f_euv = final_result.pulses.pulse_t_euv, final_result.pulses.pulse_f_euv
        

        trace = trace/jnp.max(trace)
        measured_trace = measured_trace/jnp.max(measured_trace)
        trace_difference = (measured_trace - trace)

        fig=plt.figure(figsize=(22,16))
        ax1=plt.subplot(2,3,1)
        ax1.plot(time_nir, np.abs(vectorpotential_t), label="Vectorpotential (Amp)")
        ax1.plot(time_euv, np.abs(pulse_t_euv), label="EUV-Pulse (Amp)", c="tab:red")
        ax1.set_xlabel(r"Time [fs]")
        ax1.legend(loc=2)
        ax1.set_title("Pulse Time-Domain")

        ax2 = ax1.twinx()
        ax2.plot(time_nir, np.unwrap(np.angle(vectorpotential_t))*1/np.pi, label="Vectorpotential (Phase)", c="tab:orange")
        ax2.plot(time_euv, np.unwrap(np.angle(pulse_t_euv))*1/np.pi, label="EUV-Pulse (Phase)", c="tab:green")
        ax2.set_ylabel(r"Phase [$\pi$]")
        ax2.legend(loc=1)

        

        ax1=plt.subplot(2,3,2)
        ax1.plot(frequency_nir, jnp.abs(vectorpotential_f), label="Vectorpotential (Amp)")
        ax1.plot(frequency_euv, np.abs(pulse_f_euv), label="EUV-Pulse (Amp)", c="tab:red")
        ax1.set_xlabel(r"Frequency [PHz]")
        ax1.legend(loc=2)
        ax1.set_title("Pulse Frequency-Domain")

        ax2 = ax1.twinx()
        ax2.plot(frequency_nir, jnp.unwrap(jnp.angle(vectorpotential_f))*1/np.pi, label="Vectorpotential (Phase)",  c="tab:orange")
        ax2.plot(frequency_euv, np.unwrap(np.angle(pulse_f_euv))*1/np.pi, label="EUV-Pulse (Phase)", c="tab:green")
        ax2.set_ylabel(r"Phase [$\pi$]")
        ax2.legend(loc=1)

        

        plt.subplot(2,3,3)
        plt.plot(error_arr)
        plt.yscale("log")
        plt.title("Trace Error")
        plt.xlabel("Iteration No.")

        plt.subplot(2,3,4)
        plt.pcolormesh(delay_fs, momentum_au, measured_trace.T)
        plt.xlabel("Delay [fs]")
        plt.ylabel("Momentum [a.u.]")
        plt.title("Measured Trace")

        plt.subplot(2,3,5)
        plt.pcolormesh(delay_fs, momentum_au, trace.T)
        plt.xlabel("Delay [fs]")
        plt.ylabel("Momentum [a.u.]")
        plt.title("Retrieved Trace")

        plt.subplot(2,3,6)
        plt.pcolormesh(delay_fs, momentum_au, trace_difference.T)
        plt.xlabel("Delay [fs]")
        plt.ylabel("Momentum [a.u.]")
        plt.colorbar()
        plt.title("Normalized Difference Traces")

        plt.tight_layout()
        plt.show()



        local_or_global = None

        if self.descent_info.optimize_calibration_curve._local==True:
            local_or_global="_local"

        if self.descent_info.optimize_calibration_curve._global==True:
            local_or_global="_global"

        if local_or_global!=None:
            fig=plt.figure()
            plt.plot(momentum_au, getattr(final_result.mu, local_or_global))
            plt.xlabel("Momentum [a.u.]")
            plt.ylabel("Scaling Factor [arb.u.]")
            plt.title("Calibration Curve")
            plt.tight_layout()
            plt.show()


        if self.measurement_info.retrieve_dtme==True:
            fig=plt.figure()
            plt.subplot(1,2,1)
            lines_abs = plt.plot(final_result.momentum_dtme, jnp.abs(final_result.dtme).T)
            for i, line in enumerate(lines_abs):
                line.set_label(str(i))
            plt.legend()
            plt.xlabel("Momentum [a.u.]")
            plt.ylabel("Magnitude DTME [arb.u.]")
            
            plt.subplot(1,2,2)
            lines_phase = plt.plot(final_result.momentum_dtme, jnp.unwrap(jnp.angle(final_result.dtme)).T)
            for i, line in enumerate(lines_phase):
                line.set_label(str(i))
            plt.legend()
            plt.xlabel("Momentum [a.u.]")
            plt.ylabel(r"Phase DTME [$\pi$]")

            plt.suptitle("Dipole transition matrix elements")
            plt.tight_layout()
            plt.show()


