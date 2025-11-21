import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c0
import refractiveindex

import jax
import jax.numpy as jnp

from equinox import tree_at

from src.utilities import MyNamespace, center_signal, get_sk_rn, do_interpolation_1d, calculate_gate, calculate_trace, calculate_trace_error, project_onto_amplitude
from .create_population import create_population_classic
from src.core.initial_guess_doublepulse import make_population_doublepulse
from src.core.phase_matrix_funcs import phase_func_dict, calculate_phase_matrix, calculate_phase_matrix_material




class RetrievePulses:
    """
    The Base-Class for all reconstruction methods. Defines general initialization, preprocessing and postprocessing.

    Attributes:
        nonlinear_method: str, SHG, THG, PG or SD
        f0: float, rarely some solvers need the central frequency to be zero. This saves the original central frequency.
        doubleblind: bool, whether the reconstruction is supposed to yield the gate in addition to the pulse.
        spectrum_is_being_used: bool,
        momentum_is_being_used: bool,
        measurement_info: Pytree, a container of variable (but static) structure. Holds measurement data and parameters.
        descent_info: Pytree, a container of variable (but static) structure. Holds parameters of the reconstruction algorithm.
        descent_state: Pytree, a container of variable (but static) structure. Contains the current state of the solver.
        prng_seed: int, seed for the key
        key: jnp.array, a jax.random.PRNGKey
        factor: int, for SHG/THG the a correction factor of 2/3 needs to applied occasionally.

        x_arr: jnp.array, an alias for the shifts/delays, internally indexed via m
        time: jnp.array, the time axis, internally indexed via k
        frequency: jnp.array, the frequency axis, internally indexed via n
        measured_trace: jnp.array, 2D-array with the measured data. axis=0 corresponds to shift/delay (index m), axis=1 correpsonds to the frequencies (index n)

    """

    def __init__(self, nonlinear_method, *args, seed=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.nonlinear_method = nonlinear_method
        self.f0 = 0
        self.doubleblind = False 

        self.measurement_info = MyNamespace(nonlinear_method = self.nonlinear_method, 
                                            spectral_amplitude = MyNamespace(pulse=None, gate=None), 
                                            central_f = MyNamespace(pulse=None, gate=None),
                                            real_fields = False)
        self.descent_info = MyNamespace(measured_spectrum_is_provided = MyNamespace(pulse=False, gate=False))
        self.descent_state = MyNamespace()


        if seed==None:
            self.prng_seed = np.random.randint(0, 1e9)
        else:
            self.prng_seed = int(seed)

        self.update_PRNG_key(self.prng_seed)

        if nonlinear_method=="shg":
            self.factor = 2
        elif nonlinear_method=="thg":
            self.factor = 3
        else:
            self.factor = 1


    def update_PRNG_key(self, seed):
        self.prng_seed = seed
        self.key = jax.random.PRNGKey(seed)


    def get_data(self, x_arr, frequency, measured_trace):
        """ Prepare/Convert data. """
        measured_trace = measured_trace/jnp.linalg.norm(measured_trace)

        self.x_arr = jnp.asarray(x_arr)
        self.frequency = jnp.asarray(frequency)
        self.time = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(self.frequency), jnp.mean(jnp.diff(self.frequency))))
        self.measured_trace = jnp.asarray(measured_trace)
        return self.x_arr, self.time, self.frequency, self.measured_trace




    def get_spectral_amplitude(self, measured_frequency, measured_spectrum, pulse_or_gate):
        """ Used to provide a measured pulse spectrum. A spectrum for the gate pulse can also be provided. """
        frequency = self.frequency

        spectral_intensity = do_interpolation_1d(frequency, measured_frequency-self.f0/self.factor, measured_spectrum)
        spectral_amplitude = jnp.sqrt(jnp.abs(spectral_intensity))*jnp.sign(spectral_intensity)
        
        if pulse_or_gate=="pulse":
            self.measurement_info.spectral_amplitude.pulse = spectral_amplitude
            self.descent_info.measured_spectrum_is_provided.pulse = True

        elif pulse_or_gate=="gate":
            self.measurement_info.spectral_amplitude.gate = spectral_amplitude
            self.descent_info.measured_spectrum_is_provided.gate = True

        else:
            raise ValueError(f"pulse_or_gate needs to be pulse or gate. Not {pulse_or_gate}")

        return spectral_amplitude
    


    def get_gate_pulse(self, frequency, gate_f):
        """ For crosscorrelation=True the actual gate pulse has to be provided. """
        gate_f = do_interpolation_1d(self.frequency, frequency, gate_f)
        self.gate = self.ifft(gate_f, self.sk, self.rn)
        self.measurement_info = self.measurement_info.expand(gate = self.gate)
        return self.gate
    


    def create_initial_population(self, population_size=1, guess_type="random"):
        """ 
        Creates an initial population.

        Args:
            population_size: int,
            guess_type: str, can be one of random, random_phase, constant or constant_phase

        Returns:
            tuple[jnp.array, jnp.array or None], initial populations for the pulse and possibly the gate-pulse

        """
        self.key, subkey = jax.random.split(self.key, 2)
        pulse_f_arr = create_population_classic(subkey, population_size, guess_type, self.measurement_info)

        if self.doubleblind==True:
            self.key, subkey = jax.random.split(self.key, 2)
            gate_f_arr = create_population_classic(subkey, population_size, guess_type, self.measurement_info)
        else:
            gate_f_arr=None

        self.descent_info = self.descent_info.expand(population_size=population_size)
        return pulse_f_arr, gate_f_arr
    

    def get_individual_from_idx(self, idx, population):
        # idx can also be an array (i think, didnt test)
        leaves, treedef = jax.tree.flatten(population)
        leaves_individual = [leaves[i][idx] for i in range(len(leaves))]
        individual = jax.tree.unflatten(treedef, leaves_individual)
        return individual
    

    


    def plot_results(self, final_result, exact_pulse=None):
        pulse_t, pulse_f, trace = final_result.pulse_t, final_result.pulse_f, final_result.trace
        error_arr = final_result.error_arr

        x_arr, time, frequency, measured_trace = final_result.x_arr, final_result.time, final_result.frequency, final_result.measured_trace
        frequency_exp = final_result.frequency_exp
        
        trace=trace/jnp.max(trace)
        measured_trace=measured_trace/jnp.max(measured_trace)
        trace_difference=measured_trace-trace

        fig=plt.figure(figsize=(22,14))
        ax1=plt.subplot(2,3,1)
        ax1.plot(time, np.abs(pulse_t), label="Amplitude")
        ax1.set_xlabel(r"Time [arb. u.]")
        ax1.legend(loc=2)

        ax2 = ax1.twinx()
        ax2.plot(time, np.unwrap(np.angle(pulse_t))*1/np.pi, c="tab:orange", label="Phase")
        ax2.set_ylabel(r"Phase [$\pi$]")
        ax2.legend(loc=1)

        if exact_pulse!=None:
            ax1.plot(exact_pulse.time, np.abs(exact_pulse.pulse_t)*np.max(np.abs(pulse_t))/np.max(np.abs(exact_pulse.pulse_t)), 
                     "--", c="black", label="Exact Amplitude")
            ax2.plot(exact_pulse.time, np.unwrap(np.angle(exact_pulse.pulse_t)), "--", c="black", label="Exact Phase", alpha=0.5)

        ax1=plt.subplot(2,3,2)
        ax1.plot(frequency,jnp.abs(pulse_f), label="Amplitude")
        ax1.set_xlabel(r"Frequency [arb. u.]")
        ax1.legend(loc=2)

        ax2 = ax1.twinx()
        ax2.plot(frequency, jnp.unwrap(jnp.angle(pulse_f))*1/np.pi, c="tab:orange", label="Phase")
        ax2.set_ylabel(r"Phase [$\pi$]")
        ax2.legend(loc=1)

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

        plt.subplot(2,3,5)
        plt.pcolormesh(x_arr, frequency_exp, trace.T)
        plt.xlabel("Shift [arb. u.]")
        plt.ylabel("Frequency [arb. u.]")

        plt.subplot(2,3,6)
        plt.pcolormesh(x_arr, frequency_exp, trace_difference.T)
        plt.xlabel("Shift [arb. u.]")
        plt.ylabel("Frequency [arb. u.]")
        plt.colorbar()







    def get_idx_best_individual(self, descent_state):
        """ Calculates trace error for whole population. Returns the idx of the individual with the smallest error."""
        measurement_info, descent_info = self.measurement_info, self.descent_info 

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)        
        trace = calculate_trace(self.fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn))
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measurement_info.measured_trace)
        idx = jnp.argmin(trace_error)
        return idx
    




    def post_process_center_pulse_and_gate(self, pulse_t, gate_t):
        """ This essentially removes the linear phase. But only approximately since no fits are done. """
        sk, rn = self.measurement_info.sk, self.measurement_info.rn

        pulse_t = center_signal(pulse_t)
        gate_t = center_signal(gate_t)

        pulse_f = self.fft(pulse_t, sk, rn)
        gate_f = self.fft(gate_t, sk, rn)

        return pulse_t, gate_t, pulse_f, gate_f
    
    

    def post_process_create_trace(self, individual):
        """ Post processing to get the final trace """
        sk, rn = self.measurement_info.sk, self.measurement_info.rn
        transform_arr = self.measurement_info.transform_arr
    
        signal_t = self.calculate_signal_t(individual, transform_arr, self.measurement_info)
        signal_f = self.fft(signal_t.signal_t, sk, rn)
        trace = calculate_trace(signal_f)
        return trace
    



    def post_process(self, descent_state, error_arr):
        """ Creates the final_result object from the final descent_state. """
        self.descent_state = descent_state

        pulse_t, gate_t, pulse_f, gate_f = self.post_process_get_pulse_and_gate(descent_state, self.measurement_info, self.descent_info)
        pulse_t, gate_t, pulse_f, gate_f = self.post_process_center_pulse_and_gate(pulse_t, gate_t)

        measured_trace = self.measurement_info.measured_trace
        measured_trace = measured_trace/jnp.linalg.norm(measured_trace)

        if isinstance(self, (RetrievePulsesFROG, RetrievePulses2DSI)):
            individual = MyNamespace(pulse=pulse_t, gate=gate_t)
        elif isinstance(self, RetrievePulsesCHIRPSCAN):
            individual = MyNamespace(pulse=pulse_f, gate=None)
        else:
            raise ValueError(f"self doesnt match. Is {self}")
        
        trace = self.post_process_create_trace(individual)
        trace = trace/jnp.linalg.norm(trace)

        x_arr = self.measurement_info.x_arr
        time, frequency = self.measurement_info.time, self.measurement_info.frequency + self.f0

        final_result_population = self.get_final_result_population()
        final_result = MyNamespace(x_arr=x_arr, time=time, frequency=frequency, frequency_exp=frequency,
                                 pulse_t=pulse_t, pulse_f=pulse_f, gate_t=gate_t, gate_f=gate_f,
                                 trace=trace, measured_trace=measured_trace,
                                 error_arr=error_arr, population=final_result_population)
        return final_result
    


    def get_final_result_population(self):
        time, frequency = self.measurement_info.time, self.measurement_info.frequency
        N = self.descent_info.population_size

        final_result_arr = []
        for idx in range(N):
            pulse_t, gate_t, pulse_f, gate_f = self.post_process_get_pulse_and_gate(self.descent_state, self.measurement_info, self.descent_info, idx=idx)
            pulse_t, gate_t, pulse_f, gate_f = self.post_process_center_pulse_and_gate(pulse_t, gate_t)
            final_result = MyNamespace(time=time, frequency=frequency, pulse_t=pulse_t, pulse_f=pulse_f, gate_t=gate_t, gate_f=gate_f)
            final_result_arr.append(final_result)

        return final_result_arr




    
    
    































class RetrievePulsesFROG(RetrievePulses):
    """
    The reconstruction class for FROG. Inherits from RetrievePulses.

    Attributes:
        tau_arr: jnp.array, the delays
        gate: jnp.array, the gate-pulse (if its known).
        transform_arr: jnp.array, an alias for tau_arr
        idx_arr: jnp.array, an array with indices for tau_arr
        dt: float,
        df: float,
        sk: jnp.array, correction values for FFT->DFT
        rn: jnp.array, correction values for FFT->DFT
        cross_correlation: bool,
        ifrog: bool, 

    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, ifrog=False, **kwargs):
        
        super().__init__(nonlinear_method, **kwargs)

        self.tau_arr, self.time, self.frequency, self.measured_trace = self.get_data(delay, frequency, measured_trace)
        self.gate = jnp.zeros(jnp.size(self.time))

        self.transform_arr = self.tau_arr
        self.idx_arr = jnp.arange(jnp.shape(self.transform_arr)[0])   
        

        self.dt = jnp.mean(jnp.diff(self.time))
        self.df = jnp.mean(jnp.diff(self.frequency))
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        self.cross_correlation = cross_correlation
        self.ifrog = ifrog

        if self.cross_correlation=="doubleblind":
            self.doubleblind = True
            self.cross_correlation = False
        elif self.cross_correlation==True or self.cross_correlation==False:
            pass
        else: 
            raise ValueError(f"cross_correlation can only take one of doubleblind, True or False. Got {self.cross_correlation}") 

        self.measurement_info = self.measurement_info.expand(tau_arr = self.tau_arr,
                                                             frequency = self.frequency,
                                                             time = self.time,
                                                             measured_trace = self.measured_trace,
                                                             cross_correlation = self.cross_correlation,
                                                             doubleblind = self.doubleblind,
                                                             ifrog = self.ifrog,
                                                             dt = self.dt,
                                                             df = self.df,
                                                             sk = self.sk,
                                                             rn = self.rn,
                                                             gate = self.gate,
                                                             transform_arr = self.transform_arr,
                                                             x_arr = self.x_arr)
        


    def create_initial_population(self, population_size=1, guess_type="random"):
        """ 
        Creates an initial guess. The guess is in the time domain.

        Args:
            population_size: int, the number of guesses
            guess_type: str, the guess type. Can be one of random, random_phase, constant or constant_phase.

        Returns:
            Pytree
        
        """

        pulse_f_arr, gate_f_arr = super().create_initial_population(population_size, guess_type)

        sk, rn = self.sk, self.rn
        pulse_t_arr = self.ifft(pulse_f_arr, sk, rn)

        if self.measurement_info.doubleblind==True:
            gate_t_arr = self.ifft(gate_f_arr, sk, rn)
        else:
            gate_t_arr = None

        population = MyNamespace(pulse=pulse_t_arr, gate=gate_t_arr)
        return population



    def create_initial_population_doublepulse(self, population_size, **kwargs):
        """ 
        Calls initial_guess_doublepulse.make_population_doublepulse to create an initial guess.
        The guess is in the time domain. Assumes an autocorrelation FROG.
        
        Args:
            population_size: int,
            **kwargs: passed to make_population_doublepulse()

        Returns:
            Pytree

        """
        measurement_info = self.measurement_info
        assert measurement_info.doubleblind==False, "Only implemented for doubleblind=False"
        
        self.key, subkey = jax.random.split(self.key, 2)

        tau_arr, frequency, measured_trace = measurement_info.tau_arr, measurement_info.frequency, measurement_info.measured_trace
        nonlinear_method = measurement_info.nonlinear_method
        pulse_t_arr = make_population_doublepulse(subkey, population_size, tau_arr, frequency, measured_trace, nonlinear_method, **kwargs)

        population = MyNamespace(pulse=pulse_t_arr, gate=None)
        return population
    





    def shift_signal_in_time(self, signal, tau, frequency, sk, rn):
        """ The Fourier-Shift theorem. """
        signal_f = self.fft(signal, sk, rn)
        signal_f = signal_f*jnp.exp(-1j*2*jnp.pi*frequency*tau)
        signal = self.ifft(signal_f, sk, rn)
        return signal


    def calculate_shifted_signal(self, signal, frequency, tau_arr, time, in_axes=(None, 0, None, None, None)):
        """ The Fourier-Shift theorem applied to a list of signals. """

        # im really unhappy with this, but this re-definition/calculation of sk, rn is necessary?
        # in the original case a global phase shift dependent on tau and f[0] occured, which i couldnt figure out
        frequency = frequency - (frequency[-1] + frequency[0])/2

        N = jnp.size(frequency)
        pad_arr = [(0,0)]*(signal.ndim-1) + [(0,N)]
        signal = jnp.pad(signal, pad_arr)
        
        frequency = jnp.linspace(jnp.min(frequency), jnp.max(frequency), 2*N)
        time = jnp.fft.fftshift(jnp.fft.fftfreq(2*N, jnp.mean(jnp.diff(frequency))))

        sk, rn = get_sk_rn(time, frequency)

        signal_shifted = jax.vmap(self.shift_signal_in_time, in_axes=in_axes)(signal, tau_arr, frequency, sk, rn)
        return signal_shifted[ ... , :N]




    def calculate_signal_t(self, individual, tau_arr, measurement_info):
        """
        Calculates the signal field of a FROG in the time domain. 

        Args:
            individual: Pytree, a population containing only one member. (jax.vmap over whole population)
            tau_arr: jnp.array, the delays
            measurement_info: Pytree, contains the measurement parameters (e.g. nonlinear method, interferometric, ... )

        Returns:
            Pytree, contains the signal field in the time domain as well as the fields used to calculate it.
        """

        time, frequency = measurement_info.time, measurement_info.frequency
        cross_correlation, doubleblind, ifrog = measurement_info.cross_correlation, measurement_info.doubleblind, measurement_info.ifrog
        frogmethod = measurement_info.nonlinear_method

        pulse, gate = individual.pulse, individual.gate


        pulse_t_shifted=self.calculate_shifted_signal(pulse, frequency, tau_arr, time)

        if cross_correlation==True:
            gate_pulse_shifted =self.calculate_shifted_signal(measurement_info.gate, frequency, tau_arr, time)
            gate_shifted = calculate_gate(gate_pulse_shifted, frogmethod)

        elif doubleblind==True:
            gate_pulse_shifted = self.calculate_shifted_signal(gate, frequency, tau_arr, time)
            gate_shifted = calculate_gate(gate_pulse_shifted, frogmethod)

        else:
            gate_pulse_shifted=None
            gate_shifted=calculate_gate(pulse_t_shifted, frogmethod)


        if ifrog==True and cross_correlation==False and doubleblind==False:
            signal_t=(pulse + pulse_t_shifted)*calculate_gate(pulse + pulse_t_shifted, frogmethod)
        elif ifrog==True:
            signal_t=(pulse + gate_pulse_shifted)*calculate_gate(pulse + gate_pulse_shifted, frogmethod)
        else:
            signal_t=pulse*gate_shifted
            

        signal_t = MyNamespace(signal_t=signal_t, pulse_t_shifted=pulse_t_shifted, gate_shifted=gate_shifted, gate_pulse_shifted=gate_pulse_shifted)
        return signal_t




    def generate_signal_t(self, descent_state, measurement_info, descent_info):
        """ Applies calculate_signal_t to a whole population via jax.vmap """
        tau_arr = measurement_info.tau_arr
        population = descent_state.population
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,None,None))(population, tau_arr, measurement_info)
        return signal_t
    




    def post_process_get_pulse_and_gate(self, descent_state, measurement_info, descent_info, idx=None):
        """ FROG specific post processing to get the final pulse/gate """
        sk, rn = measurement_info.sk, measurement_info.rn

        if idx==None:
            idx = self.get_idx_best_individual(descent_state)

        individual = self.get_individual_from_idx(idx, descent_state.population)
        pulse_t = individual.pulse
        pulse_f = self.fft(pulse_t, sk, rn)

        if measurement_info.doubleblind==True:
            gate_t = individual.gate
            gate_f = self.fft(gate_t, sk, rn)
        else:
            gate_t, gate_f = pulse_t, pulse_f

        return pulse_t, gate_t, pulse_f, gate_f
    


    
    def apply_spectrum(self, pulse_t, spectrum, sk, rn):
        """ FROG specific method to project the pulse guess onto a measured spectrum. """
        pulse_f = self.fft(pulse_t, sk, rn)
        pulse_f_new = project_onto_amplitude(pulse_f, spectrum)
        pulse_t = self.ifft(pulse_f_new, sk, rn)
        return pulse_t
    









class RetrievePulsesCHIRPSCAN(RetrievePulses):
    """
    The reconstruction class for Chirp-Scan methods. Inherits from RetrievePulses.

    Attributes:
        z_arr: jnp.array, the shifts
        dt: float,
        df: float,
        sk: jnp.array, correction values for FFT->DFT
        rn: jnp.array, correction values for FFT->DFT
        phase_matrix: jnp.array, a 2D-array with the phase values applied to pulse
        parameters: tuple, parameters for the chirp function
        transform_arr: jnp.array, an alias for phase_matrix
        idx_arr: jnp.array, indices for z_arr

    """
    
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, **kwargs):
        super().__init__(nonlinear_method, **kwargs)

        self.z_arr, self.time, self.frequency, self.measured_trace = self.get_data(z_arr, frequency, measured_trace)

        self.dt = jnp.mean(jnp.diff(self.time))
        self.df = jnp.mean(jnp.diff(self.frequency))
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)


        self.measurement_info = self.measurement_info.expand(z_arr = self.z_arr,
                                                             frequency = self.frequency,
                                                             time = self.time,
                                                             measured_trace = self.measured_trace,
                                                             doubleblind = self.doubleblind,
                                                             dt = self.dt,
                                                             df = self.df,
                                                             sk = self.sk,
                                                             rn = self.rn,
                                                             x_arr = self.x_arr)
        

        self.phase_type = phase_type
        self.phase_matrix = self.get_phase_matrix(chirp_parameters)
        



    def get_phase_matrix(self, parameters):
        """ Calls phase_matrix_func in order to calculate the phase matrix. """
        self.parameters = parameters

        if self.phase_type=="material":
            self.phase_matrix = calculate_phase_matrix_material(self.measurement_info, parameters)

        elif type(self.phase_type)==str:
            self.phase_matrix = calculate_phase_matrix(self.measurement_info, parameters, phase_func=phase_func_dict[self.phase_type])

        elif callable(self.phase_type)==True:
            self.phase_matrix = self.phase_type(self.measurement_info, parameters)

        else:
            raise ValueError(f"phase_type needs to be a string or a callable. Not {self.phase_type}")


        self.transform_arr = self.phase_matrix
        self.idx_arr = jnp.arange(jnp.shape(self.transform_arr)[0])   

        self.measurement_info = self.measurement_info.expand(phase_matrix = self.phase_matrix,
                                                             transform_arr = self.transform_arr)
        return self.phase_matrix
        





    def create_initial_population(self, population_size=1, guess_type="random"):
        """ 
        Creates an initial guess. The guess is in the frequency domain.

        Args:
            population_size: int, the number of guesses
            guess_type: str, the guess type. Can be one of random, random_phase, constant or constant_phase.

        Returns:
            Pytree
        
        """

        pulse_f_arr, gate_f_arr = super().create_initial_population(population_size, guess_type)

        if self.doubleblind==True:
            gate_arr = gate_f_arr
        else:
            gate_arr = None

        population = MyNamespace(pulse=pulse_f_arr, gate=gate_arr)
        return population
    



    def get_dispersed_pulse_t(self, pulse_f, phase_matrix, sk, rn):
        """ Applies phase-matrix to a signal. """
        
        pulse_f = pulse_f*jnp.exp(1j*phase_matrix)
        pulse_t_disp = self.ifft(pulse_f, sk, rn)

        return pulse_t_disp, phase_matrix
    



    def calculate_signal_t(self, individual, phase_matrix, measurement_info):
        """
        Calculates the signal field of a Chirp-Scan in the time domain. 

        Args:
            individual: Pytree, a population containing only one member. (jax.vmap over whole population)
            phase_matrix: jnp.array, the applied phases
            measurement_info: Pytree, contains the measurement parameters (e.g. nonlinear method, ... )

        Returns:
            Pytree, contains the signal field in the time domain as well as the fields used to calculate it.
        """

        pulse = individual.pulse

        pulse_t_disp, phase_matrix = self.get_dispersed_pulse_t(pulse, phase_matrix, measurement_info.sk, measurement_info.rn)
        gate_disp = calculate_gate(pulse_t_disp, measurement_info.nonlinear_method)
        signal_t = pulse_t_disp*gate_disp

        signal_t = MyNamespace(signal_t=signal_t, pulse_t_disp=pulse_t_disp, gate_disp=gate_disp)
        return signal_t
    

    def generate_signal_t(self, descent_state, measurement_info, descent_info):
        """ Applies calculate_signal_t to a whole population via jax.vmap """
        phase_matrix = measurement_info.phase_matrix
        population = descent_state.population
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,None,None))(population, phase_matrix, measurement_info)
        return signal_t





    


    def post_process_get_pulse_and_gate(self, descent_state, measurement_info, descent_info, idx=None):
        """ Chirp-Scan specific post processing to get the final pulse/gate """
        sk, rn =  measurement_info.sk, measurement_info.rn

        if idx==None:
            idx = self.get_idx_best_individual(descent_state)

        individual = self.get_individual_from_idx(idx, descent_state.population)
        pulse_f = individual.pulse
        pulse_t = self.ifft(pulse_f, sk, rn)

        if measurement_info.doubleblind==True:
            gate_f = individual.gate
            gate_t = self.ifft(gate_f, sk, rn)
        else:
            gate_f, gate_t = pulse_f, pulse_t

        return pulse_t, gate_t, pulse_f, gate_f
    

    def apply_spectrum(self, pulse, spectrum, sk, rn):
        """ Chirp-Scan specific method to project the pulse guess onto a measured spectrum. """
        pulse = project_onto_amplitude(pulse, spectrum)
        return pulse










class RetrievePulses2DSI(RetrievePulsesFROG):
    """
    The reconstruction class for 2DSI. Inherits from RetrievePulsesFROG.

    Attributes:
        anc1_frequency: float, the first ancillary frequency
        anc2_frequency: float, the second ancillary frequency
        c0: float, the speed of light
        refractive_index: refractiveindex.RefractiveIndexMaterial, returns the refractive index for a material given a wavelength in um
        phase_matrix: jnp.array, a 2D-array with phase values that could potentially have been applied to a pulse

    """

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=True, anc1_frequency=None, anc2_frequency=None, 
                 material_thickness=0, refractive_index = refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, ifrog=False, **kwargs)

        self.anc1_frequency = anc1_frequency
        self.anc2_frequency = anc2_frequency
        self.c0 = c0
        self.refractive_index = refractive_index

        self.measurement_info = self.measurement_info.expand(c0=self.c0)
        self.phase_matrix = self.get_phase_matrix(self.refractive_index, material_thickness, self.measurement_info)
        self.measurement_info = self.measurement_info.expand(anc1_frequency = self.anc1_frequency, anc2_frequency = self.anc2_frequency, 
                                                             phase_matrix=self.phase_matrix)
        



    def get_anc_pulse(self, frequency, anc_f, anc_no=1):
        """ For cross_correlation instead of the gate pulse the two-acillae pulses need to be provided. """
        anc_f = do_interpolation_1d(self.frequency, frequency, anc_f)
        anc = self.ifft(anc_f, self.sk, self.rn)

        anc_dict = {1: self.measurement_info.expand(anc_1=anc), 
                    2: self.measurement_info.expand(anc_2=anc)}
        self.measurement_info = anc_dict[anc_no]
        return anc
    


    def get_phase_matrix(self, refractive_index, material_thickness, measurement_info):
        """ 
        Calculates the phase matrix that is applied of a pulse passes through a material.
        """
        frequency, c0 = measurement_info.frequency, measurement_info.c0
        c0 = c0*1e-12 # speed of light in mm/fs
        wavelength = c0/(frequency+1e-15)
        n_arr = refractive_index.material.getRefractiveIndex(jnp.abs(wavelength)*1e6 + 1e-9, bounds_error=False) # wavelength needs to be in nm
        n_arr = jnp.where(jnp.isnan(n_arr)==False, n_arr, 1.0)
        k_arr = 2*jnp.pi/(wavelength + 1e-9)*n_arr
        phase_matrix = k_arr*material_thickness
        return phase_matrix



    def apply_phase(self, pulse_t, measurement_info, sk, rn):
        """
        For an auto-correlation 2DSI reconstruction one may need to consider effects of a beam splitter in the interferometer.
        This applies a dispersion based on phase_matrix in order to achieve this.
        """
        
        pulse_f = self.fft(pulse_t, sk, rn)
        pulse_f=pulse_f*jnp.exp(1j*measurement_info.phase_matrix)
        pulse_t_disp=self.ifft(pulse_f, sk, rn)
        return pulse_t_disp
    


    def calculate_signal_t(self, individual, tau_arr, measurement_info):
        """
        Calculates the signal field of 2DSI in the time domain. 

        Args:
            individual: Pytree, a population containing only one member. (jax.vmap over whole population)
            tau_arr: jnp.array, the delays
            measurement_info: Pytree, contains the measurement parameters (e.g. nonlinear method, ... )

        Returns:
            Pytree, contains the signal field in the time domain as well as the fields used to calculate it.
        """

        time, frequency, nonlinear_method = measurement_info.time, measurement_info.frequency, measurement_info.nonlinear_method

        pulse_t = individual.pulse

        if measurement_info.cross_correlation==True:
            gate1, gate2 = measurement_info.anc_1, measurement_info.anc_2

        elif measurement_info.doubleblind==True:
            gate1 = gate2 = individual.gate

        else:
            sk, rn = measurement_info.sk, measurement_info.rn
            gate1 = gate2 = self.apply_phase(pulse_t, measurement_info, sk, rn)
            

        gate2_shifted = self.calculate_shifted_signal(gate2, frequency, tau_arr, time)
        gate_pulses = gate1 + gate2_shifted
        gate = calculate_gate(gate_pulses, nonlinear_method)
        signal_t = pulse_t*gate

        signal_t = MyNamespace(signal_t=signal_t, gate_pulses=gate_pulses, gate=gate)
        return signal_t


















