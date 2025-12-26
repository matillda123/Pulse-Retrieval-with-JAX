import jax
import jax.numpy as jnp
import numpy as np

from jax.tree_util import Partial
from equinox import tree_at

from src.utilities import MyNamespace, do_fft, do_ifft, calculate_trace, run_scan, get_com, loss_function_modifications, project_onto_amplitude
from .bsplines_1d import get_prefactor, get_M, make_bsplines
from .create_population import create_population_general

from .base_classes_methods import RetrievePulsesFROG, RetrievePulsesCHIRPSCAN, RetrievePulsesTDP, RetrievePulses2DSI, RetrievePulsesVAMPIRE


class AlgorithmsBASE:
    """
    The Base-Class for all solvers.

    Attributes:
        jit (bool): enables/disables jax.jit
        spectrum_is_being_used (bool): what the name says
        fft (Callable): performs an fft, performs an fft of a signal. (Needs to expect signal, sk, rn, axis)
        ifft (Callable): performs an ifft, performs an ifft of a signal. (Needs to expect signal, sk, rn, axis)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.jit = False
        self.spectrum_is_being_used = False


        self.fft = do_fft
        self.ifft = do_ifft
        self._name = "AlgorithmsBASE"



    def run(self, init_vals, no_iterations=1):
        """ This function is invoked by most solvers to perform the iterative reconstruction. """
        if self.spectrum_is_being_used==True:
            assert self.descent_info.measured_spectrum_is_provided.pulse==True or self.descent_info.measured_spectrum_is_provided.gate==True, "you need to provide a spectrum"
        
        if self.measurement_info.doubleblind==True:
            if self.descent_info.measured_spectrum_is_provided.pulse==False or self.descent_info.measured_spectrum_is_provided.gate==False:
                print("Doubleblind Retrieval has uniqueness issues. You should provide spectra for pulse and gate-pulse.")

        carry, do_scan = self.initialize_run(init_vals)
        carry, error_arr = run_scan(do_scan, carry, no_iterations, self.jit)

        error_arr = jnp.squeeze(error_arr)
        final_result = self.post_process(carry, error_arr)
        return final_result
    


    def do_step_and_apply_spectrum(self, descent_state, measurement_info, descent_info, do_step):
        """ If a spectrum is provided this wraps around the step-method of all solvers and projects the current guess onto the measured spectrum. """
        population = descent_state.population
        
        if descent_info.measured_spectrum_is_provided.pulse==True:
            norm_pulse = jnp.linalg.norm(population.pulse, axis=-1)
            pulse = jax.vmap(self.apply_spectrum, in_axes=(0,None,None,None))(population.pulse, measurement_info.spectral_amplitude.pulse, 
                                                                              measurement_info.sk, measurement_info.rn)
            
            pulse = pulse/jnp.linalg.norm(pulse,axis=-1)[:,jnp.newaxis]*norm_pulse[:,jnp.newaxis]
            population = tree_at(lambda x: x.pulse, population, pulse)

        if descent_info.measured_spectrum_is_provided.gate==True:
            norm_gate = jnp.linalg.norm(population.gate, axis=-1)
            gate = jax.vmap(self.apply_spectrum, in_axes=(0,None,None,None))(population.gate, measurement_info.spectral_amplitude.gate, 
                                                                             measurement_info.sk, measurement_info.rn)
            gate = gate/jnp.linalg.norm(gate,axis=-1)[:,jnp.newaxis]*norm_gate[:,jnp.newaxis]
            population = tree_at(lambda x: x.gate, population, gate)
            
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state, trace_error = do_step(descent_state, measurement_info, descent_info)
        return descent_state, trace_error





    def use_measured_spectrum(self, frequency, spectrum, pulse_or_gate="pulse"):
        """ 
        Needs to be called if a pulse spectrum is meant to be used in the reconstruction. 

        Args:
            frequency: jnp.array, the frequency axis of spectrum
            spectrum: jnp.array, the spectrum
            pulse_or_gate (str): whether the spectrum is from the pulse or the gate-pulse.
        
        Returns:
            the class instance
        """
        _ = self.get_spectral_amplitude(frequency, spectrum, pulse_or_gate)

        if self.spectrum_is_being_used==True:
            return self
        else:
            names_list = ["DifferentialEvolution", "Evosax", "AutoDiff", "DirectReconstruction"]

            if self._name=="COPRA" or self._name=="PtychographicIterativeEngine":
                self._local_step = self.local_step
                self._global_step = self.global_step
                self.local_step = Partial(self.do_step_and_apply_spectrum, do_step=self._local_step)
                self.global_step = Partial(self.do_step_and_apply_spectrum, do_step=self._global_step)

            elif any([self._name==name for name in names_list])==True:
                # in these classes the spectrum is applied directly
                pass
            else:
                self._step = self.step
                self.step = Partial(self.do_step_and_apply_spectrum, do_step=self._step)

            self.spectrum_is_being_used = True
            return self
        



    def apply_spectrum(self, pulse, spectrum, sk, rn):
        if isinstance(self, (RetrievePulsesFROG, RetrievePulsesTDP, RetrievePulses2DSI, RetrievePulsesVAMPIRE)):
            pulse = self.apply_spectrum_time_domain(pulse, spectrum, sk, rn)
        elif isinstance(self, RetrievePulsesCHIRPSCAN):
            pulse = self.apply_spectrum_frequency_domain(pulse, spectrum, sk, rn)
        else:
            raise ValueError(f"""self needs to be one of RetrievePulsesFROG, RetrievePulsesTDP, RetrievePulses2DSI, 
                             RetrievePulsesVAMPIRE or RetrievePulsesChripScan. Not {self}""")
        return pulse

    
    def apply_spectrum_time_domain(self, pulse_t, spectrum, sk, rn):
        pulse_f = self.fft(pulse_t, sk, rn)
        pulse_f_new = project_onto_amplitude(pulse_f, spectrum)
        pulse_t = self.ifft(pulse_f_new, sk, rn)
        return pulse_t
    
    def apply_spectrum_frequency_domain(self, pulse, spectrum, sk, rn):
        pulse = project_onto_amplitude(pulse, spectrum)
        return pulse
    





    def __repr__(self):
        mydict = self.__dict__
        keys = mydict.keys()

        myoutput = {}
        for key in keys:
            value = mydict[key]

            if isinstance(value, MyNamespace):
                myoutput[key] = value.__repr__()
            else:
                if isinstance(value, (jax.Array,tuple,list,np.ndarray)):
                    #myoutput.append([key, jnp.shape(jnp.asarray(value)), value.dtype])
                    myoutput[key] = ["shape=", jnp.shape(jnp.asarray(value)), value.dtype]
                else:
                    #myoutput.append([key, value, value.dtype])
                    try:
                        myoutput[key] = [value, value.dtype]
                    except:
                        myoutput[key] = [value, type(value).__name__]
                    
        return f"{self._name}(" + f"{myoutput}".replace("\'","").replace("\"", "").replace("shape=, ","shape=")[1:-1] + ")"



    








class ClassicAlgorithmsBASE(AlgorithmsBASE):
    """
    The Base-Class for all classical solvers (e.g. Generalized Projection, ...). Inherits from AlgorithmsBASE.
    
    Attributes:
        local_gamma (float): step size for local iterations
        global_gamma (float): step size for global iterations

        linesearch (bool, str): enables/disables a linesearch. Can be False, backtracking or zoom.
        max_steps_linesearch (int): maximum number of linesearch steos
        c1 (float): constant for the Armijo-condition
        c2 (float): constant for the strong Wolfe-condition
        delta_gamma (float): a factor which by which gamma is increased each iteration

        local_newton (bool, str): enables/disables the use of a hessian in local iterations. Can be False, lbfgs or diagonal.
        global_newton (bool, str): enables/disables the use of a hessian in global iterations. Can be False, lbfgs, diagonal or full.
        lambda_lm (float): a Levenberg-Marquardt style damping coefficient.
        lbfgs_memory (int): the number of past iterations to use in LBFGS
        linalg_solver (str, lineax-solver): chooses a library/method for inverting the hessian. Can be scipy, lineax or a specific lineax solver.

        conjugate_gradients (bool, str): enables/diables the use of the Nonlinear Conjugate Gradients method. Can be False, Fletcher-Reeves, 
                                            Hestenes-Stiefel, Dai-Yuan, Polak-Ribiere or average.

        r_local_method (str): chooses the method on the calculation of S_prime in local iterations. Can be projection or iteration.
        r_global_method (str): chooses the method on the calculation of S_prime in global iterations. Can be projection or iteration.
        r_gradient (str): if r_method=iteration, chooses the type of residual to optimize. Can be amplitude or intensity.
        r_newton (bool): enables/diables the use of the diagonal hessian if r_method=iteration
        r_weights (float, jnp.array): allows the weigthing of residuals
        r_no_iterations (int): the number of iterations if r_method=iteration
        r_step_scaling (str): the type of adpative step-size scaling to use if r_method=iteration

        xi (float): a damping coefficient for adaptive step-sizes, avoids division by zero
        local_adaptive_scaling (bool, str): enables/disables adaptive step sized in local iterations. Can be one of False, pade_10 (linear), pade_20 (nonlinear),
                                            pade_11, pade_01 or pade_02
        global_adaptive_scaling (bool, str): enables/disables adaptive step sized in global iterations. Can be one of False, pade_10 (linear), pade_20 (nonlinear),
                                            pade_11, pade_01 or pade_02

        momentum_is_being_used (bool): what the name says


    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "ClassicAlgorithmsBASE"

        self.local_gamma = 1
        self.global_gamma = 1

        self.linesearch = False
        self.max_steps_linesearch = 15
        self.c1 = 1e-4
        self.c2 = 0.9
        self.delta_gamma = 0.5

        self.local_newton = False
        self.global_newton = False
        self.lambda_lm = 1e-3
        self.lbfgs_memory = 10
        self.linalg_solver = "lineax"

        self.conjugate_gradients = False

        self.r_local_method = "projection"
        self.r_global_method = "projection"
        self.r_gradient = "intensity"
        self.r_newton = False
        self.r_weights = 1.0
        self.r_no_iterations = 1
        self.r_step_scaling = "linear"

        self.local_adaptive_scaling = False
        self.global_adaptive_scaling = False
        self.xi = 1e-12

        self.momentum_is_being_used = False




    def shuffle_data_along_m(self, descent_state, measurement_info, descent_info):
        """ 
        Some solvers randomize local iterations. This is done through this method. 
        It returns shuffled but consistent(!) data.
        """
        descent_state.key, subkey=jax.random.split(descent_state.key, 2)
        keys = jax.random.split(subkey, descent_info.population_size)

        idx_arr = jnp.arange(jnp.shape(measurement_info.transform_arr)[0])   
        idx_arr=jax.vmap(jax.random.permutation, in_axes=(0,None))(keys, idx_arr)

        transform_arr, measured_trace = measurement_info.transform_arr, measurement_info.measured_trace
        
        transform_arr = jax.vmap(Partial(jnp.take, axis=0), in_axes=(None, 0), out_axes=1)(transform_arr, idx_arr)
        measured_trace = jax.vmap(Partial(jnp.take, axis=0), in_axes=(None, 0), out_axes=1)(measured_trace, idx_arr)

        transform_arr = jnp.expand_dims(transform_arr, axis=2)
        measured_trace = jnp.expand_dims(measured_trace, axis=2)
        return transform_arr, measured_trace, descent_state





    def do_step_and_apply_momentum(self, descent_state, measurement_info, descent_info, do_step):
        """ If momentum is being used this wraps around the step-method of all solvers and updates the current guess accordingly. """
        population = descent_state.population
        momentum = descent_state.momentum

        population_pulse, momentum_pulse = self.apply_momentum(population.pulse, momentum.pulse, descent_info.eta)
        population = tree_at(lambda x: x.pulse, population, population_pulse)
        momentum = tree_at(lambda x: x.pulse, momentum, momentum_pulse)

        if measurement_info.doubleblind==True:
            population_gate, momentum_gate = self.apply_momentum(population.gate, momentum.gate, descent_info.eta)
            population = tree_at(lambda x: x.gate, population, population_gate)
            momentum = tree_at(lambda x: x.gate, momentum, momentum_gate)

        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x.momentum, descent_state, momentum)

        descent_state, trace_error = do_step(descent_state, measurement_info, descent_info)
        return descent_state, trace_error
    



    def momentum(self, population_size, eta):
        """ 
        Needs to be called if momentum is meant to be used in the reconstruction. 

        Args:
            population_size: int, is needed for some initialization 
            eta (float): parameter that controls the momentum strength

        Returns:
            the class instance
        """
        if self.momentum_is_being_used==True:
            return self
        else:
            shape=(population_size, jnp.size(self.frequency))
            init_arr = jnp.zeros(shape, dtype=jnp.complex64)
            
            self.descent_info = self.descent_info.expand(eta=eta)
            self.descent_state = self.descent_state.expand(momentum = MyNamespace(pulse = MyNamespace(update_for_velocity_map=init_arr, velocity_map=init_arr), 
                                                                                  gate = MyNamespace(update_for_velocity_map=init_arr, velocity_map=init_arr)))
            
            if self._name=="COPRA" or self._name=="PtychographicIterativeEngine":
                self._local_step = self.local_step
                self._global_step = self.global_step
                self.local_step = Partial(self.do_step_and_apply_momentum, do_step = self._local_step)
                self.global_step = Partial(self.do_step_and_apply_momentum, do_step = self._global_step)
                
            else:
                self._step = self.step
                self.step = Partial(self.do_step_and_apply_momentum, do_step=self._step)
            
            self.momentum_is_being_used = True
            return self
    


    def apply_momentum(self, signal, momentum, eta):
        """ 
        Applies momentum to a signal. 

        Args:
            signal: jnp.array,
            momentum: Pytree, contains the velocity map and its update
            eta (float): the strength of the momentum

        Returns:
            tuple[jnp.array, Pytree], the updated signal and momentum state
        """
        update_for_velocity_map, velocity_map = momentum.update_for_velocity_map, momentum.velocity_map

        velocity_map = eta*velocity_map + (signal - update_for_velocity_map)
        signal = signal + eta*velocity_map

        momentum = MyNamespace(update_for_velocity_map=signal, velocity_map=velocity_map)
        return signal, momentum


    











class GeneralOptimizationBASE(AlgorithmsBASE):
    """
    The Base-Class for all general solvers. Inherits from AlgorithmsBASE.

    Attributes:
        fd_grad (bool, int): the loss function can be calcualted with the residuals of the finite-differences of the trace
        amplitude_or_intensity (str, int, float): the loss function can be calculated with the residuals of fields to arbitrary powers
        error_metric (Callable): an arbitrary loss-function, needs to expect (trace, measured_trace) as input
    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "GeneralOptimizationBASE"

        self.fd_grad = False
        self.amplitude_or_intensity = "intensity"
        self.error_metric = self.trace_error


    def create_initial_population(self, population_size, amp_type="gaussian", phase_type="polynomial", no_funcs_amp=5, no_funcs_phase=6):
        """ 
        Creates an initial guess either explicit or parametrized. 

        Args:
            population_size: int, the number of individuals
            amp_type (str): the representation of the spectral amplitude, can be one of gaussian, lorentzian, bsplines or discrete
            phase_type (str): the representation of the spectral phase, can be one of polynomial, sinusoidal, sigmoidal, bsplines or discrete
            no_funcs_amp: int, the number of basis functions for the spectral amplitude (if parametrized)
            no_funcs_phase: int, the number of basis functions for the spectral phase (if parametrized)

        Returns:
            Pytree, the initial guess population
        
        """

        self.key, subkey = jax.random.split(self.key, 2)

        population=MyNamespace(pulse=MyNamespace(amp=None, phase=None), 
                               gate=MyNamespace(amp=None, phase=None))
    
        if phase_type[:-2]=="bsplines":
            k, phase_type = int(phase_type[-1]), phase_type[:-2]
            f, M = get_prefactor(k), get_M(k)

            # spline order and number of f-points restrict valid number of control-points
            N = jnp.size(self.measurement_info.frequency)
            n = no_funcs_phase
            nn = jnp.divide(N, jnp.linspace(1, jnp.ceil(N/n), int(jnp.ceil(N/n))))
            no_funcs_phase = int(nn[jnp.round(nn%1, 5)==0][-1]) + (k-1)

            Nx = N/(no_funcs_phase-k+1) + 1 # how many points per spline?
            self.make_bsplines_phase = Partial(make_bsplines, k=k, M=M, f=f, Nx=int(Nx))

        if amp_type[:-2]=="bsplines":
            k, amp_type = int(amp_type[-1]), amp_type[:-2]
            f, M = get_prefactor(k), get_M(k)
            
            N = jnp.size(self.measurement_info.frequency)
            n = no_funcs_amp
            nn = jnp.divide(N, jnp.linspace(1, jnp.ceil(N/n), int(jnp.ceil(N/n))))
            no_funcs_amp = int(nn[jnp.round(nn%1, 5)==0][-1]) + (k-1)

            Nx = N/(no_funcs_amp-k+1) + 1
            self.make_bsplines_amp = Partial(make_bsplines, k=k, M=M, f=f, Nx=int(Nx))


        subkey, population_pulse = create_population_general(subkey, amp_type, phase_type, population.pulse, population_size, no_funcs_amp, no_funcs_phase, 
                                                             self.descent_info.measured_spectrum_is_provided.pulse, self.measurement_info)
        population = tree_at(lambda x: x.pulse, population, population_pulse)
        
        if self.doubleblind==True:
            subkey, population_gate = create_population_general(subkey, amp_type, phase_type, population.gate, population_size, no_funcs_amp, no_funcs_phase, 
                                                                self.descent_info.measured_spectrum_is_provided.gate, self.measurement_info)
            population = tree_at(lambda x: x.gate, population, population_gate, is_leaf=lambda x: x is None)
            

        classical_guess_types=["random", "random_phase", "constant", "constant_phase"]
        if any([guess==phase_type for guess in classical_guess_types])==True:
            phase_type = "continuous"
            
        if any([guess==amp_type for guess in classical_guess_types])==True:
            amp_type = "continuous"
        
        self.descent_info = self.descent_info.expand(population_size=population_size, phase_type=phase_type, amp_type=amp_type)
        return population
    


    def split_population_in_amp_and_phase(self, population):
        """ Splits a population into an amplitude and phase population. """
        population_amp = MyNamespace(pulse=MyNamespace(amp=population.pulse.amp, phase=None), 
                                     gate=MyNamespace(amp=population.gate.amp, phase=None))
        
        population_phase = MyNamespace(pulse=MyNamespace(amp=None, phase=population.pulse.phase), 
                                       gate=MyNamespace(amp=None, phase=population.gate.phase))

        return population_amp, population_phase
    

    def merge_population_from_amp_and_phase(self, population_amp, population_phase):
        """ Undoes split_population_in_amp_and_phase() """
        population = MyNamespace(pulse=MyNamespace(amp=population_amp.pulse.amp, phase=population_phase.pulse.phase), 
                                 gate=MyNamespace(amp=population_amp.gate.amp, phase=population_phase.gate.phase))
        return population




    def polynomial_term(self, coefficient, order, x0, x):
        return coefficient*(2*jnp.pi*(x-x0))**order
    

    def polynomial_phase(self, coefficients, central_f, measurement_info):
        phase = jax.vmap(self.polynomial_term, in_axes=(0, 0, None, None))(coefficients, jnp.arange(jnp.size(coefficients))+1, central_f, measurement_info.frequency)
        phase = jnp.sum(phase, axis=0)
        return phase
    

    def sinusoidal_term(self, a, b, c, x):
        phase = a*jnp.sin(2*jnp.pi*b*x+c)
        return phase
    
    def sinusoidal_phase(self, coefficients, central_f, measurement_info):
        a, b, c = coefficients.a, coefficients.b, coefficients.c
        phase_arr = jax.vmap(self.sinusoidal_term, in_axes=(0, 0, 0, None))(a, b, c, measurement_info.frequency)
        phase = jnp.sum(phase_arr, axis=0)
        return phase
    

    def discrete_phase(self, coefficients, central_f, measurement_info):
        return coefficients
    

    def tanh_term(self, c, k, x):
        return 0.5*(1+jnp.tanh((x-c)/k))
    

    def tanh_phase(self, coefficients, central_f, measurement_info):
        a, c, k = coefficients.a, coefficients.c, coefficients.k
        phase_arr=jax.vmap(self.tanh_term, in_axes=(0, 0, None))(c, k, measurement_info.frequency)
        phase = jnp.sum(a[:, jnp.newaxis]*phase_arr, axis=0)
        return phase
    

    def bspline_phase(self, coefficients, central_f, measurement_info):
        phase = self.make_bsplines_phase(coefficients.c)
        return phase



    def gaussian_term(self, a, b, c, frequency):
        b = b/2.355 # go from fwhm to sigma
        return a*jnp.exp(-(frequency-c)**2/(2*b**2+1e-9))
    

    def gaussian_amplitude(self, coefficients, measurement_info):
        a, b, c = coefficients.a, coefficients.b, coefficients.c
        amp_f = jax.vmap(self.gaussian_term, in_axes=(0, 0, 0, None))(a, b, c, measurement_info.frequency)
        amp_f = jnp.sum(amp_f, axis=0)
        return amp_f



    
    def lorentzian_term(self, a, b, c, frequency):
        b = b/2 # go from fwhm to sigma
        return a/((frequency-c)**2+b**2)
    

    def lorentzian_amplitude(self, coefficients, measurement_info):
        a, b, c = coefficients.a, coefficients.b, coefficients.c
        amp_f = jax.vmap(self.lorentzian_term, in_axes=(0, 0, 0, None))(a, b, c, measurement_info.frequency)
        amp_f = jnp.sum(amp_f, axis=0)
        return amp_f
    

    

    def discrete_amplitude(self, coefficients, measurement_info):
        amp_f = coefficients
        return amp_f
    

    def bspline_amplitude(self, coefficients, measurement_info):
        amp_f = self.make_bsplines_amp(coefficients.c)
        return amp_f
    



    def get_phase(self, coefficients, central_f, measurement_info, descent_info):
        """ Evaluates the spectral phase onto the frequency axis. """
        spectral_phase_func_dict={"polynomial": self.polynomial_phase,
                                  "sinusoidal": self.sinusoidal_phase,
                                  "sigmoidal": self.tanh_phase,
                                  "bsplines": self.bspline_phase,
                                  "continuous": self.discrete_phase}
        
        spectral_phase_func = spectral_phase_func_dict[descent_info.phase_type]
        return spectral_phase_func(coefficients, central_f, measurement_info)
    


    def get_amplitude(self, coefficients, measurement_info, descent_info):
        """ Evaluates the spectral amplitude onto the frequency axis. """
        amp_func_dict={"gaussian": self.gaussian_amplitude,
                       "lorentzian": self.lorentzian_amplitude,
                       "bsplines": self.bspline_amplitude,
                       "continuous": self.discrete_amplitude}
            
        amp_func = amp_func_dict[descent_info.amp_type]
        amp_f = amp_func(coefficients, measurement_info)

        frequency = measurement_info.frequency
        idx_arr = jnp.arange(jnp.size(frequency))
        idx = jnp.sum(idx_arr*jnp.abs(amp_f))/jnp.sum(jnp.abs(amp_f))
        central_f = frequency[idx.astype(int)]
        return amp_f, central_f
    



    def make_pulse_f_from_individual(self, individual, measurement_info, descent_info, pulse_or_gate="pulse"):
        """ Evaluates a parametrized individual onto the frequency axis. """
        individual = getattr(individual, pulse_or_gate)

        if getattr(descent_info.measured_spectrum_is_provided, pulse_or_gate)==True:
            amp = getattr(measurement_info.spectral_amplitude, pulse_or_gate)
            central_f = getattr(measurement_info.central_f, pulse_or_gate)

        else:
            amp, central_f = self.get_amplitude(individual.amp, measurement_info, descent_info)
    
        phase = self.get_phase(individual.phase, central_f, measurement_info, descent_info)
        signal_f = amp*jnp.exp(1j*2*jnp.pi*phase)
        return signal_f
    


    def get_pulses_f_from_population(self, population, measurement_info, descent_info):
        """ Evaluates a parametrized population onto the frequency axis. """
        make_pulse = Partial(self.make_pulse_f_from_individual, measurement_info=measurement_info, descent_info=descent_info, pulse_or_gate="pulse")
        pulse_f_arr = jax.vmap(make_pulse)(population)

        if measurement_info.doubleblind==True:
            make_gate = Partial(self.make_pulse_f_from_individual, measurement_info=measurement_info, descent_info=descent_info, pulse_or_gate="gate")
            gate_arr = jax.vmap(make_gate)(population)
        else:
            gate_arr = pulse_f_arr

        return pulse_f_arr, gate_arr
    
    

    def make_pulse_t_from_individual(self, individual, measurement_info, descent_info, pulse_or_gate="pulse"):
        """ Evaluates a parametrized individual onto the time axis. """
        signal_f = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, pulse_or_gate)
        signal = self.ifft(signal_f, measurement_info.sk, measurement_info.rn)
        return signal
    
    

    def get_pulses_t_from_population(self, population, measurement_info, descent_info):
        """ Evaluates a parametrized population onto the time axis. """
        make_pulse = Partial(self.make_pulse_t_from_individual, measurement_info=measurement_info, descent_info=descent_info, pulse_or_gate="pulse")
        pulse_t_arr = jax.vmap(make_pulse)(population)

        if measurement_info.doubleblind==True:
            make_gate = Partial(self.make_pulse_t_from_individual, measurement_info=measurement_info, descent_info=descent_info, pulse_or_gate="gate")
            gate_arr = jax.vmap(make_gate)(population)
        else:
            gate_arr = pulse_t_arr

        return pulse_t_arr, gate_arr
    
    



    def construct_trace(self, individual, measurement_info, descent_info):
        """ Generates a trace for a given individual. Calls the method specific function for calculating the nonlinear signal fields. """
        x_arr, frequency = measurement_info.x_arr, measurement_info.frequency
        
        signal_t = self.calculate_signal_t(individual, measurement_info.transform_arr, measurement_info)
        trace = calculate_trace(signal_t.signal_f)
        return x_arr, frequency, trace
    

    def trace_error(self, trace, measured_trace):
        """ The mean least squares error. """
        return jnp.mean(jnp.abs(trace - measured_trace)**2)


    def calculate_error_individual(self, individual, measurement_info, descent_info):
        """ Calculates the error of an individual based on its trace. 
        Allows modification of the error-function via error_metric() and loss_function_modification(). """
        measured_trace = measurement_info.measured_trace
        amplitude_or_intensity, fd_grad = descent_info.amplitude_or_intensity, descent_info.fd_grad
        error_metric = descent_info.error_metric

        x_arr, y_arr, trace = self.construct_trace(individual, measurement_info, descent_info)
        trace, measured_trace = loss_function_modifications(trace, measured_trace, x_arr, y_arr, amplitude_or_intensity, fd_grad)

        if fd_grad!=False:
            trace_error = jax.vmap(error_metric)(trace, measured_trace)
            trace_error = jnp.sum(trace_error, axis=0)
        else:
            trace_error = error_metric(trace, measured_trace)
        
        return trace_error
            
    


    def calculate_error_population(self, population, measurement_info, descent_info):
        """ Calls jax.vmap over calculate_error_individual() for an entire population. """
        pulse_arr, gate_arr = self.get_pulses_from_population(population, measurement_info, descent_info)
        error_arr = jax.vmap(self.calculate_error_individual, in_axes=(0, None, None))(MyNamespace(pulse=pulse_arr, gate=gate_arr), measurement_info, descent_info)
        return error_arr
    




    def initialize_general_optimizer(self, population):
        """ A common initialization step for all general solvers. """
        if self.descent_info.measured_spectrum_is_provided.pulse==True:
            spectrum = self.measurement_info.spectral_amplitude.pulse
            idx = get_com(spectrum, jnp.arange(jnp.size(spectrum)))
            self.measurement_info = tree_at(lambda x: x.central_f.pulse, self.measurement_info, self.frequency[int(idx)], is_leaf=lambda x: x is None)

        if self.descent_info.measured_spectrum_is_provided.gate==True:
            spectrum = self.measurement_info.spectral_amplitude.gate
            idx = get_com(spectrum, jnp.arange(jnp.size(spectrum)))
            self.measurement_info = tree_at(lambda x: x.central_f.gate, self.measurement_info, self.frequency[int(idx)], is_leaf=lambda x: x is None)


        self.descent_info = self.descent_info.expand(fd_grad = self.fd_grad,
                                                     amplitude_or_intensity = self.amplitude_or_intensity,
                                                     error_metric = self.error_metric)

        self.descent_state = self.descent_state.expand(population = population)




    def get_idx_best_individual(self, descent_state):
        """ Calculates the error for a population. Returns the index of the lowest individual. """
        population = descent_state.population
        pulse_arr, gate_arr = self.get_pulses_from_population(population, self.measurement_info, self.descent_info)
        error_arr = jax.vmap(self.calculate_error_individual, in_axes=(0, None, None))(MyNamespace(pulse=pulse_arr, gate=gate_arr), 
                                                                                       self.measurement_info, self.descent_info)
        idx = jnp.argmin(error_arr)
        return idx



    def post_process_get_pulse_and_gate(self, descent_state, measurement_info, descent_info, idx=None):
        """ Post-processing to evaluate parametrized individuals. """
        if idx==None:
            idx = self.get_idx_best_individual(descent_state)
        individual = self.get_individual_from_idx(idx, descent_state.population)

        pulse_t = self.make_pulse_t_from_individual(individual, measurement_info, descent_info, "pulse")
        pulse_f = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, "pulse")

        if measurement_info.doubleblind==True:
            gate_t = self.make_pulse_t_from_individual(individual, measurement_info, descent_info, "gate")
            gate_f = self.make_pulse_f_from_individual(individual, measurement_info, descent_info, "gate")
        else:
            gate_t, gate_f = pulse_t, pulse_f

        return pulse_t, gate_t, pulse_f, gate_f
    
