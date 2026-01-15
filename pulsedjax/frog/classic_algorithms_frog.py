import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from equinox import tree_at

from pulsedjax.core.base_classes_methods import RetrievePulsesFROG 
from pulsedjax.core.base_classes_algorithms import ClassicAlgorithmsBASE
from pulsedjax.core.base_classic_algorithms import LSGPABASE, CPCGPABASE, GeneralizedProjectionBASE, PtychographicIterativeEngineBASE, COPRABASE, initialize_S_prime_params

from pulsedjax.utilities import scan_helper, get_com, get_sk_rn, calculate_gate, calculate_trace, calculate_mu, calculate_trace_error, do_interpolation_1d
from pulsedjax.core.construct_s_prime import calculate_S_prime

from pulsedjax.core.gradients.frog_z_error_gradients import calculate_Z_gradient
from pulsedjax.core.hessians.frog_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error
from pulsedjax.core.hessians.pie_pseudo_hessian import PIE_get_pseudo_newton_direction






class Vanilla(ClassicAlgorithmsBASE, RetrievePulsesFROG):
    """
    The Vanilla-FROG Algorithm as described by R. Trebino.

    R. Trebino, "Frequency-Resolved Optical Gating: The Measurement of Ultrashort Laser Pulses", 10.1007/978-1-4615-1181-6 (2000)

    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        if cross_correlation=="doubleblind":
            print("Vanilla/LSGPA dont work for doubleblind.")
            # which is weird because lsgpa was invented for attosecond-streaking -> is doubleblind by definition. 

        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)
        self._name = "Vanilla"        

        # for some reason vanilla only works with central_f=0. No idea why. Is undone when using LSGPA.
        idx = get_com(jnp.mean(self.measured_trace, axis=0), jnp.arange(jnp.size(self.frequency)))
        self.f0 = frequency[int(idx)]
        self.frequency = self.frequency - self.f0

        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        self.measurement_info = tree_at(lambda x: x.sk, self.measurement_info, self.sk)
        self.measurement_info = tree_at(lambda x: x.rn, self.measurement_info, self.rn)
        self.measurement_info = tree_at(lambda x: x.frequency, self.measurement_info, self.frequency)




    def update_pulse(self, pulse, signal_t_new, gate_shifted, measurement_info, descent_info):
        """ Generates an new (maybe improoved) guess for the pulse. """
        pulse_t = jnp.sum(signal_t_new, axis=1)
        return pulse_t
    
    def update_gate(self, gate, signal_t_new, pulse_t_shifted, measurement_info, descent_info):
        """ Generates an new (maybe improoved) guess for the gate. """
        gate = jnp.sum(signal_t_new, axis=2)
        gate = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(measurement_info.time, measurement_info.tau_arr, gate)
        return gate

    
        
    def step(self, descent_state, measurement_info, descent_info):
        """ 
        Performs one iteration of the Vanilla Algorithm. 

        Args:
            descent_state: Pytree,
            measurement_info: Pytree,
            descent_info: Pytree,
        
        Returns:
            tuple[Pytree, jnp.array], the updated descent state and the current errors

        """
        measured_trace = measurement_info.measured_trace

        population = descent_state.population
        
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(signal_t.signal_f)

        mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,0,None,0,None,None,None))(signal_t.signal_t,signal_t.signal_f, measured_trace, mu, measurement_info, descent_info, "_global")
        
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)
        population_pulse = self.update_pulse(population.pulse, signal_t_new, signal_t.gate_shifted, measurement_info, descent_info)
        population_pulse = population_pulse/jnp.linalg.norm(population_pulse,axis=-1)[:,jnp.newaxis]
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population_pulse)


        if measurement_info.doubleblind==True:
            population_gate = self.update_gate(population.gate, signal_t_new, signal_t.pulse_t_shifted, measurement_info, descent_info)
            population_gate = population_gate/jnp.linalg.norm(population_gate,axis=-1)[:,jnp.newaxis]
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population_gate)

        return descent_state, trace_error.reshape(-1,1)



    def initialize_run(self, population):
        """
        Prepares all provided data and parameters for the reconstruction. 
        Here the final shape/structure of descent_state, measurement_info and descent_info are determined. 

        Args:
            population: Pytree, the initial guess as created by self.create_initial_population()
        
        Returns:
            tuple[Pytree, Callable], the initial descent state and the step-function of the algorithm.

        """
        measurement_info = self.measurement_info

        s_prime_params = initialize_S_prime_params(self)
        self.descent_info = self.descent_info.expand(s_prime_params=s_prime_params)
        descent_info = self.descent_info

        self.descent_state = self.descent_state.expand(population=population)
        descent_state = self.descent_state

        do_step = Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step = Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step









class LSGPA(LSGPABASE, RetrievePulsesFROG):
    __doc__ = LSGPABASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)
        




class CPCGPA(CPCGPABASE, RetrievePulsesFROG):
    __doc__ = CPCGPABASE.__doc__

    def __init__(self, delay, frequency, trace, nonlinear_method, cross_correlation=False, constraints=False, svd=False, antialias=False, **kwargs):
        super().__init__(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation, constraints=constraints, svd=svd, antialias=antialias, **kwargs)
        

    def calculate_gate(self, gate_pulse, measurement_info):
        return calculate_gate(gate_pulse, measurement_info.nonlinear_method)
        







class GeneralizedProjection(GeneralizedProjectionBASE, RetrievePulsesFROG):
    __doc__ = GeneralizedProjectionBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, interferometric=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, interferometric=interferometric, **kwargs)


    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, population.pulse, signal_t.pulse_t_shifted, signal_t.gate_shifted, tau_arr, 
                                    measurement_info, pulse_or_gate)
        return grad


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """
        
        descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, descent_state.population.pulse, signal_t.pulse_t_shifted, signal_t.gate_shifted, 
                                                                         signal_t.signal_t, signal_t_new, tau_arr, measurement_info, 
                                                                         descent_state.newton, descent_info.newton, full_or_diagonal, pulse_or_gate)
        return descent_direction, newton_state


    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Updates an individual based on a descent_direction and step size. """
        sk, rn = measurement_info.sk, measurement_info.rn

        pulse_f = self.fft(getattr(individual, pulse_or_gate), sk, rn)
        pulse_f = pulse_f + gamma*descent_direction
        pulse = self.ifft(pulse_f, sk, rn)

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, pulse)
        return individual













class PtychographicIterativeEngine(PtychographicIterativeEngineBASE, RetrievePulsesFROG):
    __doc__ = PtychographicIterativeEngineBASE.__doc__

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)
        assert self.interferometric==False, "Dont use interferometric with PIE. its not meant or made for that"


    def reverse_transform_grad(self, signal, tau_arr, measurement_info):
        """ For reconstruction of the gate-pulse the shift has to be undone. """
        frequency, time = measurement_info.frequency, measurement_info.time

        signal = self.calculate_shifted_signal(signal, frequency, -1*tau_arr, time, in_axes=(0, 0, None, None, None))
        return signal
    
    

    def modify_grad_for_gate_pulse(self, grad_all_m, gate_pulse_shifted, nonlinear_method):
        """ For reconstruction of the gate-pulse the gradient depends on the nonlinear method. """
        if nonlinear_method=="shg":
            pass
        elif nonlinear_method=="thg":
            grad_all_m = grad_all_m*jnp.conjugate(2*gate_pulse_shifted)
        elif nonlinear_method=="pg":
            grad_all_m = grad_all_m*gate_pulse_shifted
        elif nonlinear_method=="sd":
            grad_all_m = jnp.conjugate(grad_all_m*2*gate_pulse_shifted)
        elif nonlinear_method[-2:]=="hg":
            n = int(nonlinear_method[0])
            grad_all_m = grad_all_m*jnp.conjugate((n-1)*gate_pulse_shifted**(n-2))
        else:
            raise NotImplementedError(f"nonlinear_method={nonlinear_method} is not available.")

        return grad_all_m


    def calculate_PIE_descent_direction_m(self, signal_t, signal_t_new, tau, measured_trace, population, pie_method, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the PIE direction for pulse or gate-pulse for a given shift. """
        alpha = descent_info.alpha

        difference_signal_t = signal_t_new - signal_t.signal_t

        if pulse_or_gate=="pulse":
            probe = signal_t.gate_shifted
            grad = -1*jnp.conjugate(probe)*difference_signal_t
            U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(probe, alpha, pie_method)
            grad_U = grad*U
            
        elif pulse_or_gate=="gate":
            probe = jnp.broadcast_to(population.pulse[:,jnp.newaxis,:], jnp.shape(difference_signal_t))
            grad = -1*jnp.conjugate(probe)*difference_signal_t
            U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(probe, alpha, pie_method)

            grad = self.modify_grad_for_gate_pulse(grad, signal_t.gate_pulse_shifted, measurement_info.nonlinear_method)
            grad_U = jax.vmap(self.reverse_transform_grad, in_axes=(0,0,None))(grad*U, tau, measurement_info)

        return grad_U
    


    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Updates an individual based on a descent direction and step size. """
        signal = getattr(individual, pulse_or_gate)
        signal = signal + gamma*descent_direction

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal)
        return individual


    def get_gate_probe_for_hessian(self, pulse_t, gate_pulse_shifted, nonlinear_method):
        """ For the reconstruction of the gate pulse, the probe depends on the nonlinear method for the hessian calculation. """
        if nonlinear_method=="shg":
            probe = pulse_t
        elif nonlinear_method=="thg":
            probe = pulse_t*2*gate_pulse_shifted
        elif nonlinear_method=="pg":
            probe = pulse_t*jnp.conjugate(gate_pulse_shifted)
        elif nonlinear_method=="sd":
            probe = jnp.conjugate(pulse_t)*2*gate_pulse_shifted
        elif nonlinear_method[-2:]=="hg":
            n = int(nonlinear_method[0])
            probe = pulse_t*(n-1)*gate_pulse_shifted**(n-2)
        else:
             raise NotImplementedError(f"nonlinear_method={nonlinear_method} is not available.")

        return probe



    def calculate_PIE_newton_direction(self, grad, signal_t, tau_arr, measured_trace, population, local_or_global_state, measurement_info, descent_info, 
                                       pulse_or_gate, local_or_global):
        """ Calculates the newton direction for a population. """
        
        newton_direction_prev = getattr(local_or_global_state.newton, pulse_or_gate).newton_direction_prev
        
        if pulse_or_gate=="pulse":
            probe = signal_t.gate_shifted

        elif pulse_or_gate=="gate":
            pulse_t = jnp.broadcast_to(population.pulse[:,jnp.newaxis,:], jnp.shape(signal_t.signal_t))
            probe = self.get_gate_probe_for_hessian(pulse_t, signal_t.gate_pulse_shifted, measurement_info.nonlinear_method)

        # if local_or_global=="_local": # it would be nicer to fix this generally. 
        #     measured_trace = measured_trace[jnp.newaxis, :]


        reverse_transform = None

        # signal_f = self.fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        descent_direction, newton_state = PIE_get_pseudo_newton_direction(grad, probe, signal_t.signal_f, tau_arr, measured_trace, reverse_transform, newton_direction_prev, 
                                                                     measurement_info, descent_info, pulse_or_gate, local_or_global)
        return descent_direction, newton_state
    
















class COPRA(COPRABASE, RetrievePulsesFROG):
    __doc__ = COPRABASE.__doc__
    
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, interferometric=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, interferometric=interferometric, **kwargs)



    def update_individual(self, individual, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        """ Updates an individual based on a descent direction and a step size. """
        sk, rn = measurement_info.sk, measurement_info.rn

        signal = getattr(individual, pulse_or_gate)
        signal_f = self.fft(signal, sk, rn)
        signal_f = signal_f + gamma*descent_direction
        signal = self.ifft(signal_f, sk, rn)

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal)
        return individual


    def get_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, population.pulse, signal_t.pulse_t_shifted, 
                                    signal_t.gate_shifted, tau_arr, measurement_info, pulse_or_gate)
        return grad



    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, tau_arr, population, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """


        newton_state = local_or_global_state.newton
        descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, population.pulse, signal_t.pulse_t_shifted, signal_t.gate_shifted, 
                                                                         signal_t.signal_t, signal_t_new, tau_arr, measurement_info, 
                                                                         newton_state, descent_info.newton, full_or_diagonal, pulse_or_gate)
        return descent_direction, newton_state

