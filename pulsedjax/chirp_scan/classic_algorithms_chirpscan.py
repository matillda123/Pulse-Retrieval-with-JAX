import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from equinox import tree_at

from pulsedjax.core.base_classes_methods import RetrievePulsesCHIRPSCAN
from pulsedjax.core.base_classes_algorithms import ClassicAlgorithmsBASE
from pulsedjax.core.base_classic_algorithms import GeneralizedProjectionBASE, PtychographicIterativeEngineBASE, COPRABASE


from pulsedjax.utilities import scan_helper, calculate_mu, calculate_trace, calculate_trace_error, integrate_signal_1D, do_interpolation_1d, get_sk_rn
from pulsedjax.core.construct_s_prime import calculate_S_prime

from pulsedjax.core.gradients.chirpscan_z_error_gradients import calculate_Z_gradient
from pulsedjax.core.hessians.chirpscan_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error
from pulsedjax.core.hessians.pie_pseudo_hessian import PIE_get_pseudo_newton_direction

from pulsedjax.core.phase_matrix_funcs import phase_func_dict, calc_GDD



class MIIPS(ClassicAlgorithmsBASE, RetrievePulsesCHIRPSCAN):
    """
    The (improoved)-MIIPS Algorithm. Doesnt seem to work though.

    [1] A. Comin et al., Opt. Express 24, 2505-2512 (2016)

    Attributes:
        integration_method (str): the integration method to use, can be cumsum or euler_maclaurin_k
        integration_order (None, int): the order to which euler_maclaurin is performed. Is infered from integration_method.
        global_gamma (float): the step size
    
    """
    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters, **kwargs)

        self.name = "MIIPS"

        self.integration_method = "euler_maclaurin_3"
        self.integration_order = None

        self.global_gamma = 0.5


    def calc_approximate_phase_individual(self, trace, measurement_info, descent_info):
        theta, omega, parameters, phase_type = measurement_info.theta, measurement_info.omega, measurement_info.chirp_paramaters, measurement_info.phase_type
        factor = measurement_info.factor
        integration_method, integration_order = descent_info.integration_method, descent_info.integration_order

        idx = jnp.argmax(trace, axis=0)
        delta_m = jnp.take(theta, idx)
        gdd = calc_GDD(omega, delta_m, parameters, phase_func_dict[phase_type])
        gd = integrate_signal_1D(gdd, omega, integration_method, integration_order)
        phase = integrate_signal_1D(gd, omega, integration_method, integration_order)
        phase = do_interpolation_1d(omega, omega/factor, phase)
        return phase
    

    def update_individual(self, individual, phase_prime, measurement_info, descent_info):
        pulse = individual.population.pulse
        phase = individual.phases
        phase = phase + descent_info.gamma*(phase_prime - phase)
        pulse = jnp.abs(pulse)*jnp.exp(1j*phase)
        return tree_at(lambda x: x.pulse, individual.population, pulse)



    def step(self, descent_state, measurement_info, descent_info):
        phase_prime = jax.vmap(self.calc_approximate_phase_individual, in_axes=(0,None,None))(descent_state.traces, measurement_info, descent_info)
        population = jax.vmap(self.update_individual, in_axes=(0,0,None,None))(descent_state, phase_prime, measurement_info, descent_info)

        descent_state = tree_at(lambda x: x.population, descent_state, population)
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(signal_t.signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measurement_info.measured_trace)

        descent_state = tree_at(lambda x: x.phases, descent_state, phase_prime)
        descent_state = tree_at(lambda x: x.traces, descent_state, trace)
        return descent_state, trace_error.reshape(-1,1)
    
    

    def initialize_run(self, population):
        assert self.descent_info.measured_spectrum_is_provided.pulse==True, "A spectrum is needed."


        if self.measurement_info.nonlinear_method=="shg":
            factor=2
        elif self.measurement_info.nonlinear_method=="thg":
            factor=3
        else:
            factor=1

        omega = 2*jnp.pi*self.measurement_info.frequency
        self.measurement_info = self.measurement_info.expand(phase_type = self.phase_type,
                                                             chirp_paramaters = self.parameters,
                                                             omega = omega,
                                                             factor = factor)
        measurement_info = self.measurement_info


        if self.integration_method[:-2]=="euler_maclaurin":
            self.integration_order = int(self.integration_method[-1])
            self.integration_method = "euler_maclaurin"
        self.descent_info = self.descent_info.expand(integration_method = self.integration_method, 
                                                     integration_order = self.integration_order,
                                                     gamma = self.global_gamma)
        descent_info = self.descent_info


        self.descent_state = self.descent_state.expand(population = population,
                                                       phases = jnp.zeros(jnp.shape(population.pulse), dtype=jnp.complex64),
                                                       traces = jnp.broadcast_to(measurement_info.measured_trace, 
                                                                                 (jnp.shape(population.pulse)[0],) + jnp.shape(measurement_info.measured_trace)))
        descent_state = self.descent_state

        do_step = Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step = Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step
    








class Basic(ClassicAlgorithmsBASE, RetrievePulsesCHIRPSCAN):
    """
    The Basic Reconstruction Algorithm.
    
    [1] M. Miranda et al., J. Opt. Soc. Am. B 34, 190-197 (2017) 

    """

    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters, **kwargs)

        self.name = "Basic"


    def update_pulse(self, signal_t_new, gate, phase_matrix, nonlinear_method, sk, rn):
        """ Creates an updated guess for the pulse. """
        signal_t_new = signal_t_new*jnp.conjugate(gate)

        if nonlinear_method=="shg":
            n=3
        else: 
            n=5
        signal_t_new = jnp.abs(signal_t_new)**(1/n)*jnp.exp(1j*jnp.angle(signal_t_new))

        signal_f_new = self.fft(signal_t_new, sk, rn)
        signal_f_new = signal_f_new*jnp.exp(-1j*phase_matrix)

        pulse_f = jnp.mean(signal_f_new, axis=0)
        return pulse_f
    
    

    def step(self, descent_state, measurement_info, descent_info):
        """ 
        Performs one iteration of the Basic Algorithm. 

        Args:
            descent_state: Pytree,
            measurement_info: Pytree,
            descent_info: Pytree,
        
        Returns:
            tuple[Pytree, jnp.array], the updated descent state and the current errors

        """
        nonlinear_method, sk, rn = measurement_info.nonlinear_method, measurement_info.sk, measurement_info.rn
        phase_matrix = measurement_info.phase_matrix
        measured_trace = measurement_info.measured_trace

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(signal_t.signal_f)

        mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,0,None,0,None,None,None))(signal_t.signal_t,signal_t.signal_f, measured_trace, mu, measurement_info, descent_info, "_global")
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)
        
        pulse = jax.vmap(self.update_pulse, in_axes=(0,0,None,None,None,None))(signal_t_new, signal_t.gate_disp, phase_matrix, nonlinear_method, sk, rn)

        descent_state = tree_at(lambda x: x.population.pulse, descent_state, pulse)
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

        self.descent_state = self.descent_state.expand(population = population)
       
        measurement_info = self.measurement_info
        descent_info = self.descent_info
        descent_state = self.descent_state

        do_step = Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step = Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step
    

    






class GeneralizedProjection(GeneralizedProjectionBASE, RetrievePulsesCHIRPSCAN):
    __doc__ = GeneralizedProjectionBASE.__doc__

    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters, **kwargs)


    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, population, phase_matrix, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t.pulse_t_disp, signal_t.signal_t, signal_t_new, phase_matrix, measurement_info)
        return grad 


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, phase_matrix, descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """

        descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, signal_t.pulse_t_disp, signal_t.signal_t, signal_t_new, phase_matrix, 
                                                                        measurement_info, descent_state.newton, descent_info.newton, full_or_diagonal)
        return descent_direction, newton_state
    

    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Updates an individual based on a descent direction and a step size. """
        pulse = individual.pulse + gamma*descent_direction
        individual = tree_at(lambda x: x.pulse, individual, pulse)
        return individual








class PtychographicIterativeEngine(PtychographicIterativeEngineBASE, RetrievePulsesCHIRPSCAN):
    __doc__ = PtychographicIterativeEngineBASE.__doc__

    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters, **kwargs)


    def reverse_transform_grad(self, signal, phase_matrix, measurement_info):
        """ For Chirp-Scan the effects of the phase matrix have to be undone to obtain the actual PIE-direction. """
        sk, rn = measurement_info.sk, measurement_info.rn
        signal_f = self.fft(signal, sk, rn)
        signal_f = signal_f*jnp.exp(-1j*phase_matrix)
        signal = self.ifft(signal_f, sk, rn)
        return signal




    # def reverse_transform_full_hessian(self, hessian_all_m, phase_matrix, measurement_info):
    #     # time, frequency = measurement_info.time, measurement_info.frequency
    
    #     # frequency = frequency - (frequency[-1] + frequency[0])/2
    #     # N = jnp.size(frequency)
    #     # hessian_all_m = jnp.pad(hessian_all_m, ((0,0), (0,N), (0,N))) 

    #     # frequency = jnp.linspace(jnp.min(frequency), jnp.max(frequency), 2*N)
    #     # time = jnp.fft.fftshift(jnp.fft.fftfreq(2*N, jnp.mean(jnp.diff(frequency))))
    #     # sk, rn = get_sk_rn(time, frequency)

    #     sk, rn = measurement_info.sk, measurement_info.rn

    #     # convert hessian to (m, n, n) -> frequency domain 
    #     hessian_all_m = self.fft(hessian_all_m, sk, rn, axis=-1)
    #     hessian_all_m = self.fft(hessian_all_m, sk, rn, axis=-2) 

    #     #phi_mn = -1*phase_matrix
    #     phi_mn = phase_matrix
    #     phi = phi_mn[:,:,jnp.newaxis] - phi_mn[:,jnp.newaxis,:]
    #     exp_arr = jnp.exp(1j*phi)
    #     hessian_all_m = hessian_all_m * exp_arr

    #     # convert hessian to (N, m, k, k) -> time domain 
    #     hessian_all_m = self.ifft(hessian_all_m, sk, rn, axis=-1)
    #     hessian_all_m = self.ifft(hessian_all_m, sk, rn, axis=-2) 
    #     return hessian_all_m#[:, :N, :N]
    

    # def reverse_transform_diagonal_hessian(self, hessian_all_m, phase_matrix, measurement_info):
    #     # # i think a backtransform is not needed since the transform matrix phi is zero for these entries
    #     return hessian_all_m




    def calculate_PIE_descent_direction_m(self, signal_t, signal_t_new, phase_matrix_m, measured_trace, population, pie_method, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the PIE direction for a given shift. """
        alpha = descent_info.alpha

        probe = signal_t.gate_disp
        difference_signal_t = signal_t_new - signal_t.signal_t

        grad = -1*jnp.conjugate(probe)*difference_signal_t
        U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(probe, alpha, pie_method)

        grad_U = jax.vmap(self.reverse_transform_grad, in_axes=(0,0,None))(grad*U, phase_matrix_m, measurement_info)
        return grad_U



    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Updates an individual based on a descent direction and a step size. """
        sk, rn = measurement_info.sk, measurement_info.rn
        
        pulse_t=self.ifft(individual.pulse, sk, rn)
        pulse_t=pulse_t + gamma*descent_direction
        pulse = self.fft(pulse_t, sk, rn)

        individual = tree_at(lambda x: x.pulse, individual, pulse)
        return individual
    

    def calculate_PIE_newton_direction(self, grad, signal_t, phase_matrix, measured_trace, population, local_or_global_state, measurement_info, 
                                                descent_info, pulse_or_gate, local_or_global):
        """ Calculates the PIE newton direction for a population. """
        
        assert getattr(descent_info.newton, local_or_global)!="full", """Dont use full hessian. Its not implemented. 
        It requires the derivative with respect to the unmodified pulse. Which seems hard for the hessian."""
        
        newton_direction_prev = local_or_global_state.newton.pulse.newton_direction_prev

        # reverse_transform_hessian = {"diagonal": self.reverse_transform_diagonal_hessian,
        #                              "full": self.reverse_transform_full_hessian}
        # reverse_transform = Partial(reverse_transform_hessian[getattr(descent_info.newton, local_or_global)], measurement_info=measurement_info)
        reverse_transform = None

        descent_direction, newton_state = PIE_get_pseudo_newton_direction(grad, signal_t.gate_disp, signal_t.signal_f, phase_matrix, measured_trace, reverse_transform, 
                                                                     newton_direction_prev, measurement_info, descent_info, "gate", local_or_global)
        return descent_direction, newton_state















class COPRA(COPRABASE, RetrievePulsesCHIRPSCAN):
    __doc__ = COPRABASE.__doc__

    
    def __init__(self, theta, frequency, measured_trace, nonlinear_method, phase_type=None, chirp_parameters=None, **kwargs):
        super().__init__(theta, frequency, measured_trace, nonlinear_method, phase_type=phase_type, chirp_parameters=chirp_parameters, **kwargs)


    def update_individual(self, individual, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        """ Updates an individual based on a desent direction and a step size. """
        pulse = individual.pulse + gamma*descent_direction
        individual = tree_at(lambda x: x.pulse, individual, pulse)
        return individual


    def get_Z_gradient_individual(self, signal_t, signal_t_new, population, phase_matrix, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t.pulse_t_disp, signal_t.signal_t, signal_t_new, phase_matrix, measurement_info)
        return grad


    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, phase_matrix, population, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """

        newton_state = local_or_global_state.newton
        descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, signal_t.pulse_t_disp, signal_t.signal_t, signal_t_new, phase_matrix, measurement_info, 
                                                                         newton_state, descent_info.newton, full_or_diagonal)
        return descent_direction, newton_state
