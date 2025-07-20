import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from BaseClasses import RetrievePulsesDSCAN, AlgorithmsBASE
from classic_algorithms_base import GeneralizedProjectionBASE, TimeDomainPtychographyBASE, COPRABASE


from utilities import scan_helper, MyNamespace, do_fft, do_ifft, calculate_mu, calculate_S_prime, calculate_trace, calculate_trace_error, calculate_Z_error


from dscan_z_error_gradients import calculate_Z_gradient
from dscan_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error
from pie_pseudo_hessian import PIE_get_pseudo_newton_direction









class Basic(RetrievePulsesDSCAN, AlgorithmsBASE):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)

        self.child_class="Basic"


    def update_pulse(self, signal_t_new, gate, phase_matrix, nonlinear_method, sk, rn):
        signal_t_new=signal_t_new*jnp.conjugate(gate)

        if nonlinear_method=="shg":
            n=3
        else: 
            n=5
        signal_t_new=jnp.abs(signal_t_new)**(1/n)*jnp.exp(1j*jnp.angle(signal_t_new))

        signal_f_new=do_fft(signal_t_new, sk, rn)
        signal_f_new=signal_f_new*jnp.exp(-1j*phase_matrix)

        pulse_f=jnp.mean(signal_f_new, axis=0)
        return pulse_f
    
    

    def step(self, descent_state, measurement_info, descent_info):
        nonlinear_method, sk, rn = measurement_info.nonlinear_method, measurement_info.sk, measurement_info.rn
        phase_matrix = measurement_info.phase_matrix
        measured_trace=measurement_info.measured_trace

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_f=do_fft(signal_t.signal_t, sk, rn)
        trace=calculate_trace(signal_f)


        mu=jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        signal_t_new=jax.vmap(calculate_S_prime, in_axes=(0,None,0,None))(signal_t.signal_t, measured_trace, mu, measurement_info)
        trace_error=jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)
        
        pulse = jax.vmap(self.update_pulse, in_axes=(0,0,None,None,None,None))(signal_t_new, signal_t.gate_disp, phase_matrix, nonlinear_method, sk, rn)

        descent_state.population.pulse=pulse
        return descent_state, trace_error.reshape(-1,1)
    


    def initialize_run(self, population):

        self.descent_state.population = population
       
        measurement_info=self.measurement_info
        descent_info=self.descent_info
        descent_state=self.descent_state

        do_step=Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step=Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)

        return descent_state, do_step
    

    






class GeneralizedProjection(RetrievePulsesDSCAN, GeneralizedProjectionBASE):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)



    # def calc_Z_error_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
    #     descent_direction, signal_t_new = linesearch_info.descent_direction, linesearch_info.signal_t_new
    #     phase_matrix = measurement_info.phase_matrix

    #     pulse = linesearch_info.population.pulse + gamma*descent_direction
        
    #     individual = MyNamespace(pulse=pulse, gate=None)
    #     signal_t = self.calculate_signal_t(individual, phase_matrix, measurement_info)
    #     Z_error_new=calculate_Z_error(signal_t.signal_t, signal_t_new)
    #     return Z_error_new
    


    def calc_Z_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        descent_direction, signal_t_new = linesearch_info.descent_direction, linesearch_info.signal_t_new
        phase_matrix = measurement_info.phase_matrix

        individual = linesearch_info.population

        individual = self.update_individual(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, phase_matrix, measurement_info)
        grad = calculate_Z_gradient(signal_t.pulse_t_disp, signal_t.signal_t, signal_t_new, phase_matrix, measurement_info)
        return jnp.sum(grad, axis=0)



    def calculate_Z_error_gradient(self, signal_t_new, signal_t, population, phase_matrix, measurement_info, pulse_or_gate):
        grad = jax.vmap(calculate_Z_gradient, in_axes=(0,0,0,None,None))(signal_t.pulse_t_disp, signal_t.signal_t, signal_t_new, 
                                                                                   phase_matrix, measurement_info)
        return grad


    def calculate_Z_error_newton_direction(self, grad, signal_t_new, signal_t, phase_matrix, descent_state, measurement_info, descent_info, use_hessian, pulse_or_gate):
        newton_direction = get_pseudo_newton_direction_Z_error(grad, signal_t.pulse_t_disp, signal_t.signal_t, signal_t_new, phase_matrix, 
                                                               descent_state, measurement_info, descent_info, use_hessian)
        return newton_direction



    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        individual.pulse = individual.pulse + gamma*descent_direction
        return individual







class TimeDomainPtychography(RetrievePulsesDSCAN, TimeDomainPtychographyBASE):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, PIE_method="rPIE", **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)

        self.PIE_method=PIE_method




    def reverse_transform(self, signal, phase_matrix, measurement_info):
        sk, rn = measurement_info.sk, measurement_info.rn
        signal_f = do_fft(signal, sk, rn)
        signal_f = signal_f*jnp.exp(-1j*phase_matrix)
        signal = do_ifft(signal_f, sk, rn)
        return signal



    def update_population_local(self, population, signal_t, signal_t_new, phase_matrix, measurement_info, descent_info, pulse_or_gate):
        alpha, beta, PIE_method = descent_info.alpha, descent_info.beta, descent_info.PIE_method
        sk, rn = measurement_info.sk, measurement_info.rn

        difference_signal_t = signal_t_new - signal_t.signal_t
        grad = -1*jnp.conjugate(signal_t.gate_disp)*difference_signal_t
        U = self.get_PIE_weights(signal_t.gate_disp, alpha, PIE_method)

        descent_direction = self.reverse_transform(U*grad, phase_matrix, measurement_info)

        pulse_t = do_ifft(population.pulse, sk, rn)
        pulse_t = pulse_t - beta*descent_direction
        population.pulse = do_fft(pulse_t, sk, rn)
        return population




    def update_individual_global(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        sk, rn = measurement_info.sk, measurement_info.rn
        
        pulse_t=do_ifft(individual.pulse, sk, rn)
        pulse_t=pulse_t + gamma*descent_direction
        pulse = do_fft(pulse_t, sk, rn)

        individual = MyNamespace(pulse=pulse, gate=None)
        return individual
    

    def update_population_global(self, population, gamma, descent_direction, measurement_info, pulse_or_gate):
        population = jax.vmap(self.update_individual_global, in_axes=(0,0,0,None,None))(population, gamma, descent_direction, measurement_info, pulse_or_gate)
        return population




    def calculate_PIE_descent_direction(self, population, signal_t, signal_t_new, measurement_info, descent_info, pulse_or_gate):
        phase_matrix = measurement_info.phase_matrix
        alpha, PIE_method = descent_info.alpha, descent_info.PIE_method
        
        U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(signal_t.gate_disp, alpha, PIE_method)
        grad_all_m = -1*jnp.conjugate(signal_t.gate_disp)*(signal_t_new - signal_t.signal_t)

        U = self.reverse_transform(U, phase_matrix, measurement_info)
        grad_all_m = self.reverse_transform(grad_all_m, phase_matrix, measurement_info)
        return grad_all_m, U



    def calculate_PIE_descent_direction_hessian(self, grad, signal_t, descent_state, measurement_info, descent_info, pulse_or_gate):
        signal_f = do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        
        reverse_transform = Partial(self.reverse_transform, phase_matrix=measurement_info.phase_matrix, measurement_info=measurement_info)
        newton_direction_prev = descent_state.hessian_state.newton_direction_prev.pulse
        descent_direction = PIE_get_pseudo_newton_direction(grad, signal_t.gate_disp, signal_f, newton_direction_prev, 
                                                            measurement_info, descent_info, "gate", reverse_transform)
        return descent_direction









class COPRA(RetrievePulsesDSCAN, COPRABASE):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)



    def update_population_local(self, population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        beta = descent_info.beta
        population.pulse=population.pulse + beta*gamma[:,jnp.newaxis]*descent_direction
        return population



    def update_individual_global(self, individual, alpha, eta, descent_direction, measurement_info, descent_info, pulse_or_gate):
        individual.pulse = individual.pulse + alpha*eta*descent_direction
        return individual



    def calculate_Z_gradient(self, signal_t_new, signal_t, population, phase_matrix, measurement_info, pulse_or_gate, local=False):
        if local==True:
            in_axes=(0,0,0,0,None)
        else:
            in_axes=(0,0,0,None,None)

        grad = jax.vmap(calculate_Z_gradient, in_axes=in_axes)(signal_t.pulse_t_disp, signal_t.signal_t, signal_t_new, phase_matrix, measurement_info)
        return grad


    def calculate_Z_error_newton_direction(self, grad, signal_t_new, signal_t, phase_matrix, descent_state, measurement_info, descent_info, 
                                           use_hessian, pulse_or_gate, local=False):
        if local==True:
            in_axes=(0,0,0,0,None,None)
            phase_matrix = phase_matrix[:,jnp.newaxis,:]
            grad = grad[:,jnp.newaxis,:]
            pulse_t_disp = signal_t.pulse_t_disp[:,jnp.newaxis,:]
            signal_t = signal_t.signal_t[:,jnp.newaxis,:]
            signal_t_new = signal_t_new[:,jnp.newaxis,:]
        else:
            in_axes=(0,0,0,None,None,None)
            pulse_t_disp = signal_t.pulse_t_disp
            signal_t = signal_t.signal_t
        
        newton_direction = get_pseudo_newton_direction_Z_error(grad, pulse_t_disp, signal_t, signal_t_new, phase_matrix, 
                                                               descent_state, measurement_info, descent_info, use_hessian, in_axes=in_axes)
        return newton_direction
            
    

    

    def calc_Z_grad_for_linesearch(self, alpha, linesearch_info, measurement_info, descent_info, pulse_or_gate):
        eta, descent_direction, signal_t_new = linesearch_info.eta, linesearch_info.descent_direction, linesearch_info.signal_t_new 
        phase_matrix = measurement_info.phase_matrix
        
        individual = linesearch_info.population
        
        individual = self.update_individual_global(individual, alpha, eta, descent_direction, measurement_info, descent_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, phase_matrix, measurement_info)
        grad = calculate_Z_gradient(signal_t.pulse_t_disp, signal_t.signal_t, signal_t_new, phase_matrix, measurement_info)
        return jnp.sum(grad, axis=1)
