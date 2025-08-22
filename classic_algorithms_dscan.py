import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from equinox import tree_at

from BaseClasses import RetrievePulsesDSCAN, AlgorithmsBASE
from classic_algorithms_base import GeneralizedProjectionBASE, TimeDomainPtychographyBASE, COPRABASE


from utilities import scan_helper, MyNamespace, get_sk_rn, do_fft, do_ifft, calculate_mu, calculate_S_prime, calculate_trace, calculate_trace_error, calculate_Z_error


from dscan_z_error_gradients import calculate_Z_gradient
from dscan_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error
from pie_pseudo_hessian import PIE_get_pseudo_newton_direction









class Basic(RetrievePulsesDSCAN, AlgorithmsBASE):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)

        self.name = "Basic"


    def update_pulse(self, signal_t_new, gate, phase_matrix, nonlinear_method, sk, rn):
        signal_t_new = signal_t_new*jnp.conjugate(gate)

        if nonlinear_method=="shg":
            n=3
        else: 
            n=5
        signal_t_new = jnp.abs(signal_t_new)**(1/n)*jnp.exp(1j*jnp.angle(signal_t_new))

        signal_f_new = do_fft(signal_t_new, sk, rn)
        signal_f_new = signal_f_new*jnp.exp(-1j*phase_matrix)

        pulse_f = jnp.mean(signal_f_new, axis=0)
        return pulse_f
    
    

    def step(self, descent_state, measurement_info, descent_info):
        nonlinear_method, sk, rn = measurement_info.nonlinear_method, measurement_info.sk, measurement_info.rn
        phase_matrix = measurement_info.phase_matrix
        measured_trace = measurement_info.measured_trace

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_f = do_fft(signal_t.signal_t, sk, rn)
        trace = calculate_trace(signal_f)


        mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,None,0,None))(signal_t.signal_t, measured_trace, mu, measurement_info)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)
        
        pulse = jax.vmap(self.update_pulse, in_axes=(0,0,None,None,None,None))(signal_t_new, signal_t.gate_disp, phase_matrix, nonlinear_method, sk, rn)

        descent_state = tree_at(lambda x: x.population.pulse, descent_state, pulse)
        return descent_state, trace_error.reshape(-1,1)
    


    def initialize_run(self, population):

        self.descent_state = self.descent_state.expand(population = population)
       
        measurement_info=self.measurement_info
        descent_info=self.descent_info
        descent_state=self.descent_state

        do_step=Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step=Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)

        return descent_state, do_step
    

    






class GeneralizedProjection(RetrievePulsesDSCAN, GeneralizedProjectionBASE):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)



    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, population, phase_matrix, measurement_info, pulse_or_gate):
        grad = calculate_Z_gradient(signal_t.pulse_t_disp, signal_t.signal_t, signal_t_new, phase_matrix, measurement_info)
        return grad 


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, phase_matrix, descent_state, measurement_info, descent_info, use_hessian, pulse_or_gate):
        descent_direction, hessian = get_pseudo_newton_direction_Z_error(grad, signal_t.pulse_t_disp, signal_t.signal_t, signal_t_new, phase_matrix, 
                                                                        measurement_info, descent_state.hessian, descent_info.hessian, use_hessian)
        return descent_direction, hessian
    

    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        pulse = individual.pulse + gamma*descent_direction
        individual = tree_at(lambda x: x.pulse, individual, pulse)
        return individual








class TimeDomainPtychography(RetrievePulsesDSCAN, TimeDomainPtychographyBASE):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, pie_method="rPIE", **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)

        self.pie_method = pie_method




    def reverse_transform_grad(self, signal, phase_matrix, measurement_info):
        sk, rn = measurement_info.sk, measurement_info.rn
        signal_f = do_fft(signal, sk, rn)
        signal_f = signal_f*jnp.exp(-1j*phase_matrix)
        signal = do_ifft(signal_f, sk, rn)
        return signal


    def reverse_transform_full_hessian(self, hessian_all_m, phase_matrix, measurement_info):
        time, frequency = measurement_info.time, measurement_info.frequency
    
        frequency = frequency - (frequency[-1] + frequency[0])/2
        N = jnp.size(frequency)
        hessian_all_m = jnp.pad(hessian_all_m, ((0,0), (0,N), (0,N))) 

        frequency = jnp.linspace(jnp.min(frequency), jnp.max(frequency), 2*N)
        time = jnp.fft.fftshift(jnp.fft.fftfreq(2*N, jnp.mean(jnp.diff(frequency))))
        sk, rn = get_sk_rn(time, frequency)

        # convert hessian to (m, n, n) -> frequency domain 
        hessian_all_m = do_fft(hessian_all_m, sk, rn, axis=-1)
        hessian_all_m = do_fft(hessian_all_m, sk, rn, axis=-2) 

        phi_mn = -1*phase_matrix
        phi = phi_mn[:,:,jnp.newaxis] - phi_mn[:,jnp.newaxis,:]
        exp_arr = jnp.exp(1j*phi)
        hessian_all_m = hessian_all_m * exp_arr

        # convert hessian to (N, m, k, k) -> time domain 
        hessian_all_m = do_ifft(hessian_all_m, sk, rn, axis=-1)
        hessian_all_m = do_ifft(hessian_all_m, sk, rn, axis=-2) 
        return hessian_all_m[:, :N, :N]
    


    def reverse_transform_diagonal_hessian(self, hessian_all_m, phase_matrix, measurement_info):
        # # i think a backtransform is not needed since the transform matrix phi is zero for these entries
        return hessian_all_m



    def calculate_PIE_descent_direction_m(self, signal_t, signal_t_new, phase_matrix_m, population, pie_method, measurement_info, descent_info):
        alpha = descent_info.alpha

        probe = signal_t.gate_disp
        difference_signal_t = signal_t_new - signal_t.signal_t

        grad = -1*jnp.conjugate(probe)*difference_signal_t
        U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(probe, alpha, pie_method)

        U = jax.vmap(self.reverse_transform_grad, in_axes=(0,0,None))(U, phase_matrix_m, measurement_info)
        grad = jax.vmap(self.reverse_transform_grad, in_axes=(0,0,None))(grad, phase_matrix_m, measurement_info)
        return grad, U




    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        sk, rn = measurement_info.sk, measurement_info.rn
        
        pulse_t=do_ifft(individual.pulse, sk, rn)
        pulse_t=pulse_t + gamma*descent_direction
        pulse = do_fft(pulse_t, sk, rn)

        individual = tree_at(lambda x: x.pulse, individual, pulse)
        return individual
    

    def calculate_PIE_newton_direction(self, grad, signal_t, phase_matrix, measured_trace, population, local_or_global_state, measurement_info, 
                                                descent_info, pulse_or_gate, local_or_global):

        reverse_transform_hessian = {"diagonal": self.reverse_transform_diagonal_hessian,
                                     "full": self.reverse_transform_full_hessian}
        reverse_transform = Partial(reverse_transform_hessian[getattr(descent_info.hessian, local_or_global)], measurement_info=measurement_info)

        signal_f = do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        descent_direction, hessian = PIE_get_pseudo_newton_direction(grad, signal_t.gate_disp, signal_f, phase_matrix, measured_trace, reverse_transform, 
                                                                     local_or_global_state, measurement_info, descent_info, "gate", local_or_global)
        return descent_direction, hessian















class COPRA(RetrievePulsesDSCAN, COPRABASE):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, **kwargs):
        super().__init__(z_arr, frequency, measured_trace, nonlinear_method, **kwargs)




    def update_individual(self, individual, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        pulse = individual.pulse + gamma*descent_direction
        individual = tree_at(lambda x: x.pulse, individual, pulse)
        return individual



    def get_Z_gradient_individual(self, signal_t, signal_t_new, population, phase_matrix, measurement_info):
        grad = calculate_Z_gradient(signal_t.pulse_t_disp, signal_t.signal_t, signal_t_new, phase_matrix, measurement_info)
        return grad



    def calculate_Z_error_newton_direction(self, grad, signal_t, signal_t_new, phase_matrix, population, local_or_global_state, measurement_info, descent_info, 
                                           use_hessian, pulse_or_gate):

        hessian_state = local_or_global_state.hessian
        descent_direction, hessian = get_pseudo_newton_direction_Z_error(grad, signal_t.pulse_t_disp, signal_t.signal_t, signal_t_new, phase_matrix, measurement_info, 
                                                                         hessian_state, descent_info.hessian, use_hessian)
        return descent_direction, hessian
