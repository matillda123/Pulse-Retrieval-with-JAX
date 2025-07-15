import jax
import jax.numpy as jnp
from jax.tree_util import Partial



from BaseClasses import RetrievePulsesFROG, AlgorithmsBASE
from classic_algorithms_base import GeneralizedProjectionBASE, TimeDomainPtychographyBASE, COPRABASE

from utilities import scan_helper, get_com, MyNamespace, get_sk_rn, do_fft, do_ifft, calculate_trace, calculate_mu, calculate_S_prime, calculate_trace_error, calculate_Z_error, do_interpolation_1d

from frog_z_error_gradients import calculate_Z_gradient
from frog_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error
from pie_pseudo_hessian import PIE_get_pseudo_newton_direction









class Vanilla(RetrievePulsesFROG, AlgorithmsBASE):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, xfrog=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, xfrog, **kwargs)
        

        # for some reason vanilla only works when the trace is centered around f=0. No idea why. Is undone when using LSGPA.
        idx = get_com(jnp.mean(self.measured_trace, axis=0), jnp.arange(jnp.size(self.frequency)))
        idx=int(idx)
        self.f0=frequency[idx]
        self.frequency = self.frequency - self.f0

        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        self.measurement_info.sk=self.sk
        self.measurement_info.rn=self.rn
        self.measurement_info.frequency=self.frequency




    def update_pulse(self, pulse, signal_t_new, gate_shifted, measurement_info, descent_info):
        pulse_t=jnp.sum(signal_t_new, axis=1)
        return pulse_t
    
    def update_gate(self, gate, signal_t_new, pulse_t_shifted, measurement_info, descent_info):
        gate = jnp.sum(signal_t_new, axis=2)
        gate = jax.vmap(do_interpolation_1d, in_axes=(None,None,0))(measurement_info.time, measurement_info.tau_arr, gate)
        return gate

    
        
    def step(self, descent_state, measurement_info, descent_info):
        measured_trace = measurement_info.measured_trace
        sk, rn = measurement_info.sk, measurement_info.rn

        population = descent_state.population
        
        signal_t=self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace=calculate_trace(do_fft(signal_t.signal_t, sk, rn))

        mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        signal_t_new=jax.vmap(calculate_S_prime, in_axes=(0,None,0,None))(signal_t.signal_t, measured_trace, mu, measurement_info)

        trace_error=jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)
        descent_state.population.pulse = self.update_pulse(population.pulse, signal_t_new, signal_t.gate_shifted, measurement_info, descent_info)

        if measurement_info.doubleblind==True:
            signal_t=self.generate_signal_t(descent_state, measurement_info, descent_info)
            trace=calculate_trace(do_fft(signal_t.signal_t, sk, rn))

            mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
            signal_t_new=jax.vmap(calculate_S_prime, in_axes=(0,None,0,None))(signal_t.signal_t, measured_trace, mu, measurement_info)
            descent_state.population.gate = self.update_gate(population.gate, signal_t_new, signal_t.pulse_t_shifted, measurement_info, descent_info)

        descent_state.population.pulse = jax.vmap(lambda x: x/jnp.linalg.norm(x))(descent_state.population.pulse)
        return descent_state, trace_error.reshape(-1,1)



    def initialize_run(self, population):
        if hasattr(self, "lambda_lm"):
            self.descent_info.lambda_lm = self.lambda_lm
        if hasattr(self, "beta"):
            self.descent_info.beta = self.beta

        measurement_info=self.measurement_info
        descent_info=self.descent_info

        self.descent_state.population=population
        descent_state=self.descent_state

        do_step=Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step=Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step









class LSGPA(Vanilla):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, xfrog=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, xfrog, **kwargs)


        self.frequency = self.frequency + self.f0
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        self.measurement_info.sk=self.sk
        self.measurement_info.rn=self.rn
        self.measurement_info.frequency=self.frequency

        self.f0=0


        self.lambda_lm = 1e-3
        self.beta=0.1

        


    def update_pulse(self, pulse, signal_t_new, gate_shifted, measurement_info, descent_info):
        pulse=jnp.sum(signal_t_new*jnp.conjugate(gate_shifted), axis=1)/(jnp.sum(jnp.abs(gate_shifted)**2, axis=1) + descent_info.lambda_lm)
        return pulse
    
    
    def update_gate(self, gate, signal_t_new, pulse_t_shifted, measurement_info, descent_info):
        gate=jnp.sum(signal_t_new*jnp.conjugate(pulse_t_shifted), axis=1)/(jnp.sum(jnp.abs(pulse_t_shifted)**2, axis=1) + descent_info.lambda_lm)
        return gate
        

    
    # nonlinear least squares -> maybe this performs better on doubleblind
    #   - use only gate of this -> treats gate and pulse update unequally
    #
    # def update_pulse(self, pulse, signal_t_new, gate_shifted, measurement_info, descent_info):
    #     beta=descent_info.beta
    #     t1 = 2*(gate_shifted*pulse[:,jnp.newaxis,:] - signal_t_new)*jnp.conjugate(gate_shifted)
    #     t2 = jnp.abs(gate_shifted)**2
    #     pulse = pulse - beta*jnp.sum(t1, axis=1)/(jnp.sum(t2, axis=1) + descent_info.lambda_lm)
    #     return pulse
    
    
    # def update_gate(self, gate, signal_t_new, pulse_t_shifted, measurement_info, descent_info):
    #     beta=descent_info.beta
    #     t1 = 2*(pulse_t_shifted*gate[:,jnp.newaxis,:] - signal_t_new)*jnp.conjugate(pulse_t_shifted)
    #     t2 = jnp.abs(pulse_t_shifted)**2
    #     gate = gate - beta*jnp.sum(t1, axis=1)/(jnp.sum(t2, axis=1) + descent_info.lambda_lm)
    #     return gate











class GeneralizedProjection(RetrievePulsesFROG, GeneralizedProjectionBASE):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, xfrog=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, xfrog=xfrog, **kwargs)



    def calc_Z_error_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        population, descent_direction, signal_t_new = linesearch_info.population, linesearch_info.descent_direction, linesearch_info.signal_t_new
        pulse = population.pulse
        gate = population.gate

        tau_arr = measurement_info.tau_arr
        sk, rn = measurement_info.sk, measurement_info.rn

        if pulse_or_gate=="pulse":
            pulse_f=do_fft(pulse, sk, rn)
            pulse_f=pulse_f+gamma*descent_direction
            pulse=do_ifft(pulse_f, sk, rn)

        elif pulse_or_gate=="gate":
            gate=do_fft(gate, sk, rn)
            gate=gate+gamma*descent_direction
            gate=do_ifft(gate, sk, rn)

        individual = MyNamespace(pulse=pulse, gate=gate)
        signal_t=self.calculate_signal_t(individual, tau_arr, measurement_info)
        Z_error_new=calculate_Z_error(signal_t.signal_t, signal_t_new)
        return Z_error_new
    

    def calc_Z_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        population, descent_direction, signal_t_new = linesearch_info.population, linesearch_info.descent_direction, linesearch_info.signal_t_new
        pulse = population.pulse
        gate = population.gate

        tau_arr = measurement_info.tau_arr
        sk, rn = measurement_info.sk, measurement_info.rn
        measured_trace = measurement_info.measured_trace

        if pulse_or_gate=="pulse":
            pulse_f=do_fft(pulse, sk, rn)
            pulse_f=pulse_f+gamma*descent_direction
            pulse=do_ifft(pulse_f, sk, rn)

        elif pulse_or_gate=="gate":
            gate=do_fft(gate, sk, rn)
            gate=gate+gamma*descent_direction
            gate=do_ifft(gate, sk, rn)

        individual = MyNamespace(pulse=pulse, gate=gate)
        signal_t=self.calculate_signal_t(individual, tau_arr, measurement_info)
        trace = calculate_trace(do_fft(signal_t.signal_t, sk, rn))
        mu = calculate_mu(trace, measured_trace)
        signal_t_new = calculate_S_prime(signal_t.signal_t, measured_trace, mu, measurement_info)

        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, pulse, signal_t.pulse_t_shifted, 
                                    signal_t.gate_shifted, tau_arr, measurement_info, pulse_or_gate)
        return jnp.sum(grad, axis=0) 



    def calculate_Z_error_gradient(self, signal_t_new, signal_t, population, tau_arr, measurement_info, pulse_or_gate):
        grad = jax.vmap(calculate_Z_gradient, in_axes=(0, 0, 0, 0, 0, None, None, None))(signal_t.signal_t, signal_t_new, population.pulse, signal_t.pulse_t_shifted, 
                                                                                         signal_t.gate_shifted, tau_arr, measurement_info, pulse_or_gate)
        return grad


    def calculate_Z_error_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, use_hessian, pulse_or_gate):
        newton_direction = get_pseudo_newton_direction_Z_error(grad, signal_t.pulse_t_shifted, signal_t.gate_shifted, signal_t.signal_t, signal_t_new, tau_arr, 
                                                               descent_state, measurement_info, descent_info, use_hessian, pulse_or_gate)
        return newton_direction



    def update_population(self, population, gamma_new, descent_direction, measurement_info, pulse_or_gate):
        sk, rn = measurement_info.sk, measurement_info.rn

        pulse_f = do_fft(getattr(population, pulse_or_gate), sk, rn)
        pulse_f = pulse_f + gamma_new[:,jnp.newaxis]*descent_direction
        pulse = do_ifft(pulse_f, sk, rn)

        setattr(population, pulse_or_gate, pulse)  
        return population
    






class TimeDomainPtychography(RetrievePulsesFROG, TimeDomainPtychographyBASE):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, PIE_method="rPIE", xfrog=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, xfrog=xfrog, **kwargs)

        self.PIE_method=PIE_method



    def update_population_local(self, population, signal_t, signal_t_new, tau, measurement_info, descent_info, pulse_or_gate):
        alpha, beta, PIE_method = descent_info.alpha, descent_info.beta, descent_info.PIE_method
        pulse = population.pulse
        gate_shifted = jnp.squeeze(signal_t.gate_shifted)

        difference_signal_t = signal_t_new - jnp.squeeze(signal_t.signal_t)

        if pulse_or_gate=="pulse":
            grad = -1*jnp.conjugate(gate_shifted)*difference_signal_t
            U = self.get_PIE_weights(gate_shifted, alpha, PIE_method)
            population.pulse = pulse - beta*U*grad

        elif pulse_or_gate=="gate":
            time, frequency = measurement_info.time, measurement_info.frequency
            grad = -1*jnp.conjugate(pulse)*difference_signal_t
            U = self.get_PIE_weights(pulse, alpha, PIE_method)
            
            gate_shifted = gate_shifted - beta*U*grad
            gate=jax.vmap(self.reverse_time_shift_for_gate, in_axes=(0,0,None,None))(gate_shifted[:,jnp.newaxis,:], tau, frequency, time)
            population.gate=jnp.squeeze(gate)
            
        return population



    def update_population_global(self, signal_t, population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        if pulse_or_gate=="pulse":
            pulse = population.pulse + gamma[:,jnp.newaxis]*descent_direction
            population.pulse=pulse
            
        elif pulse_or_gate=="gate":
            time, frequency = measurement_info.time, measurement_info.frequency
            gate_shifted = signal_t.gate_shifted + gamma[:,jnp.newaxis,jnp.newaxis]*descent_direction
            gate_shifted_back = jax.vmap(self.reverse_time_shift_for_gate, in_axes=(0,None,None,None))(gate_shifted, measurement_info.tau_arr, frequency, time)
            population.gate=jnp.mean(gate_shifted_back, axis=1)

        return population




    def calc_error_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        tau_arr, time, frequency, measured_trace = measurement_info.tau_arr, measurement_info.time, measurement_info.frequency, measurement_info.measured_trace
        sk, rn = measurement_info.sk, measurement_info.rn

        individual, descent_direction = linesearch_info.population, linesearch_info.descent_direction

        if pulse_or_gate=="pulse":
            gate = individual.gate
            pulse = individual.pulse + gamma*descent_direction

        elif pulse_or_gate=="gate":
            pulse = individual.pulse
            gate_shifted = linesearch_info.signal_t.gate_shifted
            gate_shifted = gate_shifted + gamma*descent_direction
            gate_shifted_back = self.reverse_time_shift_for_gate(gate_shifted, tau_arr, frequency, time)
            gate = jnp.mean(gate_shifted_back, axis=0)

        individual = MyNamespace(pulse=pulse, gate=gate)

        signal_t=self.calculate_signal_t(individual, tau_arr, measurement_info)
        signal_f=do_fft(signal_t.signal_t, sk, rn)
        error_new=self.calculate_PIE_error(signal_f, measured_trace)
        return error_new
    



    def calculate_PIE_descent_direction(self, population, signal_t, signal_t_new, descent_info, pulse_or_gate):
        alpha, PIE_method = descent_info.alpha, descent_info.PIE_method
        difference_signal_t = signal_t_new - signal_t.signal_t

        if pulse_or_gate=="pulse":
            U=jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(signal_t.gate_shifted, alpha, PIE_method)
            grad_all_m=-1*jnp.conjugate(signal_t.gate_shifted)*difference_signal_t

        elif pulse_or_gate=="gate":
            U=jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(population.pulse, alpha, PIE_method)[:,:,jnp.newaxis]
            grad_all_m=-1*jnp.conjugate(population.pulse)[:,:,jnp.newaxis]*difference_signal_t

        return grad_all_m, U



    def calculate_PIE_descent_direction_hessian(self, grad, signal_t, descent_state, measurement_info, descent_info, pulse_or_gate):
        newton_direction_prev = getattr(descent_state.hessian_state.newton_direction_prev, pulse_or_gate)

        if pulse_or_gate=="pulse":
            probe=signal_t.gate_shifted

        elif pulse_or_gate=="gate":
            probe=descent_state.population.pulse


        signal_f = do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        descent_direction=PIE_get_pseudo_newton_direction(grad, probe, signal_f, newton_direction_prev, measurement_info, descent_info, pulse_or_gate)
        return descent_direction



    def get_descent_direction(self, grad, U, pulse_or_gate):
        if pulse_or_gate=="pulse":
            descent_direction=-1*jnp.sum(grad*U, axis=1)

        elif pulse_or_gate=="gate":
            descent_direction=-1*grad*U

        return descent_direction



    def calculate_pk_dot_gradient(self, grad, gradient_sum, descent_direction, pulse_or_gate):
        if pulse_or_gate=="pulse":
            pk_dot_gradient=jax.vmap(lambda x,y: jnp.real(jnp.dot(jnp.conjugate(x),y)), in_axes=(0,0))(descent_direction, gradient_sum)

        elif pulse_or_gate=="gate":
            pk_dot_gradient=jax.vmap(lambda x,y: jnp.sum(jnp.real(jax.vmap(jnp.dot, in_axes=(0,0))(jnp.conjugate(x), y))), in_axes=(0,0))(descent_direction, grad)

        return pk_dot_gradient

    



    def reverse_time_shift_for_gate(self, gate_shifted, x_arr, frequency, time):
        frequency = frequency - (frequency[-1] + frequency[0])/2

        N=jnp.size(frequency)
        gate_shifted = jnp.pad(gate_shifted, ((0,0),(N,N)))
        frequency = jnp.linspace(jnp.min(frequency), jnp.max(frequency), 3*N)
        time = jnp.fft.fftshift(jnp.fft.fftfreq(3*N, jnp.mean(jnp.diff(frequency))))

        sk, rn = get_sk_rn(time, frequency)

        gate_shifted_back=jax.vmap(self.shift_signal_in_time, in_axes=(0, 0, None, None, None))(gate_shifted, -1*x_arr, frequency, sk, rn)
        return gate_shifted_back[:, N:2*N]

    















class COPRA(RetrievePulsesFROG, COPRABASE):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, xfrog=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, xfrog=xfrog, **kwargs)



    def update_population_local(self, population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        beta = descent_info.beta
        sk, rn = measurement_info.sk, measurement_info.rn

        signal = getattr(population, pulse_or_gate)
        signal_f = do_fft(signal, sk, rn)
        signal_f = signal_f + beta*gamma[:,jnp.newaxis]*descent_direction
        signal = do_ifft(signal_f, sk, rn)

        setattr(population, pulse_or_gate, signal)
        return population




    def update_population_global(self, population, eta, descent_direction, measurement_info, descent_info, pulse_or_gate):
        alpha = descent_info.alpha
        sk, rn = measurement_info.sk, measurement_info.rn

        signal = getattr(population, pulse_or_gate)
        signal_f = do_fft(signal, sk, rn)
        signal_f = signal_f + alpha*eta[:,jnp.newaxis]*descent_direction
        signal = do_ifft(signal_f, sk, rn)

        setattr(population, pulse_or_gate, signal)
        return population



    def calculate_Z_gradient(self, signal_t_new, signal_t, population, tau_arr, measurement_info, pulse_or_gate, local=False):
        if local==True:
            in_axes=(0,0,0,0,0,0,None,None)
        else:
            in_axes=(0,0,0,0,0,None,None,None)

        grad = jax.vmap(calculate_Z_gradient, in_axes=in_axes)(signal_t.signal_t, signal_t_new, population.pulse, 
                                                                                  signal_t.pulse_t_shifted, signal_t.gate_shifted, tau_arr, measurement_info, pulse_or_gate)
        return grad



    def calculate_Z_error_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, 
                                           use_hessian, pulse_or_gate, local=False):
        if local==True:
            in_axes=(0,0,0,0,0,0,None,None,None)
        else:
            in_axes=(0,0,0,0,0,None,None,None,None)

        newton_direction = get_pseudo_newton_direction_Z_error(grad, signal_t.pulse_t_shifted, signal_t.gate_shifted, signal_t.signal_t, signal_t_new, tau_arr,
                                                               descent_state, measurement_info, descent_info, use_hessian, pulse_or_gate, in_axes=in_axes)
        return newton_direction
            
    



    