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



    # def calc_Z_error_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
    #     population, descent_direction, signal_t_new = linesearch_info.population, linesearch_info.descent_direction, linesearch_info.signal_t_new
    #     pulse = population.pulse
    #     gate = population.gate

    #     transform_arr = measurement_info.transform_arr
    #     sk, rn = measurement_info.sk, measurement_info.rn

    #     # if pulse_or_gate=="pulse":
    #     #     pulse_f=do_fft(pulse, sk, rn)
    #     #     pulse_f=pulse_f+gamma*descent_direction
    #     #     pulse=do_ifft(pulse_f, sk, rn)

    #     # elif pulse_or_gate=="gate":
    #     #     gate=do_fft(gate, sk, rn)
    #     #     gate=gate+gamma*descent_direction
    #     #     gate=do_ifft(gate, sk, rn)

    #     # individual = MyNamespace(pulse=pulse, gate=gate)

    #     individual = self.update_population(population, gamma, descent_direction, measurement_info, pulse_or_gate)

    #     signal_t=self.calculate_signal_t(individual, transform_arr, measurement_info)
    #     Z_error_new=calculate_Z_error(signal_t.signal_t, signal_t_new)
    #     return Z_error_new
    

    def calc_Z_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        individual, descent_direction, signal_t_new = linesearch_info.population, linesearch_info.descent_direction, linesearch_info.signal_t_new

        tau_arr = measurement_info.tau_arr
        sk, rn = measurement_info.sk, measurement_info.rn
        measured_trace = measurement_info.measured_trace

        individual = self.update_individual(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t=self.calculate_signal_t(individual, tau_arr, measurement_info)
        trace = calculate_trace(do_fft(signal_t.signal_t, sk, rn))
        mu = calculate_mu(trace, measured_trace)
        signal_t_new = calculate_S_prime(signal_t.signal_t, measured_trace, mu, measurement_info)

        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, individual.pulse, signal_t.pulse_t_shifted, 
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



    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        sk, rn = measurement_info.sk, measurement_info.rn

        pulse_f = do_fft(getattr(individual, pulse_or_gate), sk, rn)
        pulse_f = pulse_f + gamma*descent_direction
        pulse = do_ifft(pulse_f, sk, rn)

        individual = MyNamespace(pulse=individual.pulse, gate=individual.gate)
        setattr(individual, pulse_or_gate, pulse)
        return individual





class TimeDomainPtychography(RetrievePulsesFROG, TimeDomainPtychographyBASE):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, PIE_method="rPIE", xfrog=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, xfrog=xfrog, **kwargs)

        self.PIE_method=PIE_method


    def reverse_transform(self, signal, tau_arr, measurement_info, local):
        frequency, time = measurement_info.frequency, measurement_info.time
        
        if local==True:
            signal = self.calculate_shifted_signal(signal, frequency, -1*tau_arr, time, in_axes=(0, 0, None, None, None))
        elif local==False:
            signal = self.calculate_shifted_signal(signal, frequency, -1*tau_arr, time, in_axes=(1, 0, None, None, None))
            signal = jnp.transpose(signal, axes=(1,0,2))

        return signal
    

    def get_grad_for_gate_pulse(self, grad_all_m, gate_pulse_shifted, nonlinear_method):
        if nonlinear_method=="shg":
            grad = 1
        elif nonlinear_method=="thg":
            grad = 2*gate_pulse_shifted
        elif nonlinear_method=="pg":
            grad = jnp.conjugate(gate_pulse_shifted)
        elif nonlinear_method=="sd":
            grad = 1 # trying to incorprate this doesnt work, the gradient with respect to the actual pulse is zero.
        else:
            print("soethong is wrong")

        grad_all_m = grad_all_m*jnp.conjugate(grad)
        return grad_all_m


    def update_population_local(self, population, signal_t, signal_t_new, tau, measurement_info, descent_info, pulse_or_gate):
        alpha, beta, PIE_method = descent_info.alpha, descent_info.beta, descent_info.PIE_method

        pulse = population.pulse
        gate = population.gate

        gate_shifted = jnp.squeeze(signal_t.gate_shifted)
        difference_signal_t = signal_t_new - jnp.squeeze(signal_t.signal_t)

        if pulse_or_gate=="pulse":
            grad = -1*jnp.conjugate(gate_shifted)*difference_signal_t
            U = self.get_PIE_weights(gate_shifted, alpha, PIE_method)
            population.pulse = pulse - beta*U*grad

        elif pulse_or_gate=="gate":
            grad = -1*jnp.conjugate(pulse)*difference_signal_t
            U = self.get_PIE_weights(pulse, alpha, PIE_method)

            grad = self.get_grad_for_gate_pulse(grad, jnp.squeeze(signal_t.gate_pulse_shifted), measurement_info.nonlinear_method)

            descent_direction = self.reverse_transform(U*grad, tau, measurement_info, local=True)
            population.gate = gate - beta*descent_direction

        return population
    


    def update_individual_global(self, individual, beta, descent_direction, measurement_info, pulse_or_gate):
        signal = getattr(individual, pulse_or_gate)
        signal = signal + beta*descent_direction

        individual = MyNamespace(pulse=individual.pulse, gate=individual.gate)
        setattr(individual, pulse_or_gate, signal)
        return individual


    def update_population_global(self, population, beta, descent_direction, measurement_info, pulse_or_gate):
        population = jax.vmap(self.update_individual_global, in_axes=(0,0,0,None,None))(population, beta, descent_direction, measurement_info, pulse_or_gate)
        return population




    def calculate_PIE_descent_direction(self, population, signal_t, signal_t_new, measurement_info, descent_info, pulse_or_gate):
        tau_arr = measurement_info.tau_arr

        alpha, PIE_method = descent_info.alpha, descent_info.PIE_method
        difference_signal_t = signal_t_new - signal_t.signal_t

        if pulse_or_gate=="pulse":
            U=jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(signal_t.gate_shifted, alpha, PIE_method)
            grad_all_m=-1*jnp.conjugate(signal_t.gate_shifted)*difference_signal_t

        elif pulse_or_gate=="gate":
            pulse = jnp.broadcast_to(population.pulse[:,jnp.newaxis,:], jnp.shape(difference_signal_t))
            U=jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(pulse, alpha, PIE_method)
            grad_all_m=-1*jnp.conjugate(pulse)*difference_signal_t

            grad_all_m = self.get_grad_for_gate_pulse(grad_all_m, signal_t.gate_pulse_shifted, measurement_info.nonlinear_method)

            U=self.reverse_transform(U, tau_arr, measurement_info, local=False)
            grad_all_m=self.reverse_transform(grad_all_m, tau_arr, measurement_info, local=False)

        return grad_all_m, U



    def calculate_PIE_descent_direction_hessian(self, grad, signal_t, descent_state, measurement_info, descent_info, pulse_or_gate):
        newton_direction_prev = getattr(descent_state.hessian_state.newton_direction_prev, pulse_or_gate)

        if pulse_or_gate=="pulse":
            probe = signal_t.gate_shifted

        elif pulse_or_gate=="gate":
            probe = jnp.broadcast_to(descent_state.population.pulse[:,jnp.newaxis,:], jnp.shape(signal_t.signal_t))

        tau_arr = measurement_info.tau_arr
        reverse_transform = Partial(self.reverse_transform, tau_arr=tau_arr, measurement_info=measurement_info, local=False)

        signal_f = do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        descent_direction=PIE_get_pseudo_newton_direction(grad, probe, signal_f, newton_direction_prev, 
                                                          measurement_info, descent_info, pulse_or_gate, reverse_transform)
        return descent_direction
    









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



    def update_individual_global(self, individual, gamma, eta, descent_direction, measurement_info, descent_info, pulse_or_gate):
        sk, rn = measurement_info.sk, measurement_info.rn

        signal = getattr(individual, pulse_or_gate)
        signal_f = do_fft(signal, sk, rn)
        signal_f = signal_f + gamma*eta*descent_direction
        signal = do_ifft(signal_f, sk, rn)

        individual = MyNamespace(pulse=individual.pulse, gate=individual.gate)
        setattr(individual, pulse_or_gate, signal)
        return individual



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
            
    







    def calc_Z_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate):
        tau_arr = measurement_info.tau_arr

        signal_t_new, eta, descent_direction = linesearch_info.signal_t_new, linesearch_info.eta, linesearch_info.descent_direction

        individual = self.update_individual_global(linesearch_info.population, gamma, eta, descent_direction, measurement_info, descent_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, tau_arr, measurement_info)
        
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, individual.pulse, signal_t.pulse_t_shifted, signal_t.gate_shifted, 
                                    tau_arr, measurement_info, pulse_or_gate)        
        return jnp.sum(grad, axis=1)



    