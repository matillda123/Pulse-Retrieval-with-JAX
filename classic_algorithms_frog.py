import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from equinox import tree_at



from BaseClasses import RetrievePulsesFROG, AlgorithmsBASE
from classic_algorithms_base import GeneralizedProjectionBASE, TimeDomainPtychographyBASE, COPRABASE

from utilities import scan_helper, get_com, MyNamespace, get_sk_rn, do_fft, do_ifft, calculate_trace, calculate_mu, calculate_S_prime, calculate_trace_error, calculate_Z_error, do_interpolation_1d

from frog_z_error_gradients import calculate_Z_gradient
from frog_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error
from pie_pseudo_hessian import PIE_get_pseudo_newton_direction









class Vanilla(RetrievePulsesFROG, AlgorithmsBASE):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, xfrog=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, xfrog, **kwargs)
        self.name = "Vanilla"
        

        # for some reason vanilla only works when the trace is centered around f=0. No idea why. Is undone when using LSGPA.
        idx = get_com(jnp.mean(self.measured_trace, axis=0), jnp.arange(jnp.size(self.frequency)))
        idx=int(idx)
        self.f0=frequency[idx]
        self.frequency = self.frequency - self.f0

        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        self.measurement_info = tree_at(lambda x: x.sk, self.measurement_info, self.sk)
        self.measurement_info = tree_at(lambda x: x.rn, self.measurement_info, self.rn)
        self.measurement_info = tree_at(lambda x: x.frequency, self.measurement_info, self.frequency)




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
        
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(do_fft(signal_t.signal_t, sk, rn))

        mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,None,0,None))(signal_t.signal_t, measured_trace, mu, measurement_info)

        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)
        population_pulse = self.update_pulse(population.pulse, signal_t_new, signal_t.gate_shifted, measurement_info, descent_info)
        population_pulse = jax.vmap(lambda x: x/jnp.linalg.norm(x))(population_pulse)
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population_pulse)

        if measurement_info.doubleblind==True:
            signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
            trace = calculate_trace(do_fft(signal_t.signal_t, sk, rn))

            mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
            signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,None,0,None))(signal_t.signal_t, measured_trace, mu, measurement_info)
            population_gate = self.update_gate(population.gate, signal_t_new, signal_t.pulse_t_shifted, measurement_info, descent_info)
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population_gate)

        return descent_state, trace_error.reshape(-1,1)



    def initialize_run(self, population):
        measurement_info = self.measurement_info
        descent_info = self.descent_info

        self.descent_state = self.descent_state.expand(population=population)
        descent_state=self.descent_state

        do_step=Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step=Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step









class LSGPA(Vanilla):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, xfrog=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, xfrog, **kwargs)
        self.name="LSGPA"


        self.frequency = self.frequency + self.f0
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)
        
        self.measurement_info = tree_at(lambda x: x.sk, self.measurement_info, self.sk)
        self.measurement_info = tree_at(lambda x: x.rn, self.measurement_info, self.rn)
        self.measurement_info = tree_at(lambda x: x.frequency, self.measurement_info, self.frequency)

        self.f0=0

        self.lambda_lm = 1e-3
        self.beta = 0.1

        


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


    

    def calc_Z_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        individual, descent_direction, signal_t_new = linesearch_info.population, linesearch_info.descent_direction, linesearch_info.signal_t_new

        tau_arr = measurement_info.tau_arr

        individual = self.update_individual(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, tau_arr, measurement_info)
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, individual.pulse, signal_t.pulse_t_shifted, 
                                    signal_t.gate_shifted, tau_arr, measurement_info, pulse_or_gate)
        return jnp.sum(grad, axis=0) 



    def calculate_Z_error_gradient(self, signal_t_new, signal_t, population, tau_arr, measurement_info, pulse_or_gate):
        grad = jax.vmap(calculate_Z_gradient, in_axes=(0, 0, 0, 0, 0, None, None, None))(signal_t.signal_t, signal_t_new, population.pulse, signal_t.pulse_t_shifted, 
                                                                                         signal_t.gate_shifted, tau_arr, measurement_info, pulse_or_gate)
        return grad


    def calculate_Z_error_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, use_hessian, pulse_or_gate):
        newton_direction = get_pseudo_newton_direction_Z_error(grad, signal_t.pulse_t_shifted, signal_t.gate_shifted, signal_t.signal_t, signal_t_new, tau_arr, 
                                                               descent_state, measurement_info, descent_info.hessian, use_hessian, pulse_or_gate,
                                                               in_axes=(0,0,0,0,0,None,None,None,None))
        return newton_direction



    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        sk, rn = measurement_info.sk, measurement_info.rn

        pulse_f = do_fft(getattr(individual, pulse_or_gate), sk, rn)
        pulse_f = pulse_f + gamma*descent_direction
        pulse = do_ifft(pulse_f, sk, rn)

        #individual = MyNamespace(pulse=individual.pulse, gate=individual.gate) # i dont need to do this, since tree_at will make a copy.
        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, pulse)
        return individual







class TimeDomainPtychography(RetrievePulsesFROG, TimeDomainPtychographyBASE):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, PIE_method="rPIE", xfrog=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, xfrog=xfrog, **kwargs)

        self.PIE_method=PIE_method


    def reverse_transform_grad(self, signal, tau_arr, measurement_info, local):
        frequency, time = measurement_info.frequency, measurement_info.time
        
        if local==True:
            signal = self.calculate_shifted_signal(signal, frequency, -1*tau_arr, time, in_axes=(0, 0, None, None, None))

        elif local==False:
            signal = self.calculate_shifted_signal(signal, frequency, -1*tau_arr, time, in_axes=(1, 0, None, None, None))
            signal = jnp.transpose(signal, axes=(1,0,2))

        return signal
    


    def reverse_transform_full_hessian(self, hessian_all_m, tau_arr, measurement_info):
        time, frequency = measurement_info.time, measurement_info.frequency
    
        frequency = frequency - (frequency[-1] + frequency[0])/2
        N = jnp.size(frequency)
        hessian_all_m = jnp.pad(hessian_all_m, ((0,0), (0,0), (N,N), (N,N))) 

        frequency = jnp.linspace(jnp.min(frequency), jnp.max(frequency), 3*N)
        time = jnp.fft.fftshift(jnp.fft.fftfreq(3*N, jnp.mean(jnp.diff(frequency))))
        sk, rn = get_sk_rn(time, frequency)

        # convert hessian to (N, m, n, n) -> frequency domain 
        hessian_all_m = do_fft(hessian_all_m, sk, rn, axis=-1)
        hessian_all_m = do_fft(hessian_all_m, sk, rn, axis=-2) 

        phi_mn = -1*2*jnp.pi*jnp.outer(tau_arr, frequency)
        phi = phi_mn[:,:,jnp.newaxis] - phi_mn[:,jnp.newaxis,:]
        exp_arr = jnp.exp(-1j*phi)
        hessian_all_m = hessian_all_m * exp_arr[jnp.newaxis,:,:,:]

        # convert hessian to (N, m, k, k) -> time domain 
        hessian_all_m = do_ifft(hessian_all_m, sk, rn, axis=-1)
        hessian_all_m = do_ifft(hessian_all_m, sk, rn, axis=-2) 
        return hessian_all_m[:, :, N:2*N, N:2*N]
    

    def reverse_transform_diagonal_hessian(self, hessian_all_m, tau_arr, measurement_info):
        # # i think a backtransform is not needed since the transform matrix phi is zero for these entries

        # time, frequency = measurement_info.time, measurement_info.frequency
        # frequency = frequency - (frequency[-1] + frequency[0])/2
        # N = jnp.size(frequency)
        # hessian_all_m = jnp.pad(hessian_all_m, ((0,0), (0,0), (N,N))) 

        # frequency = jnp.linspace(jnp.min(frequency), jnp.max(frequency), 3*N)
        # time = jnp.fft.fftshift(jnp.fft.fftfreq(3*N, jnp.mean(jnp.diff(frequency))))
        # sk, rn = get_sk_rn(time, frequency)

        # # convert hessian to (N, m, n) -> frequency domain 
        # hessian_all_m = do_fft(hessian_all_m, sk, rn, axis=-1)

        # phi_mn = -1*2*jnp.pi*jnp.outer(tau_arr, frequency)
        # phi = phi_mn[:,:,jnp.newaxis] - phi_mn[:,jnp.newaxis,:]
        # exp_arr = jnp.exp(-1j*phi)
        # hessian_all_m = hessian_all_m * exp_arr[jnp.newaxis,:,:,:]

        # # convert hessian to (N, m, k) -> time domain 
        # hessian_all_m = do_ifft(hessian_all_m, sk, rn, axis=-1)
        return hessian_all_m
    

    def modify_grad_for_gate_pulse(self, grad_all_m, gate_pulse_shifted, nonlinear_method):
        if nonlinear_method=="shg":
            grad = 1
        elif nonlinear_method=="thg":
            grad = 2*gate_pulse_shifted
        elif nonlinear_method=="pg":
            grad = jnp.conjugate(gate_pulse_shifted)
        elif nonlinear_method=="sd":
            print("check again if this is really correct")
            grad = 2*jnp.conjugate(gate_pulse_shifted)
        else:
            print("somethong is wrong")

        grad_all_m = grad_all_m*jnp.conjugate(grad)
        return grad_all_m


    def update_population_local(self, population, signal_t, signal_t_new, tau, PIE_method, measurement_info, descent_info, pulse_or_gate):
        alpha, gamma = descent_info.alpha, descent_info.gamma

        pulse = population.pulse
        gate = population.gate

        gate_shifted = jnp.squeeze(signal_t.gate_shifted)
        difference_signal_t = signal_t_new - jnp.squeeze(signal_t.signal_t)

        if pulse_or_gate=="pulse":
            grad = -1*jnp.conjugate(gate_shifted)*difference_signal_t
            U = self.get_PIE_weights(gate_shifted, alpha, PIE_method)
            pulse = pulse - gamma*U*grad
            population = tree_at(lambda x: x.pulse, population, pulse)

        elif pulse_or_gate=="gate":
            grad = -1*jnp.conjugate(pulse)*difference_signal_t
            U = self.get_PIE_weights(pulse, alpha, PIE_method)

            grad = self.modify_grad_for_gate_pulse(grad, jnp.squeeze(signal_t.gate_pulse_shifted), measurement_info.nonlinear_method)

            descent_direction = self.reverse_transform_grad(U*grad, tau, measurement_info, local=True)
            gate = gate - gamma*descent_direction
            population = tree_at(lambda x: x.gate, population, gate)

        return population
    


    def update_individual_global(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        signal = getattr(individual, pulse_or_gate)
        signal = signal + gamma*descent_direction

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal)
        return individual


    def update_population_global(self, population, gamma, descent_direction, measurement_info, pulse_or_gate):
        population = jax.vmap(self.update_individual_global, in_axes=(0,0,0,None,None))(population, gamma, descent_direction, measurement_info, pulse_or_gate)
        return population




    def calculate_PIE_descent_direction(self, population, signal_t, signal_t_new, PIE_method, measurement_info, descent_info, pulse_or_gate):
        tau_arr = measurement_info.tau_arr

        alpha = descent_info.alpha
        difference_signal_t = signal_t_new - signal_t.signal_t

        if pulse_or_gate=="pulse":
            U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(signal_t.gate_shifted, alpha, PIE_method)
            grad_all_m = -1*jnp.conjugate(signal_t.gate_shifted)*difference_signal_t

        elif pulse_or_gate=="gate":
            pulse = jnp.broadcast_to(population.pulse[:,jnp.newaxis,:], jnp.shape(difference_signal_t))
            U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(pulse, alpha, PIE_method)
            grad_all_m = -1*jnp.conjugate(pulse)*difference_signal_t

            grad_all_m = self.modify_grad_for_gate_pulse(grad_all_m, signal_t.gate_pulse_shifted, measurement_info.nonlinear_method)

            U = self.reverse_transform_grad(U, tau_arr, measurement_info, local=False)
            grad_all_m = self.reverse_transform_grad(grad_all_m, tau_arr, measurement_info, local=False)

        return grad_all_m, U



    def calculate_PIE_descent_direction_hessian(self, grad, signal_t, descent_state, measurement_info, descent_info, pulse_or_gate):
        newton_direction_prev = getattr(descent_state.hessian.newton_direction_prev, pulse_or_gate)

        if pulse_or_gate=="pulse":
            probe = signal_t.gate_shifted

        elif pulse_or_gate=="gate":
            probe = jnp.broadcast_to(descent_state.population.pulse[:,jnp.newaxis,:], jnp.shape(signal_t.signal_t))
            probe = self.modify_grad_for_gate_pulse(jnp.conjugate(probe), signal_t.gate_pulse_shifted, measurement_info.nonlinear_method)
            probe = jnp.conjugate(probe)

        
        if descent_info.hessian.use_hessian=="diagonal":
            reverse_transform_hessian = self.reverse_transform_diagonal_hessian
        elif descent_info.hessian.use_hessian=="full":
            reverse_transform_hessian = self.reverse_transform_full_hessian
        else:
            print("something is very wrong if you can read this")

        reverse_transform = Partial(reverse_transform_hessian, tau_arr=measurement_info.tau_arr, measurement_info=measurement_info)

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

        population = tree_at(lambda x: getattr(x, pulse_or_gate), population, signal)
        return population



    def update_individual_global(self, individual, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        sk, rn = measurement_info.sk, measurement_info.rn

        signal = getattr(individual, pulse_or_gate)
        signal_f = do_fft(signal, sk, rn)
        signal_f = signal_f + gamma*descent_direction
        signal = do_ifft(signal_f, sk, rn)

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal)
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
                                                               descent_state, measurement_info, descent_info.hessian, use_hessian, pulse_or_gate, in_axes=in_axes)
        return newton_direction
            
    







    def calc_Z_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate):
        tau_arr = measurement_info.tau_arr

        signal_t_new, eta, descent_direction = linesearch_info.signal_t_new, linesearch_info.eta, linesearch_info.descent_direction

        individual = self.update_individual_global(linesearch_info.population, gamma, eta, descent_direction, measurement_info, descent_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, tau_arr, measurement_info)
        
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, individual.pulse, signal_t.pulse_t_shifted, signal_t.gate_shifted, 
                                    tau_arr, measurement_info, pulse_or_gate)        
        return jnp.sum(grad, axis=1)



    