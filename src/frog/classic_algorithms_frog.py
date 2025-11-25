import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from equinox import tree_at

from src.core.base_classes_methods import RetrievePulsesFROG 
from src.core.base_classes_algorithms import ClassicAlgorithmsBASE
from src.core.base_classic_algorithms import GeneralizedProjectionBASE, TimeDomainPtychographyBASE, COPRABASE, initialize_S_prime_params

from src.utilities import MyNamespace, scan_helper, get_com, get_sk_rn, calculate_gate, calculate_trace, calculate_mu, calculate_trace_error, do_interpolation_1d
from src.core.construct_s_prime import calculate_S_prime_projection, calculate_S_prime

from src.gradients.frog_z_error_gradients import calculate_Z_gradient
from src.hessians.frog_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error
from src.hessians.pie_pseudo_hessian import PIE_get_pseudo_newton_direction






class Vanilla(ClassicAlgorithmsBASE, RetrievePulsesFROG):
    """
    The Vanilla-FROG Algorithm as described by R. Trebino. Inherits from ClassicAlgorithmsBASE and RetrievePulsesFROG.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)
        self.name = "Vanilla"        

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
        sk, rn = measurement_info.sk, measurement_info.rn

        population = descent_state.population
        
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(self.fft(signal_t.signal_t, sk, rn))

        mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        signal_t_new = jax.vmap(calculate_S_prime_projection, in_axes=(0,None,0,None))(signal_t.signal_t, measured_trace, mu, measurement_info)

        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)
        population_pulse = self.update_pulse(population.pulse, signal_t_new, signal_t.gate_shifted, measurement_info, descent_info)
        population_pulse = jax.vmap(lambda x: x/jnp.linalg.norm(x))(population_pulse)
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population_pulse)


        if measurement_info.doubleblind==True:
            signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
            trace = calculate_trace(self.fft(signal_t.signal_t, sk, rn))

            mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
            signal_t_new = jax.vmap(calculate_S_prime_projection, in_axes=(0,None,0,None))(signal_t.signal_t, measured_trace, mu, measurement_info)
            population_gate = self.update_gate(population.gate, signal_t_new, signal_t.pulse_t_shifted, measurement_info, descent_info)
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
        descent_info = self.descent_info

        self.descent_state = self.descent_state.expand(population=population)
        descent_state = self.descent_state

        do_step = Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step = Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step









class LSGPA(Vanilla):
    # this could actually be a standalone classic algorithm. But its probably not worth it.
    """
    The Least-Squares Generalized Projection Algorithm as described by J. Gagnon et al., Appl. Phys. B 92, 25-32 (2008). https://doi.org/10.1007/s00340-008-3063-x
    Inherits from Vanilla.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)
        self.name = "LSGPA"

        self.frequency = self.frequency + self.f0
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)
        
        self.measurement_info = tree_at(lambda x: x.sk, self.measurement_info, self.sk)
        self.measurement_info = tree_at(lambda x: x.rn, self.measurement_info, self.rn)
        self.measurement_info = tree_at(lambda x: x.frequency, self.measurement_info, self.frequency)

        self.f0 = 0

        # self.lambda_lm = 1e-3
        # self.beta = 0.1


    def update_pulse(self, pulse, signal_t_new, gate_shifted, measurement_info, descent_info):
        """ Generates an new (maybe improoved) guess for the pulse. """
        pulse=jnp.sum(signal_t_new*jnp.conjugate(gate_shifted), axis=1)/(jnp.sum(jnp.abs(gate_shifted)**2, axis=1) + 1e-12)
        return pulse
    
    
    def update_gate(self, gate, signal_t_new, pulse_t_shifted, measurement_info, descent_info):
        """ Generates an new (maybe improoved) guess for the gate. """
        gate=jnp.sum(signal_t_new*jnp.conjugate(pulse_t_shifted), axis=1)/(jnp.sum(jnp.abs(pulse_t_shifted)**2, axis=1) + 1e-12)
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















class CPCGPA(ClassicAlgorithmsBASE, RetrievePulsesFROG):

    def __init__(self, delay, frequency, trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)
        assert self.ifrog==False, "PCGPA is not intended for interferometric measurements."

        self.name = "CPCGPA"
        
        self.idx_arr = jnp.arange(jnp.size(self.frequency))
        self.measurement_info = self.measurement_info.expand(idx_arr = self.idx_arr)

        self.constraints = False
        self.svd = False
        self.antialias = False


    def do_anti_alias(self,opf, half_N):
        opf = opf - jnp.tril(opf, -half_N) - jnp.triu(opf, half_N)
        return opf

    
    def calculate_opf(self, pulse_t, gate, pulse_t_prime, gate_prime, iteration, nonlinear_method, measurement_info):
        if nonlinear_method=="shg" or nonlinear_method=="thg":
            opf = jnp.outer(pulse_t, gate) + jnp.outer(pulse_t_prime, gate) + jnp.outer(pulse_t, gate_prime)
        elif nonlinear_method=="pg" or nonlinear_method=="sd":
            is_ac = 1 # (1-measurement_info.cross_correlation) & (1-measurement_info.doubleblind)
            opf = jnp.outer(pulse_t, gate.conj())
            opf = opf + (1-iteration%2)*(jnp.outer(pulse_t_prime, gate.conj()) + jnp.outer(pulse_t, gate_prime.conj()))*is_ac
        else:
            raise ValueError(f"nonlinear_method needs to be shg, thg, pg or sd. Not {nonlinear_method}")
        
        return opf


    @Partial(jax.vmap, in_axes=(None, 0, 0))
    def shift_rows(self, row, idx):
        return jnp.roll(row, idx)
    
    def convert_opf_to_signal_t(self, opf, idx_arr):
        temp = self.shift_rows(opf,-idx_arr)
        signal_t = jnp.roll(jnp.fliplr(jnp.fft.fftshift(temp,axes=1)), 1, axis=1)
        return signal_t
    

    def calculate_signal_t_using_opf(self, individual, iteration, measurement_info, descent_info):
        idx_arr = measurement_info.idx_arr

        pulse_t, pulse_t_prime = individual.pulse, individual.pulse_prime

        if measurement_info.doubleblind==True:
            gate, gate_prime = individual.gate, individual.gate_prime

        elif measurement_info.cross_correlation==True:
            gate = gate_prime = calculate_gate(measurement_info.gate, measurement_info.nonlinear_method)

        else:
            gate = calculate_gate(pulse_t, measurement_info.nonlinear_method)
            gate_prime = calculate_gate(pulse_t_prime, measurement_info.nonlinear_method)

        
        opf = self.calculate_opf(pulse_t, gate, pulse_t_prime, gate_prime, iteration, measurement_info.nonlinear_method, measurement_info)

        if descent_info.antialias==True:
            half_N = jnp.size(opf[0])//2
            opf = self.do_anti_alias(opf, half_N)


        signal_t = self.convert_opf_to_signal_t(opf, idx_arr)
        # transpose for consistency
        return jnp.transpose(signal_t)



    def convert_signal_t_to_opf(self, signal_t, idx_arr):
        signal_t = jnp.transpose(signal_t)
        signal_t = jnp.roll(signal_t, -1, axis=1)
        temp = jnp.fft.fftshift(jnp.fliplr(signal_t), axes=1)
        opf = self.shift_rows(temp, idx_arr)
        return opf


    def decompose_opf(self, opf, pulse_t, gate, measurement_info, descent_info):
        if descent_info.svd==True:
            U, S, Vh = jnp.linalg.svd(opf)
            pulse_t = U[:,0]

            if measurement_info.doubleblind==True:
                gate = Vh[0].conj()
            else:
                gate = None

        else:
            pulse_t = jnp.dot(opf, jnp.dot(opf.T.conj(), pulse_t))
            pulse_t = pulse_t/jnp.linalg.norm(pulse_t) # needed. otherwise amplitude goes to zero.

            if measurement_info.doubleblind==True:
                gate = jnp.dot(opf.T.conj(), jnp.dot(opf, gate))
                gate = gate/jnp.linalg.norm(gate) # needed. otherwise amplitude goes to zero.
                # is fine, since amplitudes factor out -> wouldnt be fine for ifrog
            else:
                gate = None

        return pulse_t, gate
    
    

    def impose_constraints(self, pulse_t, gate, opf, measurement_info):
        # these are the additional constraints in C-PCGPA
            # opf maps from gate to pulse_t_prime
            # opf^dagger maps from pulse_t to gate_prime

        nonlinear_method = measurement_info.nonlinear_method

        if measurement_info.cross_correlation==True:
            gate = calculate_gate(measurement_info.gate, nonlinear_method)
            pulse_t_prime = jnp.dot(opf, gate).astype(jnp.complex64)
            gate_prime = None

        elif measurement_info.doubleblind==True:
            # this is suggested by the c-pcgpa paper but im not sure its an actual improvement
            # if nonlinear_method=="pg":
            #     #gate = jnp.abs(gate)
            #     pulse_t_prime = jnp.dot(opf, jnp.abs(pulse_t)**2).astype(jnp.complex64)
            #     gate_prime = (jnp.abs(jnp.dot(opf, gate))**2).astype(jnp.complex64)
            # else:
            pulse_t_prime = jnp.dot(opf, gate).astype(jnp.complex64)
            gate_prime = jnp.dot(opf.T.conj(), pulse_t).astype(jnp.complex64)

        else:
            gate = calculate_gate(pulse_t, nonlinear_method)
            pulse_t_prime = jnp.dot(opf, gate).astype(jnp.complex64)
            gate_prime = None

        return pulse_t_prime, gate_prime



    def update_population(self, opf, individual, measurement_info, descent_info):
        pulse_t, gate = individual.pulse, individual.gate

        if measurement_info.cross_correlation==True:
            gate = calculate_gate(measurement_info.gate, measurement_info.nonlinear_method)
        elif measurement_info.doubleblind==True:
            pass
        else:
            pass
        

        pulse_t, gate = self.decompose_opf(opf, pulse_t, gate, measurement_info, descent_info)

        if descent_info.constraints==True:
            pulse_t_prime, gate_prime = self.impose_constraints(pulse_t, gate, opf, measurement_info)

        else:
            pulse_t_prime, gate_prime = pulse_t, gate

        # it seems more sensible to declare pulse_prime as pulse. Applying constraints should make guess more accurate
        return MyNamespace(pulse=pulse_t_prime, pulse_prime=pulse_t, gate=gate_prime, gate_prime=gate)




    def step(self, descent_state, measurement_info, descent_info):
        sk, rn, idx_arr, measured_trace = measurement_info.sk, measurement_info.rn, measurement_info.idx_arr, measurement_info.measured_trace
        population, iteration = descent_state.population, descent_state.iteration

        signal_t = jax.vmap(self.calculate_signal_t_using_opf, in_axes=(0,None,None,None))(population, iteration, measurement_info, descent_info)

        signal_f = self.fft(signal_t, sk, rn)
        trace = calculate_trace(signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)

        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,None,None,None,None,None))(signal_t, measured_trace, 1, measurement_info, descent_info, "_global")
        opf = jax.vmap(self.convert_signal_t_to_opf, in_axes=(0,None))(signal_t_new, idx_arr)

        if descent_info.antialias==True:
            half_N = jnp.size(opf[0])//2
            opf = self.do_anti_alias(opf, half_N)

        population = jax.vmap(self.update_population, in_axes=(0,0,None,None))(opf, population, measurement_info, descent_info)

        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x.iteration, descent_state, iteration+1)
        return descent_state, trace_error.reshape(-1,1)
    



    def initialize_run(self, population):
        measurement_info = self.measurement_info

        s_prime_params = initialize_S_prime_params(self)
        self.descent_info = self.descent_info.expand(svd=self.svd, 
                                                     constraints=self.constraints,
                                                     s_prime_params = s_prime_params,
                                                     antialias = self.antialias)
        descent_info = self.descent_info


        population = MyNamespace(pulse=population.pulse, pulse_prime=population.pulse,
                                 gate=population.gate, gate_prime=population.gate)
        self.descent_state = self.descent_state.expand(population = population, 
                                                       iteration = 0)

        descent_state = self.descent_state

        do_step = Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step = Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step
    


    def post_process_create_trace(self, individual):
        iteration = self.descent_state.iteration
        sk, rn = self.measurement_info.sk, self.measurement_info.rn

        individual = MyNamespace(pulse=individual.pulse, pulse_prime=individual.pulse, 
                                 gate=individual.gate, gate_prime=individual.gate)
        signal_t = self.calculate_signal_t_using_opf(individual, iteration, self.measurement_info, self.descent_info)
        signal_f = self.fft(signal_t, sk, rn)
        trace = calculate_trace(signal_f)
        return trace















class GeneralizedProjection(GeneralizedProjectionBASE, RetrievePulsesFROG):
    """
    The Generalized Projection Algorithm for FROG. Inherits from GeneralizedProjectionBASE and RetrievePulsesFROG.
    
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)


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













class TimeDomainPtychography(TimeDomainPtychographyBASE, RetrievePulsesFROG):
    """
    The Ptychographic Iterative Engine (PIE) for FROG. Inherits from TimeDomainPtychographyBASE and RetrievePulsesFROG.

    Attributes:
        pie_method: None or str, specifies the PIE variant. Can be one of None, PIE, ePIE, rPIE.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, pie_method="rPIE", cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)
        assert self.ifrog==False, "Dont use ifrog with PIE. its not meant or made for that"

        self.pie_method=pie_method


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
        else:
            raise NotImplementedError(f"nonlinear_method={nonlinear_method} is not available.")

        return grad_all_m


    def calculate_PIE_descent_direction_m(self, signal_t, signal_t_new, tau, population, pie_method, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the PIE direction for pulse or gate-pulse for a given shift. """
        alpha = descent_info.alpha

        difference_signal_t = signal_t_new - signal_t.signal_t

        if pulse_or_gate=="pulse":
            probe = signal_t.gate_shifted
            grad = -1*jnp.conjugate(probe)*difference_signal_t
            U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(probe, alpha, pie_method)
            
        elif pulse_or_gate=="gate":
            probe = jnp.broadcast_to(population.pulse, jnp.shape(difference_signal_t))
            grad = -1*jnp.conjugate(probe)*difference_signal_t
            U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(probe, alpha, pie_method)

            grad = self.modify_grad_for_gate_pulse(grad, signal_t.gate_pulse_shifted, measurement_info.nonlinear_method)

            U = self.reverse_transform_grad(U, tau, measurement_info)
            grad = self.reverse_transform_grad(grad, tau, measurement_info)

        return grad, U
    


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

        signal_f = self.fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        descent_direction, newton_state = PIE_get_pseudo_newton_direction(grad, probe, signal_f, tau_arr, measured_trace, reverse_transform, newton_direction_prev, 
                                                                     measurement_info, descent_info, pulse_or_gate, local_or_global)
        return descent_direction, newton_state
    
















class COPRA(COPRABASE, RetrievePulsesFROG):
    """
    The Common Pulse Retrieval Algorithm (COPRA) for FROG. Inherits from COPRABASE and RetrievePulsesFROG.
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)



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

