import jax
import jax.numpy as jnp

from jax.tree_util import Partial
from equinox import tree_at


from .stepsize import do_linesearch, adaptive_step_size
from .nonlinear_cg import get_nonlinear_CG_direction
from .lbfgs import get_quasi_newton_direction

from pulsedjax.utilities import scan_helper, MyNamespace, calculate_mu, calculate_trace, calculate_trace_error, calculate_Z_error, run_scan, do_checks_before_running
from .base_classes_algorithms import ClassicAlgorithmsBASE

from .construct_s_prime import calculate_S_prime






def initialize_CG_state(shape, measurement_info):
    init_arr = jnp.zeros(shape, dtype=jnp.complex64)

    cg_pulse = MyNamespace(CG_direction_prev = init_arr, 
                           descent_direction_prev = init_arr)

    if measurement_info.doubleblind==True:
        cg_gate = MyNamespace(CG_direction_prev = init_arr, 
                              descent_direction_prev = init_arr)
    else:
        cg_gate = None

    return MyNamespace(pulse=cg_pulse, gate=cg_gate)



def initialize_pseudo_newton_state(shape, measurement_info):
    init_arr1 = jnp.zeros(shape, dtype=jnp.complex64)

    newton_pulse = MyNamespace(newton_direction_prev=init_arr1)
    if measurement_info.doubleblind==True:
        newton_gate = MyNamespace(newton_direction_prev=init_arr1)
    else:
        newton_gate = None

    return MyNamespace(pulse=newton_pulse, gate=newton_gate)



def initialize_lbfgs_state(shape, measurement_info, descent_info):
    N = shape[0]
    n = shape[1]
    m = descent_info.newton.lbfgs_memory

    init_arr1 = jnp.zeros((N,m,n), dtype=jnp.complex64)
    init_arr2 = jnp.zeros((N,m,1), dtype=jnp.float32)

    lbfgs_init_pulse = MyNamespace(grad_prev = init_arr1, newton_direction_prev = init_arr1, step_size_prev = init_arr2)
    if measurement_info.doubleblind==True:
        lbfgs_init_gate = MyNamespace(grad_prev = init_arr1, newton_direction_prev = init_arr1, step_size_prev = init_arr2)
    else:
        lbfgs_init_gate = None
        
    return MyNamespace(pulse=lbfgs_init_pulse, gate=lbfgs_init_gate)


def update_lbfgs_state(lbfgs_state, gamma, grad, descent_direction):
    step_size_arr = lbfgs_state.step_size_prev
    step_size_arr = step_size_arr.at[:,1:].set(step_size_arr[:,:-1])
    step_size_arr = step_size_arr.at[:,0].set(gamma[:, jnp.newaxis])

    grad_arr = lbfgs_state.grad_prev
    grad_arr = grad_arr.at[:,1:].set(grad_arr[:,:-1])
    grad_arr = grad_arr.at[:,0].set(grad)

    newton_arr = lbfgs_state.newton_direction_prev
    newton_arr = newton_arr.at[:,1:].set(newton_arr[:,:-1])
    newton_arr = newton_arr.at[:,0].set(descent_direction)

    lbfgs_state = MyNamespace(grad_prev = grad_arr, 
                            newton_direction_prev = newton_arr,
                            step_size_prev = step_size_arr)
    return lbfgs_state



def initialize_linesearch_info(optimizer):
    linesearch_params = MyNamespace(linesearch=optimizer.linesearch, 
                                    c1=optimizer.c1, 
                                    c2=optimizer.c2, 
                                    max_steps=optimizer.max_steps_linesearch, 
                                    delta_gamma=optimizer.delta_gamma)
    return linesearch_params



def initialize_S_prime_params(optimizer):
    s_prime_params = MyNamespace(_local=optimizer.r_local_method, 
                                 _global=optimizer.r_global_method, 
                                 number_of_iterations=optimizer.r_no_iterations, 
                                 r_gradient=optimizer.r_gradient, 
                                 r_newton=optimizer.r_newton, 
                                 weights=optimizer.r_weights)
    return s_prime_params



def initialize_newton_info(optimizer):
    newton = MyNamespace(_local=optimizer.local_newton, 
                        _global=optimizer.global_newton, 
                        linalg_solver=optimizer.linalg_solver, 
                        lambda_lm=optimizer.lambda_lm,
                        lbfgs_memory=optimizer.lbfgs_memory)

    return newton




def initialize_descent_info(optimizer):
    linesearch_params = initialize_linesearch_info(optimizer)
    newton = initialize_newton_info(optimizer)
    s_prime_params = initialize_S_prime_params(optimizer)
    
    descent_info = optimizer.descent_info.expand(gamma = MyNamespace(_local=optimizer.local_gamma, _global=optimizer.global_gamma),
                                                 conjugate_gradients = optimizer.conjugate_gradients,
                                                 linesearch_params = linesearch_params,
                                                 newton = newton,
                                                 s_prime_params = s_prime_params,
                                                 xi = optimizer.xi,
                                                 adaptive_scaling = MyNamespace(_local=MyNamespace(order=optimizer.local_adaptive_scaling, 
                                                                                                   factor=optimizer.local_adaptive_scaling_factor), 
                                                                                _global=MyNamespace(order=optimizer.global_adaptive_scaling, 
                                                                                                    factor=optimizer.global_adaptive_scaling_factor)))
    return descent_info









class LSGPABASE(ClassicAlgorithmsBASE):
    # for chirp-scan one would end up with somehting related to the pie i think.
    """
    The Least-Squares Generalized Projection Algorithm.
    Only available for delay based non-interferometric methods.
     
    J. Gagnon et al., Appl. Phys. B 92, 25-32, 10.1007/s00340-008-3063-x (2008)
    
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)
        assert self.interferometric==False, "LSGPA is not intended for interferometric measurements."
        assert self.doubleblind==False, "LSGPA is not intended for doubelblind."

        self._name = "LSGPA"


    def update_pulse(self, pulse, signal_t_new, gate_shifted, measurement_info, descent_info):
        """ Generates an new (maybe improoved) guess for the pulse. """
        pulse=jnp.sum(signal_t_new*jnp.conjugate(gate_shifted), axis=1)/(jnp.sum(jnp.abs(gate_shifted)**2, axis=1) + 1e-12)
        return pulse
    
    
    def update_gate(self, gate, signal_t_new, pulse_t_shifted, measurement_info, descent_info):
        """ Generates an new (maybe improoved) guess for the gate. """
        gate=jnp.sum(signal_t_new*jnp.conjugate(pulse_t_shifted), axis=1)/(jnp.sum(jnp.abs(pulse_t_shifted)**2, axis=1) + 1e-12)
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
        #population_pulse = population_pulse/jnp.linalg.norm(population_pulse,axis=-1)[:,jnp.newaxis]
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population_pulse)


        if measurement_info.doubleblind==True:
            population_gate = self.update_gate(population.gate, signal_t_new, signal_t.pulse_t_shifted, measurement_info, descent_info)
            #population_gate = population_gate/jnp.linalg.norm(population_gate,axis=-1)[:,jnp.newaxis]
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








class CPCGPABASE(ClassicAlgorithmsBASE):
    """
    The Constrained-PCGP-Algorithms.
    Only available for delay based non-interferometric methods.

    D. J. Kane and A. B. Vakhtin, Prog. Quantum Electron. 81 (100364), 10.1016/j.pquantelec.2021.100364 (2022)

    Attributes:
        constraints (bool): if true the operator based constraints are used.
        svd (bool): if true a full SVD is performed instead of a single iteration of the power method
        antialias (bool): if true anti-aliasing is applied to the outer-product-matrix-form
    
    """
    def __init__(self, delay, frequency, trace, nonlinear_method, cross_correlation=False, constraints=False, svd=False, antialias=False, **kwargs):
        super().__init__(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)
        assert self.interferometric==False, "PCGPA is not intended for interferometric measurements."
        assert nonlinear_method!="sd", "Doesnt work for SD. Which is weird."

        self._name = "CPCGPA"
        
        self.idx_arr = jnp.arange(jnp.size(self.frequency))
        self.measurement_info = self.measurement_info.expand(idx_arr = self.idx_arr)

        self.constraints = constraints
        self.svd = svd
        self.antialias = antialias


    
    def get_spectral_amplitude(self, measured_frequency, measured_spectrum, pulse_or_gate):
        """ Used to provide a measured pulse spectrum. A spectrum for the gate pulse can also be provided. """

        if self.measurement_info.doubleblind==True:
            print("Actually for doubleblind, C-PCGPA probably performs better without spectrum constraints.")

        frequency = self.frequency
        f0 = frequency[jnp.argmax(jnp.sum(self.measured_trace, axis=0))]

        if pulse_or_gate=="pulse":
            f0_p = measured_frequency[jnp.argmax(jnp.abs(measured_spectrum))]

        elif pulse_or_gate=="gate" and self.descent_info.measured_spectrum_is_provided.pulse==True:
            f0_p = frequency[jnp.argmax(jnp.abs(self.measurement_info.spectral_amplitude.pulse))]

        elif pulse_or_gate=="gate" and self.descent_info.measured_spectrum_is_provided.pulse==False:
            raise ValueError(f"For C-PCGPA you must provide a spectrum for the pulse first.")
        else:
            raise ValueError(f"pulse_or_gate needs to be pulse or gate. Not {pulse_or_gate}")
        
        return super().get_spectral_amplitude(measured_frequency+(f0-f0_p), measured_spectrum, pulse_or_gate)
    



    
    def calculate_opf(self, pulse_t, gate, pulse_t_prime, gate_prime, iteration, nonlinear_method, measurement_info):
        """ Calculates the opf given a pulse and gate. """
        if nonlinear_method=="shg" or nonlinear_method=="thg" or nonlinear_method[-2:]=="hg":
            opf = jnp.outer(pulse_t, gate) + jnp.outer(pulse_t_prime, gate) + jnp.outer(pulse_t, gate_prime)
        elif nonlinear_method=="pg" or nonlinear_method=="sd":
            opf = jnp.outer(pulse_t, gate)
            opf = opf + (1-iteration%2)*(jnp.outer(pulse_t_prime, gate) + jnp.outer(pulse_t, gate_prime))
        # elif nonlinear_method=="sd":
        #     opf = jnp.outer(pulse_t, gate)# + 
        #     #opf = jnp.outer(pulse_t_prime, gate) + jnp.outer(pulse_t, gate_prime)
        #     #opf = opf + (1-iteration%2)*(jnp.outer(pulse_t_prime, gate) + jnp.outer(pulse_t, gate_prime))
        else:
            raise ValueError(f"nonlinear_method needs to be shg, thg, pg or sd. Not {nonlinear_method}")
        
        return opf
    

    
    def calculate_signal_t_using_opf(self, individual, iteration, measurement_info, descent_info):
        """ Calculates signal_t for and individual via the opf. """
        idx_arr = measurement_info.idx_arr

        pulse_t, pulse_t_prime = individual.pulse, individual.pulse_prime

        if measurement_info.doubleblind==True:
            gate, gate_prime = individual.gate, individual.gate_prime

        elif measurement_info.cross_correlation==True:
            gate = gate_prime = self.calculate_gate(measurement_info.gate, measurement_info)

        else:
            gate = self.calculate_gate(pulse_t, measurement_info)
            gate_prime = self.calculate_gate(pulse_t_prime, measurement_info)

        
        opf = self.calculate_opf(pulse_t, gate, pulse_t_prime, gate_prime, iteration, measurement_info.nonlinear_method, measurement_info)

        if descent_info.antialias==True:
            half_N = jnp.size(opf[0])//2
            opf = self.do_anti_alias(opf, half_N)

        signal_t = self.convert_opf_to_signal_t(opf, idx_arr)
        signal_t = jnp.transpose(signal_t) # transpose for consistency
        signal_f = self.fft(signal_t, measurement_info.sk, measurement_info.rn)
        return MyNamespace(signal_t=signal_t, signal_f=signal_f)
    

    def do_anti_alias(self, opf, half_N):
        """ Performs anti-aliasing to the opf by setting a lower and upper an triangle to zero. """
        opf = opf - jnp.tril(opf, -half_N) - jnp.triu(opf, half_N)
        return opf


    @Partial(jax.vmap, in_axes=(None, 0, 0))
    def shift_rows(self, row, idx):
        return jnp.roll(row, idx)
    
    def convert_opf_to_signal_t(self, opf, idx_arr):
        """ Transforms opf to signal field, by shifting along the time axis. Switching and flipping the two halfs around. """
        temp = self.shift_rows(opf, -idx_arr)
        signal_t = jnp.roll(jnp.fliplr(jnp.fft.fftshift(temp,axes=1)), 1, axis=1)
        return signal_t
    

    def convert_signal_t_to_opf(self, signal_t, idx_arr):
        """ Converts a signal field into an opf by reversing the operations from  convert_opf_to_signal_t(). """
        signal_t = jnp.transpose(signal_t) # is needed since calculate_signal_t_using_opf() applies a transpose.
        signal_t = jnp.roll(signal_t, -1, axis=1)
        temp = jnp.fft.fftshift(jnp.fliplr(signal_t), axes=1)
        opf = self.shift_rows(temp, idx_arr)
        return opf


    def decompose_opf(self, opf, pulse_t, gate, measurement_info, descent_info):
        """ Decomposes the opf into its dominant components via an SVD or the Power-Method. """
        if descent_info.svd==True:
            U, S, Vh = jnp.linalg.svd(opf)
            pulse_t = U[:,0]

            if measurement_info.doubleblind==True:
                gate = Vh[0].conj()
            else:
                gate = None

        else:
            # if measurement_info.nonlinear_method=="sd":
            #     pulse_t = jnp.dot(opf.conj, jnp.dot(opf.T.conj(), pulse_t))
            # else:
            pulse_t = jnp.dot(opf, jnp.dot(opf.T.conj(), pulse_t))
            pulse_t = pulse_t/jnp.linalg.norm(pulse_t) # needed. otherwise amplitude goes to zero.

            if measurement_info.doubleblind==True:
                gate = jnp.dot(opf.T.conj(), jnp.dot(opf, gate))
                gate = gate/jnp.linalg.norm(gate) # needed. otherwise amplitude goes to zero.
                # is fine, since amplitudes factor out -> wouldnt be fine for interferometric
            else:
                gate = None

        return pulse_t, gate
    
    

    def impose_constraints(self, pulse_t, gate, opf, measurement_info):
        """ Applies additional constraints according to the operator formalism of PCGP. """
        # these are the additional constraints in C-PCGPA
            # opf maps from gate to pulse_t_prime
            # opf^dagger maps from pulse_t to gate_prime

        if measurement_info.cross_correlation==True:
            gate = self.calculate_gate(measurement_info.gate, measurement_info)
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
            gate = self.calculate_gate(pulse_t, measurement_info)
            pulse_t_prime = jnp.dot(opf, gate).astype(jnp.complex64)
            gate_prime = None

        return pulse_t_prime, gate_prime



    def update_individual(self, opf, individual, measurement_info, descent_info):
        """ Updates and individual using an updated opf. """
        pulse_t, gate = individual.pulse, individual.gate

        if measurement_info.cross_correlation==True:
            gate = self.calculate_gate(measurement_info.gate, measurement_info)
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
        """ 
        Performs one iteration of the C-PCGP Algorithm. 

        Args:
            descent_state: Pytree,
            measurement_info: Pytree,
            descent_info: Pytree,
        
        Returns:
            tuple[Pytree, jnp.array], the updated descent state and the current errors

        """
        idx_arr, measured_trace = measurement_info.idx_arr, measurement_info.measured_trace
        population, iteration = descent_state.population, descent_state.iteration

        signal_t = jax.vmap(self.calculate_signal_t_using_opf, in_axes=(0,None,None,None))(population, iteration, measurement_info, descent_info)
        trace = calculate_trace(signal_t.signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)

        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,0,None,None,None,None,None))(signal_t.signal_t, signal_t.signal_f, measured_trace, 1, measurement_info, descent_info, "_global")
        opf = jax.vmap(self.convert_signal_t_to_opf, in_axes=(0,None))(signal_t_new, idx_arr)

        if descent_info.antialias==True:
            half_N = jnp.size(opf[0])//2
            opf = self.do_anti_alias(opf, half_N)

        population = jax.vmap(self.update_individual, in_axes=(0,0,None,None))(opf, population, measurement_info, descent_info)

        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x.iteration, descent_state, iteration+1)
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
        self.descent_info = self.descent_info.expand(svd = self.svd, 
                                                     constraints = self.constraints,
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
        """ For PCGP the trace is constructed using the opf. """
        iteration = self.descent_state.iteration
        individual = MyNamespace(pulse=individual.pulse, pulse_prime=individual.pulse, 
                                 gate=individual.gate, gate_prime=individual.gate)
        signal_t = self.calculate_signal_t_using_opf(individual, iteration, self.measurement_info, self.descent_info)
        trace = calculate_trace(signal_t.signal_f)
        return trace









class GeneralizedProjectionBASE(ClassicAlgorithmsBASE):
    """
    Implements the Generalized Projection Algorithm.

    K. W. DeLong et al., Opt. Lett. 19, 2152-2154 (1994) 

    
    Attributes:
        no_steps_descent (int): the numer of descent steps per iteration

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._name = "GeneralizedProjection"

        self.local_gamma = None
        self.global_gamma = 1

        self.no_steps_descent = 15

        self.r_local_method = None


    
    def update_population(self, population, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Applies the descent based update to the population. """
        population = jax.vmap(self.update_individual, in_axes=(0,0,0,None,None))(population, gamma, descent_direction, measurement_info, pulse_or_gate)
        return population
    

    
    def get_Z_gradient(self, signal_t, signal_t_new, population, transform_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for the entire population. """
        grad = jax.vmap(self.calculate_Z_gradient_individual, in_axes=(0, 0, 0, 0, None, None))(signal_t, signal_t_new, population, transform_arr, 
                                                                                                measurement_info, pulse_or_gate)
        return grad

    

    def calc_Z_error_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        """ Calculates the Z-error such that it can be called in a linesearch. """
        individual, descent_direction, signal_t_new = linesearch_info.population, linesearch_info.descent_direction, linesearch_info.signal_t_new
       
        transform_arr = measurement_info.transform_arr

        individual = self.update_individual(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        Z_error_new = calculate_Z_error(signal_t.signal_t, signal_t_new)
        return Z_error_new
    

    def calc_Z_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient such that it can be called in a linesearch. """
        individual, descent_direction, signal_t_new = linesearch_info.population, linesearch_info.descent_direction, linesearch_info.signal_t_new

        transform_arr = measurement_info.transform_arr

        individual = self.update_individual(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        grad = self.calculate_Z_gradient_individual(signal_t, signal_t_new, individual, transform_arr, measurement_info, pulse_or_gate)
        return jnp.sum(grad, axis=0)


    def descent_Z_error_step(self, signal_t, signal_t_new, Z_error, descent_state, measurement_info, descent_info, pulse_or_gate): 
        """ 
        Performs a descent step in order to minimize the Z-error. 
        Employs gradient descent, nonlinear conjugate gradients, LBFGS or damped Newtons method (diagonal or full).
        The step size is determined via a fixed/adaptive step size, a backtracking or a zoom linesearch.
        """       

        newton_info, conjugate_gradients = descent_info.newton._global, descent_info.conjugate_gradients

        population = descent_state.population
        transform_arr = measurement_info.transform_arr
        transform_arr = jnp.broadcast_to(transform_arr, (descent_info.population_size, ) + jnp.shape(transform_arr))

        grad = self.get_Z_gradient(signal_t, signal_t_new, population, transform_arr, measurement_info, pulse_or_gate)
        grad_sum = jnp.sum(grad, axis=1)


        if newton_info=="diagonal" or newton_info=="full":
            descent_direction, newton_state = self.calculate_Z_newton_direction(grad, signal_t_new, signal_t, transform_arr, descent_state, 
                                                                                       measurement_info, descent_info, newton_info, pulse_or_gate)
            descent_state = tree_at(lambda x: getattr(x.newton, pulse_or_gate), descent_state, newton_state)

        elif newton_info=="lbfgs":
            lbfgs_state = getattr(descent_state.lbfgs, pulse_or_gate)
            descent_direction, lbfgs_state = get_quasi_newton_direction(grad_sum, lbfgs_state, descent_info)

        else:
            descent_direction = -1*grad_sum


        if conjugate_gradients!=False:
            cg = getattr(descent_state.cg, pulse_or_gate)
            descent_direction, cg =jax.vmap(get_nonlinear_CG_direction, in_axes=(0,0,None))(descent_direction, cg, conjugate_gradients)
            descent_state = tree_at(lambda x: getattr(x.cg, pulse_or_gate), descent_state, cg)

        adaptive_scaling_info = getattr(descent_info.adaptive_scaling, "_global")
        if adaptive_scaling_info.order!=False:
            descent_direction, descent_state = jax.vmap(adaptive_step_size, in_axes=(0,0,0,None,None,None,None,None), out_axes=(0,None))(Z_error, grad_sum, descent_direction,
                                                                                                                                    descent_info.xi,
                                                                                                                    descent_state,
                                                                                                                    adaptive_scaling_info,
                                                                                                                    pulse_or_gate, "_global")

        if descent_info.linesearch_params.linesearch!=False:
            pk_dot_gradient = jax.vmap(lambda x,y: jnp.real(jnp.vdot(x,y)), in_axes=(0,0))(descent_direction, grad_sum)
            
            linesearch_info=MyNamespace(population=population, descent_direction=descent_direction, signal_t_new=signal_t_new, 
                                        error=Z_error, pk_dot_gradient=pk_dot_gradient)
            
            gamma = jax.vmap(do_linesearch, in_axes=(0,None,None,None,None,None))(linesearch_info, measurement_info, descent_info, 
                                                                             Partial(self.calc_Z_error_for_linesearch, pulse_or_gate=pulse_or_gate),
                                                                             Partial(self.calc_Z_grad_for_linesearch, pulse_or_gate=pulse_or_gate), "_global")
        else:
            gamma = descent_info.gamma._global
            if jnp.size(gamma)==1:
                gamma = jnp.broadcast_to(gamma, (descent_info.population_size, ))
            elif jnp.size(gamma)==descent_info.population_size:
                pass
            else:
                raise ValueError(f"Size of gamma has to be 1 or the population size. Not {jnp.size(gamma)}")

        if newton_info=="lbfgs":
            lbfgs_state = update_lbfgs_state(lbfgs_state, gamma, grad_sum, descent_direction)
            descent_state = tree_at(lambda x: getattr(x.lbfgs, pulse_or_gate), descent_state, lbfgs_state)

        population = self.update_population(population, gamma, descent_direction, measurement_info, pulse_or_gate) 
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        return descent_state
    



    def do_descent_Z_error_step(self, descent_state, signal_t_new, measurement_info, descent_info):
        """ Does one Z-error descent step. Calls descent_Z_error_step for pulse and or gate. """
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        Z_error = jax.vmap(calculate_Z_error, in_axes=(0,0))(signal_t.signal_t, signal_t_new)

        descent_state = self.descent_Z_error_step(signal_t, signal_t_new, Z_error, descent_state, measurement_info, descent_info, "pulse")
        population_pulse = descent_state.population.pulse/jnp.linalg.norm(descent_state.population.pulse,axis=-1)[:,jnp.newaxis]
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population_pulse)

        if measurement_info.doubleblind==True:
            descent_state=self.descent_Z_error_step(signal_t, signal_t_new, Z_error, descent_state, measurement_info, descent_info, "gate")

            if measurement_info.interferometric==False:
                population_gate = descent_state.population.gate/jnp.linalg.norm(descent_state.population.gate,axis=-1)[:,jnp.newaxis]
                descent_state = tree_at(lambda x: x.population.gate, descent_state, population_gate)

        return descent_state, None



    def do_descent_Z_error(self, descent_state, signal_t_new, measurement_info, descent_info):
        """ Performs a descent based optimization to find the pulse/gate that are able to produce S_prime. """
        
        shape_pulse = jnp.shape(descent_state.population.pulse)
        cg_state = initialize_CG_state(shape_pulse, measurement_info)
        newton_state = initialize_pseudo_newton_state(shape_pulse, measurement_info)
        lbfgs_state = initialize_lbfgs_state(shape_pulse, measurement_info, descent_info)
        descent_state = tree_at(lambda x: x.cg, descent_state, cg_state)
        descent_state = tree_at(lambda x: x.newton, descent_state, newton_state)
        descent_state = tree_at(lambda x: x.lbfgs, descent_state, lbfgs_state)
        
        do_gradient_descent_step = Partial(self.do_descent_Z_error_step, signal_t_new=signal_t_new, measurement_info=measurement_info, descent_info=descent_info)
        do_gradient_descent_step = Partial(scan_helper, actual_function=do_gradient_descent_step, number_of_args=1, number_of_xs=0)
        descent_state, _ = jax.lax.scan(do_gradient_descent_step, descent_state, length=descent_info.no_steps_descent)

        return descent_state
    
    

    def step(self, descent_state, measurement_info, descent_info):
        """
        Performs one iteration of the Generalized Projection Algorithm.
        
        Args:
            descent_state (Pytree):
            measurement_info (Pytree):
            descent_info (Pytree):

        Returns:
            tuple[Pytree, jnp.array], the updated descent state and the current trace errors of the population.
        """
        measured_trace = measurement_info.measured_trace
        sk, rn = measurement_info.sk, measurement_info.rn

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(signal_t.signal_f)

        mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,0,None,0,None,None,None))(signal_t.signal_t, signal_t.signal_f, measured_trace, mu, measurement_info, descent_info, "_global")

        descent_state = self.do_descent_Z_error(descent_state, signal_t_new, measurement_info, descent_info)

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(signal_t.signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)

        return descent_state, trace_error.reshape(-1,1)
    


    def initialize_run(self, population):
        """
        Prepares all provided data and parameters for the reconstruction. 
        Here the final shape/structure of descent_state, measurement_info and descent_info are determined. 

        Args:
            population (Pytree): the initial guess as created by self.create_initial_population()
        
        Returns:
            tuple[Pytree, Callable], the initial descent state and the step-function of the algorithm.

        """

        measurement_info = self.measurement_info

        self.descent_info = initialize_descent_info(self).expand(no_steps_descent = self.no_steps_descent)
        descent_info = self.descent_info

        self.descent_state = self.descent_state.expand(population = population)

        shape_pulse = jnp.shape(self.descent_state.population.pulse)
        cg_state = initialize_CG_state(shape_pulse, measurement_info)
        newton_state = initialize_pseudo_newton_state(shape_pulse, measurement_info)
        lbfgs_state = initialize_lbfgs_state(shape_pulse, measurement_info, descent_info)
        self.descent_state = self.descent_state.expand(cg = cg_state, newton=newton_state, lbfgs=lbfgs_state)

        descent_state = self.descent_state

        do_step = Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step = Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step
    
















class PtychographicIterativeEngineBASE(ClassicAlgorithmsBASE):
    """
    Implements a version of the Ptychographic Iterative Engine (PIE).

    A. Maiden et al., Optica 4, 736-745 (2017) 
    T. Schweizer, "Time-Domain Ptychography and its Applications in Ultrafast Science", PhD Thesis, Bern (2021)

    Attributes:
        alpha (float): a regularization parameter
        pie_method (None, str): specifies the PIE variant. Can be one of None, PIE, ePIE, rPIE. Where None indicates that the pure gradient is used.

    """

    def __init__(self, *args, pie_method="rPIE", **kwargs):
        super().__init__(*args, **kwargs)

        self._name = "PtychographicIterativeEngine"
        self.alpha = 0.5
        self.pie_method = pie_method



    def calculate_PIE_error(self, signal_f, measured_trace):
        """ Calculates the normalized least-squares error using the amplitude residuals. """
        return jnp.mean(jnp.abs(jnp.sqrt(jnp.abs(measured_trace))*jnp.sign(measured_trace) - jnp.abs(signal_f))**2)


    def get_PIE_weights(self, probe, alpha, pie_method):
        """ Calculates the weight-functions for the differen PIE-version. """

        #U=2/(jnp.abs(probe_shifted)**2+1e-6) # -> rPIE is eqivalent to pseudo-gauss-newton/levenberg-marquardt for small gamma. 
        #gamma=>1 -> rPIE=>ePIE

        p2 = jnp.abs(probe)**2
        if pie_method=="PIE":
            U = 1/(p2 + alpha*jnp.max(p2))*jnp.abs(probe)/jnp.max(jnp.abs(probe))

        elif pie_method=="ePIE":
            U = jnp.ones(jnp.shape(probe))/jnp.max(p2)

        elif pie_method=="rPIE":
            U = 1/((1-alpha)*p2 + alpha*jnp.max(p2))

        elif pie_method==None:
            U = jnp.ones(jnp.shape(probe))

        else:
            raise ValueError(f"pie_method needs to be one of PIE, ePIE, rPIE or None. Not {pie_method}")
        
        return U


    def update_population(self, population, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Applies the PIE update to the population. """
        population = jax.vmap(self.update_individual, in_axes=(0,0,0,None,None))(population, gamma, descent_direction, measurement_info, pulse_or_gate)
        return population
    


    def calculate_PIE_descent_direction(self, population, signal_t, signal_t_new, transform_arr, measured_trace, pie_method, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the descent direction based on the PIE version. """
        get_descent_direction = Partial(self.calculate_PIE_descent_direction_m, population=population, pie_method=pie_method, 
                                        measurement_info=measurement_info, descent_info=descent_info, pulse_or_gate=pulse_or_gate)

        grad_U = get_descent_direction(signal_t, signal_t_new, transform_arr, measured_trace)
        return grad_U






    def calc_error_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        """ Calculates the PIE-error such that it can be called in a linesearch. """

        transform_arr, measured_trace = linesearch_info.transform_arr, linesearch_info.measured_trace
        individual, descent_direction = linesearch_info.population, linesearch_info.descent_direction

        individual = self.update_individual(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        error_new = self.calculate_PIE_error(signal_t.signal_f, measured_trace)
        return error_new
    


    def calc_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate, local_or_global):
        """ Calculates the PIE direction such that it can be called in a linesearch. """
        transform_arr, measured_trace = linesearch_info.transform_arr[jnp.newaxis, ... ], linesearch_info.measured_trace[jnp.newaxis, ... ]
        individual, descent_direction = linesearch_info.population, linesearch_info.descent_direction
        
        individual = self.update_individual(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(None,0,None))(individual, transform_arr, measurement_info)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,0,0,None,None,None, None))(signal_t.signal_t,signal_t.signal_f, measured_trace, 1, measurement_info, 
                                                                                       descent_info, local_or_global)

        grad_U = self.calculate_PIE_descent_direction(individual, signal_t, signal_t_new, transform_arr, measured_trace, descent_info.pie_method, 
                                                             measurement_info, descent_info, pulse_or_gate)
        return jnp.sum(grad_U, axis=1)
    


    
    def do_iteration(self, signal_t, signal_t_new, transform_arr, measured_trace, pie_error, population, local_or_global_state, 
                     measurement_info, descent_info, pulse_or_gate, local_or_global):
        
        """ 
        Performs one local/global iteration of the PIE. 
        On top of the different PIE-version nonlinear conjugate gradients, LBFGS or damped Newtons method (diagonal or full) may be used.
        The step size is determined via a fixed/adaptive step size, a backtracking or a zoom linesearch.

        Newtons method with a full newton is not available for the reconstruction of the gate.
        """
        

        if local_or_global=="_global":
            N = descent_info.population_size
            shape = (N,) + jnp.shape(measured_trace)
            measured_trace = jnp.broadcast_to(measured_trace, shape)
            shape = (N, ) + jnp.shape(transform_arr)
            transform_arr = jnp.broadcast_to(transform_arr, shape)
            
        

        pie_method = descent_info.pie_method
        conjugate_gradients = descent_info.conjugate_gradients
        newton_info = getattr(descent_info.newton, local_or_global)

        grad_U = self.calculate_PIE_descent_direction(population, signal_t, signal_t_new, transform_arr, measured_trace, pie_method, measurement_info, descent_info, pulse_or_gate)
        grad_sum = jnp.sum(grad_U, axis=1)

        if newton_info=="diagonal" or (newton_info=="full" and pulse_or_gate=="pulse"):
            descent_direction, newton_state = self.calculate_PIE_newton_direction(grad_U, signal_t, transform_arr, measured_trace, population, local_or_global_state, 
                                                                                   measurement_info, descent_info, pulse_or_gate, local_or_global)
            local_or_global_state = tree_at(lambda x: getattr(x.newton, pulse_or_gate), local_or_global_state, newton_state)

        elif newton_info=="lbfgs":
            lbfgs_state = getattr(local_or_global_state.lbfgs, pulse_or_gate)
            descent_direction, lbfgs_state = get_quasi_newton_direction(grad_sum, lbfgs_state, descent_info)

        else:
            descent_direction = -1*grad_sum #-1*jnp.sum(grad*U, axis=1)



        if conjugate_gradients!=False:
            cg = getattr(local_or_global_state.cg, pulse_or_gate)
            descent_direction, cg = jax.vmap(get_nonlinear_CG_direction, in_axes=(0,0,None))(descent_direction, cg, conjugate_gradients)
            local_or_global_state = tree_at(lambda x: getattr(x.cg, pulse_or_gate), local_or_global_state, cg)


        adaptive_scaling_info = getattr(descent_info.adaptive_scaling, local_or_global)
        if adaptive_scaling_info.order!=False:
            descent_direction, local_or_global_state = jax.vmap(adaptive_step_size, in_axes=(0,0,0,None,0,None,None,None))(pie_error, grad_sum, descent_direction, 
                                                                                                                      descent_info.xi,
                                                                                                                            local_or_global_state, 
                                                                                                                            adaptive_scaling_info,
                                                                                                                            pulse_or_gate, local_or_global)


        if descent_info.linesearch_params.linesearch!=False and local_or_global=="_global":
            pk_dot_gradient=jax.vmap(lambda x,y: jnp.real(jnp.vdot(x,y)), in_axes=(0,0))(descent_direction, grad_sum)

            linesearch_info=MyNamespace(population=population, signal_t=signal_t, descent_direction=descent_direction, 
                                        pk_dot_gradient=pk_dot_gradient, error=pie_error,
                                        transform_arr=transform_arr, measured_trace=measured_trace)     

            gamma = jax.vmap(do_linesearch, in_axes=(0, None, None, None, None, None))(linesearch_info, measurement_info, descent_info, 
                                                                                Partial(self.calc_error_for_linesearch, pulse_or_gate=pulse_or_gate),
                                                                                Partial(self.calc_grad_for_linesearch, descent_info=descent_info, 
                                                                                        pulse_or_gate=pulse_or_gate, local_or_global=local_or_global), 
                                                                                local_or_global)
            
        else:
            gamma = jnp.broadcast_to(getattr(descent_info.gamma, local_or_global), (descent_info.population_size, ))
            if jnp.size(gamma)==1:
                gamma = jnp.broadcast_to(gamma, (descent_info.population_size, ))
            elif jnp.size(gamma)==descent_info.population_size:
                pass
            else:
                raise ValueError(f"Size of {local_or_global} gamma has to be 1 or the population size. Not {jnp.size(gamma)}")


        if newton_info=="lbfgs":
            lbfgs_state = update_lbfgs_state(lbfgs_state, gamma, grad_sum, descent_direction)
            local_or_global_state = tree_at(lambda x: getattr(x.lbfgs, pulse_or_gate), local_or_global_state, lbfgs_state)

        population = self.update_population(population, gamma, descent_direction, measurement_info, pulse_or_gate)
        return local_or_global_state, population



    

    def local_iteration(self, descent_state, transform_arr_m, trace_line, measurement_info, descent_info):
        """ Peforms one local iteration. Calls do_iteration() with the appropriate (randomized) signal fields. """
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,0,0,None,None,None,None))(signal_t.signal_t,signal_t.signal_f, trace_line, 1, measurement_info, descent_info, "_local")

        pie_error = jax.vmap(self.calculate_PIE_error, in_axes=(0,None))(signal_t.signal_f, trace_line)

        local_state, population = descent_state._local, descent_state.population
        local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, trace_line, pie_error, population, local_state, 
                                                      measurement_info, descent_info, "pulse", "_local")

        # population_pulse = jax.vmap(lambda x,y: x/jnp.linalg.norm(x)*jnp.linalg.norm(y))(population.pulse, signal_t_new)
        # population = tree_at(lambda x: x.pulse, population, population_pulse)

        if measurement_info.doubleblind==True:
            local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, trace_line, pie_error, population, local_state, 
                                                      measurement_info, descent_info, "gate", "_local")
            
            # if measurement_info.interferometric==False:
            #     population_gate = jax.vmap(lambda x: x/jnp.linalg.norm(x))(population.gate)
            #     population = tree_at(lambda x: x.gate, population, population_gate)
        
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x._local, descent_state, local_state)
        return descent_state, None
    


    def local_step(self, descent_state, measurement_info, descent_info):
        """
        Performs one local iteration of the PIE. 
        This means the method loops over the randomized measurement data once and updates the population using each data point individually.
        
        Args:
            descent_state (Pytree):
            measurement_info (Pytree):
            descent_info (Pytree):

        Returns:
            tuple[Pytree, jnp.array], the updated descent state and the current trace errors of the population.
        """

        transform_arr, measured_trace, descent_state = self.shuffle_data_along_m(descent_state, measurement_info, descent_info)

        local_iteration=Partial(self.local_iteration, measurement_info=measurement_info, descent_info=descent_info)
        local_iteration=Partial(scan_helper, actual_function=local_iteration, number_of_args=1, number_of_xs=2)

        descent_state, _ = jax.lax.scan(local_iteration, descent_state, (transform_arr, measured_trace))


        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace=calculate_trace(signal_t.signal_f)
        trace_error=jax.vmap(calculate_trace_error, in_axes=(0, None))(trace, measurement_info.measured_trace)

        return descent_state, trace_error.reshape(-1,1)
    



    
    
    def global_step(self, descent_state, measurement_info, descent_info):
        """
        Performs one global iteration of the PIE. 
        This means the method updates the population once using all measured data at once.
        
        Args:
            descent_state (Pytree):
            measurement_info (Pytree):
            descent_info (Pytree):

        Returns:
            tuple[Pytree, jnp.array], the updated descent state and the current trace errors of the population.
        """

        measured_trace = measurement_info.measured_trace

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,0,None,None,None,None,None))(signal_t.signal_t,signal_t.signal_f, measured_trace, 1, measurement_info, 
                                                                                         descent_info, "_global")
        
        pie_error = jax.vmap(self.calculate_PIE_error, in_axes=(0,None))(signal_t.signal_f, measured_trace)

        global_state, population = descent_state._global, descent_state.population 
        global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, measured_trace, pie_error, 
                                                      population, global_state, measurement_info, descent_info, "pulse", "_global")
        
        # population_pulse = population.pulse/jnp.linalg.norm(population.pulse,axis=-1)[:,jnp.newaxis]*jnp.linalg.norm(signal_t_new,axis=(-2,-1))[:,jnp.newaxis]
        # population = tree_at(lambda x: x.pulse, population, population_pulse)

        if measurement_info.doubleblind==True:
            global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, measured_trace, pie_error, 
                                                          population, global_state, measurement_info, descent_info, "gate", "_global")

            # if measurement_info.interferometric==True:
            #     population_gate = population.gate/jnp.linalg.norm(population.gate,axis=-1)[:,jnp.newaxis]
            #     population = tree_at(lambda x: x.gate, population, population_gate)
        

        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x._global, descent_state, global_state)

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace=calculate_trace(signal_t.signal_f)
        trace_error=jax.vmap(calculate_trace_error, in_axes=(0, None))(trace, measured_trace)

        return descent_state, trace_error.reshape(-1,1)






    def initialize_run(self, population):
        """
        Prepares all provided data and parameters for the reconstruction. 
        Here the final shape/structure of descent_state, measurement_info and descent_info are determined. 

        Args:
            population (Pytree): the initial guess as created by self.create_initial_population()
        
        Returns:
            tuple[Pytree, Callable, Callable], the initial descent state, the local and global step-functions of the algorithm.

        """

        measurement_info = self.measurement_info

        self.descent_info = initialize_descent_info(self).expand(alpha = self.alpha,
                                                                 pie_method = self.pie_method)
        descent_info = self.descent_info

        shape = jnp.shape(population.pulse)
        cg_state_local = initialize_CG_state(shape, measurement_info)
        newton_state_local = initialize_pseudo_newton_state(shape, measurement_info)
        lbfgs_state_local = initialize_lbfgs_state(shape, measurement_info, descent_info)

        cg_state_global = initialize_CG_state(shape, measurement_info)
        newton_state_global = initialize_pseudo_newton_state(shape, measurement_info)
        lbfgs_state_global = initialize_lbfgs_state(shape, measurement_info, descent_info)

        init_arr = jnp.zeros(shape[0])
        self.descent_state = self.descent_state.expand(key = self.key, 
                                                       population = population, 
                                                       _local=MyNamespace(cg=cg_state_local, newton=newton_state_local, lbfgs=lbfgs_state_local, 
                                                                          max_scaling = MyNamespace(pulse=init_arr, gate=init_arr)),
                                                       _global=MyNamespace(cg=cg_state_global, newton=newton_state_global, lbfgs=lbfgs_state_global))
    
        descent_state=self.descent_state

        do_local=Partial(self.local_step, measurement_info=measurement_info, descent_info=descent_info)
        do_local=Partial(scan_helper, actual_function=do_local, number_of_args=1, number_of_xs=0)

        do_global=Partial(self.global_step, measurement_info=measurement_info, descent_info=descent_info)
        do_global=Partial(scan_helper, actual_function=do_global, number_of_args=1, number_of_xs=0)
        return descent_state, do_local, do_global
    



    def run(self, population, no_iterations_local, no_iterations_global, **kwargs):
        """ 
        The Algorithm can use a local and a global sequentially.
        
        Args:
            population (Pytree): the initial guess
            no_iterations_local: int, the number of local iterations. Accepts zero as a value.
            no_iterations_global: int, the number of globale iterations. Accepts zero as a value.

        Returns:
            Pytree, the final result
        """

        do_checks_before_running(self, **kwargs)

        descent_state, do_local, do_global = self.initialize_run(population)

        descent_state, error_arr_local = run_scan(do_local, descent_state, no_iterations_local, self.jit)
        descent_state, error_arr_global = run_scan(do_global, descent_state, no_iterations_global, self.jit)

        error_arr = jnp.concatenate([error_arr_local, error_arr_global], axis=0)
        error_arr = jnp.squeeze(error_arr)

        final_result = self.post_process(descent_state, error_arr)
        return final_result
    













class COPRABASE(ClassicAlgorithmsBASE):
    """
    Implements a version of the Common Pulse Retrieval Algorithm (COPRA).

    N. C. Geib et al., Optica 6, 495-505 (2019) 

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._name = "COPRA"

        self.global_gamma = 0.25
        self.r_global_method = "iteration"

        self.local_adaptive_scaling = "linear"
        self.global_adaptive_scaling = "linear"



    def update_population(self, population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        """ Applies the a descent based update to the population. """
        population = jax.vmap(self.update_individual, in_axes=(0,0,0,None,None,None))(population, gamma, descent_direction, 
                                                                                      measurement_info, descent_info, pulse_or_gate)
        return population
    

    

    def get_Z_gradient(self, signal_t, signal_t_new, population, transform_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for the current population. """
        grad = jax.vmap(self.get_Z_gradient_individual, in_axes=(0,0,0,0,None,None))(signal_t, signal_t_new, population, transform_arr, measurement_info, pulse_or_gate)
        return grad




    def calc_Z_error_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the Z-error such that it can be called in a linesearch. """
        transform_arr = linesearch_info.transform_arr
        signal_t_new, descent_direction = linesearch_info.signal_t_new, linesearch_info.descent_direction

        individual = self.update_individual(linesearch_info.population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        error = calculate_Z_error(signal_t.signal_t, signal_t_new)
        return error
    


    def calc_Z_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the Z-error gradient such that it can be called in a linesearch. """
        transform_arr = linesearch_info.transform_arr
        signal_t_new, descent_direction = linesearch_info.signal_t_new, linesearch_info.descent_direction

        individual = self.update_individual(linesearch_info.population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        
        grad = self.get_Z_gradient_individual(signal_t, signal_t_new, individual, transform_arr, measurement_info, pulse_or_gate)
        return jnp.sum(grad, axis=0)
    




    def do_iteration(self, signal_t, signal_t_new, transform_arr, population, local_or_global_state, measurement_info, descent_info, 
                            pulse_or_gate, local_or_global):
        
        """ 
        Performs one local/global iteration of the Common Pulse Retrieval Algorithm. 
        Uses gradient descent, nonlinear conjugate gradients, LBFGS or damped Newtons method (diagonal or full).
        The step size is determined via a fixed/adaptive step size, a backtracking or a zoom linesearch.
        """
        
        gamma, newton_info = getattr(descent_info.gamma, local_or_global), getattr(descent_info.newton, local_or_global)
        
        if local_or_global=="_global":
            shape = (descent_info.population_size, ) + jnp.shape(transform_arr)
            transform_arr = jnp.broadcast_to(transform_arr, shape)

        grad = self.get_Z_gradient(signal_t, signal_t_new, population, transform_arr, measurement_info, pulse_or_gate)
        grad_sum = jnp.sum(grad, axis=1)

        if newton_info=="diagonal" or newton_info=="full":
            descent_direction, newton_state = self.get_Z_newton_direction(grad, signal_t, signal_t_new, transform_arr, population, local_or_global_state, 
                                                                                       measurement_info, descent_info, newton_info, pulse_or_gate)

            local_or_global_state = tree_at(lambda x: getattr(x.newton, pulse_or_gate), local_or_global_state, newton_state)

        elif newton_info=="lbfgs":
            lbfgs_state = getattr(local_or_global_state.lbfgs, pulse_or_gate)
            descent_direction, lbfgs_state = get_quasi_newton_direction(grad_sum, lbfgs_state, descent_info)

        else: 
            descent_direction = -1*grad_sum


        if descent_info.conjugate_gradients!=False:
            cg = getattr(local_or_global_state.cg, pulse_or_gate)
            descent_direction, cg = jax.vmap(get_nonlinear_CG_direction, in_axes=(0,0,None))(descent_direction, cg, descent_info.conjugate_gradients)
            local_or_global_state = tree_at(lambda x: getattr(x.cg, pulse_or_gate), local_or_global_state, cg)


        Z_error = jax.vmap(calculate_Z_error, in_axes=(0,0))(signal_t.signal_t, signal_t_new)
        adaptive_scaling_info = getattr(descent_info.adaptive_scaling, local_or_global)
        if adaptive_scaling_info.order!=False:
            descent_direction, local_or_global_state = jax.vmap(adaptive_step_size, in_axes=(0,0,0,None,0,None,None,None))(Z_error, grad_sum, descent_direction,
                                                                                                                      descent_info.xi, 
                                                                                                                            local_or_global_state, 
                                                                                                                            adaptive_scaling_info,
                                                                                                                            pulse_or_gate, local_or_global)
            
        if descent_info.linesearch_params.linesearch!=False and local_or_global=="_global":
            pk_dot_gradient = jax.vmap(lambda x,y: jnp.real(jnp.vdot(x,y)), in_axes=(0,0))(descent_direction, grad_sum)        
            linesearch_info=MyNamespace(population=population, signal_t_new=signal_t_new, descent_direction=descent_direction, error=Z_error, 
                                        pk_dot_gradient=pk_dot_gradient, transform_arr=transform_arr)
            
            gamma = jax.vmap(do_linesearch, in_axes=(0,None,None,None,None, None))(linesearch_info, measurement_info, descent_info, 
                                                                            Partial(self.calc_Z_error_for_linesearch, descent_info=descent_info, 
                                                                                    pulse_or_gate=pulse_or_gate),
                                                                            Partial(self.calc_Z_grad_for_linesearch, descent_info=descent_info, 
                                                                                    pulse_or_gate=pulse_or_gate), local_or_global)
        else:
            if jnp.size(gamma)==1:
                gamma = jnp.broadcast_to(gamma, (descent_info.population_size, ))
            elif jnp.size(gamma)==descent_info.population_size:
                pass
            else:
                raise ValueError(f"Size of {local_or_global} gamma has to be 1 or the population size. Not {jnp.size(gamma)}")
            

        if newton_info=="lbfgs":
            lbfgs_state = update_lbfgs_state(lbfgs_state, gamma, grad_sum, descent_direction)
            local_or_global_state = tree_at(lambda x: getattr(x.lbfgs, pulse_or_gate), local_or_global_state, lbfgs_state)

        population = self.update_population(population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate)
        return local_or_global_state, population
    




    def local_iteration(self, descent_state, transform_arr_m, trace_line, measurement_info, descent_info):
        """ Peforms one local iteration. Calls do_iteration() with the appropriate (randomized) signal fields. """
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,0,0,0,None,None,None))(signal_t.signal_t, signal_t.signal_f, trace_line, descent_state._local.mu, measurement_info, 
                                                                                   descent_info, "_local")


        population, local_state = descent_state.population, descent_state._local
        local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, population, local_state, measurement_info, descent_info, 
                                                   "pulse", "_local")
        
        population_pulse = population.pulse/jnp.linalg.norm(population.pulse,axis=-1)[:,jnp.newaxis]
        population = tree_at(lambda x: x.pulse, population, population_pulse)

        if measurement_info.doubleblind==True:
            local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, population, local_state, measurement_info, descent_info, 
                                                        "gate", "_local")
            if measurement_info.interferometric==False:
                population_gate = population.gate/jnp.linalg.norm(population.gate,axis=-1)[:,jnp.newaxis]
                population = tree_at(lambda x: x.pulse, population, population_gate)
            
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x._local, descent_state, local_state)
        return descent_state, None
    

    

    def local_step(self, descent_state, measurement_info, descent_info):
        """
        Performs one local iteration of the Common Pulse Retrieval Algorithm. 
        This means the method loops over the randomized measurement data once and updates the population using each data point individually.
        
        Args:
            descent_state (Pytree):
            measurement_info (Pytree):
            descent_info (Pytree):

        Returns:
            tuple[Pytree, jnp.array], the updated descent state and the current trace errors of the population.
        """

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(signal_t.signal_f)
        local_mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measurement_info.measured_trace)
        descent_state = tree_at(lambda x: x._local.mu, descent_state, local_mu)

        one_local_iteration=Partial(self.local_iteration, measurement_info=measurement_info, descent_info=descent_info)
        one_local_iteration=Partial(scan_helper, actual_function=one_local_iteration, number_of_args=1, number_of_xs=2)

        transform_arr, measured_trace, descent_state = self.shuffle_data_along_m(descent_state, measurement_info, descent_info)
        descent_state, _ = jax.lax.scan(one_local_iteration, descent_state, (transform_arr, measured_trace))


        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(signal_t.signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measurement_info.measured_trace)

        return descent_state, trace_error.reshape(-1,1)
    




    def global_step(self, descent_state, measurement_info, descent_info):
        """
        Performs one global iteration of the Common Pulse Retrieval Algorithm. 
        This means the method updates the population once using all measured data at once.
        
        Args:
            descent_state (Pytree):
            measurement_info (Pytree):
            descent_info (Pytree):

        Returns:
            tuple[Pytree, jnp.array], the updated descent state and the current trace errors of the population.
        """

        measured_trace = measurement_info.measured_trace

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(signal_t.signal_f)
        mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,0,None,0,None,None,None))(signal_t.signal_t, signal_t.signal_f, measured_trace, mu, measurement_info, 
                                                                                      descent_info, "_global")


        population, global_state = descent_state.population, descent_state._global
        global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, population, global_state, measurement_info, 
                                                     descent_info, "pulse", "_global")
        
        population_pulse = population.pulse/jnp.linalg.norm(population.pulse,axis=-1)[:,jnp.newaxis]
        population = tree_at(lambda x: x.pulse, population, population_pulse)

        if measurement_info.doubleblind==True:
            global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, population, global_state, measurement_info, 
                                                         descent_info, "gate", "_global")
            
            if measurement_info.interferometric==False:
                population_gate = population.gate/jnp.linalg.norm(population.gate,axis=-1)[:,jnp.newaxis]
                population = tree_at(lambda x: x.gate, population, population_gate)

            
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x._global, descent_state, global_state)

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(signal_t.signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)

        return descent_state, trace_error.reshape(-1,1)
    
    
    


    def initialize_run(self, population):
        """
        Prepares all provided data and parameters for the reconstruction. 
        Here the final shape/structure of descent_state, measurement_info and descent_info are determined. 

        Args:
            population (Pytree): the initial guess as created by create_initial_population()
        
        Returns:
            tuple[Pytree, Callable, Callable], the initial descent state, the local and global step-functions of the algorithm.

        """

        measurement_info = self.measurement_info

        self.descent_info = initialize_descent_info(self)
        descent_info = self.descent_info

        shape = jnp.shape(population.pulse)
        cg_state_local = initialize_CG_state(shape, measurement_info)
        newton_state_local = initialize_pseudo_newton_state(shape, measurement_info)
        lbfgs_state_local = initialize_lbfgs_state(shape, measurement_info, descent_info)

        cg_state_global = initialize_CG_state(shape, measurement_info)
        newton_state_global = initialize_pseudo_newton_state(shape, measurement_info)
        lbfgs_state_global = initialize_lbfgs_state(shape, measurement_info, descent_info)

        init_arr = jnp.zeros(shape[0])
        self.descent_state = self.descent_state.expand(key = self.key, 
                                                       population = population, 
                                                       _local=MyNamespace(cg=cg_state_local, newton=newton_state_local, lbfgs=lbfgs_state_local, 
                                                                          max_scaling = MyNamespace(pulse=init_arr, gate=init_arr),
                                                                          mu = jnp.ones(shape[0])),
                                                       _global=MyNamespace(cg=cg_state_global, newton=newton_state_global, lbfgs=lbfgs_state_global))
        
        descent_state = self.descent_state

        do_local=Partial(self.local_step, measurement_info=measurement_info, descent_info=descent_info)
        do_local=Partial(scan_helper, actual_function=do_local, number_of_args=1, number_of_xs=0)

        do_global=Partial(self.global_step, measurement_info=measurement_info, descent_info=descent_info)
        do_global=Partial(scan_helper, actual_function=do_global, number_of_args=1, number_of_xs=0)
        return descent_state, do_local, do_global
    




    def run(self, population, no_iterations_local, no_iterations_global, **kwargs):
        """ 
        The Algorithm can use a local and a global sequentially.

        Args:
            population (Pytree): the initial guess
            no_iterations_local: int, the number of local iterations. Accepts zero as a value.
            no_iterations_global: int, the number of globale iterations. Accepts zero as a value.

        Returns:
            Pytree, the final result
        """

        do_checks_before_running(self, **kwargs)

        descent_state, do_local, do_global = self.initialize_run(population)

        descent_state, error_arr_local = run_scan(do_local, descent_state, no_iterations_local, self.jit)
        descent_state, error_arr_global = run_scan(do_global, descent_state, no_iterations_global, self.jit)

        error_arr = jnp.concatenate([error_arr_local, error_arr_global], axis=0)
        error_arr = jnp.squeeze(error_arr)

        final_result = self.post_process(descent_state, error_arr)
        return final_result



