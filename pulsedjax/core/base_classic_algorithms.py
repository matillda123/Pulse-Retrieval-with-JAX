import jax
import jax.numpy as jnp

from functools import partial as Partial
from equinox import tree_at


from pulsedjax.core.stepsize import do_linesearch, adaptive_step_size
from pulsedjax.core.nonlinear_cg import get_nonlinear_CG_direction
from pulsedjax.core.lbfgs import get_quasi_newton_direction

from pulsedjax.utilities import scan_helper, MyNamespace, calculate_mu, _calculate_mu, _calculate_mu_f, calculate_trace, calculate_trace_error, calculate_Z_error, run_scan, do_interpolation_1d, initialize_mu
from pulsedjax.core.base_classes_algorithms import ClassicAlgorithmsBASE




def normalize_population(population, measurement_info, pulse_or_gate):
    if pulse_or_gate=="pulse":
        population_pulse = population.pulse/jnp.linalg.norm(population.pulse,axis=-1)[:,jnp.newaxis]
        population_gate = population.gate

    elif pulse_or_gate=="gate":
        population_pulse = population.pulse
        if measurement_info.interferometric==False:
            population_gate = population.gate/jnp.linalg.norm(population.gate,axis=-1)[:,jnp.newaxis]
        else:
            population_gate = population.gate

    return MyNamespace(pulse=population_pulse, gate=population_gate)




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
                                 r_newton=optimizer.r_newton)
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
                                                 conjugate_gradients = MyNamespace(_local=optimizer.local_conjugate_gradients, 
                                                                                   _global=optimizer.global_conjugate_gradients),
                                                 linesearch_params = linesearch_params,
                                                 newton = newton,
                                                 s_prime_params = s_prime_params,
                                                 xi = optimizer.xi,
                                                 adaptive_scaling = MyNamespace(_local=MyNamespace(order=optimizer.local_adaptive_scaling, 
                                                                                                   factor=optimizer.local_adaptive_scaling_factor), 
                                                                                _global=MyNamespace(order=optimizer.global_adaptive_scaling, 
                                                                                                    factor=optimizer.global_adaptive_scaling_factor)),
                                                optimize_calibration_curve = MyNamespace(_local=optimizer.local_optimize_calibration_curve,
                                                                                         _global=optimizer.global_optimize_calibration_curve),
                                                eta_spectral_amplitude = optimizer.eta_spectral_amplitude
                                                )
    return descent_info









class LSGPABASE(ClassicAlgorithmsBASE):
    # for chirp-scan one would end up with somehting related to the pie i think.

    # i think doubleblind doesnt work here because i am trying to retrive the gate-pulse instead of the gate 
    # -> shg should work though

    """
    The Least-Squares Generalized Projection Algorithm.
    Only available for delay based non-interferometric methods.
     
    J. Gagnon et al., Appl. Phys. B 92, 25-32, 10.1007/s00340-008-3063-x (2008)
    
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)
        assert self.interferometric==False, "LSGPA is not intended for interferometric measurements."
        assert self.doubleblind==False, "LSGPA doesnt work with doubleblind"
        #or nonlinear_method=="shg", "LSGPA doesnt work with doubleblind unless its SHG."

        self._name = "LSGPA"


    def update_pulse(self, signal_t_new, gate_shifted, measurement_info):
        """ Generates an new (maybe improoved) guess for the pulse. """
        _lambda = 1e-12
        pulse = jnp.sum(signal_t_new*jnp.conjugate(gate_shifted), axis=1)/(jnp.sum(jnp.abs(gate_shifted)**2, axis=1) + _lambda)
        return self.fft(pulse, measurement_info.sk, measurement_info.rn)
    
    def update_gate(self, signal_t_new, pulse_t_shifted, measurement_info):
        """ Generates an new (maybe improoved) guess for the gate. """
        _lambda = 1e-12
        # maybe there is an error here -> its the same formula as for the pulse, doesnt seem right 
        # rederive formula, also take mu into account
        # for non-shg one would need to reverse the gate function
        gate = jnp.sum(signal_t_new*jnp.conjugate(pulse_t_shifted), axis=1)/(jnp.sum(jnp.abs(pulse_t_shifted)**2, axis=1) + _lambda)
        return self.fft(gate, measurement_info.sk, measurement_info.rn)
    


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
        
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        _calculate_trace = Partial(calculate_trace, measured_trace=measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_global")
        trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        signal_t_new = self.calculate_S_prime_population(signal_t, measured_trace, 
                                                         mu, measurement_info, descent_info, "_global", 
                                                         axes=(0,None,0,None,None,None))
    
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,0,None))(mu, trace, measured_trace)
        population_pulse = self.update_pulse(signal_t_new, signal_t.gate_shifted, measurement_info)
        population_pulse = population_pulse/jnp.linalg.norm(population_pulse,axis=-1)[:,jnp.newaxis]
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population_pulse)

        if measurement_info.doubleblind==True:
            population_gate = self.update_gate(signal_t_new, signal_t.pulse_t_shifted, measurement_info)
            population_gate = population_gate/jnp.linalg.norm(population_gate,axis=-1)[:,jnp.newaxis]
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population_gate)

        descent_state = tree_at(lambda x: x.mu, descent_state, mu)
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
        self.descent_info = self.descent_info.expand(s_prime_params = s_prime_params,
                                                     xi = self.xi,
                                                     gamma = MyNamespace(_local=None, _global=self.global_gamma),
                                                     optimize_calibration_curve = MyNamespace(_local=None,
                                                                                              _global=self.global_optimize_calibration_curve),
                                                    eta_spectral_amplitude=self.eta_spectral_amplitude)
        descent_info = self.descent_info

        mu_init_local, mu_init_global = initialize_mu(self, measurement_info, descent_info)
        self.descent_state = self.descent_state.expand(population = population, 
                                                       mu = mu_init_global)
        descent_state = self.descent_state

        do_step = Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step = Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step








class CPCGPABASE(ClassicAlgorithmsBASE):
    """
    The Constrained-PCGP-Algorithm.
    Only available for delay based non-interferometric methods. Transforms population to time domain during initialization.

    D. J. Kane and A. B. Vakhtin, Prog. Quantum Electron. 81 (100364), 10.1016/j.pquantelec.2021.100364 (2022)

    Attributes:
        constraints (bool): if true the operator based constraints are used.
        svd (bool): if true a full SVD is performed instead of a single iteration of the power method
        antialias (bool): if true anti-aliasing is applied to the outer-product-matrix-form
    
    """
    def __init__(self, delay, frequency, trace, nonlinear_method, cross_correlation=False, constraints=False, svd=False, antialias=False, **kwargs):
        assert jnp.shape(trace)[0]==jnp.shape(trace)[1], "For CPCGPA the trace needs to be symmetric and FFT conform."
        
        super().__init__(delay, frequency, trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)
        assert self.interferometric==False, "PCGPA is not intended for interferometric measurements."
        assert nonlinear_method!="sd", "Doesnt work for SD. Which is weird."

        self._name = "CPCGPA"
        
        self.idx_arr = jnp.arange(jnp.size(self.frequency)) 
        self.measurement_info = self.measurement_info.expand(idx_arr = self.idx_arr)

        self.constraints = constraints
        self.svd = svd
        self.antialias = antialias


    
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

            if measurement_info.doubleblind==True:
                gate = jnp.dot(opf.T.conj(), jnp.dot(opf, gate))
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

        return MyNamespace(pulse=pulse_t, pulse_prime=pulse_t_prime, gate=gate, gate_prime=gate_prime)




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

        population = jax.tree.map(lambda x: self.ifft(x, measurement_info.sk, measurement_info.rn), population)

        signal_t = jax.vmap(self.calculate_signal_t_using_opf, in_axes=(0,None,None,None))(population, iteration, measurement_info, descent_info)
        _calculate_trace = Partial(calculate_trace, measured_trace=measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_global")
        trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,0,None))(mu, trace, measured_trace)
        
        # mu isnt used here, as a factor it is destabilizing, as a calibration curve it messes retrieval up.
        signal_t_new = self.calculate_S_prime_population(signal_t, measured_trace, 1, 
                                                         measurement_info, descent_info, "_global", 
                                                         axes=(0,None,None,None,None,None))
        
        opf = jax.vmap(self.convert_signal_t_to_opf, in_axes=(0,None))(signal_t_new, idx_arr)

        if descent_info.antialias==True:
            half_N = jnp.size(opf[0])//2
            opf = self.do_anti_alias(opf, half_N)

        population = jax.vmap(self.update_individual, in_axes=(0,0,None,None))(opf, population, measurement_info, descent_info)        
        population = jax.tree.map(lambda x: self.fft(x, measurement_info.sk, measurement_info.rn), population)
        population = jax.tree.map(lambda x: x/jnp.linalg.norm(x,axis=-1)[:,None], population)
        
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x.iteration, descent_state, iteration+1)
        descent_state = tree_at(lambda x: x.mu, descent_state, mu)
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
        if self.descent_info.measured_spectrum_is_provided.gate==True and self.nonlinear_method!="shg":
            print("PCGPA retrieves the gate, not the gate-pulse. Thus providing the spectrum isnt correct.")

        if self.global_optimize_calibration_curve==True:
            print("Calibration curve optimization isnt working in PCGPA.")


        measurement_info = self.measurement_info

        s_prime_params = initialize_S_prime_params(self)
        self.descent_info = self.descent_info.expand(svd = self.svd, 
                                                     constraints = self.constraints,
                                                     antialias = self.antialias,
                                                     s_prime_params = s_prime_params,
                                                     xi = self.xi,
                                                     gamma = MyNamespace(_local=None, _global=self.global_gamma),
                                                     optimize_calibration_curve = MyNamespace(_local=None,
                                                                                              _global=self.global_optimize_calibration_curve),
                                                    eta_spectral_amplitude=self.eta_spectral_amplitude)
        descent_info = self.descent_info

        population = MyNamespace(pulse=population.pulse, pulse_prime=population.pulse,
                                 gate=population.gate, gate_prime=population.gate)
        
        _, mu_init_global = initialize_mu(self, measurement_info, descent_info)
        self.descent_state = self.descent_state.expand(population = population, 
                                                       iteration = 0,
                                                       mu = mu_init_global)

        descent_state = self.descent_state

        do_step = Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step = Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step
    


    def post_process_create_trace(self, descent_state, measurement_info, descent_info, idx):
        """ For PCGP the trace is constructed using the opf. """
        iteration = descent_state.iteration
        population = jax.tree.map(lambda x: self.ifft(x, measurement_info.sk, measurement_info.rn), descent_state.population)

        signal_t = jax.vmap(self.calculate_signal_t_using_opf, in_axes=(0,None,None,None))(population, iteration, measurement_info, descent_info)
        _calculate_trace = Partial(calculate_trace, measured_trace=measurement_info.measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_global")
        trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        trace = mu[:,None,None]*trace
        return trace[idx]


    

    def do_step_and_apply_spectral_amplitude(self, descent_state, measurement_info, descent_info, do_step):
        """ If a spectrum is provided this wraps around the step-method of all solvers and projects the current guess onto the measured spectrum. """
        descent_state, trace_error = do_step(descent_state, measurement_info, descent_info)
        
        population = descent_state.population
        eta = descent_info.eta_spectral_amplitude
        
        if descent_info.measured_spectrum_is_provided.pulse==True:
            _apply_spectrum = Partial(self.apply_spectrum, spectral_amplitude=measurement_info.spectral_amplitude.pulse, eta=eta)
            population_pulse = jax.tree.map(_apply_spectrum, population.pulse)
            population = tree_at(lambda x: x.pulse, population, population_pulse)

            population_pulse = jax.tree.map(_apply_spectrum, population.pulse_prime)
            population = tree_at(lambda x: x.pulse_prime, population, population_pulse)

        if descent_info.measured_spectrum_is_provided.gate==True:
            _apply_spectrum = Partial(self.apply_spectrum, spectral_amplitude=measurement_info.spectral_amplitude.gate, eta=eta)
            population_gate = jax.tree.map(_apply_spectrum, population.gate)
            population = tree_at(lambda x: x.gate, population, population_gate)

            population_gate = jax.tree.map(_apply_spectrum, population.gate_prime)
            population = tree_at(lambda x: x.gate_prime, population, population_gate)
            
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        return descent_state, trace_error


    









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


    def update_individual(self, individual, gamma, descent_direction, pulse_or_gate):
        pulse_f = getattr(individual, pulse_or_gate)
        pulse_f = pulse_f + gamma*descent_direction

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, pulse_f)
        return individual

    
    def update_population(self, population, gamma, descent_direction, pulse_or_gate):
        """ Applies the descent based update to the population. """
        return self.update_individual(population, gamma[:,None], descent_direction, pulse_or_gate)
        

    
    def get_Z_gradient(self, signal_t, signal_t_new, transform_arr, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the Z-error gradient for the entire population. """
        grad = jax.vmap(self.calculate_Z_gradient_individual, in_axes=(0,0,0,None,None))(signal_t, signal_t_new, 
                                                                                           transform_arr, measurement_info, 
                                                                                           pulse_or_gate)
        
        if getattr(descent_info.measured_spectrum_is_provided, pulse_or_gate)==True:
            grad = grad*getattr(measurement_info.spectral_amplitude, pulse_or_gate)

        return grad

    

    def calc_Z_error_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        """ Calculates the Z-error such that it can be called in a linesearch. """
        individual, descent_direction, signal_t_new = linesearch_info.population, linesearch_info.descent_direction, linesearch_info.signal_t_new
       
        transform_arr = measurement_info.transform_arr

        individual = self.update_individual(individual, gamma, descent_direction, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        Z_error_new = calculate_Z_error(signal_t.signal_t, signal_t_new)
        return Z_error_new
    

    def calc_Z_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the Z-error gradient such that it can be called in a linesearch. """
        individual, descent_direction, signal_t_new = linesearch_info.population, linesearch_info.descent_direction, linesearch_info.signal_t_new

        transform_arr = measurement_info.transform_arr

        individual = self.update_individual(individual, gamma, descent_direction, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        grad = self.calculate_Z_gradient_individual(signal_t, signal_t_new, transform_arr, measurement_info, pulse_or_gate)
        
        if getattr(descent_info.measured_spectrum_is_provided, pulse_or_gate)==True:
            grad = grad*getattr(measurement_info.spectral_amplitude, pulse_or_gate)

        return jnp.sum(grad, axis=0)


    def descent_Z_error_step(self, signal_t, signal_t_new, Z_error, descent_state, measurement_info, descent_info, pulse_or_gate): 
        """ 
        Performs a descent step in order to minimize the Z-error. 
        Employs gradient descent, nonlinear conjugate gradients, LBFGS or damped Newtons method (diagonal or full).
        The step size is determined via a fixed/adaptive step size, a backtracking or a zoom linesearch.
        """       

        newton_info, conjugate_gradients = descent_info.newton._global, descent_info.conjugate_gradients._global

        population = descent_state.population
        transform_arr = measurement_info.transform_arr
        transform_arr = jnp.broadcast_to(transform_arr, (descent_info.population_size, ) + jnp.shape(transform_arr))

        grad = self.get_Z_gradient(signal_t, signal_t_new, transform_arr, measurement_info, descent_info, pulse_or_gate)
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
            #pk_dot_gradient = jax.vmap(lambda x,y: jnp.real(jnp.vdot(x,y)), in_axes=(0,0))(descent_direction, grad_sum)
            pk_dot_gradient = jnp.real(jnp.vecdot(descent_direction, grad_sum)) # should be the same
            
            linesearch_info=MyNamespace(population=population, descent_direction=descent_direction, signal_t_new=signal_t_new, 
                                        error=Z_error, pk_dot_gradient=pk_dot_gradient)
            
            gamma = jax.vmap(do_linesearch, in_axes=(0,None,None,None,None,None))(linesearch_info, measurement_info, descent_info, 
                                                                             Partial(self.calc_Z_error_for_linesearch, pulse_or_gate=pulse_or_gate),
                                                                             Partial(self.calc_Z_grad_for_linesearch, descent_info=descent_info, 
                                                                                     pulse_or_gate=pulse_or_gate), "_global")
        else:
            gamma = jnp.broadcast_to(descent_info.gamma._global, (descent_info.population_size))

        if newton_info=="lbfgs":
            lbfgs_state = update_lbfgs_state(lbfgs_state, gamma, grad_sum, descent_direction)
            descent_state = tree_at(lambda x: getattr(x.lbfgs, pulse_or_gate), descent_state, lbfgs_state)

        population = self.update_population(population, gamma, descent_direction, pulse_or_gate) 
        return population
    



    def do_descent_Z_error_step(self, descent_state, signal_t_new, measurement_info, descent_info):
        """ Does one Z-error descent step. Calls descent_Z_error_step for pulse and or gate. """

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        Z_error = jax.vmap(calculate_Z_error, in_axes=(0,0))(signal_t.signal_t, signal_t_new)

        population = self.descent_Z_error_step(signal_t, signal_t_new, Z_error, descent_state, measurement_info, descent_info, "pulse")
        population = normalize_population(population, measurement_info, "pulse")
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population.pulse)
        
        if measurement_info.doubleblind==True:
            population = self.descent_Z_error_step(signal_t, signal_t_new, Z_error, descent_state, 
                                                            measurement_info, descent_info, "gate")
            population = normalize_population(population, measurement_info, "gate")
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population.gate)
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
        _calculate_trace = Partial(calculate_trace, measured_trace=measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_global")

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        signal_t_new = self.calculate_S_prime_population(signal_t, measured_trace, mu, 
                                                         measurement_info, descent_info, "_global", 
                                                         axes=(0,None,0,None,None,None))
        descent_state = self.do_descent_Z_error(descent_state, signal_t_new, measurement_info, descent_info)

        #signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,0,None))(mu,trace, measured_trace)

        descent_state = tree_at(lambda x: x.mu, descent_state, mu)
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

        shape_pulse = jnp.shape(population.pulse)
        cg_state = initialize_CG_state(shape_pulse, measurement_info)
        newton_state = initialize_pseudo_newton_state(shape_pulse, measurement_info)
        lbfgs_state = initialize_lbfgs_state(shape_pulse, measurement_info, descent_info)

        _, mu_init_global = initialize_mu(self, measurement_info, descent_info)
        self.descent_state = self.descent_state.expand(population=population,
                                                       cg=cg_state, newton=newton_state, 
                                                       lbfgs=lbfgs_state, mu=mu_init_global)

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

        # this corresponds to the pie-error
        self.r_gradient = "amplitude"




    def calculate_PIE_mu(self, signal_f_abs, trace_root, measurement_info, descent_info, local_or_global):
        if descent_info.calibration_curve_is_provided==True:
            mu = jnp.sqrt(measurement_info.calibration_curve)
        else:
            if getattr(descent_info.optimize_calibration_curve, local_or_global)==True:
                mu = _calculate_mu_f(signal_f_abs, trace_root)
            else:
                mu = _calculate_mu(signal_f_abs, trace_root)
        return mu


    def calculate_PIE_trace(self, signal_f, measured_trace, measurement_info, descent_info, local_or_global):
        signal_f_abs = jnp.abs(signal_f)
        trace_root = jnp.sqrt(jnp.abs(measured_trace))*jnp.sign(measured_trace)
        mu_pie = self.calculate_PIE_mu(signal_f_abs, trace_root, measurement_info, descent_info, local_or_global)
        mu_pie2 = mu_pie**2 # is needed since usually mu is incorporated via roots but in pie that would be wrong
        return signal_f_abs, mu_pie2



    def calculate_PIE_error(self, mu, signal_f, measured_trace):
        """ Calculates the normalized least-squares error using the amplitude residuals. """
        trace_root = jnp.sqrt(jnp.abs(measured_trace)/mu)*jnp.sign(measured_trace)
        return jnp.mean(jnp.abs(trace_root - jnp.abs(signal_f))**2)


    def get_PIE_weights(self, probe, alpha, pie_method):
        """ Calculates the weight-functions for the differen PIE-version. """

        # U=2/(jnp.abs(probe_shifted)**2+1e-6) 
        # # -> rPIE is eqivalent to pseudo-gauss-newton/levenberg-marquardt for small gamma. 
        # gamma => 1 -> rPIE => ePIE

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


    def update_individual(self, individual, gamma, descent_direction, pulse_or_gate):
        signal_f = getattr(individual, pulse_or_gate)
        signal_f = signal_f + gamma*descent_direction

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal_f)
        return individual


    def update_population(self, population, gamma, descent_direction, pulse_or_gate):
        """ Applies the PIE update to the population. """
        return self.update_individual(population, gamma[:,None], descent_direction, pulse_or_gate)


    def calculate_PIE_descent_direction(self, signal_t, signal_t_new, transform_arr, pie_method, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the descent direction based on the PIE version. """
        return jax.vmap(self.calculate_PIE_descent_direction_m, in_axes=(0,0,0,None,None,None,None))(signal_t, signal_t_new, transform_arr, pie_method, measurement_info, descent_info, pulse_or_gate)




    def calc_error_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        """ Calculates the PIE-error such that it can be called in a linesearch. """

        transform_arr, measured_trace, mu = linesearch_info.transform_arr, linesearch_info.measured_trace, linesearch_info.mu
        individual, descent_direction = linesearch_info.population, linesearch_info.descent_direction

        individual = self.update_individual(individual, gamma, descent_direction, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        error_new = self.calculate_PIE_error(mu, signal_t.signal_f, measured_trace)
        return error_new
    


    def calc_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate, local_or_global):
        """ Calculates the PIE direction such that it can be called in a linesearch. """
        transform_arr, measured_trace = linesearch_info.transform_arr, linesearch_info.measured_trace
        individual, descent_direction = linesearch_info.population, linesearch_info.descent_direction
        
        individual = self.update_individual(individual, gamma, descent_direction, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        signal_t_new = self.calculate_S_prime_individual(signal_t, measured_trace, linesearch_info.mu, 
                                                         measurement_info, descent_info, local_or_global)
        
        # None, is the correct choice since it yields the steepest descent direction of pie
        grad_U = self.calculate_PIE_descent_direction_m(signal_t, signal_t_new, transform_arr, None, 
                                                        measurement_info, descent_info, pulse_or_gate)
        
        grad_U = self.fft(grad_U, measurement_info.sk, measurement_info.rn)
        if getattr(descent_info.measured_spectrum_is_provided, pulse_or_gate)==True:
            grad_U = grad_U*getattr(measurement_info.spectral_amplitude, pulse_or_gate)

        return jnp.sum(grad_U, axis=0)
    


    
    def do_iteration(self, signal_t, signal_t_new, transform_arr, measured_trace, population, local_or_global_state, 
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
        conjugate_gradients = getattr(descent_info.conjugate_gradients, local_or_global)
        newton_info = getattr(descent_info.newton, local_or_global)

        grad_U = self.calculate_PIE_descent_direction(signal_t, signal_t_new, transform_arr, pie_method, measurement_info, descent_info, pulse_or_gate)
        
        grad_U = self.fft(grad_U, measurement_info.sk, measurement_info.rn)
        if getattr(descent_info.measured_spectrum_is_provided, pulse_or_gate)==True:
            grad_U = grad_U*getattr(measurement_info.spectral_amplitude, pulse_or_gate)
            
        grad_sum = jnp.sum(grad_U, axis=1)


        if newton_info=="diagonal" or newton_info=="full":
            measured_trace_for_newton = jax.vmap(lambda x,y: x/(y + 1e-9))(measured_trace, local_or_global_state.mu)
            descent_direction, newton_state = self.calculate_PIE_newton_direction(grad_U, signal_t, transform_arr, measured_trace_for_newton, local_or_global_state, 
                                                                                   measurement_info, descent_info, pulse_or_gate, local_or_global)
            local_or_global_state = tree_at(lambda x: getattr(x.newton, pulse_or_gate), local_or_global_state, newton_state)

        elif newton_info=="lbfgs":
            lbfgs_state = getattr(local_or_global_state.lbfgs, pulse_or_gate)
            descent_direction, lbfgs_state = get_quasi_newton_direction(grad_sum, lbfgs_state, descent_info)

        else:
            descent_direction = -1*grad_sum


        if conjugate_gradients!=False:
            cg = getattr(local_or_global_state.cg, pulse_or_gate)
            descent_direction, cg = jax.vmap(get_nonlinear_CG_direction, in_axes=(0,0,None))(descent_direction, cg, conjugate_gradients)
            local_or_global_state = tree_at(lambda x: getattr(x.cg, pulse_or_gate), local_or_global_state, cg)



        pie_error = jax.vmap(self.calculate_PIE_error, in_axes=(0,0,None))(local_or_global_state.mu, signal_t.signal_f, measured_trace)
        adaptive_scaling_info = getattr(descent_info.adaptive_scaling, local_or_global)
        if adaptive_scaling_info.order!=False:
            descent_direction, local_or_global_state = jax.vmap(adaptive_step_size, in_axes=(0,0,0,None,0,None,None,None))(pie_error, grad_sum, descent_direction, 
                                                                                                                      descent_info.xi,
                                                                                                                            local_or_global_state, 
                                                                                                                            adaptive_scaling_info,
                                                                                                                            pulse_or_gate, local_or_global)


        if descent_info.linesearch_params.linesearch!=False and local_or_global=="_global":
            #pk_dot_gradient = jax.vmap(lambda x,y: jnp.real(jnp.vdot(x,y)), in_axes=(0,0))(descent_direction, grad_sum)
            pk_dot_gradient = jnp.real(jnp.vecdot(descent_direction, grad_sum)) # should be the same

            linesearch_info=MyNamespace(population=population, signal_t=signal_t, descent_direction=descent_direction, 
                                        pk_dot_gradient=pk_dot_gradient, error=pie_error,
                                        transform_arr=transform_arr, measured_trace=measured_trace, mu=local_or_global_state.mu)     

            gamma = jax.vmap(do_linesearch, in_axes=(0, None, None, None, None, None))(linesearch_info, measurement_info, descent_info, 
                                                                                Partial(self.calc_error_for_linesearch, pulse_or_gate=pulse_or_gate),
                                                                                Partial(self.calc_grad_for_linesearch, descent_info=descent_info, 
                                                                                        pulse_or_gate=pulse_or_gate, local_or_global=local_or_global), 
                                                                                local_or_global)
            
        else:
            gamma = jnp.broadcast_to(getattr(descent_info.gamma, local_or_global), descent_info.population_size)

        if newton_info=="lbfgs":
            lbfgs_state = update_lbfgs_state(lbfgs_state, gamma, grad_sum, descent_direction)
            local_or_global_state = tree_at(lambda x: getattr(x.lbfgs, pulse_or_gate), local_or_global_state, lbfgs_state)
        
        population = self.update_population(population, gamma, descent_direction, pulse_or_gate)
        return local_or_global_state, population



    

    def local_iteration(self, descent_state, transform_arr_m, trace_line, measurement_info, descent_info):
        """ Peforms one local iteration. Calls do_iteration() with the appropriate (randomized) signal fields. """
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
        signal_t_new = self.calculate_S_prime_population(signal_t, trace_line, descent_state._local.mu, 
                                                         measurement_info, descent_info, "_local", 
                                                         axes=(0,0,0,None,None,None))

        local_state = descent_state._local
        local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, trace_line, 
                                                    descent_state.population, local_state, 
                                                      measurement_info, descent_info, "pulse", "_local")
        population = normalize_population(population, measurement_info, "pulse")
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population.pulse)

        if measurement_info.doubleblind==True:
            signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
            signal_t_new = self.calculate_S_prime_population(signal_t, trace_line, descent_state._local.mu, 
                                                            measurement_info, descent_info, "_local", 
                                                            axes=(0,0,0,None,None,None))
            
            local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, trace_line,
                                                        descent_state.population, local_state, 
                                                      measurement_info, descent_info, "gate", "_local")
            
            population = normalize_population(population, measurement_info, "gate")
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population.gate)
        
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
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        _calculate_trace = Partial(self.calculate_PIE_trace, measured_trace=measurement_info.measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_local")
        _, local_mu_pie2 = jax.vmap(_calculate_trace)(signal_t.signal_f)
        descent_state = tree_at(lambda x: x._local.mu, descent_state, local_mu_pie2)

        local_iteration = Partial(self.local_iteration, measurement_info=measurement_info, descent_info=descent_info)
        local_iteration = Partial(scan_helper, actual_function=local_iteration, number_of_args=1, number_of_xs=2)

        transform_arr, measured_trace, descent_state = self.shuffle_data_along_m(descent_state, measurement_info, descent_info)
        descent_state, _ = jax.lax.scan(local_iteration, descent_state, (transform_arr, measured_trace))


        # signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        _calculate_trace = Partial(calculate_trace, measured_trace=measurement_info.measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_local")
        trace, local_mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,0,None))(local_mu, trace, measurement_info.measured_trace)
        
        #trace_error = jax.vmap(self.calculate_PIE_error, in_axes=(0,0,None))(local_mu_pie, trace_pie, measurement_info.measured_trace)
        descent_state = tree_at(lambda x: x._local.mu, descent_state, local_mu)
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
        _calculate_trace = Partial(self.calculate_PIE_trace, measured_trace=measurement_info.measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_global")
        _, mu_pie2 = jax.vmap(_calculate_trace)(signal_t.signal_f)
        signal_t_new = self.calculate_S_prime_population(signal_t, measured_trace, mu_pie2, 
                                                         measurement_info, descent_info, "_global", 
                                                         axes=(0,None,0,None,None,None))
        
        global_state = descent_state._global
        global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, measured_trace,
                                                      descent_state.population, global_state, 
                                                      measurement_info, descent_info, "pulse", "_global")
        population = normalize_population(population, measurement_info, "pulse")
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population.pulse)

        if measurement_info.doubleblind==True:
            signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
            _calculate_trace = Partial(self.calculate_PIE_trace, measured_trace=measurement_info.measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_global")
            _, mu_pie2 = jax.vmap(_calculate_trace)(signal_t.signal_f)
            signal_t_new = self.calculate_S_prime_population(signal_t, measured_trace, mu_pie2, 
                                                            measurement_info, descent_info, "_global", 
                                                            axes=(0,None,0,None,None,None))
            
            global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, 
                                                         measured_trace, 
                                                          descent_state.population, global_state, 
                                                          measurement_info, descent_info, "gate", "_global")
            
            population = normalize_population(population, measurement_info, "gate")
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population.gate)

        descent_state = tree_at(lambda x: x._global, descent_state, global_state)

        # signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        _calculate_trace = Partial(calculate_trace, measured_trace=measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_global")
        trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,0,None))(mu, trace, measured_trace)

        #trace_error = jax.vmap(self.calculate_PIE_error, in_axes=(0,0,None))(mu_pie, trace_pie, measured_trace)
        descent_state = tree_at(lambda x: x._global.mu, descent_state, mu)
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

        mu_init_local, mu_init_global = initialize_mu(self, measurement_info, descent_info)

        init_arr = jnp.zeros(shape[0])
        self.descent_state = self.descent_state.expand(key = self.key, 
                                                       population = population, 
                                                       _local=MyNamespace(cg=cg_state_local, newton=newton_state_local, lbfgs=lbfgs_state_local, 
                                                                          max_scaling = MyNamespace(pulse=init_arr, gate=init_arr),
                                                                          mu=mu_init_local),
                                                       _global=MyNamespace(cg=cg_state_global, newton=newton_state_global, 
                                                                           lbfgs=lbfgs_state_global, mu=mu_init_global))
    
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

        self.do_checks_before_running(**kwargs)

        descent_state, do_local, do_global = self.initialize_run(population)

        descent_state, error_arr_local = run_scan(do_local, descent_state, no_iterations_local)
        descent_state, error_arr_global = run_scan(do_global, descent_state, no_iterations_global)

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



    def update_individual(self, individual, gamma, descent_direction, pulse_or_gate):
        """ Updates an individual based on a descent direction and a step size. """

        signal_f = getattr(individual, pulse_or_gate)
        signal_f = signal_f + gamma*descent_direction
        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal_f)
        return individual


    def update_population(self, population, gamma, descent_direction, pulse_or_gate):
        """ Applies the a descent based update to the population. """
        return self.update_individual(population, gamma[:,None], descent_direction, pulse_or_gate)
    

    

    def get_Z_gradient(self, signal_t, signal_t_new, transform_arr, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the Z-error gradient for the current population. """
        grad = jax.vmap(self.get_Z_gradient_individual, in_axes=(0,0,0,None,None))(signal_t, signal_t_new, 
                                                                                     transform_arr, measurement_info, 
                                                                                     pulse_or_gate)
        
        if getattr(descent_info.measured_spectrum_is_provided, pulse_or_gate)==True:
          grad = grad*getattr(measurement_info.spectral_amplitude, pulse_or_gate)
          # this is technically more accurate 
          # grad = grad*jnp.abs(self.fft(signal_t.pulse_t, measurement_info.sk, measurement_info.rn))[:,None,:]

        return grad




    def calc_Z_error_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the Z-error such that it can be called in a linesearch. """
        transform_arr = linesearch_info.transform_arr
        signal_t_new, descent_direction = linesearch_info.signal_t_new, linesearch_info.descent_direction

        individual = self.update_individual(linesearch_info.population, gamma, descent_direction, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        error = calculate_Z_error(signal_t.signal_t, signal_t_new)
        return error
    


    def calc_Z_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the Z-error gradient such that it can be called in a linesearch. """
        transform_arr = linesearch_info.transform_arr
        signal_t_new, descent_direction = linesearch_info.signal_t_new, linesearch_info.descent_direction

        individual = self.update_individual(linesearch_info.population, gamma, descent_direction, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        grad = self.get_Z_gradient_individual(signal_t, signal_t_new, transform_arr, measurement_info, pulse_or_gate)
        
        if getattr(descent_info.measured_spectrum_is_provided, pulse_or_gate) == True:
            grad = grad*getattr(measurement_info.spectral_amplitude, pulse_or_gate)
            # grad = grad*jnp.abs(self.fft(signal_t.pulse_t, measurement_info.sk, measurement_info.rn))

        return jnp.sum(grad, axis=0)
    




    def do_iteration(self, signal_t, signal_t_new, transform_arr, population, local_or_global_state, measurement_info, descent_info, 
                            pulse_or_gate, local_or_global):
        
        """ 
        Performs one local/global iteration of the Common Pulse Retrieval Algorithm. 
        Uses gradient descent, nonlinear conjugate gradients, LBFGS or damped Newtons method (diagonal or full).
        The step size is determined via a fixed/adaptive step size, a backtracking or a zoom linesearch.
        """
        
        newton_info = getattr(descent_info.newton, local_or_global)
        conjugate_gradients = getattr(descent_info.conjugate_gradients, local_or_global)
        
        if local_or_global=="_global":
            shape = (descent_info.population_size, ) + jnp.shape(transform_arr)
            transform_arr = jnp.broadcast_to(transform_arr, shape)

        grad = self.get_Z_gradient(signal_t, signal_t_new, transform_arr, measurement_info, descent_info, pulse_or_gate)
        grad_sum = jnp.sum(grad, axis=1)

        if newton_info=="diagonal" or newton_info=="full":
            descent_direction, newton_state = self.get_Z_newton_direction(grad, signal_t, signal_t_new, transform_arr, local_or_global_state, 
                                                                          measurement_info, descent_info, newton_info, pulse_or_gate)

            local_or_global_state = tree_at(lambda x: getattr(x.newton, pulse_or_gate), local_or_global_state, newton_state)

        elif newton_info=="lbfgs":
            lbfgs_state = getattr(local_or_global_state.lbfgs, pulse_or_gate)
            descent_direction, lbfgs_state = get_quasi_newton_direction(grad_sum, lbfgs_state, descent_info)

        else: 
            descent_direction = -1*grad_sum


        if conjugate_gradients!=False:
            cg = getattr(local_or_global_state.cg, pulse_or_gate)
            descent_direction, cg = jax.vmap(get_nonlinear_CG_direction, in_axes=(0,0,None))(descent_direction, cg, conjugate_gradients)
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
            #pk_dot_gradient = jax.vmap(lambda x,y: jnp.real(jnp.vdot(x,y)), in_axes=(0,0))(descent_direction, grad_sum)
            pk_dot_gradient = jnp.real(jnp.vecdot(descent_direction, grad_sum)) # should be the same
            
            linesearch_info=MyNamespace(population=population, signal_t_new=signal_t_new, descent_direction=descent_direction, error=Z_error, 
                                        pk_dot_gradient=pk_dot_gradient, transform_arr=transform_arr)
            
            gamma = jax.vmap(do_linesearch, in_axes=(0,None,None,None,None, None))(linesearch_info, measurement_info, descent_info, 
                                                                            Partial(self.calc_Z_error_for_linesearch, descent_info=descent_info, 
                                                                                    pulse_or_gate=pulse_or_gate),
                                                                            Partial(self.calc_Z_grad_for_linesearch, descent_info=descent_info, 
                                                                                    pulse_or_gate=pulse_or_gate), local_or_global)
        else:
            gamma = jnp.broadcast_to(getattr(descent_info.gamma, local_or_global), descent_info.population_size)


        if newton_info=="lbfgs":
            lbfgs_state = update_lbfgs_state(lbfgs_state, gamma, grad_sum, descent_direction)
            local_or_global_state = tree_at(lambda x: getattr(x.lbfgs, pulse_or_gate), local_or_global_state, lbfgs_state)
            
        population = self.update_population(population, gamma, descent_direction, pulse_or_gate)
        return local_or_global_state, population
    




    def local_iteration(self, descent_state, transform_arr_m, trace_line, measurement_info, descent_info):
        """ Peforms one local iteration. Calls do_iteration() with the appropriate (randomized) signal fields. """
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
        signal_t_new = self.calculate_S_prime_population(signal_t, trace_line, descent_state._local.mu, 
                                                         measurement_info, descent_info, "_local", 
                                                         axes=(0,0,0,None,None,None))
        
        local_state = descent_state._local
        local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, 
                                                    descent_state.population, local_state, 
                                                    measurement_info, descent_info, 
                                                   "pulse", "_local")
        
        population = normalize_population(population, measurement_info, "pulse")
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population.pulse)

        if measurement_info.doubleblind==True:
            signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
            signal_t_new = self.calculate_S_prime_population(signal_t, trace_line, descent_state._local.mu, 
                                                            measurement_info, descent_info, "_local", 
                                                            axes=(0,0,0,None,None,None))
            local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, 
                                                        descent_state.population, local_state, 
                                                        measurement_info, descent_info, 
                                                        "gate", "_local")
            population = normalize_population(population, measurement_info, "gate")
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population.gate)

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
        _calculate_trace = Partial(calculate_trace, measured_trace=measurement_info.measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_local")

        # signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        # trace, local_mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        # descent_state = tree_at(lambda x: x._local.mu, descent_state, local_mu)

        one_local_iteration = Partial(self.local_iteration, measurement_info=measurement_info, descent_info=descent_info)
        one_local_iteration = Partial(scan_helper, actual_function=one_local_iteration, number_of_args=1, number_of_xs=2)

        transform_arr, measured_trace, descent_state = self.shuffle_data_along_m(descent_state, measurement_info, descent_info)
        descent_state, _ = jax.lax.scan(one_local_iteration, descent_state, (transform_arr, measured_trace))


        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace, local_mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,0,None))(local_mu, trace, measurement_info.measured_trace)

        descent_state = tree_at(lambda x: x._local.mu, descent_state, local_mu)
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
        _calculate_trace = Partial(calculate_trace, measured_trace=measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_global")
        
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        signal_t_new = self.calculate_S_prime_population(signal_t, measured_trace, mu, 
                                                         measurement_info, descent_info, "_global", 
                                                         axes=(0,None,0,None,None,None))

        global_state = descent_state._global
        global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, 
                                                     descent_state.population, global_state, measurement_info, 
                                                     descent_info, "pulse", "_global")
        
        population = normalize_population(population, measurement_info, "pulse")
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population.pulse)

        if measurement_info.doubleblind==True:
            signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
            trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
            signal_t_new = self.calculate_S_prime_population(signal_t, measured_trace, mu, 
                                                            measurement_info, descent_info, "_global", 
                                                            axes=(0,None,0,None,None,None))
        
            global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, 
                                                         descent_state.population, global_state, measurement_info, 
                                                         descent_info, "gate", "_global")
            
            population = normalize_population(population, measurement_info, "gate")
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population.gate)
            
        descent_state = tree_at(lambda x: x._global, descent_state, global_state)

        #signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        #trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,0,None))(mu, trace, measured_trace)

        descent_state = tree_at(lambda x: x._global.mu, descent_state, mu)
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

        mu_init_local, mu_init_global = initialize_mu(self, measurement_info, descent_info)

        init_arr = jnp.zeros(shape[0])
        self.descent_state = self.descent_state.expand(key = self.key, 
                                                       population = population, 
                                                       _local=MyNamespace(cg=cg_state_local, newton=newton_state_local, lbfgs=lbfgs_state_local, 
                                                                          max_scaling = MyNamespace(pulse=init_arr, gate=init_arr),
                                                                          mu = mu_init_local),
                                                       _global=MyNamespace(cg=cg_state_global, newton=newton_state_global, lbfgs=lbfgs_state_global, 
                                                                           mu=mu_init_global))
        
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

        self.do_checks_before_running(**kwargs)

        descent_state, do_local, do_global = self.initialize_run(population)

        descent_state, error_arr_local = run_scan(do_local, descent_state, no_iterations_local)
        descent_state, error_arr_global = run_scan(do_global, descent_state, no_iterations_global)

        error_arr = jnp.concatenate([error_arr_local, error_arr_global], axis=0)
        error_arr = jnp.squeeze(error_arr)

        final_result = self.post_process(descent_state, error_arr)
        return final_result





















class LSFBASE(ClassicAlgorithmsBASE):
    # maybe there is an issue here. 
    # In one iteration all bisection steps are done first on the pulse and only then on the gate. Maybe this should be alternating?

    """
    Implements a version of the Linesearch FROG Algorithm (LSF). Despite its name the algorithm is NOT restricted to FROG. 

    C. O. Krook and V. Pasiskevicius, Opt. Express 33, 33258-33269 (2025) 

    Attributes:
        no_sections (int): number of sections to split the search line into
        number_of_disection_iterations (int): as the name says
        direction_mode (str): can be random or continuous
        ratio_points_for_continuous (int): smaller value means more randomness/eratic
        only_allow_improvements (bool): if true, only steps that decrease the error will be accepted
       
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._name = "LSF"

        self.no_sections = 2
        self.number_of_disection_iterations = 12

        self.direction_mode = "random"
        self.ratio_points_for_continuous = 0.25

        self.boundary = 1 
        self.only_allow_improvements = True


    def apply_spectrum(self, pulse_f, spectral_amplitude, eta):
        """ Projects onto measured spectrum by dividing the current spectral amplitude out. """
        spectral_amplitude = spectral_amplitude/jnp.max(spectral_amplitude)

        norm_in = jnp.max(jnp.abs(pulse_f), axis=-1)[:,None]
        pulse_f = pulse_f/norm_in
        phase_factor = pulse_f/jnp.abs(pulse_f)
        amp_f = (1-eta)*jnp.abs(pulse_f) + eta*spectral_amplitude

        pulse_f_new = amp_f*phase_factor
        norm_out = jnp.max(jnp.abs(pulse_f_new), axis=-1)[:,None]
        pulse_f_new = pulse_f_new*norm_in/norm_out
        return pulse_f_new



    def get_random_values(self, key, shape, minval, maxval, descent_info):
        """ LSF requires random directions. These are produced here. """
        mode = descent_info.direction_mode
        ratio_points_for_continuous = descent_info.ratio_points_for_continuous

        if mode=="random":
            values = jax.random.uniform(key, shape, minval=minval, maxval=maxval)

        elif mode=="continuous":
            x_new = jnp.linspace(0, 1, shape[0])
            N = int(ratio_points_for_continuous*shape[0])

            key1, key2 = jax.random.split(key, 2)
            x = jnp.sort(jax.random.choice(key1, x_new, (N, ), replace=False))
            points = jax.random.uniform(key2, (N, ), minval=minval, maxval=maxval)

            values = do_interpolation_1d(x_new, x, points)
            values = values/jnp.max(jnp.abs(values))*jnp.maximum(jnp.abs(minval), jnp.abs(maxval))
        else:
            raise NotImplementedError(f"direction_mode needs to be random or continuous. Not {mode}")

        return values
    


    def get_search_direction_individual(self, keys, individual, measurement_info, descent_info):
        """ Creates a pytree with random search directions for one individual. """
        key1, key2 = keys.pulse
        direction = MyNamespace(pulse=None, gate=None)

        shape_pulse = jnp.shape(individual.pulse)
        d_pulse_re = self.get_random_values(key1, shape_pulse, -1, 1, descent_info)
        d_pulse_im = self.get_random_values(key2, shape_pulse, -1, 1, descent_info)
        d = d_pulse_re + 1j*d_pulse_im
        direction_pulse = d/jnp.linalg.norm(d)
        direction = tree_at(lambda x: x.pulse, direction, direction_pulse, is_leaf=lambda x: x is None)

        if measurement_info.doubleblind==True:
            key3, key4 = keys.gate

            shape_gate = jnp.shape(individual.gate)
            d_gate_re = self.get_random_values(key3, shape_gate, -1, 1, descent_info)
            d_gate_im = self.get_random_values(key4, shape_gate, -1, 1, descent_info)
            d = d_gate_re + 1j*d_gate_im
            direction_gate = d/jnp.linalg.norm(d)
            direction = tree_at(lambda x: x.gate, direction, direction_gate, is_leaf=lambda x: x is None)

        return direction



    def get_search_direction(self, key, population, measurement_info, descent_info):
        """ Creates a pytree with random search directions for an entire population. """
        # maybe one can use jax.tree.map here
        leaves, treedef = jax.tree.flatten(population)
        keys = jax.random.split(key, len(leaves))
        keys = [jax.random.split(keys[i], jnp.shape(leaves[i])[0]*2).reshape(jnp.shape(leaves[i])[0], 2, 2) for i in range(len(leaves))]
        key_tree = jax.tree.unflatten(treedef, keys)
        return jax.vmap(self.get_search_direction_individual, in_axes=(0, 0, None, None))(key_tree, population, measurement_info, descent_info)




    def get_scalars(self, direction, signal, descent_info):
        """ Calculates scalars to identify the min/max of a search direction. """
        # solve jnp.abs(signal + s*direction)**2 = jnp.abs(boundary)**2
        p = 2*jnp.real(signal*jnp.conjugate(direction))/(jnp.abs(direction)**2 + 1e-9)
        q = (jnp.abs(signal)**2 - jnp.abs(descent_info.boundary)**2)/(jnp.abs(direction)**2 + 1e-9)

        diskriminante = p**2/4 - q
        diskriminante = jnp.maximum(diskriminante, 0)
        s1 = -p/2 - jnp.sqrt(diskriminante)
        s2 = -p/2 + jnp.sqrt(diskriminante)
        return jnp.max(s1, axis=1)[:, jnp.newaxis], jnp.min(s2, axis=1)[:, jnp.newaxis]


    

    def select_new_brackets_N_section(self, E_arr, error_arr, descent_state, pulse_or_gate):
        """ Selects the two trial-individuals with the lowest errors in order to narrow the search bracket. """
        idx1 = jnp.argmin(error_arr, axis=0)
        mask_func = lambda x,y: jnp.where(jnp.arange(jnp.shape(x)[0])==y, jnp.inf, x)
        error_arr_masked = jax.vmap(mask_func, in_axes=(1,0), out_axes=1)(error_arr, idx1)
        idx2 = jnp.argmin(error_arr_masked, axis=0)
        El = jax.vmap(Partial(jnp.take, axis=0), in_axes=(1,0))(E_arr, idx1)
        Er = jax.vmap(Partial(jnp.take, axis=0), in_axes=(1,0))(E_arr, idx2)

        return El, Er


    def select_new_brackets_bisection(self, E_arr, error_arr, descent_state, pulse_or_gate):
        """
        Narrows the search bracket by removing the trial-individual with the largest error or 
        the one that is the furthest from the trial guess.
        """

        def bisection_step_logic_0(El, Em, Er, signal):
            return Em, Er

        def bisection_step_logic_1(El, Em, Er, signal):
            dl = jnp.linalg.norm(signal - El)
            dr = jnp.linalg.norm(signal - Er)

            dl_smaller_dr = (dl < dr)
            El = El*dl_smaller_dr + Em*(1-dl_smaller_dr)
            Er = Em*dl_smaller_dr + Er*(1-dl_smaller_dr)
            return El, Er

        def bisection_step_logic_2(El, Em, Er, signal):
            return El, Em


        idx = jnp.argmax(error_arr, axis=0)
        population = getattr(descent_state.population, pulse_or_gate)

        El, Em, Er = E_arr
        El, Er = jax.vmap(jax.lax.switch, in_axes=(0, None, 0, 0, 0, None))(idx, 
                                                                            [bisection_step_logic_0, 
                                                                             bisection_step_logic_1, 
                                                                             bisection_step_logic_2], 
                                                                            El, Em, Er,
                                                                            population
                                                                            )
        return El, Er



    def generate_E_arr(self, El, Er, n):
        """ Takes in a range El, Er and disects it into n equal sections. """
        d = (Er - El)/n
        E_arr = El[None,:,:] + d[None,:,:]*jnp.arange(n+1)[:,None,None]
        return E_arr


    def bisection_step(self, El, Er, descent_state, measurement_info, descent_info, pulse_or_gate):
        """ Does one bisection step of the LSF algorithm. """

        E_arr = self.generate_E_arr(El, Er, descent_info.no_sections)
        error_arr = jax.vmap(self.calculate_error, in_axes=(0, None, None, None, None))(E_arr, descent_state, measurement_info, descent_info, pulse_or_gate)
        
        if descent_info.no_sections==2:
            El, Er = self.select_new_brackets_bisection(E_arr, error_arr, descent_state, pulse_or_gate)
        else:
            El, Er = self.select_new_brackets_N_section(E_arr, error_arr, descent_state, pulse_or_gate)

        return (El, Er), None
    



    def do_bisection_search(self, direction, descent_state, measurement_info, descent_info, pulse_or_gate):
        """ 
        Performs one bisection search to find the minimum along a given search direction. 
        The number of iterations is set through self.number_of_bisection_iterations
        """
        s1, s2 = self.get_scalars(direction, getattr(descent_state.population, pulse_or_gate), descent_info)

        El = getattr(descent_state.population, pulse_or_gate) + s1*direction
        Er = getattr(descent_state.population, pulse_or_gate) + s2*direction

        no_iterations = descent_info.number_of_disection_iterations
        do_bisection_step = Partial(self.bisection_step, descent_state=descent_state, measurement_info=measurement_info, 
                                    descent_info=descent_info, pulse_or_gate=pulse_or_gate)
        
        do_step = Partial(scan_helper, actual_function=do_bisection_step, number_of_args=2, number_of_xs=0)
        E_arr, _ = jax.lax.scan(do_step, (El, Er), length=no_iterations) 

        El, Er = E_arr
        E_arr = self.generate_E_arr(El, Er, descent_info.no_sections)
        error_arr = jax.vmap(self.calculate_error, in_axes=(0, None, None, None, None))(E_arr, descent_state, measurement_info, descent_info, pulse_or_gate)
        
        if descent_info.only_allow_improvements==True:
            # only improvements are accepted if this is active -> may cause stagnation?
            # it seems very unlikely to truly stagnate in such a high dimensional optimization problem
            error_arr = jnp.concatenate((error_arr, descent_state.error_arr[None,:]), axis=0)
            E_arr = jnp.concatenate((E_arr, getattr(descent_state.population, pulse_or_gate)[None,:,:]), axis=0)
        
        idx = jnp.argmin(error_arr, axis=0)
        population_new = jax.vmap(Partial(jnp.take, axis=0), in_axes=(1,0))(E_arr, idx)
        return population_new



    def search_along_direction(self, direction, descent_state, measurement_info, descent_info):
        """ Performs a bisection search along one direction for pulse and the for gate. """

        if descent_info.measured_spectrum_is_provided.pulse==True:
            direction_pulse = self.apply_spectrum(direction.pulse, measurement_info.spectral_amplitude.pulse, 
                                                  descent_info.eta_spectral_amplitude)
        else:
            direction_pulse = direction.pulse
            
        population_pulse = self.do_bisection_search(direction_pulse, descent_state, measurement_info, descent_info, "pulse")
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population_pulse)
        
        if measurement_info.doubleblind==True:  
            if descent_info.measured_spectrum_is_provided.gate==True:
                direction_gate = self.apply_spectrum(direction.gate, measurement_info.spectral_amplitude.gate, 
                                                     descent_info.eta_spectral_amplitude)
            else:
                direction_gate = direction.gate

            population_gate = self.do_bisection_search(direction_gate, descent_state, measurement_info, descent_info, "gate")
            descent_state = tree_at(lambda x: x.population.gate, descent_state, population_gate)

        return descent_state
    
    

    def make_population_bisection_search(self, E_arr, population, pulse_or_gate):
        """ Combines the trial populations with the current population such that the errors may be computed. """
        temp_dict = {"pulse": (E_arr, population.gate),
                     "gate": (population.pulse, E_arr)}
        
        pulse_arr, gate_arr = temp_dict[pulse_or_gate]
        return MyNamespace(pulse=pulse_arr, gate=gate_arr)




    def calculate_error(self, E_arr, descent_state, measurement_info, descent_info, pulse_or_gate):
        """ 
        Calculates the trace error. Since pulse and gate are optimized independently the population as provided to calculate_error_individual() 
        needs to be constructed from the current population and the fields in the cureent optimization.
        """
        population = self.make_population_bisection_search(E_arr, descent_state.population, pulse_or_gate)
        signal_t = self.generate_signal_t(MyNamespace(population=population), measurement_info, descent_info)
        _calculate_trace = Partial(calculate_trace, measured_trace=measurement_info.measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_local")
        
        trace, _ = jax.vmap(_calculate_trace)(signal_t.signal_f)
        error_arr = jax.vmap(calculate_trace_error, in_axes=(0,0,None))(descent_state.mu, trace, measurement_info.measured_trace)
        return error_arr



    def step(self, descent_state, measurement_info, descent_info):
        """
        Performs one step of the LSF-algorithm.

        Args:
            descent_state: Pytree, 
            measurement_info: Pytree,
            descent_info: Pytree,

        Returns:
            tuple[Pytree, jnp.array], the updated descent state, errors of the current population

        """
        
        key, subkey = jax.random.split(descent_state.key, 2)
        descent_state = tree_at(lambda x: x.key, descent_state, key)
        
        direction = self.get_search_direction(subkey, descent_state.population, measurement_info, descent_info)
        descent_state = self.search_along_direction(direction, descent_state, measurement_info, descent_info)

        population = jax.tree.map(lambda x: x/jnp.linalg.norm(x, axis=-1)[:,None], descent_state.population)
        descent_state = tree_at(lambda x: x.population, descent_state, population)

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)        
        _calculate_trace = Partial(calculate_trace, measured_trace=measurement_info.measured_trace, measurement_info=measurement_info, descent_info=descent_info, local_or_global="_global")
        trace, mu = jax.vmap(_calculate_trace)(signal_t.signal_f)
        error_arr = jax.vmap(calculate_trace_error, in_axes=(0,0,None))(mu, trace, measurement_info.measured_trace)
        
        descent_state = tree_at(lambda x: x.mu, descent_state, mu)
        descent_state = tree_at(lambda x: x.error_arr, descent_state, error_arr)
        return descent_state, error_arr.reshape(-1,1)
    



    def initialize_run(self, population):
        """
        Prepares all provided data and parameters for the reconstruction.
        Here the final shape/structure of descent_state, measurement_info and descent_info are determined. 

        Args:
            population: Pytree, the initial guess as created by self.create_initial_population()
        
        Returns:
            tuple[Pytree, Callable], the initial descent state, the step-function of the algorithm.

        """

        assert (0 < self.ratio_points_for_continuous < 1) | (self.direction_mode!="continuous")
        assert type(self.no_sections)==int, "Number of sections needs to be an int."
        assert self.no_sections > 1, "Number of sections needs to be greater than one."

        measurement_info = self.measurement_info
        self.descent_info = self.descent_info.expand(no_sections = self.no_sections,
                                                     number_of_disection_iterations = self.number_of_disection_iterations,
                                                     ratio_points_for_continuous = self.ratio_points_for_continuous,
                                                     direction_mode = self.direction_mode,
                                                     boundary = self.boundary,
                                                     optimize_calibration_curve = MyNamespace(_local=None,
                                                                                              _global=self.global_optimize_calibration_curve),
                                                    eta_spectral_amplitude=self.eta_spectral_amplitude,
                                                    only_allow_improvements=self.only_allow_improvements)
        descent_info = self.descent_info

        # normalization seems to be needed, i guess the calculation of s1, s2 is faulty otherwise
        population = jax.tree.map(lambda x: x/jnp.linalg.norm(x, axis=-1)[:,None], population)
        _, mu_init_global = initialize_mu(self, measurement_info, descent_info)
        self.descent_state = self.descent_state.expand(population = population,
                                                       key = self.key, 
                                                       mu = mu_init_global,
                                                       error_arr = jnp.full((descent_info.population_size,), jnp.inf))
        descent_state = self.descent_state

        do_step = Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step = Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step

