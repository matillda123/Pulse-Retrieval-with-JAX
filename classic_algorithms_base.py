import jax
import jax.numpy as jnp

from jax.tree_util import Partial


from linesearch import do_linesearch
from nonlinear_cg import get_nonlinear_CG_direction
from lbfgs import get_pseudo_newton_direction

from utilities import scan_helper, MyNamespace, do_fft, do_ifft, calculate_S_prime, calculate_mu, calculate_trace, calculate_trace_error, calculate_Z_error
from BaseClasses import AlgorithmsBASE





class GeneralizedProjectionBASE(AlgorithmsBASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.no_steps_gradient_descent = 15
        self.max_steps_linesearch = 25
        self.wolfe_linesearch = False
        self.gamma = 1e3
        self.delta_gamma = (0.5, 1.5)
        
        self.c1 = 1e-4
        self.c2 = 0.9

        self.use_hessian = False
        self.lambda_lm = 1e-3
        self.lbfgs_memory = 10

        self.use_conjugate_gradients = False
        self.beta_parameter_version = "fletcher_reeves"

        self.linalg_solver="lineax"



    
    def update_population(self, population, gamma, descent_direction, measurement_info, pulse_or_gate):
        population = jax.vmap(self.update_individual, in_axes=(0,0,0,None,None))(population, gamma, descent_direction, measurement_info, pulse_or_gate)
        return population
    

    def calc_Z_error_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        individual, descent_direction, signal_t_new = linesearch_info.population, linesearch_info.descent_direction, linesearch_info.signal_t_new
       
        transform_arr = measurement_info.transform_arr

        individual = self.update_individual(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        Z_error_new = calculate_Z_error(signal_t.signal_t, signal_t_new)
        return Z_error_new


    def gradient_descent_Z_error_step(self, signal_t, signal_t_new, Z_error, descent_state, measurement_info, descent_info, pulse_or_gate):        
        use_hessian, use_conjugate_gradients = descent_info.use_hessian, descent_info.use_conjugate_gradients

        population = descent_state.population
        transform_arr = measurement_info.transform_arr


        grad = self.calculate_Z_error_gradient(signal_t_new, signal_t, population, transform_arr, measurement_info, pulse_or_gate)
        gradient_sum = jnp.sum(grad, axis=1)


        if use_hessian=="diagonal" or use_hessian=="full":
            newton_direction = self.calculate_Z_error_newton_direction(grad, signal_t_new, signal_t, transform_arr, descent_state, measurement_info, descent_info, 
                                                                 use_hessian, pulse_or_gate)
            setattr(descent_state.hessian_state.newton_direction_prev, pulse_or_gate, newton_direction)

            descent_direction = -1*newton_direction

        elif use_hessian=="lbfgs":
            newton_direction, lbfgs_state = get_pseudo_newton_direction(gradient_sum, getattr(descent_state.lbfgs, pulse_or_gate), descent_info)
            descent_direction = -1*newton_direction

        else:
            descent_direction = -1*gradient_sum



        if use_conjugate_gradients==True:
            beta = descent_info.beta_parameter_version
            cg=descent_state.cg
            descent_direction_prev, CG_direction_prev = getattr(cg.descent_direction_prev, pulse_or_gate), getattr(cg.CG_direction_prev, pulse_or_gate)

            CG_direction=jax.vmap(get_nonlinear_CG_direction, in_axes=(0,0,0,None))(-1*descent_direction, descent_direction_prev, CG_direction_prev, beta)

            setattr(cg.descent_direction_prev, pulse_or_gate, -1*descent_direction)
            setattr(cg.CG_direction_prev, pulse_or_gate, CG_direction)
            descent_state.cg = cg
            
            descent_direction = CG_direction


        pk_dot_gradient = jax.vmap(lambda x,y: jnp.real(jnp.vdot(x,y)), in_axes=(0,0))(descent_direction, gradient_sum)
        
        linesearch_info=MyNamespace(population=population, descent_direction=descent_direction, signal_t_new=signal_t_new, 
                                    error=Z_error, pk_dot_gradient=pk_dot_gradient, pk=descent_direction)
        
        gamma = jax.vmap(do_linesearch, in_axes=(0,None,None,None,None))(linesearch_info, measurement_info, descent_info, 
                                                                           Partial(self.calc_Z_error_for_linesearch, pulse_or_gate=pulse_or_gate),
                                                                           Partial(self.calc_Z_grad_for_linesearch, pulse_or_gate=pulse_or_gate))

        if use_hessian=="lbfgs":
           step_size_arr = lbfgs_state.step_size_prev
           step_size_arr = step_size_arr.at[:,1:].set(step_size_arr[:,:-1])
           step_size_arr = step_size_arr.at[:,0].set(gamma[:,jnp.newaxis])

           lbfgs_state.step_size_prev = step_size_arr 
           setattr(descent_state.lbfgs, pulse_or_gate, lbfgs_state)

        descent_state.population = self.update_population(population, gamma, descent_direction, measurement_info, pulse_or_gate) 
        return descent_state
    



    def do_gradient_descent_Z_error_step(self, descent_state, signal_t_new, measurement_info, descent_info):
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        Z_error=jax.vmap(calculate_Z_error, in_axes=(0,0))(signal_t.signal_t, signal_t_new)


        descent_state=self.gradient_descent_Z_error_step(signal_t, signal_t_new, Z_error, descent_state, measurement_info, descent_info, "pulse")
        descent_state.population.pulse = jax.vmap(lambda x: x/jnp.linalg.norm(x))(descent_state.population.pulse)

        if measurement_info.doubleblind==True:
            descent_state=self.gradient_descent_Z_error_step(signal_t, signal_t_new, Z_error, descent_state, measurement_info, descent_info, "gate")

        return descent_state, None



    def do_gradient_descent_Z_error(self, descent_state, signal_t_new, measurement_info, descent_info):

        descent_state = self.initialize_CG(descent_state, measurement_info)
        descent_state = self.initialize_pseudo_newton(descent_state, measurement_info)    
        descent_state = self.initialize_lbfgs(descent_state, measurement_info, descent_info)
        
        do_gradient_descent_step=Partial(self.do_gradient_descent_Z_error_step, signal_t_new=signal_t_new, measurement_info=measurement_info, descent_info=descent_info)
        
        do_gradient_descent_step=Partial(scan_helper, actual_function=do_gradient_descent_step, number_of_args=1, number_of_xs=0)
        descent_state, _ =jax.lax.scan(do_gradient_descent_step, descent_state, length=descent_info.no_steps_gradient_descent)

        return descent_state
    
    

    def step(self, descent_state, measurement_info, descent_info):
        measured_trace = measurement_info.measured_trace
        sk, rn = measurement_info.sk, measurement_info.rn

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_f=do_fft(signal_t.signal_t, sk, rn)
        trace=calculate_trace(signal_f)

        mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,None,0,None))(signal_t.signal_t, measured_trace, mu, measurement_info)
        descent_state=self.do_gradient_descent_Z_error(descent_state, signal_t_new, measurement_info, descent_info)

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_f=do_fft(signal_t.signal_t, sk, rn)
        trace=calculate_trace(signal_f)
        trace_error=jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)

        return descent_state, trace_error.reshape(-1,1)
    




    def initialize_CG(self, descent_state, measurement_info):
        shape = jnp.shape(descent_state.population.pulse)
        init_arr = jnp.zeros(shape, dtype=jnp.complex64)

        descent_state.cg=MyNamespace(CG_direction_prev=MyNamespace(pulse=None, gate=None), 
                                     descent_direction_prev=MyNamespace(pulse=None, gate=None))
        descent_state.cg.descent_direction_prev.pulse = init_arr
        descent_state.cg.CG_direction_prev.pulse = init_arr

        if measurement_info.doubleblind==True:
            descent_state.cg.descent_direction_prev.gate = init_arr
            descent_state.cg.CG_direction_prev.gate = init_arr

        return descent_state
    
    
    def initialize_pseudo_newton(self, descent_state, measurement_info):
        shape = jnp.shape(descent_state.population.pulse)
        init_arr = jnp.zeros(shape, dtype=jnp.complex64)

        descent_state.hessian_state=MyNamespace(newton_direction_prev = MyNamespace(pulse=None, gate=None))
        descent_state.hessian_state.newton_direction_prev.pulse = init_arr

        if measurement_info.doubleblind==True:
            descent_state.hessian_state.newton_direction_prev.gate = init_arr

        return descent_state
    

    def initialize_lbfgs(self, descent_state, measurement_info, descent_info):
        shape = jnp.shape(descent_state.population.pulse)
        N = shape[0]
        n = shape[1]
        m = descent_info.lbfgs_memory

        init_arr1 = jnp.zeros((N,m,n), dtype=jnp.complex64)
        init_arr2 = jnp.zeros((N,m,1), dtype=jnp.float32)

        lbfgs_init_pulse = MyNamespace(grad_prev = init_arr1, newton_direction_prev = init_arr1, step_size_prev = init_arr2)
        if measurement_info.doubleblind==True:
            lbfgs_init_gate = MyNamespace(grad_prev = init_arr1, newton_direction_prev = init_arr1, step_size_prev = init_arr2)
        else:
            lbfgs_init_gate = None

        descent_state.lbfgs=MyNamespace(pulse=lbfgs_init_pulse, gate=lbfgs_init_gate)
        return descent_state


    def initialize_run(self, population):
        measurement_info=self.measurement_info

        # parameters for linesearch
        self.descent_info.c1=self.c1
        self.descent_info.c2=self.c2
        self.descent_info.wolfe_linesearch=self.wolfe_linesearch
        self.descent_info.gamma=self.gamma
        self.descent_info.delta_gamma=self.delta_gamma
        self.descent_info.max_steps_linesearch=self.max_steps_linesearch

        # parameters for gradient descent/damped newton method
        self.descent_info.use_hessian=self.use_hessian
        self.descent_info.lambda_lm=self.lambda_lm
        self.descent_info.no_steps_gradient_descent=self.no_steps_gradient_descent
        self.descent_info.linalg_solver=self.linalg_solver
        self.descent_info.lbfgs_memory = self.lbfgs_memory

        # parameters for nonlinear conjugate gradient method 
        self.descent_info.beta_parameter_version=self.beta_parameter_version
        self.descent_info.use_conjugate_gradients=self.use_conjugate_gradients
        descent_info=self.descent_info


        self.descent_state.population=population

        self.descent_state = self.initialize_CG(self.descent_state, measurement_info)
        self.descent_state = self.initialize_pseudo_newton(self.descent_state, measurement_info)
        self.descent_state = self.initialize_lbfgs(self.descent_state, measurement_info, descent_info)

        descent_state=self.descent_state

        do_step=Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step=Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step
    















class TimeDomainPtychographyBASE(AlgorithmsBASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_global_step=False
        self.use_hessian=False#"diagonal"

        self.alpha=1
        self.beta=1

        self.max_steps_linesearch=25
        self.gamma=self.beta

        self.c1=1e-3
        self.lambda_lm=1e-2

        self.linalg_solver = "lineax"


        self.delta_gamma = (0.5, 1.5)
        self.wolfe_linesearch = False
        self.c2 = 0.9




    def calculate_PIE_error(self, signal_f, measured_trace):
        return jnp.mean(jnp.abs(jnp.sqrt(jnp.abs(measured_trace))*jnp.sign(measured_trace) - jnp.abs(signal_f))**2)


    def get_PIE_weights(self, probe_shifted, alpha, PIE_method):

        """
        elif PIE_method=="lm":
            U=2/(jnp.abs(probe_shifted)**2+1e-6) # -> rPIE is eqivalent to pseudo-gauss-newton/levenberg-marquardt for small gamma. 

        gamma=>1 -> rPIE=>ePIE
        """
        
        if PIE_method=="PIE":
            U = 1/(jnp.abs(probe_shifted)**2+alpha*jnp.max(jnp.abs(probe_shifted)**2))*jnp.abs(probe_shifted)/jnp.max(jnp.abs(probe_shifted))
        elif PIE_method=="ePIE":
            U = 1/jnp.max(jnp.abs(probe_shifted)**2)
            U = U*jnp.ones(jnp.shape(probe_shifted))
        elif PIE_method=="rPIE":
            U = 1/((1-alpha)*jnp.abs(probe_shifted)**2+alpha*jnp.max(jnp.abs(probe_shifted)**2))
        elif PIE_method=="gradient":
            U = jnp.ones(jnp.shape(probe_shifted))
        else:
            print(f"PIE_method={PIE_method} not defined.")
        
        return U



    def do_local_iteration(self, descent_state, transform_arr_m, trace_line, measurement_info, descent_info):
        population = descent_state.population
        
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(population, transform_arr_m, measurement_info)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,0,None,None))(jnp.squeeze(signal_t.signal_t), trace_line, 1, measurement_info)


        population = self.update_population_local(population, signal_t, signal_t_new, transform_arr_m, measurement_info, descent_info, "pulse")

        if measurement_info.doubleblind==True:
            population = self.update_population_local(population, signal_t, signal_t_new, transform_arr_m, measurement_info, descent_info, "gate")

        descent_state.population = population
        return descent_state, None
    





    def get_descent_direction(self, grad, U):
        descent_direction=-1*jnp.sum(grad*U, axis=1)
        return descent_direction


    def calculate_pk_dot_gradient(self, gradient_sum, descent_direction):
        pk_dot_gradient=jax.vmap(lambda x,y: jnp.real(jnp.dot(jnp.conjugate(x),y)), in_axes=(0,0))(descent_direction, gradient_sum)
        return pk_dot_gradient
        

    def calc_error_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        transform_arr, measured_trace = measurement_info.transform_arr, measurement_info.measured_trace
        sk, rn = measurement_info.sk, measurement_info.rn

        individual, descent_direction = linesearch_info.population, linesearch_info.descent_direction

        individual = self.update_individual_global(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        error_new=self.calculate_PIE_error(do_fft(signal_t.signal_t, sk, rn), measured_trace)
        return error_new
    

    def calc_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate):
        transform_arr, measured_trace = measurement_info.transform_arr, measurement_info.measured_trace
        individual, descent_direction = linesearch_info.population, linesearch_info.descent_direction
        
        individual = self.update_individual_global(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        signal_t_new = calculate_S_prime(signal_t.signal_t, measured_trace, 1, measurement_info)
        grad_all_m,U = self.calculate_PIE_descent_direction(individual, signal_t, signal_t_new, measurement_info, descent_info, pulse_or_gate)
        return jnp.sum(grad_all_m, axis=1)
    

    def do_global_step(self, descent_state, measurement_info, descent_info, pulse_or_gate):
        measured_trace = measurement_info.measured_trace
        sk, rn = measurement_info.sk, measurement_info.rn

        use_hessian = descent_info.use_hessian
        population = descent_state.population

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,None,None,None))(signal_t.signal_t, measured_trace, 1, measurement_info)
        signal_f = do_fft(signal_t.signal_t, sk, rn)
        pie_error = jax.vmap(self.calculate_PIE_error, in_axes=(0,None))(signal_f, measured_trace)

        grad, U = self.calculate_PIE_descent_direction(population, signal_t, signal_t_new, measurement_info, descent_info, pulse_or_gate)
        gradient_sum = jnp.sum(grad, axis=1)

        
        if use_hessian!=False:
            descent_direction = self.calculate_PIE_descent_direction_hessian(grad, signal_t, descent_state, measurement_info, descent_info, pulse_or_gate)
            setattr(descent_state.hessian_state.newton_direction_prev, pulse_or_gate, -1*descent_direction)

        else:
            descent_direction = self.get_descent_direction(grad, U)


        pk_dot_gradient = self.calculate_pk_dot_gradient(gradient_sum, descent_direction)

        linesearch_info=MyNamespace(population=population, signal_t=signal_t, descent_direction=descent_direction, 
                                    pk_dot_gradient=pk_dot_gradient, pk=gradient_sum, error=pie_error)     


        gamma = jax.vmap(do_linesearch, in_axes=(0, None, None, None, None))(linesearch_info, measurement_info, descent_info, 
                                                                             Partial(self.calc_error_for_linesearch, pulse_or_gate=pulse_or_gate),
                                                                             Partial(self.calc_grad_for_linesearch, descent_info=descent_info, pulse_or_gate=pulse_or_gate),)

        descent_state.population = self.update_population_global(population, gamma, descent_direction, measurement_info, pulse_or_gate)
        return descent_state     


    
    
    def global_step(self, descent_state, measurement_info, descent_info):
        descent_state=self.do_global_step(descent_state, measurement_info, descent_info, "pulse")

        if measurement_info.doubleblind==True:
            descent_state=self.do_global_step(descent_state, measurement_info, descent_info, "gate")

        return descent_state





    def step(self, descent_state, measurement_info, descent_info):
        sk, rn = measurement_info.sk, measurement_info.rn

        transform_arr, measured_trace, descent_state = self.shuffle_data_along_m(descent_state, measurement_info, descent_info)

        local_iteration=Partial(self.do_local_iteration, measurement_info=measurement_info, descent_info=descent_info)
        local_iteration=Partial(scan_helper, actual_function=local_iteration, number_of_args=1, number_of_xs=2)

        descent_state, _ = jax.lax.scan(local_iteration, descent_state, (transform_arr, measured_trace))

        if descent_info.use_global_step!=False:
            descent_state=self.global_step(descent_state, measurement_info, descent_info)


        descent_state.population.pulse = jax.vmap(lambda x: x/jnp.linalg.norm(x))(descent_state.population.pulse)

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_f=do_fft(signal_t.signal_t, sk, rn)
        trace=calculate_trace(signal_f)
        trace_error=jax.vmap(calculate_trace_error, in_axes=(0, None))(trace, measurement_info.measured_trace)

        return descent_state, trace_error.reshape(-1,1)



    def initialize_run(self, population):
        measurement_info=self.measurement_info

        # basic parameters for PIE
        self.descent_info.alpha=self.alpha
        self.descent_info.beta=self.beta
        self.descent_info.PIE_method=self.PIE_method

        # parameters for optional global step
        self.descent_info.use_global_step=self.use_global_step
        self.descent_info.use_hessian=self.use_hessian
        self.descent_info.lambda_lm=self.lambda_lm

        # parameters for linesearch in global step
        self.descent_info.c1=self.c1
        self.descent_info.gamma=self.gamma
        self.descent_info.delta_gamma=self.delta_gamma
        self.descent_info.max_steps_linesearch=self.max_steps_linesearch

        self.descent_info.c2 = self.c2
        self.descent_info.wolfe_linesearch = self.wolfe_linesearch

        # randomize local iterations with these 
        self.descent_info.idx_arr=self.idx_arr
        self.descent_info.linalg_solver=self.linalg_solver

        descent_info=self.descent_info

        self.descent_state.key=self.key
        self.descent_state.population=population

        self.descent_state.hessian_state=MyNamespace(newton_direction_prev=MyNamespace(pulse=None,gate=None))
        self.descent_state.hessian_state.newton_direction_prev.pulse = jnp.zeros(jnp.shape(population.pulse), dtype=jnp.complex64)
        if self.measurement_info.doubleblind==True:
            self.descent_state.hessian_state.newton_direction_prev.gate = jnp.zeros(jnp.shape(population.gate), dtype=jnp.complex64)
    
        descent_state=self.descent_state

        do_step=Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step=Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step
    













class COPRABASE(AlgorithmsBASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.weights=1.0

        self.gamma=0.25
        self.beta=1

        self.delta_gamma=0.5
        self.max_steps_linesearch=25
        self.c1=1e-4
        self.c2=0.9
        self.wolfe_linesearch=False


        self.use_hessian="diagonal"
        self.lambda_lm=1e-3

        self.r_gradient="intensity"

        self.child_class="COPRA"
        self.xi=1e-9

        self.linalg_solver="lineax"



    def one_local_iteration(self, descent_state, transform_arr_m, trace_line, measurement_info, descent_info, pulse_or_gate):
        use_hessian = descent_info.hessian
        mu = descent_state.local.mu
        population = descent_state.population

        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(population, transform_arr_m, measurement_info)
        signal_t_new=jax.vmap(calculate_S_prime, in_axes=(0,0,0,None))(signal_t.signal_t, trace_line, mu, measurement_info)

        grad = self.calculate_Z_gradient(signal_t_new, signal_t, population, transform_arr_m, measurement_info, pulse_or_gate, local=True)
        
        if use_hessian.local_hessian!=False:
            newton_direction = self.calculate_Z_error_newton_direction(grad, signal_t_new, signal_t, transform_arr_m, descent_state, measurement_info, descent_info, 
                                                                       use_hessian.local_hessian, pulse_or_gate, local=True)
            setattr(descent_state.hessian_state.newton_direction_prev, pulse_or_gate, newton_direction)

            descent_direction = -1*newton_direction
        else:
            descent_direction = -1*jnp.squeeze(grad)
            

        grad_norm = jnp.sum(jnp.abs(descent_direction)**2)
        max_grad_norm = getattr(descent_state.local.max_grad_norm, pulse_or_gate)
        max_grad_norm = jnp.greater(grad_norm, max_grad_norm)*grad_norm + jnp.greater(max_grad_norm, grad_norm)*max_grad_norm
        setattr(descent_state.local.max_grad_norm, pulse_or_gate, max_grad_norm)

        Z_error = jax.vmap(calculate_Z_error, in_axes=(0,0))(signal_t.signal_t, signal_t_new)
        gamma = Z_error/max_grad_norm

        descent_state.population = self.update_population_local(population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate)
        return  descent_state, Z_error
    




    def do_one_local_iteration(self, descent_state, tau, trace_line, measurement_info, descent_info):
        descent_state, Z_error=self.one_local_iteration(descent_state, tau, trace_line, measurement_info, descent_info, "pulse")

        if measurement_info.doubleblind==True:
           descent_state, Z_error = self.one_local_iteration(descent_state, tau, trace_line, measurement_info, descent_info, "gate")

        return descent_state, Z_error
    

    

    def step_local_iteration(self, descent_state, measurement_info, descent_info):
        sk, rn = measurement_info.sk, measurement_info.rn

        signal_t=self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace=calculate_trace(do_fft(signal_t.signal_t, sk, rn))
        descent_state.local.mu=jax.vmap(calculate_mu, in_axes=(0,None))(trace, measurement_info.measured_trace)

        one_local_iteration=Partial(self.do_one_local_iteration, measurement_info=measurement_info, descent_info=descent_info)
        one_local_iteration=Partial(scan_helper, actual_function=one_local_iteration, number_of_args=1, number_of_xs=2)

        transform_arr, measured_trace, descent_state = self.shuffle_data_along_m(descent_state, measurement_info, descent_info)
        descent_state, Z_error = jax.lax.scan(one_local_iteration, descent_state, (transform_arr, measured_trace))


        signal_t=self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace=calculate_trace(do_fft(signal_t.signal_t, sk, rn))
        trace_error=jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measurement_info.measured_trace)

        return descent_state, trace_error
    



    def update_population_global(self, population, gamma, eta, descent_direction, measurement_info, descent_info, pulse_or_gate):
        population = jax.vmap(self.update_individual_global, in_axes=(0,0,0,0,None,None,None))(population, gamma, eta, descent_direction, 
                                                                                               measurement_info, descent_info, pulse_or_gate)
        return population

    
    def calculate_r_gradient_intensity(self, signal_f, trace, measured_trace, weights, sk, rn):
        mu=calculate_mu(trace, measured_trace)
        grad_r=-4*mu*do_ifft(signal_f*(measured_trace-mu*trace)*weights**2, sk, rn)
        return grad_r 
    
    
    def calculate_r_gradient_amplitude(self, signal_f, measured_trace, weights, sk, rn):
        mu=jnp.sum(jnp.sqrt(measured_trace)*jnp.abs(signal_f))/jnp.sum(jnp.abs(signal_f)**2)
        grad_r=-4*mu*do_ifft((jnp.sqrt(measured_trace)*jnp.exp(1j*jnp.angle(signal_f))-mu*signal_f)*weights, sk, rn)
        return grad_r
    


    def calculate_r_gradient(self, signal_f, trace, measurement_info, descent_info):
        measured_trace, sk, rn = measurement_info.measured_trace, measurement_info.sk, measurement_info.rn
        weights = descent_info.weights

        if descent_info.r_gradient=="amp":
            gradient=jax.vmap(self.calculate_r_gradient_amplitude, in_axes=(0,None,None,None,None))(signal_f, measured_trace, weights, sk, rn)
        elif descent_info.r_gradient=="intensity":
            gradient=jax.vmap(self.calculate_r_gradient_intensity, in_axes=(0,0,None,None,None,None))(signal_f, trace, measured_trace, weights, sk, rn)
        else:
            print("r_gradient has to be amp or intensity")

        return gradient



    def calculate_r_error(self, trace, measured_trace):
        mu=calculate_mu(trace, measured_trace)
        return jnp.sum(jnp.abs(mu*trace-measured_trace)**2)
    


    def calc_r_error_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        measured_trace, sk, rn = measurement_info.measured_trace, measurement_info.sk, measurement_info.rn 
        
        descent_direction, eta = linesearch_info.descent_direction, linesearch_info.eta
        signal_t = linesearch_info.signal_t

        signal_t_new = signal_t.signal_t + gamma*eta*descent_direction
        trace = calculate_trace(do_fft(signal_t_new, sk, rn))
        error = self.calculate_r_error(trace, measured_trace)
        return error
    

    def calc_r_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate):
        measured_trace, sk, rn = measurement_info.measured_trace, measurement_info.sk, measurement_info.rn 
        weights = descent_info.weights
        
        descent_direction, eta = linesearch_info.descent_direction, linesearch_info.eta
        signal_t = linesearch_info.signal_t

        signal_t_new = signal_t.signal_t + gamma*eta*descent_direction
        signal_f = do_fft(signal_t_new, sk, rn)
        trace = calculate_trace(signal_f)

        if descent_info.r_gradient=="amp":
            gradient = self.calculate_r_gradient_amplitude(signal_f, measured_trace, weights, sk, rn)
        elif descent_info.r_gradient=="intensity":
            gradient = self.calculate_r_gradient_intensity(signal_f, trace, measured_trace, weights, sk, rn)
        else:
            print("r_gradient has to be amp or intensity")
        
        return gradient
    


    def calc_Z_error_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate):
        transform_arr = measurement_info.transform_arr

        signal_t_new, eta, descent_direction = linesearch_info.signal_t_new, linesearch_info.eta, linesearch_info.descent_direction

        individual = self.update_population_global(linesearch_info.population, gamma, eta, descent_direction, measurement_info, descent_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        error = calculate_Z_error(signal_t.signal_t, signal_t_new)
        return error



    def do_global_iteration(self, descent_state, measurement_info, descent_info, pulse_or_gate):
        transform_arr, measured_trace = measurement_info.transform_arr, measurement_info.measured_trace
        gamma, use_hessian = descent_info.gamma, descent_info.hessian
        xi = descent_info.xi
        sk, rn = measurement_info.sk, measurement_info.rn

        population = descent_state.population

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_f=do_fft(signal_t.signal_t, sk, rn)
        trace=calculate_trace(signal_f)

        gradient = self.calculate_r_gradient(signal_f, trace, measurement_info, descent_info)
        descent_direction = -1*gradient

        r_error = jax.vmap(self.calculate_r_error, in_axes=(0,None))(trace, measured_trace)
        eta = r_error/(jnp.sum(jnp.abs(descent_direction)**2, axis=(1,2)) + xi)


        pk_dot_gradient = jax.vmap(lambda x,y: jnp.real(jnp.vdot(x,y)), in_axes=(0,0))(descent_direction, gradient)        
        linesearch_info=MyNamespace(signal_t=signal_t, descent_direction=descent_direction, error=r_error, 
                                    pk_dot_gradient=pk_dot_gradient, pk=descent_direction, eta=eta)
        
        gamma = jax.vmap(do_linesearch, in_axes=(0,None,None,None,None))(linesearch_info, measurement_info, descent_info, 
                                                                         Partial(self.calc_r_error_for_linesearch, descent_info=descent_info, pulse_or_gate=pulse_or_gate),
                                                                         Partial(self.calc_r_grad_for_linesearch, descent_info=descent_info, pulse_or_gate=pulse_or_gate))

        signal_t_new = signal_t.signal_t + gamma*eta[:, jnp.newaxis, jnp.newaxis]*descent_direction



        grad = self.calculate_Z_gradient(signal_t_new, signal_t, population, transform_arr, measurement_info, pulse_or_gate, local=False)
        grad_sum = jnp.sum(grad, axis=1)

        if use_hessian.global_hessian!=False:
            newton_direction = self.calculate_Z_error_newton_direction(grad, signal_t_new, signal_t, transform_arr, descent_state, measurement_info, descent_info, 
                                                                       use_hessian.global_hessian, pulse_or_gate, local=False)
            setattr(descent_state.hessian_state.newton_direction_prev, pulse_or_gate, newton_direction)
            descent_direction = -1*newton_direction
        else: 
            descent_direction = -1*grad_sum

        Z_error = jax.vmap(calculate_Z_error, in_axes=(0,0))(signal_t.signal_t, signal_t_new)
        eta = Z_error/(jnp.sum(jnp.abs(descent_direction)**2, axis=1) + xi)


        pk_dot_gradient = jax.vmap(lambda x,y: jnp.real(jnp.vdot(x,y)), in_axes=(0,0))(descent_direction, grad_sum)        
        linesearch_info=MyNamespace(population=population, signal_t_new=signal_t_new, descent_direction=descent_direction, error=Z_error, 
                                    pk_dot_gradient=pk_dot_gradient, pk=descent_direction, eta=eta)
        
        gamma = jax.vmap(do_linesearch, in_axes=(0,None,None,None,None))(linesearch_info, measurement_info, descent_info, 
                                                                         Partial(self.calc_Z_error_for_linesearch, descent_info=descent_info, pulse_or_gate=pulse_or_gate),
                                                                         Partial(self.calc_Z_grad_for_linesearch, descent_info=descent_info, pulse_or_gate=pulse_or_gate))


        descent_state.population = self.update_population_global(population, gamma, eta, descent_direction, measurement_info, descent_info, pulse_or_gate)
        return descent_state
    


    def step_global_iteration(self, descent_state, measurement_info, descent_info):
        measured_trace, sk, rn = measurement_info.measured_trace, measurement_info.sk, measurement_info.rn 

        descent_state=self.do_global_iteration(descent_state, measurement_info, descent_info, "pulse")

        if measurement_info.doubleblind==True:
            descent_state=self.do_global_iteration(descent_state, measurement_info, descent_info, "gate")


        descent_state.population.pulse = jax.vmap(lambda x: x/jnp.linalg.norm(x))(descent_state.population.pulse)

        signal_t=self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace=calculate_trace(do_fft(signal_t.signal_t, sk, rn))
        trace_error=jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)

        return descent_state, trace_error
    





    def initialize_local_stage(self, descent_state, measurement_info, descent_info):
        transform_arr, measured_trace = measurement_info.transform_arr, measurement_info.measured_trace
        sk, rn = measurement_info.sk, measurement_info.rn
        population = descent_state.population

        signal_t=self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace=calculate_trace(do_fft(signal_t.signal_t, sk, rn))
        mu=jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        signal_t_new=jax.vmap(calculate_S_prime, in_axes=(0,None,0,None))(signal_t.signal_t, measured_trace, mu, measurement_info)

        
        grad = self.calculate_Z_gradient(signal_t_new, signal_t, population, transform_arr, measurement_info, "pulse")
        descent_state.local.max_grad_norm.pulse=jnp.max(jnp.sum(jnp.abs(grad)**2, axis=1))

        grad = self.calculate_Z_gradient(signal_t_new, signal_t, population, transform_arr, measurement_info, "gate")
        descent_state.local.max_grad_norm.gate=jnp.max(jnp.sum(jnp.abs(grad)**2, axis=1))


        do_local=Partial(self.step_local_iteration, measurement_info=measurement_info, descent_info=descent_info)
        do_local=Partial(scan_helper, actual_function=do_local, number_of_args=1, number_of_xs=0)
        return descent_state, do_local
    


    def initialize_global_stage(self, descent_state, measurement_info, descent_info):
        do_global=Partial(self.step_global_iteration, measurement_info=measurement_info, descent_info=descent_info)
        do_global=Partial(scan_helper, actual_function=do_global, number_of_args=1, number_of_xs=0)
        return descent_state, do_global




    def initialize_run(self, population):
        print("this should be split such that everything happens inside initialize_local/global")
        measurement_info=self.measurement_info
        
        # general parameters, gamma (global), beta (local)
        self.descent_info.gamma=self.gamma
        self.descent_info.beta=self.beta
        self.descent_info.xi=self.xi

        self.descent_info.delta_gamma=self.delta_gamma
        self.descent_info.max_steps_linesearch=self.max_steps_linesearch
        self.descent_info.c1 = self.c1
        self.descent_info.c2 = self.c2
        self.descent_info.wolfe_linesearch = self.wolfe_linesearch

        # parameters for optional modifications -> damped pseudo-hessian Z-error, r-gradient for amplitude loss
        if type(self.use_hessian)==tuple or type(self.use_hessian)==list:
            local_hessian, global_hessian = self.use_hessian
        else:
            local_hessian = global_hessian = self.use_hessian

        self.descent_info.hessian = MyNamespace(local_hessian=local_hessian, global_hessian=global_hessian)
        self.descent_info.lambda_lm=self.lambda_lm
        self.descent_info.r_gradient=self.r_gradient
        self.descent_info.weights = self.weights

        # more general parameters
        self.descent_info.linalg_solver=self.linalg_solver
        self.descent_info.idx_arr=self.idx_arr

        descent_info=self.descent_info

        self.descent_state.key=self.key
        self.descent_state.population=population
        self.descent_state.hessian_state=MyNamespace(newton_direction_prev=MyNamespace(pulse=None, gate=None))
        self.descent_state.hessian_state.newton_direction_prev.pulse=jnp.zeros(jnp.shape(population.pulse), dtype=jnp.complex64)
        self.descent_state.hessian_state.newton_direction_prev.gate=jnp.zeros(jnp.shape(population.gate), dtype=jnp.complex64)

        self.descent_state.local=MyNamespace()
        self.descent_state.local.max_grad_norm=MyNamespace(pulse=0.0, gate=0.0)
        self.descent_state.local.mu=jnp.ones(self.descent_info.population_size)
        descent_state=self.descent_state


        descent_state, do_local = self.initialize_local_stage( descent_state, measurement_info, descent_info)
        descent_state, do_global = self.initialize_global_stage( descent_state, measurement_info, descent_info)

        return descent_state, do_local, do_global
    




    def run(self, population, no_iterations_local, no_iterations_global):

        descent_state, do_local, do_global = self.initialize_run(population)

        descent_state, error_arr_local = jax.lax.scan(do_local, descent_state, length=no_iterations_local)
        descent_state, error_arr_global = jax.lax.scan(do_global, descent_state, length=no_iterations_global)

        error_arr=jnp.vstack([error_arr_local, error_arr_global])

        final_result = self.post_process(descent_state, error_arr)
        return final_result



