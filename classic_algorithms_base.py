import jax
import jax.numpy as jnp

from jax.tree_util import Partial
from equinox import tree_at


from stepsize import do_linesearch, adaptive_scaling_of_step
from nonlinear_cg import get_nonlinear_CG_direction
from lbfgs import get_pseudo_newton_direction

from utilities import scan_helper, MyNamespace, do_fft, do_ifft, calculate_mu, calculate_trace, calculate_trace_error, calculate_Z_error, run_scan
from BaseClasses import AlgorithmsBASE

from update_signal import calculate_S_prime






def initialize_CG(descent_state, measurement_info):
    init_arr = jnp.zeros(jnp.shape(descent_state.population.pulse), dtype=jnp.complex64)

    cg_pulse = MyNamespace(CG_direction_prev = init_arr, 
                           descent_direction_prev = init_arr)

    if measurement_info.doubleblind==True:
        cg_gate = MyNamespace(CG_direction_prev = init_arr, 
                              descent_direction_prev = init_arr)
    else:
        cg_gate = None

    descent_state = descent_state.expand(cg = MyNamespace(pulse=cg_pulse, gate=cg_gate))
    return descent_state




def get_init_state_pseudo_newton(shape, measurement_info):
    init_arr1 = jnp.zeros(shape, dtype=jnp.complex64)
    init_arr2 = jnp.broadcast_to(jnp.eye(shape[1], dtype=jnp.complex64), (shape[0], shape[1], shape[1]))

    hessian_pulse = MyNamespace(newton_direction_prev=init_arr1, hessian=init_arr2)
    if measurement_info.doubleblind==True:
        hessian_gate = MyNamespace(newton_direction_prev=init_arr1, hessian=init_arr2)
    else:
        hessian_gate = None
    hessian=MyNamespace(pulse=hessian_pulse, gate=hessian_gate)
    return hessian


def initialize_pseudo_newton(descent_state, measurement_info):
    shape = jnp.shape(descent_state.population.pulse)
    hessian = get_init_state_pseudo_newton(shape, measurement_info)
    descent_state = descent_state.expand(hessian=hessian)
    return descent_state



def initialize_lbfgs(descent_state, measurement_info, descent_info):
    shape = jnp.shape(descent_state.population.pulse)
    N = shape[0]
    n = shape[1]
    m = descent_info.hessian.lbfgs_memory

    init_arr1 = jnp.zeros((N,m,n), dtype=jnp.complex64)
    init_arr2 = jnp.zeros((N,m,1), dtype=jnp.float32)

    lbfgs_init_pulse = MyNamespace(grad_prev = init_arr1, newton_direction_prev = init_arr1, step_size_prev = init_arr2)
    if measurement_info.doubleblind==True:
        lbfgs_init_gate = MyNamespace(grad_prev = init_arr1, newton_direction_prev = init_arr1, step_size_prev = init_arr2)
    else:
        lbfgs_init_gate = None

    descent_state = descent_state.expand(lbfgs=MyNamespace(pulse=lbfgs_init_pulse, gate=lbfgs_init_gate))
    return descent_state







def initialize_linesearch_info(optimizer):
    linesearch_params = MyNamespace(use_linesearch=optimizer.use_linesearch, 
                                    c1=optimizer.c1, 
                                    c2=optimizer.c2, 
                                    max_steps=optimizer.max_steps_linesearch, 
                                    delta_gamma=optimizer.delta_gamma)
    return linesearch_params


def initialize_S_prime_params(optimizer):
    s_prime_params = MyNamespace(local_method=optimizer.r_local_method, 
                                 global_method=optimizer.r_global_method, 
                                 number_of_iterations=optimizer.r_no_iterations, 
                                 r_gradient=optimizer.r_gradient, 
                                 r_hessian=optimizer.r_hessian, 
                                 weights=optimizer.r_weights)
    return s_prime_params



def initialize_hessian_info(optimizer):
    hessian = MyNamespace(local_hessian=optimizer.local_hessian, 
                        global_hessian=optimizer.global_hessian, 
                        linalg_solver=optimizer.linalg_solver, 
                        lambda_lm=optimizer.lambda_lm,
                        lbfgs_memory=optimizer.lbfgs_memory)

    return hessian












class GeneralizedProjectionBASE(AlgorithmsBASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "GeneralizedProjection"

        self.gamma = 1e3
        self.no_steps_descent = 15

        self.use_linesearch = False
        self.max_steps_linesearch = 25
        self.c1 = 1e-4
        self.c2 = 0.9
        self.delta_gamma = (0.5, 1.5)
    

        self.local_hessian = None
        self.global_hessian = False
        self.lambda_lm = 1e-3
        self.lbfgs_memory = 10
        self.linalg_solver="lineax"

        self.use_conjugate_gradients = False


        self.xi = 1e-9

        self.r_local_method = None
        self.r_global_method = "projection"
        self.r_gradient = "intensity"
        self.r_hessian = False
        self.r_weights = 1
        self.r_no_iterations = 1



    
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
        hessian, conjugate_gradients = descent_info.hessian, descent_info.conjugate_gradients

        population = descent_state.population
        transform_arr = measurement_info.transform_arr


        grad = self.calculate_Z_error_gradient(signal_t_new, signal_t, population, transform_arr, measurement_info, pulse_or_gate)
        gradient_sum = jnp.sum(grad, axis=1)


        if hessian.global_hessian=="diagonal" or hessian.global_hessian=="full":
            descent_direction, hessian_state = self.calculate_Z_error_newton_direction(grad, signal_t_new, signal_t, transform_arr, descent_state, 
                                                                                       measurement_info, descent_info, hessian.global_hessian, pulse_or_gate)
            descent_state = tree_at(lambda x: getattr(x.hessian, pulse_or_gate), descent_state, hessian_state)

        elif hessian.global_hessian=="lbfgs":
            descent_direction, lbfgs_state = get_pseudo_newton_direction(gradient_sum, getattr(descent_state.lbfgs, pulse_or_gate), descent_info)

        else:
            descent_direction = -1*gradient_sum



        if conjugate_gradients!=False:
            cg = getattr(descent_state.cg, pulse_or_gate)
            descent_direction, cg =jax.vmap(get_nonlinear_CG_direction, in_axes=(0,0,None))(descent_direction, cg, conjugate_gradients)
            descent_state = tree_at(lambda x: getattr(x.cg, pulse_or_gate), descent_state, cg)



        descent_direction = adaptive_scaling_of_step(descent_direction, Z_error, gradient_sum, getattr(descent_state.hessian, pulse_or_gate), descent_info)


        if descent_info.linesearch_params.use_linesearch=="backtracking" or descent_info.linesearch_params.use_linesearch=="wolfe":
            pk_dot_gradient = jax.vmap(lambda x,y: jnp.real(jnp.vdot(x,y)), in_axes=(0,0))(descent_direction, gradient_sum)
            
            linesearch_info=MyNamespace(population=population, descent_direction=descent_direction, signal_t_new=signal_t_new, 
                                        error=Z_error, pk_dot_gradient=pk_dot_gradient, pk=descent_direction)
            
            gamma = jax.vmap(do_linesearch, in_axes=(0,None,None,None,None))(linesearch_info, measurement_info, descent_info, 
                                                                             Partial(self.calc_Z_error_for_linesearch, pulse_or_gate=pulse_or_gate),
                                                                             Partial(self.calc_Z_grad_for_linesearch, pulse_or_gate=pulse_or_gate))
        else:
            gamma = jnp.ones(descent_info.population_size)*descent_info.gamma



        if hessian.global_hessian=="lbfgs":
           step_size_arr = lbfgs_state.step_size_prev
           step_size_arr = step_size_arr.at[:,1:].set(step_size_arr[:,:-1])
           step_size_arr = step_size_arr.at[:,0].set(gamma[:, jnp.newaxis])

           descent_state = tree_at(lambda x: getattr(x.lbfgs, pulse_or_gate).step_sizeprev, descent_state, step_size_arr)

        population = self.update_population(population, gamma, descent_direction, measurement_info, pulse_or_gate) 
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        return descent_state
    



    def do_gradient_descent_Z_error_step(self, descent_state, signal_t_new, measurement_info, descent_info):
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        Z_error = jax.vmap(calculate_Z_error, in_axes=(0,0))(signal_t.signal_t, signal_t_new)


        descent_state = self.gradient_descent_Z_error_step(signal_t, signal_t_new, Z_error, descent_state, measurement_info, descent_info, "pulse")
        population_pulse = jax.vmap(lambda x: x/jnp.linalg.norm(x))(descent_state.population.pulse)
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population_pulse)

        if measurement_info.doubleblind==True:
            descent_state=self.gradient_descent_Z_error_step(signal_t, signal_t_new, Z_error, descent_state, measurement_info, descent_info, "gate")

        return descent_state, None



    def do_gradient_descent_Z_error(self, descent_state, signal_t_new, measurement_info, descent_info):

        descent_state = initialize_CG(descent_state, measurement_info)
        descent_state = initialize_pseudo_newton(descent_state, measurement_info)
        descent_state = initialize_lbfgs(descent_state, measurement_info, descent_info)
        
        do_gradient_descent_step=Partial(self.do_gradient_descent_Z_error_step, signal_t_new=signal_t_new, measurement_info=measurement_info, descent_info=descent_info)
        
        do_gradient_descent_step=Partial(scan_helper, actual_function=do_gradient_descent_step, number_of_args=1, number_of_xs=0)
        descent_state, _ =jax.lax.scan(do_gradient_descent_step, descent_state, length=descent_info.no_steps_descent)

        return descent_state
    
    

    def step(self, descent_state, measurement_info, descent_info):
        measured_trace = measurement_info.measured_trace
        sk, rn = measurement_info.sk, measurement_info.rn

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_f = do_fft(signal_t.signal_t, sk, rn)
        trace = calculate_trace(signal_f)

        mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        get_S_prime = Partial(calculate_S_prime, method=descent_info.s_prime_params.global_method)
        signal_t_new = jax.vmap(get_S_prime, in_axes=(0,None,0,None,None))(signal_t.signal_t, measured_trace, mu, measurement_info, descent_info)
        descent_state = self.do_gradient_descent_Z_error(descent_state, signal_t_new, measurement_info, descent_info)

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_f = do_fft(signal_t.signal_t, sk, rn)
        trace = calculate_trace(signal_f)
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)

        return descent_state, trace_error.reshape(-1,1)
    




    


    def initialize_run(self, population):
        measurement_info = self.measurement_info


        linesearch_params = initialize_linesearch_info(self)
        hessian = initialize_hessian_info(self)
        s_prime_params = initialize_S_prime_params(self)

        self.descent_info = self.descent_info.expand(gamma = self.gamma, 
                                                     no_steps_descent = self.no_steps_descent, 
                                                     
                                                     conjugate_gradients = self.use_conjugate_gradients,
                                                     linesearch_params = linesearch_params, 
                                                     s_prime_params = s_prime_params, 
                                                     hessian = hessian, 

                                                     xi = self.xi, 
                                                     adaptive_scaling = self.adaptive_scaling)
    
        descent_info=self.descent_info


        self.descent_state = self.descent_state.expand(population = population)

        self.descent_state = initialize_CG(self.descent_state, measurement_info)
        self.descent_state = initialize_pseudo_newton(self.descent_state, measurement_info)
        self.descent_state = initialize_lbfgs(self.descent_state, measurement_info, descent_info)

        descent_state = self.descent_state

        do_step=Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step=Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step
    






class TimeDomainPtychographyBASE(AlgorithmsBASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "TimeDomainPtychography"

        self.local_hessian = None
        self.global_hessian = False
        self.lambda_lm = 1e-2
        self.lbfgs_memory = 10
        self.linalg_solver = "lineax"

        self.use_conjugate_gradients = False

        self.alpha = 1
        self.gamma = 1


        self.use_linesearch = False
        self.max_steps_linesearch = 25
        self.c1 = 1e-3
        self.c2 = 0.9
        self.delta_gamma = (0.5, 1.5)
    

        self.xi=1e-9

        self.r_local_method = "projection"
        self.r_global_method = "projection"
        self.r_gradient = "intensity"
        self.r_hessian = False
        self.r_weights = 1
        self.r_no_iterations = 1





    def calculate_PIE_error(self, signal_f, measured_trace):
        return jnp.mean(jnp.abs(jnp.sqrt(jnp.abs(measured_trace))*jnp.sign(measured_trace) - jnp.abs(signal_f))**2)


    def get_PIE_weights(self, probe_shifted, alpha, pie_method):

        """
        elif pie_method=="lm":
            U=2/(jnp.abs(probe_shifted)**2+1e-6) # -> rPIE is eqivalent to pseudo-gauss-newton/levenberg-marquardt for small gamma. 

        gamma=>1 -> rPIE=>ePIE
        """
        
        if pie_method=="PIE":
            U = 1/(jnp.abs(probe_shifted)**2 + alpha*jnp.max(jnp.abs(probe_shifted)**2, axis=1)[:,jnp.newaxis])*jnp.abs(probe_shifted)/jnp.max(jnp.abs(probe_shifted), axis=1)[:,jnp.newaxis]

        elif pie_method=="ePIE":
            U = 1/jnp.max(jnp.abs(probe_shifted)**2, axis=1)[:,jnp.newaxis]
            U = U*jnp.ones(jnp.shape(probe_shifted))

        elif pie_method=="rPIE":
            U = 1/((1-alpha)*jnp.abs(probe_shifted)**2 + alpha*jnp.max(jnp.abs(probe_shifted)**2, axis=1)[:,jnp.newaxis])

        elif pie_method==None:
            U = jnp.ones(jnp.shape(probe_shifted))

        else:
            print(f"pie_method={pie_method} not defined.")
        
        return U


    def update_population(self, population, gamma, descent_direction, measurement_info, pulse_or_gate):
        population = jax.vmap(self.update_individual, in_axes=(0,0,0,None,None))(population, gamma, descent_direction, measurement_info, pulse_or_gate)
        return population




    def do_local_iteration(self, descent_state, transform_arr_m, trace_line, measurement_info, descent_info):
        pie_method = descent_info.pie_method.local_pie
        gamma = descent_info.gamma*jnp.ones(descent_info.population_size)
        population = descent_state.population
        
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(population, transform_arr_m, measurement_info)
        get_S_prime = Partial(calculate_S_prime, method=descent_info.s_prime_params.local_method)
        signal_t_new = jax.vmap(get_S_prime, in_axes=(0,0,None,None,None))(jnp.squeeze(signal_t.signal_t), trace_line, 1, measurement_info, descent_info)

        grad, U = self.calculate_PIE_descent_direction_local(population, signal_t, signal_t_new, transform_arr_m, pie_method, measurement_info, descent_info, "pulse")
        descent_direction = -1*grad*U
        gamma = jnp.ones(descent_info.population_size)*gamma
        population = self.update_population(population, gamma, descent_direction, measurement_info, "pulse")

        if measurement_info.doubleblind==True: 
            grad, U = self.calculate_PIE_descent_direction_local(population, signal_t, signal_t_new, transform_arr_m, pie_method, measurement_info, descent_info, "gate")
            descent_direction = -1*grad*U
            population = self.update_population(population, gamma, descent_direction, measurement_info, "gate")

        descent_state = tree_at(lambda x: x.population, descent_state, population)
        return descent_state, None
    


    def local_step(self, descent_state, measurement_info, descent_info):
        transform_arr, measured_trace, descent_state = self.shuffle_data_along_m(descent_state, measurement_info, descent_info)

        local_iteration=Partial(self.do_local_iteration, measurement_info=measurement_info, descent_info=descent_info)
        local_iteration=Partial(scan_helper, actual_function=local_iteration, number_of_args=1, number_of_xs=2)

        descent_state, _ = jax.lax.scan(local_iteration, descent_state, (transform_arr, measured_trace))


        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_f=do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        trace=calculate_trace(signal_f)
        trace_error=jax.vmap(calculate_trace_error, in_axes=(0, None))(trace, measurement_info.measured_trace)

        return descent_state, trace_error.reshape(-1,1)
    




    def calc_error_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        transform_arr, measured_trace = measurement_info.transform_arr, measurement_info.measured_trace
        sk, rn = measurement_info.sk, measurement_info.rn

        individual, descent_direction = linesearch_info.population, linesearch_info.descent_direction

        individual = self.update_individual(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        error_new = self.calculate_PIE_error(do_fft(signal_t.signal_t, sk, rn), measured_trace)
        return error_new
    

    def calc_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate):
        transform_arr, measured_trace = measurement_info.transform_arr, measurement_info.measured_trace
        individual, descent_direction = linesearch_info.population, linesearch_info.descent_direction
        
        individual = self.update_individual(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        signal_t_new = calculate_S_prime(signal_t.signal_t, measured_trace, 1, measurement_info, descent_info, method=descent_info.s_prime_params.global_method)
        grad_all_m, U = self.calculate_PIE_descent_direction_global(individual, signal_t, signal_t_new, descent_info.pie_method.global_pie, 
                                                                    measurement_info, descent_info, pulse_or_gate)
        return jnp.sum(grad_all_m, axis=1)
    



    def do_global_step(self, signal_t, signal_t_new, descent_state, measurement_info, descent_info, pulse_or_gate):
        measured_trace = measurement_info.measured_trace
        sk, rn = measurement_info.sk, measurement_info.rn

        pie_method = descent_info.pie_method.global_pie
        hessian, conjugate_gradients = descent_info.hessian, descent_info.conjugate_gradients
        population = descent_state.population

        
        signal_f = do_fft(signal_t.signal_t, sk, rn)
        pie_error = jax.vmap(self.calculate_PIE_error, in_axes=(0,None))(signal_f, measured_trace)

        grad, U = self.calculate_PIE_descent_direction_global(population, signal_t, signal_t_new, pie_method, measurement_info, descent_info, pulse_or_gate)
        gradient_sum = jnp.sum(grad, axis=1)

        
        if hessian.global_hessian=="diagonal" or hessian.global_hessian=="full":
            descent_direction, hessian_state = self.calculate_PIE_descent_direction_hessian(grad*U, signal_t, descent_state, measurement_info, 
                                                                                            descent_info, pulse_or_gate)
            descent_state = tree_at(lambda x: getattr(x.hessian, pulse_or_gate), descent_state, hessian_state)

        if hessian.global_hessian=="lbfgs":
            descent_direction, lbfgs_state = get_pseudo_newton_direction(gradient_sum, getattr(descent_state.lbfgs, pulse_or_gate), descent_info)
        else:
            descent_direction = -1*jnp.sum(grad*U, axis=1)



        if conjugate_gradients!=False:
            cg = getattr(descent_state.cg, pulse_or_gate)
            descent_direction, cg = jax.vmap(get_nonlinear_CG_direction, in_axes=(0,0,None))(descent_direction, cg, conjugate_gradients)
            descent_state = tree_at(lambda x: getattr(x.cg, pulse_or_gate), descent_state, cg)



        descent_direction = adaptive_scaling_of_step(descent_direction, pie_error, gradient_sum, getattr(descent_state.hessian, pulse_or_gate), descent_info)



        if descent_info.linesearch_params.use_linesearch=="backtracking" or descent_info.linesearch_params.use_linesearch=="wolfe":
            pk_dot_gradient=jax.vmap(lambda x,y: jnp.real(jnp.dot(jnp.conjugate(x),y)), in_axes=(0,0))(descent_direction, gradient_sum)

            linesearch_info=MyNamespace(population=population, signal_t=signal_t, descent_direction=descent_direction, 
                                        pk_dot_gradient=pk_dot_gradient, pk=gradient_sum, error=pie_error)     

            gamma = jax.vmap(do_linesearch, in_axes=(0, None, None, None, None))(linesearch_info, measurement_info, descent_info, 
                                                                                Partial(self.calc_error_for_linesearch, pulse_or_gate=pulse_or_gate),
                                                                                Partial(self.calc_grad_for_linesearch, descent_info=descent_info, 
                                                                                        pulse_or_gate=pulse_or_gate))
            
        else:
            gamma = jnp.ones(descent_info.population_size)*descent_info.gamma


        if hessian.global_hessian=="lbfgs":
           step_size_arr = lbfgs_state.step_size_prev
           step_size_arr = step_size_arr.at[:,1:].set(step_size_arr[:,:-1])
           step_size_arr = step_size_arr.at[:,0].set(gamma[:, jnp.newaxis])

           descent_state = tree_at(lambda x: getattr(x.lbfgs, pulse_or_gate).step_size_prev, descent_state, step_size_arr)


        population = self.update_population(population, gamma, descent_direction, measurement_info, pulse_or_gate)
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        return descent_state     


    
    
    def global_step(self, descent_state, measurement_info, descent_info):

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        get_S_prime = Partial(calculate_S_prime, method=descent_info.s_prime_params.global_method)
        signal_t_new = jax.vmap(get_S_prime, in_axes=(0,None,None,None,None))(signal_t.signal_t, measurement_info.measured_trace, 1, measurement_info, descent_info)


        descent_state = self.do_global_step(signal_t, signal_t_new, descent_state, measurement_info, descent_info, "pulse")

        if measurement_info.doubleblind==True:
            descent_state = self.do_global_step(signal_t, signal_t_new, descent_state, measurement_info, descent_info, "gate")


        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_f=do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        trace=calculate_trace(signal_f)
        trace_error=jax.vmap(calculate_trace_error, in_axes=(0, None))(trace, measurement_info.measured_trace)

        return descent_state, trace_error.reshape(-1,1)






    def initialize_run(self, population):
        measurement_info = self.measurement_info

        if type(self.pie_method)==tuple or type(self.pie_method)==list:
            local_pie, global_pie = self.pie_method
        else:
            local_pie, global_pie = self.pie_method, self.pie_method

        pie_method = MyNamespace(local_pie=local_pie, 
                                 global_pie=global_pie)

        linesearch_params = initialize_linesearch_info(self)
        hessian = initialize_hessian_info(self)
        s_prime_params = initialize_S_prime_params(self)
        
        self.descent_info = self.descent_info.expand(alpha = self.alpha, 
                                                     gamma = self.gamma, 
                                                     pie_method = pie_method,

                                                     conjugate_gradients = self.use_conjugate_gradients,
                                                     hessian = hessian,
                                                     linesearch_params = linesearch_params,
                                                     s_prime_params = s_prime_params,

                                                     xi = self.xi,
                                                     adaptive_scaling = self.adaptive_scaling,
                                                     
                                                     idx_arr = self.idx_arr)
        
        descent_info = self.descent_info


        self.descent_state = self.descent_state.expand(key = self.key, 
                                                       population = population)
        
        self.descent_state = initialize_CG(self.descent_state, measurement_info)
        self.descent_state = initialize_pseudo_newton(self.descent_state, measurement_info)
        self.descent_state = initialize_lbfgs(self.descent_state, measurement_info, descent_info)
    
        descent_state=self.descent_state



        do_local=Partial(self.local_step, measurement_info=measurement_info, descent_info=descent_info)
        do_local=Partial(scan_helper, actual_function=do_local, number_of_args=1, number_of_xs=0)

        do_global=Partial(self.global_step, measurement_info=measurement_info, descent_info=descent_info)
        do_global=Partial(scan_helper, actual_function=do_global, number_of_args=1, number_of_xs=0)

        return descent_state, do_local, do_global
    




    def run(self, population, no_iterations_local, no_iterations_global):

        descent_state, do_local, do_global = self.initialize_run(population)

        descent_state, error_arr_local = jax.lax.scan(do_local, descent_state, length=no_iterations_local)
        descent_state, error_arr_global = jax.lax.scan(do_global, descent_state, length=no_iterations_global)

        descent_state, error_arr_local = run_scan(do_local, descent_state, no_iterations_local, self.use_jit)
        descent_state, error_arr_global = run_scan(do_global, descent_state, no_iterations_global, self.use_jit)

        error_arr = jnp.concatenate([error_arr_local, error_arr_global], axis=0)
        error_arr = jnp.squeeze(error_arr)

        final_result = self.post_process(descent_state, error_arr)
        return final_result
    













class COPRABASE(AlgorithmsBASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "COPRA"

        self.xi = 1e-9

        self.gamma=0.25
        self.beta=1

        self.use_linesearch = False
        self.max_steps_linesearch = 25
        self.c1 = 1e-4
        self.c2 = 0.9
        self.delta_gamma = (0.5, 1.5)


        self.local_hessian = False
        self.global_hessian = "diagonal"
        self.lambda_lm = 1e-3
        self.lbfgs_memory = 10
        self.linalg_solver="lineax"


        self.r_local_method = "projection"
        self.r_global_method = "iteration"
        self.r_gradient = "intensity"
        self.r_hessian = False
        self.r_weights = 1.0
        self.r_no_iterations = 1





    def one_local_iteration(self, signal_t, signal_t_new, transform_arr_m, descent_state, measurement_info, descent_info, pulse_or_gate):
        hessian = descent_info.hessian
        population = descent_state.population


        grad = self.calculate_Z_gradient(signal_t_new, signal_t, population, transform_arr_m, measurement_info, pulse_or_gate, local=True)
        
        if hessian.local_hessian!=False:
            descent_direction, hessian_state = self.calculate_Z_error_newton_direction(grad, signal_t_new, signal_t, transform_arr_m, descent_state, 
                                                                                       measurement_info, descent_info, hessian.local_hessian, 
                                                                                       pulse_or_gate, local=True)

            descent_state = tree_at(lambda x: getattr(x.local_state.hessian, pulse_or_gate), descent_state, hessian_state)
        else:
            descent_direction = -1*jnp.squeeze(grad)
            

        grad_norm2 = jnp.sum(jnp.abs(descent_direction)**2)
        max_grad_norm2 = getattr(descent_state.local_state.max_grad_norm2, pulse_or_gate)
        max_grad_norm2 = jnp.greater(grad_norm2, max_grad_norm2)*grad_norm2 + jnp.greater(max_grad_norm2, grad_norm2)*max_grad_norm2
        descent_state = tree_at(lambda x: getattr(x.local_state.max_grad_norm2, pulse_or_gate), descent_state, max_grad_norm2)

        Z_error = jax.vmap(calculate_Z_error, in_axes=(0,0))(signal_t.signal_t, signal_t_new)
        gamma = Z_error/max_grad_norm2

        population = self.update_population_local(population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate)
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        return  descent_state, Z_error
    




    def do_one_local_iteration(self, descent_state, transform_arr_m, trace_line, measurement_info, descent_info):

        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
        get_S_prime = Partial(calculate_S_prime, method=descent_info.s_prime_params.local_method)
        signal_t_new=jax.vmap(get_S_prime, in_axes=(0,0,0,None,None))(signal_t.signal_t, trace_line, descent_state.local_state.mu, measurement_info, descent_info)

        descent_state, Z_error = self.one_local_iteration(signal_t, signal_t_new, transform_arr_m, 
                                                          descent_state, measurement_info, descent_info, "pulse")

        if measurement_info.doubleblind==True:
           descent_state, Z_error = self.one_local_iteration(signal_t, signal_t_new, transform_arr_m, 
                                                             descent_state, measurement_info, descent_info, "gate")

        return descent_state, Z_error
    

    

    def step_local_iteration(self, descent_state, measurement_info, descent_info):
        sk, rn = measurement_info.sk, measurement_info.rn

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(do_fft(signal_t.signal_t, sk, rn))
        local_mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measurement_info.measured_trace)
        descent_state = tree_at(lambda x: x.local_state.mu, descent_state, local_mu)

        one_local_iteration=Partial(self.do_one_local_iteration, measurement_info=measurement_info, descent_info=descent_info)
        one_local_iteration=Partial(scan_helper, actual_function=one_local_iteration, number_of_args=1, number_of_xs=2)

        transform_arr, measured_trace, descent_state = self.shuffle_data_along_m(descent_state, measurement_info, descent_info)
        descent_state, Z_error = jax.lax.scan(one_local_iteration, descent_state, (transform_arr, measured_trace))


        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(do_fft(signal_t.signal_t, sk, rn))
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measurement_info.measured_trace)

        return descent_state, trace_error.reshape(-1,1)
    



    def update_population_global(self, population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        population = jax.vmap(self.update_individual_global, in_axes=(0,0,0,None,None,None))(population, gamma, descent_direction, 
                                                                                               measurement_info, descent_info, pulse_or_gate)
        return population




    def calc_Z_error_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate):
        transform_arr = measurement_info.transform_arr

        signal_t_new, descent_direction = linesearch_info.signal_t_new, linesearch_info.descent_direction

        individual = self.update_individual_global(linesearch_info.population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        error = calculate_Z_error(signal_t.signal_t, signal_t_new)
        return error



    def do_global_iteration(self, signal_t, signal_t_new, descent_state, measurement_info, descent_info, pulse_or_gate):
        transform_arr = measurement_info.transform_arr
        gamma, hessian = descent_info.gamma, descent_info.hessian

        population = descent_state.population

        grad = self.calculate_Z_gradient(signal_t_new, signal_t, population, transform_arr, measurement_info, pulse_or_gate, local=False)
        grad_sum = jnp.sum(grad, axis=1)

        if hessian.global_hessian=="diagonal" or hessian.global_hessian=="full":
            descent_direction, hessian_state = self.calculate_Z_error_newton_direction(grad, signal_t_new, signal_t, transform_arr, descent_state, 
                                                                                       measurement_info, descent_info, hessian.global_hessian, 
                                                                                       pulse_or_gate, local=False)

            descent_state = tree_at(lambda x: getattr(x.global_state.hessian, pulse_or_gate), descent_state, hessian_state)
        else: 
            descent_direction = -1*grad_sum


        Z_error = jax.vmap(calculate_Z_error, in_axes=(0,0))(signal_t.signal_t, signal_t_new)
        descent_direction = adaptive_scaling_of_step(descent_direction, Z_error, grad_sum, getattr(descent_state.global_state.hessian, pulse_or_gate), descent_info)


        if descent_info.linesearch_params.use_linesearch=="backtracking" or descent_info.linesearch_params.use_linesearch=="wolfe":
            pk_dot_gradient = jax.vmap(lambda x,y: jnp.real(jnp.vdot(x,y)), in_axes=(0,0))(descent_direction, grad_sum)        
            linesearch_info=MyNamespace(population=population, signal_t_new=signal_t_new, descent_direction=descent_direction, error=Z_error, 
                                        pk_dot_gradient=pk_dot_gradient, pk=descent_direction)
            
            gamma = jax.vmap(do_linesearch, in_axes=(0,None,None,None,None))(linesearch_info, measurement_info, descent_info, 
                                                                            Partial(self.calc_Z_error_for_linesearch, descent_info=descent_info, 
                                                                                    pulse_or_gate=pulse_or_gate),
                                                                            Partial(self.calc_Z_grad_for_linesearch, descent_info=descent_info, 
                                                                                    pulse_or_gate=pulse_or_gate))
        else:
            gamma = jnp.ones(descent_info.population_size)*descent_info.gamma

        population = self.update_population_global(population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate)
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        return descent_state
    


    def step_global_iteration(self, descent_state, measurement_info, descent_info):
        measured_trace, sk, rn = measurement_info.measured_trace, measurement_info.sk, measurement_info.rn 

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn))
        mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        get_S_prime = Partial(calculate_S_prime, method=descent_info.s_prime_params.global_method)
        signal_t_new = jax.vmap(get_S_prime, in_axes=(0,None,0,None,None))(signal_t.signal_t, measured_trace, mu, measurement_info, descent_info)


        descent_state = self.do_global_iteration(signal_t, signal_t_new, descent_state, measurement_info, descent_info, "pulse")
        population_pulse = jax.vmap(lambda x: x/jnp.linalg.norm(x))(descent_state.population.pulse)
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, population_pulse)

        if measurement_info.doubleblind==True:
            descent_state = self.do_global_iteration(signal_t, signal_t_new, descent_state, measurement_info, descent_info, "gate")


        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(do_fft(signal_t.signal_t, sk, rn))
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)

        return descent_state, trace_error.reshape(-1,1)
    





    def initialize_local_stage(self, descent_state, measurement_info, descent_info):
        transform_arr, measured_trace = measurement_info.transform_arr, measurement_info.measured_trace
        sk, rn = measurement_info.sk, measurement_info.rn
        population = descent_state.population

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(do_fft(signal_t.signal_t, sk, rn))
        mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        get_S_prime = Partial(calculate_S_prime, method=descent_info.s_prime_params.local_method)
        signal_t_new = jax.vmap(get_S_prime, in_axes=(0,None,0,None,None))(signal_t.signal_t, measured_trace, mu, measurement_info, descent_info)

        
        grad = self.calculate_Z_gradient(signal_t_new, signal_t, population, transform_arr, measurement_info, "pulse")
        max_grad_norm2_pulse=jnp.max(jnp.sum(jnp.abs(grad)**2, axis=1))
        descent_state = tree_at(lambda x: x.local_state.max_grad_norm2.pulse, descent_state, max_grad_norm2_pulse)

        if measurement_info.doubleblind==True:
            grad = self.calculate_Z_gradient(signal_t_new, signal_t, population, transform_arr, measurement_info, "gate")
            max_grad_norm2_gate=jnp.max(jnp.sum(jnp.abs(grad)**2, axis=1))
            descent_state = tree_at(lambda x: x.local_state.max_grad_norm2.gate, descent_state, max_grad_norm2_gate)


        do_local = Partial(self.step_local_iteration, measurement_info=measurement_info, descent_info=descent_info)
        do_local = Partial(scan_helper, actual_function=do_local, number_of_args=1, number_of_xs=0)
        return descent_state, do_local
    


    def initialize_global_stage(self, descent_state, measurement_info, descent_info):
        do_global = Partial(self.step_global_iteration, measurement_info=measurement_info, descent_info=descent_info)
        do_global = Partial(scan_helper, actual_function=do_global, number_of_args=1, number_of_xs=0)
        return descent_state, do_global
    


    def initialize_run(self, population):
        measurement_info=self.measurement_info


        linesearch_params = initialize_linesearch_info(self)
        hessian = initialize_hessian_info(self)
        s_prime_params = initialize_S_prime_params(self)
        
        self.descent_info = self.descent_info.expand(gamma = self.gamma, 
                                                     beta = self.beta, 
                                                     xi = self.xi, 
                                                     linesearch_params = linesearch_params,
                                                     hessian = hessian,
                                                     s_prime_params = s_prime_params,
                                                     adaptive_scaling = self.adaptive_scaling,
                                                     idx_arr = self.idx_arr)
        descent_info=self.descent_info


        hessian_state_local = get_init_state_pseudo_newton(jnp.shape(population.pulse), measurement_info)
        hessian_state_global = get_init_state_pseudo_newton(jnp.shape(population.pulse), measurement_info)
        self.descent_state = self.descent_state.expand(key = self.key, 
                                                       population = population, 
                                                       global_state = MyNamespace(hessian = hessian_state_global),
                                                       local_state = MyNamespace(hessian = hessian_state_local,
                                                                           max_grad_norm2 = MyNamespace(pulse=0.0, gate=0.0), 
                                                                           mu = jnp.ones(self.descent_info.population_size)))
        descent_state=self.descent_state


        descent_state, do_local = self.initialize_local_stage(descent_state, measurement_info, descent_info)
        descent_state, do_global = self.initialize_global_stage(descent_state, measurement_info, descent_info)

        return descent_state, do_local, do_global
    




    def run(self, population, no_iterations_local, no_iterations_global):

        descent_state, do_local, do_global = self.initialize_run(population)

        descent_state, error_arr_local = run_scan(do_local, descent_state, no_iterations_local, self.use_jit)
        descent_state, error_arr_global = run_scan(do_global, descent_state, no_iterations_global, self.use_jit)

        error_arr = jnp.concatenate([error_arr_local, error_arr_global], axis=0)
        error_arr = jnp.squeeze(error_arr)

        final_result = self.post_process(descent_state, error_arr)
        return final_result



