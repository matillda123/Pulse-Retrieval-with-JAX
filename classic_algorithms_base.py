import jax
import jax.numpy as jnp

from jax.tree_util import Partial
from equinox import tree_at


from stepsize import do_linesearch, adaptive_step_size
from nonlinear_cg import get_nonlinear_CG_direction
from lbfgs import get_quasi_newton_direction

from utilities import scan_helper, MyNamespace, do_fft, do_ifft, calculate_mu, calculate_trace, calculate_trace_error, calculate_Z_error, run_scan
from BaseClasses import ClassicAlgorithmsBASE

from construct_s_prime import calculate_S_prime






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

    hessian_pulse = MyNamespace(newton_direction_prev=init_arr1)
    if measurement_info.doubleblind==True:
        hessian_gate = MyNamespace(newton_direction_prev=init_arr1)
    else:
        hessian_gate = None

    return MyNamespace(pulse=hessian_pulse, gate=hessian_gate)




def initialize_lbfgs_state(shape, measurement_info, descent_info):
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
        
    return MyNamespace(pulse=lbfgs_init_pulse, gate=lbfgs_init_gate)







def initialize_linesearch_info(optimizer):
    linesearch_params = MyNamespace(use_linesearch=optimizer.use_linesearch, 
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
                                 r_hessian=optimizer.r_hessian, 
                                 weights=optimizer.r_weights)
    return s_prime_params



def initialize_hessian_info(optimizer):
    hessian = MyNamespace(_local=optimizer.local_hessian, 
                        _global=optimizer.global_hessian, 
                        linalg_solver=optimizer.linalg_solver, 
                        lambda_lm=optimizer.lambda_lm,
                        lbfgs_memory=optimizer.lbfgs_memory)

    return hessian












class GeneralizedProjectionBASE(ClassicAlgorithmsBASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "GeneralizedProjection"

        self.local_gamma = None
        self.global_gamma = 1e3

        self.no_steps_descent = 15

        self.r_local_method = None


    
    def update_population(self, population, gamma, descent_direction, measurement_info, pulse_or_gate):
        population = jax.vmap(self.update_individual, in_axes=(0,0,0,None,None))(population, gamma, descent_direction, measurement_info, pulse_or_gate)
        return population
    

    
    def get_Z_gradient(self, signal_t, signal_t_new, population, transform_arr, measurement_info, pulse_or_gate):
        grad = jax.vmap(self.calculate_Z_gradient_individual, in_axes=(0, 0, 0, None, None, None))(signal_t, signal_t_new, population, transform_arr, 
                                                                                                   measurement_info, pulse_or_gate)
        return grad

    

    def calc_Z_error_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        individual, descent_direction, signal_t_new = linesearch_info.population, linesearch_info.descent_direction, linesearch_info.signal_t_new
       
        transform_arr = measurement_info.transform_arr

        individual = self.update_individual(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        Z_error_new = calculate_Z_error(signal_t.signal_t, signal_t_new)
        return Z_error_new
    

    def calc_Z_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        individual, descent_direction, signal_t_new = linesearch_info.population, linesearch_info.descent_direction, linesearch_info.signal_t_new

        transform_arr = measurement_info.transform_arr

        individual = self.update_individual(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        grad = self.calculate_Z_gradient_individual(signal_t, signal_t_new, individual, transform_arr, measurement_info, pulse_or_gate)
        return jnp.sum(grad, axis=0)


    def gradient_descent_Z_error_step(self, signal_t, signal_t_new, Z_error, descent_state, measurement_info, descent_info, pulse_or_gate):        
        hessian_info, conjugate_gradients = descent_info.hessian._global, descent_info.conjugate_gradients

        population = descent_state.population
        transform_arr = measurement_info.transform_arr


        grad = self.get_Z_gradient(signal_t, signal_t_new, population, transform_arr, measurement_info, pulse_or_gate)
        grad_sum = jnp.sum(grad, axis=1)


        if hessian_info=="diagonal" or hessian_info=="full":
            descent_direction, hessian_state = self.calculate_Z_newton_direction(grad, signal_t_new, signal_t, transform_arr, descent_state, 
                                                                                       measurement_info, descent_info, hessian_info, pulse_or_gate)
            descent_state = tree_at(lambda x: getattr(x.hessian, pulse_or_gate), descent_state, hessian_state)

        elif hessian_info=="lbfgs":
            descent_direction, lbfgs_state = get_quasi_newton_direction(grad_sum, getattr(descent_state.lbfgs, pulse_or_gate), descent_info)

        else:
            descent_direction = -1*grad_sum



        if conjugate_gradients!=False:
            cg = getattr(descent_state.cg, pulse_or_gate)
            descent_direction, cg =jax.vmap(get_nonlinear_CG_direction, in_axes=(0,0,None))(descent_direction, cg, conjugate_gradients)
            descent_state = tree_at(lambda x: getattr(x.cg, pulse_or_gate), descent_state, cg)



        descent_direction, descent_state = jax.vmap(adaptive_step_size, in_axes=(0,0,0,None,None,None,None,None), out_axes=(0,None))(Z_error, grad_sum, descent_direction, 
                                                                                                                   descent_state, descent_info.xi, 
                                                                                                                   getattr(descent_info.adaptive_scaling, "_global"), 
                                                                                                                   pulse_or_gate, "_global")

        if descent_info.linesearch_params.use_linesearch!=False:
            pk_dot_gradient = jax.vmap(lambda x,y: jnp.real(jnp.vdot(x,y)), in_axes=(0,0))(descent_direction, grad_sum)
            
            linesearch_info=MyNamespace(population=population, descent_direction=descent_direction, signal_t_new=signal_t_new, 
                                        error=Z_error, pk_dot_gradient=pk_dot_gradient)
            
            gamma = jax.vmap(do_linesearch, in_axes=(0,None,None,None,None,None))(linesearch_info, measurement_info, descent_info, 
                                                                             Partial(self.calc_Z_error_for_linesearch, pulse_or_gate=pulse_or_gate),
                                                                             Partial(self.calc_Z_grad_for_linesearch, pulse_or_gate=pulse_or_gate), "_global")
        else:
            gamma = jnp.broadcast_to(descent_info.gamma._global, (descent_info.population_size, ))



        if hessian_info=="lbfgs":
           step_size_arr = lbfgs_state.step_size_prev
           step_size_arr = step_size_arr.at[:,1:].set(step_size_arr[:,:-1])
           step_size_arr = step_size_arr.at[:,0].set(gamma[:, jnp.newaxis])

           descent_state = tree_at(lambda x: getattr(x.lbfgs, pulse_or_gate).step_size_prev, descent_state, step_size_arr)

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

        shape_pulse = jnp.shape(descent_state.population.pulse)
        cg_state = initialize_CG_state(shape_pulse, measurement_info)
        hessian_state = initialize_pseudo_newton_state(shape_pulse, measurement_info)
        lbfgs_state = initialize_lbfgs_state(shape_pulse, measurement_info, descent_info)
        descent_state = tree_at(lambda x: x.cg, descent_state, cg_state)
        descent_state = tree_at(lambda x: x.hessian, descent_state, hessian_state)
        descent_state = tree_at(lambda x: x.lbfgs, descent_state, lbfgs_state)
        
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
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,None,0,None,None,None))(signal_t.signal_t, measured_trace, mu, measurement_info, descent_info, "_global")

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

        self.descent_info = self.descent_info.expand(gamma = MyNamespace(_local=self.local_gamma, _global=self.global_gamma), 
                                                     no_steps_descent = self.no_steps_descent, 
                                                     
                                                     conjugate_gradients = self.conjugate_gradients,
                                                     linesearch_params = linesearch_params, 
                                                     s_prime_params = s_prime_params, 
                                                     hessian = hessian, 

                                                     xi = self.xi, 
                                                     adaptive_scaling = MyNamespace(_local=self.local_adaptive_scaling, _global=self.global_adaptive_scaling))
    
        descent_info=self.descent_info


        self.descent_state = self.descent_state.expand(population = population)

        shape_pulse = jnp.shape(self.descent_state.population.pulse)
        cg_state = initialize_CG_state(shape_pulse, measurement_info)
        hessian_state = initialize_pseudo_newton_state(shape_pulse, measurement_info)
        lbfgs_state = initialize_lbfgs_state(shape_pulse, measurement_info, descent_info)
        self.descent_state = self.descent_state.expand(cg = cg_state, hessian=hessian_state, lbfgs=lbfgs_state)

        descent_state = self.descent_state

        do_step=Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_step=Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)
        return descent_state, do_step
    






class TimeDomainPtychographyBASE(ClassicAlgorithmsBASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "TimeDomainPtychography"

        self.alpha = 1



    def calculate_PIE_error(self, signal_f, measured_trace):
        return jnp.mean(jnp.abs(jnp.sqrt(jnp.abs(measured_trace))*jnp.sign(measured_trace) - jnp.abs(signal_f))**2)


    def get_PIE_weights(self, probe, alpha, pie_method):

        """
        U=2/(jnp.abs(probe_shifted)**2+1e-6) # -> rPIE is eqivalent to pseudo-gauss-newton/levenberg-marquardt for small gamma. 

        gamma=>1 -> rPIE=>ePIE
        """

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
            print(f"pie_method={pie_method} not defined.")
        
        return U


    def update_population(self, population, gamma, descent_direction, measurement_info, pulse_or_gate):
        population = jax.vmap(self.update_individual, in_axes=(0,0,0,None,None))(population, gamma, descent_direction, measurement_info, pulse_or_gate)
        return population
    


    def calculate_PIE_descent_direction(self, population, signal_t, signal_t_new, transform_arr, pie_method, measurement_info, descent_info, pulse_or_gate):
        get_descent_direction = Partial(self.calculate_PIE_descent_direction_m, population=population, pie_method=pie_method, 
                                        measurement_info=measurement_info, descent_info=descent_info, pulse_or_gate=pulse_or_gate)

        grad_all_m, U = jax.vmap(get_descent_direction, in_axes=(1,1,1), out_axes=(1,1))(signal_t, signal_t_new, transform_arr)
        return grad_all_m, U






    def calc_error_for_linesearch(self, gamma, linesearch_info, measurement_info, pulse_or_gate):
        sk, rn = measurement_info.sk, measurement_info.rn

        transform_arr, measured_trace = linesearch_info.transform_arr, linesearch_info.measured_trace
        individual, descent_direction = linesearch_info.population, linesearch_info.descent_direction

        individual = self.update_individual(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        error_new = self.calculate_PIE_error(do_fft(signal_t.signal_t, sk, rn), measured_trace)
        return error_new
    


    def calc_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate, local_or_global):
        transform_arr, measured_trace = linesearch_info.transform_arr[jnp.newaxis, ... ], linesearch_info.measured_trace[jnp.newaxis, ... ]
        individual, descent_direction = linesearch_info.population, linesearch_info.descent_direction
        
        individual = self.update_individual(individual, gamma, descent_direction, measurement_info, pulse_or_gate)
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(None,0,None))(individual, transform_arr, measurement_info)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,0,None,None,None, None))(signal_t.signal_t, measured_trace, 1, measurement_info, 
                                                                                       descent_info, local_or_global)

        grad_all_m, U = self.calculate_PIE_descent_direction(individual, signal_t, signal_t_new, transform_arr, descent_info.pie_method, 
                                                             measurement_info, descent_info, pulse_or_gate)
        return jnp.sum(grad_all_m, axis=1)
    


    
    def do_iteration(self, signal_t, signal_t_new, transform_arr, measured_trace, pie_error, population, local_or_global_state, 
                     measurement_info, descent_info, pulse_or_gate, local_or_global):
        

        if local_or_global=="_global":
            N = descent_info.population_size
            shape = (N,) + jnp.shape(measured_trace)
            measured_trace = jnp.broadcast_to(measured_trace, shape)
            shape = (N, ) + jnp.shape(transform_arr)
            transform_arr = jnp.broadcast_to(transform_arr, shape)
            
        

        pie_method = descent_info.pie_method
        conjugate_gradients = descent_info.conjugate_gradients
        hessian_info = getattr(descent_info.hessian, local_or_global)

        grad, U = self.calculate_PIE_descent_direction(population, signal_t, signal_t_new, transform_arr, pie_method, measurement_info, descent_info, pulse_or_gate)
        grad_sum = jnp.sum(grad, axis=1)


        if hessian_info=="diagonal" or (hessian_info=="full" and pulse_or_gate=="pulse"):
            descent_direction, hessian_state = self.calculate_PIE_newton_direction(grad, signal_t, transform_arr, measured_trace, population, local_or_global_state, 
                                                                                   measurement_info, descent_info, pulse_or_gate, local_or_global)
            local_or_global_state = tree_at(lambda x: getattr(x.hessian, pulse_or_gate), local_or_global_state, hessian_state)

        elif hessian_info=="lbfgs":
            lbfgs_state = getattr(local_or_global_state.lbfgs, pulse_or_gate)
            descent_direction, lbfgs_state = get_quasi_newton_direction(grad_sum, lbfgs_state, descent_info)

        else:
            descent_direction = -1*jnp.sum(grad*U, axis=1)



        if conjugate_gradients!=False:
            cg = getattr(local_or_global_state.cg, pulse_or_gate)
            descent_direction, cg = jax.vmap(get_nonlinear_CG_direction, in_axes=(0,0,None))(descent_direction, cg, conjugate_gradients)
            local_or_global_state = tree_at(lambda x: getattr(x.cg, pulse_or_gate), local_or_global_state, cg)



        descent_direction, local_or_global_state = jax.vmap(adaptive_step_size, in_axes=(0,0,0,0,None,None,None,None))(pie_error, grad_sum, descent_direction, 
                                                                                                                          local_or_global_state, descent_info.xi,
                                                                                                                        getattr(descent_info.adaptive_scaling, local_or_global) ,
                                                                                                                          pulse_or_gate, local_or_global)


        if descent_info.linesearch_params.use_linesearch!=False and local_or_global=="_global":
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


        if hessian_info=="lbfgs":
            step_size_arr = lbfgs_state.step_size_prev
            step_size_arr = step_size_arr.at[:,1:].set(step_size_arr[:,:-1])
            step_size_arr = step_size_arr.at[:,0].set(gamma[:, jnp.newaxis])

            local_or_global_state = tree_at(lambda x: getattr(x.lbfgs, pulse_or_gate).step_size_prev, local_or_global_state, step_size_arr)

        population = self.update_population(population, gamma, descent_direction, measurement_info, pulse_or_gate)
        return local_or_global_state, population



    

    def local_iteration(self, descent_state, transform_arr_m, trace_line, measurement_info, descent_info):
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,0,None,None,None,None))(signal_t.signal_t, trace_line, 1, measurement_info, descent_info, "_local")

        signal_f = do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        pie_error = jax.vmap(self.calculate_PIE_error, in_axes=(0,None))(signal_f, trace_line)

        local_state, population = descent_state._local, descent_state.population
        local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, trace_line, pie_error, population, local_state, 
                                                      measurement_info, descent_info, "pulse", "_local")

        if measurement_info.doubleblind==True:
            local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, trace_line, pie_error, population, local_state, 
                                                      measurement_info, descent_info, "gate", "_local")
        
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x._local, descent_state, local_state)
        return descent_state, None
    


    def local_step(self, descent_state, measurement_info, descent_info):
        transform_arr, measured_trace, descent_state = self.shuffle_data_along_m(descent_state, measurement_info, descent_info)

        local_iteration=Partial(self.local_iteration, measurement_info=measurement_info, descent_info=descent_info)
        local_iteration=Partial(scan_helper, actual_function=local_iteration, number_of_args=1, number_of_xs=2)

        descent_state, _ = jax.lax.scan(local_iteration, descent_state, (transform_arr, measured_trace))


        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_f=do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        trace=calculate_trace(signal_f)
        trace_error=jax.vmap(calculate_trace_error, in_axes=(0, None))(trace, measurement_info.measured_trace)

        return descent_state, trace_error.reshape(-1,1)
    



    
    
    def global_step(self, descent_state, measurement_info, descent_info):
        measured_trace = measurement_info.measured_trace

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,None,None,None,None,None))(signal_t.signal_t, measured_trace, 1, measurement_info, 
                                                                                         descent_info, "_global")
        
        signal_f = do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        pie_error = jax.vmap(self.calculate_PIE_error, in_axes=(0,None))(signal_f, measured_trace)

        global_state, population = descent_state._global, descent_state.population 
        global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, measured_trace, pie_error, 
                                                      population, global_state, measurement_info, descent_info, "pulse", "_global")

        if measurement_info.doubleblind==True:
            global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, measured_trace, pie_error, 
                                                          population, global_state, measurement_info, descent_info, "gate", "_global")
        
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x._global, descent_state, global_state)

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        signal_f=do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        trace=calculate_trace(signal_f)
        trace_error=jax.vmap(calculate_trace_error, in_axes=(0, None))(trace, measured_trace)

        return descent_state, trace_error.reshape(-1,1)






    def initialize_run(self, population):
        measurement_info = self.measurement_info

        linesearch_params = initialize_linesearch_info(self)
        hessian = initialize_hessian_info(self)
        s_prime_params = initialize_S_prime_params(self)
        
        self.descent_info = self.descent_info.expand(alpha = self.alpha, 
                                                     gamma = MyNamespace(_local=self.local_gamma, _global=self.global_gamma), 
                                                     pie_method = self.pie_method,

                                                     conjugate_gradients = self.conjugate_gradients,
                                                     hessian = hessian,
                                                     linesearch_params = linesearch_params,
                                                     s_prime_params = s_prime_params,

                                                     xi = self.xi,
                                                     adaptive_scaling = MyNamespace(_local=self.local_adaptive_scaling, _global=self.global_adaptive_scaling),
                                                     
                                                     idx_arr = self.idx_arr)
        
        descent_info = self.descent_info

        shape = jnp.shape(population.pulse)
        cg_state_local = initialize_CG_state(shape, measurement_info)
        hessian_state_local = initialize_pseudo_newton_state(shape, measurement_info)
        lbfgs_state_local = initialize_lbfgs_state(shape, measurement_info, descent_info)

        cg_state_global = initialize_CG_state(shape, measurement_info)
        hessian_state_global = initialize_pseudo_newton_state(shape, measurement_info)
        lbfgs_state_global = initialize_lbfgs_state(shape, measurement_info, descent_info)

        init_arr = jnp.zeros(shape[0])
        self.descent_state = self.descent_state.expand(key = self.key, 
                                                       population = population, 
                                                       _local=MyNamespace(cg=cg_state_local, hessian=hessian_state_local, lbfgs=lbfgs_state_local, 
                                                                          max_scaling = MyNamespace(pulse=init_arr, gate=init_arr)),
                                                       _global=MyNamespace(cg=cg_state_global, hessian=hessian_state_global, lbfgs=lbfgs_state_global))
    
        descent_state=self.descent_state



        do_local=Partial(self.local_step, measurement_info=measurement_info, descent_info=descent_info)
        do_local=Partial(scan_helper, actual_function=do_local, number_of_args=1, number_of_xs=0)

        do_global=Partial(self.global_step, measurement_info=measurement_info, descent_info=descent_info)
        do_global=Partial(scan_helper, actual_function=do_global, number_of_args=1, number_of_xs=0)

        return descent_state, do_local, do_global
    



    def run(self, population, no_iterations_local, no_iterations_global):

        descent_state, do_local, do_global = self.initialize_run(population)

        descent_state, error_arr_local = run_scan(do_local, descent_state, no_iterations_local, self.use_jit)
        descent_state, error_arr_global = run_scan(do_global, descent_state, no_iterations_global, self.use_jit)

        error_arr = jnp.concatenate([error_arr_local, error_arr_global], axis=0)
        error_arr = jnp.squeeze(error_arr)

        final_result = self.post_process(descent_state, error_arr)
        return final_result
    













class COPRABASE(ClassicAlgorithmsBASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "COPRA"

        self.global_gamma=0.25
        self.r_global_method = "iteration"

        self.local_adaptive_scaling = "original"
        self.global_adaptive_scaling = "original"



    def update_population(self, population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        population = jax.vmap(self.update_individual, in_axes=(0,0,0,None,None,None))(population, gamma, descent_direction, 
                                                                                      measurement_info, descent_info, pulse_or_gate)
        return population
    

    

    def get_Z_gradient(self, signal_t, signal_t_new, population, transform_arr, measurement_info, pulse_or_gate):
        grad = jax.vmap(self.get_Z_gradient_individual, in_axes=(0,0,0,0,None,None))(signal_t, signal_t_new, population, transform_arr, 
                                                                                     measurement_info, pulse_or_gate)
        return grad




    def calc_Z_error_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate):
        
        transform_arr = linesearch_info.transform_arr
        signal_t_new, descent_direction = linesearch_info.signal_t_new, linesearch_info.descent_direction

        individual = self.update_individual(linesearch_info.population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        error = calculate_Z_error(signal_t.signal_t, signal_t_new)
        return error
    


    def calc_Z_grad_for_linesearch(self, gamma, linesearch_info, measurement_info, descent_info, pulse_or_gate):

        transform_arr = linesearch_info.transform_arr
        signal_t_new, descent_direction = linesearch_info.signal_t_new, linesearch_info.descent_direction

        individual = self.update_individual(linesearch_info.population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate)
        signal_t = self.calculate_signal_t(individual, transform_arr, measurement_info)
        
        grad = self.get_Z_gradient_individual(signal_t, signal_t_new, individual, transform_arr, measurement_info, pulse_or_gate)
        return jnp.sum(grad, axis=0)
    




    def do_iteration(self, signal_t, signal_t_new, transform_arr, population, local_or_global_state, measurement_info, descent_info, 
                            pulse_or_gate, local_or_global):
        
        gamma, hessian_info = getattr(descent_info.gamma, local_or_global), getattr(descent_info.hessian, local_or_global)
        
        if local_or_global=="_global":
            shape = (descent_info.population_size, ) + jnp.shape(transform_arr)
            transform_arr = jnp.broadcast_to(transform_arr, shape)



        grad = self.get_Z_gradient(signal_t, signal_t_new, population, transform_arr, measurement_info, pulse_or_gate)
        grad_sum = jnp.sum(grad, axis=1)

        if hessian_info=="diagonal" or hessian_info=="full":
            descent_direction, hessian_state = self.get_Z_newton_direction(grad, signal_t, signal_t_new, transform_arr, population, local_or_global_state, 
                                                                                       measurement_info, descent_info, hessian_info, pulse_or_gate)

            local_or_global_state = tree_at(lambda x: getattr(x.hessian, pulse_or_gate), local_or_global_state, hessian_state)

        elif hessian_info=="lbfgs":
            lbfgs_state = getattr(local_or_global_state.lbfgs, pulse_or_gate)
            descent_direction, lbfgs_state = get_quasi_newton_direction(grad_sum, lbfgs_state, descent_info)

        else: 
            descent_direction = -1*grad_sum



        if descent_info.conjugate_gradients!=False:
            cg = getattr(local_or_global_state.cg, pulse_or_gate)
            descent_direction, cg = jax.vmap(get_nonlinear_CG_direction, in_axes=(0,0,None))(descent_direction, cg, descent_info.conjugate_gradients)
            local_or_global_state = tree_at(lambda x: getattr(x.cg, pulse_or_gate), local_or_global_state, cg)



        Z_error = jax.vmap(calculate_Z_error, in_axes=(0,0))(signal_t.signal_t, signal_t_new)
        descent_direction, local_or_global_state = jax.vmap(adaptive_step_size, in_axes=(0,0,0,0,None,None,None,None))(Z_error, grad_sum, descent_direction, 
                                                                                                                          local_or_global_state, descent_info.xi,
                                                                                                                        getattr(descent_info.adaptive_scaling, local_or_global),
                                                                                                                          pulse_or_gate, local_or_global)
        

        if descent_info.linesearch_params.use_linesearch!=False and local_or_global=="_global":
            pk_dot_gradient = jax.vmap(lambda x,y: jnp.real(jnp.vdot(x,y)), in_axes=(0,0))(descent_direction, grad_sum)        
            linesearch_info=MyNamespace(population=population, signal_t_new=signal_t_new, descent_direction=descent_direction, error=Z_error, 
                                        pk_dot_gradient=pk_dot_gradient, transform_arr=transform_arr)
            
            gamma = jax.vmap(do_linesearch, in_axes=(0,None,None,None,None, None))(linesearch_info, measurement_info, descent_info, 
                                                                            Partial(self.calc_Z_error_for_linesearch, descent_info=descent_info, 
                                                                                    pulse_or_gate=pulse_or_gate),
                                                                            Partial(self.calc_Z_grad_for_linesearch, descent_info=descent_info, 
                                                                                    pulse_or_gate=pulse_or_gate), local_or_global)
        else:
            gamma = jnp.broadcast_to(gamma, (descent_info.population_size, ))


        if hessian_info=="lbfgs":
            step_size_arr = lbfgs_state.step_size_prev
            step_size_arr = step_size_arr.at[:,1:].set(step_size_arr[:,:-1])
            step_size_arr = step_size_arr.at[:,0].set(gamma[:, jnp.newaxis])

            local_or_global_state = tree_at(lambda x: getattr(x.lbfgs, pulse_or_gate).step_size_prev, local_or_global_state, step_size_arr)

        population = self.update_population(population, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate)
        return local_or_global_state, population
    




    def do_one_local_iteration(self, descent_state, transform_arr_m, trace_line, measurement_info, descent_info):

        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,0,None))(descent_state.population, transform_arr_m, measurement_info)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,0,0,None,None,None))(signal_t.signal_t, trace_line, descent_state._local.mu, measurement_info, 
                                                                                   descent_info, "_local")

        population, local_state = descent_state.population, descent_state._local
        local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, population, local_state, measurement_info, descent_info, 
                                                   "pulse", "_local")

        if measurement_info.doubleblind==True:
            local_state, population = self.do_iteration(signal_t, signal_t_new, transform_arr_m, population, local_state, measurement_info, descent_info, 
                                                        "gate", "_local")
            
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x._local, descent_state, local_state)
        return descent_state, None
    

    

    def step_local_iteration(self, descent_state, measurement_info, descent_info):
        sk, rn = measurement_info.sk, measurement_info.rn

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(do_fft(signal_t.signal_t, sk, rn))
        local_mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measurement_info.measured_trace)
        descent_state = tree_at(lambda x: x._local.mu, descent_state, local_mu)

        one_local_iteration=Partial(self.do_one_local_iteration, measurement_info=measurement_info, descent_info=descent_info)
        one_local_iteration=Partial(scan_helper, actual_function=one_local_iteration, number_of_args=1, number_of_xs=2)

        transform_arr, measured_trace, descent_state = self.shuffle_data_along_m(descent_state, measurement_info, descent_info)
        descent_state, _ = jax.lax.scan(one_local_iteration, descent_state, (transform_arr, measured_trace))


        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(do_fft(signal_t.signal_t, sk, rn))
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measurement_info.measured_trace)

        return descent_state, trace_error.reshape(-1,1)
    




    def step_global_iteration(self, descent_state, measurement_info, descent_info):
        measured_trace, sk, rn = measurement_info.measured_trace, measurement_info.sk, measurement_info.rn 

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn))
        mu = jax.vmap(calculate_mu, in_axes=(0,None))(trace, measured_trace)
        signal_t_new = jax.vmap(calculate_S_prime, in_axes=(0,None,0,None,None,None))(signal_t.signal_t, measured_trace, mu, measurement_info, 
                                                                                      descent_info, "_global")


        population, global_state = descent_state.population, descent_state._global
        global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, population, global_state, measurement_info, 
                                                     descent_info, "pulse", "_global")
        
        population_pulse = jax.vmap(lambda x: x/jnp.linalg.norm(x))(population.pulse)
        population = tree_at(lambda x: x.pulse, population, population_pulse)

        if measurement_info.doubleblind==True:
            global_state, population = self.do_iteration(signal_t, signal_t_new, measurement_info.transform_arr, population, global_state, measurement_info, 
                                                         descent_info, "gate", "_global")
            
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x._global, descent_state, global_state)


        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(do_fft(signal_t.signal_t, sk, rn))
        trace_error = jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measured_trace)

        return descent_state, trace_error.reshape(-1,1)
    
    
    


    def initialize_run(self, population):
        measurement_info=self.measurement_info


        linesearch_params = initialize_linesearch_info(self)
        hessian = initialize_hessian_info(self)
        s_prime_params = initialize_S_prime_params(self)
        
        self.descent_info = self.descent_info.expand(gamma = MyNamespace(_local=self.local_gamma, _global=self.global_gamma),  
                                                     xi = self.xi, 
                                                     linesearch_params = linesearch_params,
                                                     hessian = hessian,
                                                     s_prime_params = s_prime_params,
                                                     adaptive_scaling = MyNamespace(_local=self.local_adaptive_scaling, _global=self.global_adaptive_scaling),
                                                     idx_arr = self.idx_arr,
                                                     conjugate_gradients = self.conjugate_gradients)
        descent_info=self.descent_info


        shape = jnp.shape(population.pulse)
        cg_state_local = initialize_CG_state(shape, measurement_info)
        hessian_state_local = initialize_pseudo_newton_state(shape, measurement_info)
        lbfgs_state_local = initialize_lbfgs_state(shape, measurement_info, descent_info)

        cg_state_global = initialize_CG_state(shape, measurement_info)
        hessian_state_global = initialize_pseudo_newton_state(shape, measurement_info)
        lbfgs_state_global = initialize_lbfgs_state(shape, measurement_info, descent_info)

        init_arr = jnp.zeros(shape[0])
        self.descent_state = self.descent_state.expand(key = self.key, 
                                                       population = population, 
                                                       _local=MyNamespace(cg=cg_state_local, hessian=hessian_state_local, lbfgs=lbfgs_state_local, 
                                                                          max_scaling = MyNamespace(pulse=init_arr, gate=init_arr),
                                                                          mu = jnp.ones(shape[0])),
                                                       _global=MyNamespace(cg=cg_state_global, hessian=hessian_state_global, lbfgs=lbfgs_state_global))
        
        descent_state=self.descent_state


        do_local = Partial(self.step_local_iteration, measurement_info=measurement_info, descent_info=descent_info)
        do_local = Partial(scan_helper, actual_function=do_local, number_of_args=1, number_of_xs=0)

        do_global = Partial(self.step_global_iteration, measurement_info=measurement_info, descent_info=descent_info)
        do_global = Partial(scan_helper, actual_function=do_global, number_of_args=1, number_of_xs=0)

        return descent_state, do_local, do_global
    




    def run(self, population, no_iterations_local, no_iterations_global):

        descent_state, do_local, do_global = self.initialize_run(population)

        descent_state, error_arr_local = run_scan(do_local, descent_state, no_iterations_local, self.use_jit)
        descent_state, error_arr_global = run_scan(do_global, descent_state, no_iterations_global, self.use_jit)

        error_arr = jnp.concatenate([error_arr_local, error_arr_global], axis=0)
        error_arr = jnp.squeeze(error_arr)

        final_result = self.post_process(descent_state, error_arr)
        return final_result



