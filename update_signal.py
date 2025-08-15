import jax.numpy as jnp
import jax

from jax.tree_util import Partial


from utilities import scan_helper, MyNamespace, do_fft, do_ifft, project_onto_intensity, calculate_mu, calculate_trace
from stepsize import do_linesearch, adaptive_scaling_of_step






def calculate_S_prime_projection(signal_t, measured_trace, mu, measurement_info):
    sk, rn = measurement_info.sk, measurement_info.rn
    signal_f=do_fft(signal_t, sk, rn)

    signal_f_new = project_onto_intensity(signal_f, measured_trace)

    signal_t_new = do_ifft(signal_f_new, sk, rn)*1/(jnp.sqrt(mu)+1e-12)
    return signal_t_new











def calculate_r_hessian_diagonal_intensity(trace, measured_trace):
    H_zz_diag = jnp.sum(2*trace - measured_trace, axis=-1)
    return H_zz_diag


def calculate_r_hessian_diagonal_amplitude(trace, measured_trace):
    H_zz_diag = 1
    return H_zz_diag


def calculate_r_hessian_diagonal(signal_f, measurement_info, descent_info):
    trace = calculate_trace(signal_f)
    measured_trace = measurement_info.measured_trace

    calc_r_hessian_diag_dict={"amplitude": calculate_r_hessian_diagonal_amplitude,
                              "intensity": calculate_r_hessian_diagonal_intensity}
    return calc_r_hessian_diag_dict[descent_info.s_prime_params.r_gradient](trace, measured_trace) # r_gradient is correct here, no need for extra r_hessian with amp/int




def calculate_r_gradient_intensity(signal_f, mu, measured_trace, weights, sk, rn):
    trace = calculate_trace(signal_f)
    grad_r = -4*mu*do_ifft(signal_f*(measured_trace - mu*trace)*weights**2, sk, rn)
    return grad_r 


def calculate_r_gradient_amplitude(signal_f, mu, measured_trace, weights, sk, rn):
    grad_r = -4*mu*do_ifft((jnp.sqrt(measured_trace)*jnp.exp(1j*jnp.angle(signal_f))-mu*signal_f)*weights, sk, rn)
    return grad_r



def calculate_r_gradient(signal_f, mu, measurement_info, descent_info):
    measured_trace, sk, rn = measurement_info.measured_trace, measurement_info.sk, measurement_info.rn
    weights = descent_info.s_prime_params.weights

    calc_r_grad_dict={"amplitude": calculate_r_gradient_amplitude,
                      "intensity": calculate_r_gradient_intensity}
    
    gradient = calc_r_grad_dict[descent_info.s_prime_params.r_gradient](signal_f, mu, measured_trace, weights, sk, rn)
    return gradient


def calculate_r_descent_direction(signal_f, mu, measurement_info, descent_info):
    gradient = calculate_r_gradient(signal_f, mu, measurement_info, descent_info)

    if descent_info.s_prime_params.r_hessian!=False:
        hessian = calculate_r_hessian_diagonal(signal_f, measurement_info, descent_info)
        descent_direction = -1*gradient/(hessian[:,jnp.newaxis] + 1e-12)
    else:
        hessian = jnp.ones(jnp.shape(gradient))
        descent_direction = -1*gradient
        
    return descent_direction, gradient, jax.vmap(jnp.diag)(hessian)



def calculate_r_error_intensity(trace, measured_trace, mu):
    return jnp.sum(jnp.abs(measured_trace - mu*trace)**2)


def calculate_r_error_amplitude(trace, measured_trace, mu):
    return jnp.sum(jnp.abs(jnp.sign(measured_trace)*jnp.sqrt(jnp.abs(measured_trace)) - mu*jnp.sqrt(trace))**2)


def calculate_r_error(trace, measured_trace, mu, descent_info):
    r_error_dict={"intensity": calculate_r_error_intensity,
                  "amplitude": calculate_r_error_amplitude}
    r_error=r_error_dict[descent_info.s_prime_params.r_gradient](trace, measured_trace, mu)
    return r_error



def calc_r_error_for_linesearch(gamma, linesearch_info, measurement_info, descent_info):
    measured_trace, sk, rn = measurement_info.measured_trace, measurement_info.sk, measurement_info.rn 
    
    descent_direction, mu = linesearch_info.descent_direction, linesearch_info.mu
    signal_t = linesearch_info.signal_t

    signal_t_new = signal_t + gamma*descent_direction
    trace = calculate_trace(do_fft(signal_t_new, sk, rn))
    error = calculate_r_error(trace, measured_trace, mu, descent_info)
    return error


def calc_r_grad_for_linesearch(gamma, linesearch_info, measurement_info, descent_info):
    measured_trace, sk, rn = measurement_info.measured_trace, measurement_info.sk, measurement_info.rn 
    weights = descent_info.s_prime_params.weights
    
    descent_direction, mu = linesearch_info.descent_direction, linesearch_info.mu
    signal_t = linesearch_info.signal_t

    signal_t_new = signal_t + gamma*descent_direction
    signal_f = do_fft(signal_t_new, sk, rn)

    calc_r_grad_dict={"amplitude": calculate_r_gradient_amplitude,
                      "intensity": calculate_r_gradient_intensity}
    
    gradient = calc_r_grad_dict[descent_info.s_prime_params.r_gradient](signal_f, mu, measured_trace, weights, sk, rn)
    return gradient



def calculate_S_prime_iterative_step(signal_t, measured_trace, mu, measurement_info, descent_info, local_or_global):
    sk, rn = measurement_info.sk, measurement_info.rn
    gamma = getattr(descent_info.gamma, local_or_global)

    signal_f=do_fft(signal_t, sk, rn)
    trace=calculate_trace(signal_f)

    descent_direction, gradient, hessian = calculate_r_descent_direction(signal_f, mu, measurement_info, descent_info)

    r_error = calculate_r_error(trace, measured_trace, mu, descent_info)

    descent_direction = adaptive_scaling_of_step(descent_direction, r_error, gradient, hessian, descent_info, "_global")

    if (descent_info.linesearch_params.use_linesearch=="backtracking" or descent_info.linesearch_params.use_linesearch=="wolfe") and local_or_global=="_global":
        pk_dot_gradient = jnp.sum(jnp.real(jnp.vecdot(descent_direction, gradient)))
        linesearch_info = MyNamespace(signal_t=signal_t, descent_direction=descent_direction, error=r_error, 
                                    pk_dot_gradient=pk_dot_gradient, pk=descent_direction, mu=mu)
        
        gamma = do_linesearch(linesearch_info, measurement_info, descent_info, 
                              Partial(calc_r_error_for_linesearch, descent_info=descent_info), 
                              Partial(calc_r_grad_for_linesearch, descent_info=descent_info), local_or_global)
        
    signal_t_new = signal_t + gamma*descent_direction
    return signal_t_new, None


def calculate_S_prime_iterative(signal_t, measured_trace, mu, measurement_info, descent_info, number_of_iterations, local_or_global):
    if number_of_iterations==1:
        signal_t_new, _ = calculate_S_prime_iterative_step(signal_t, measured_trace, mu, measurement_info, descent_info, local_or_global)
    else:
        step = Partial(calculate_S_prime_iterative_step, measured_trace=measured_trace, mu=mu, measurement_info=measurement_info, descent_info=descent_info, 
                       local_or_global=local_or_global)
        do_step = Partial(scan_helper, actual_function=step, number_of_args=1, number_of_xs=0)
        signal_t_new, _ = jax.lax.scan(do_step, signal_t, length=number_of_iterations)
    return signal_t_new






def calculate_S_prime(signal_t, measured_trace, mu, measurement_info, descent_info, local_or_global):
    method = getattr(descent_info.s_prime_params, local_or_global)

    if method=="projection":
        signal_t_new = calculate_S_prime_projection(signal_t, measured_trace, mu, measurement_info)

    elif method=="iteration":
        number_of_iterations = descent_info.s_prime_params.number_of_iterations
        signal_t_new = calculate_S_prime_iterative(signal_t, measured_trace, mu, measurement_info, descent_info, number_of_iterations, local_or_global)

    else:
        print("something is wrong")

    return signal_t_new