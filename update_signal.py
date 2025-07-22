import jax.numpy as jnp
import jax

from jax.tree_util import Partial


from utilities import  MyNamespace, do_fft, do_ifft, project_onto_intensity, calculate_mu, calculate_trace
from linesearch import do_linesearch












def calculate_S_prime_projection(signal_t, measured_trace, mu, measurement_info):
    sk, rn = measurement_info.sk, measurement_info.rn
    signal_f=do_fft(signal_t, sk, rn)

    signal_f_new=project_onto_intensity(signal_f, measured_trace)

    signal_t_new=do_ifft(signal_f_new, sk, rn)*1/(jnp.sqrt(mu)+1e-12)
    return signal_t_new

















def calculate_r_gradient_intensity(signal_f, measured_trace, weights, sk, rn):
    trace = calculate_trace(signal_f)
    mu = calculate_mu(trace, measured_trace)
    grad_r = -4*mu*do_ifft(signal_f*(measured_trace-mu*trace)*weights**2, sk, rn)
    return grad_r 


def calculate_r_gradient_amplitude(signal_f, measured_trace, weights, sk, rn):
    mu = jnp.sum(jnp.sqrt(measured_trace)*jnp.abs(signal_f))/jnp.sum(jnp.abs(signal_f)**2)
    grad_r = -4*mu*do_ifft((jnp.sqrt(measured_trace)*jnp.exp(1j*jnp.angle(signal_f))-mu*signal_f)*weights, sk, rn)
    return grad_r



def calculate_r_gradient(signal_f, measurement_info, descent_info):
    measured_trace, sk, rn = measurement_info.measured_trace, measurement_info.sk, measurement_info.rn
    weights = descent_info.weights

    calc_r_grad_dict={"amplitude":calculate_r_gradient_amplitude,
                      "intensity": calculate_r_gradient_intensity}
    calc_r_grad = jax.vmap(calc_r_grad_dict[descent_info.r_gradient], in_axes=(0,None,None,None,None))
    gradient = calc_r_grad(signal_f, measured_trace, weights, sk, rn)

    return gradient



def calculate_r_error(trace, measured_trace):
    mu=calculate_mu(trace, measured_trace)
    return jnp.sum(jnp.abs(measured_trace - mu*trace)**2)



def calc_r_error_for_linesearch(gamma, linesearch_info, measurement_info, descent_info):
    measured_trace, sk, rn = measurement_info.measured_trace, measurement_info.sk, measurement_info.rn 
    
    descent_direction, eta = linesearch_info.descent_direction, linesearch_info.eta
    signal_t = linesearch_info.signal_t

    signal_t_new = signal_t.signal_t + gamma*eta*descent_direction
    trace = calculate_trace(do_fft(signal_t_new, sk, rn))
    error = calculate_r_error(trace, measured_trace)
    return error



def calc_r_grad_for_linesearch(gamma, linesearch_info, measurement_info, descent_info):
    measured_trace, sk, rn = measurement_info.measured_trace, measurement_info.sk, measurement_info.rn 
    weights = descent_info.weights
    
    descent_direction, eta = linesearch_info.descent_direction, linesearch_info.eta
    signal_t = linesearch_info.signal_t

    signal_t_new = signal_t.signal_t + gamma*eta*descent_direction
    signal_f = do_fft(signal_t_new, sk, rn)

    calc_r_grad_dict={"amplitude": calculate_r_gradient_amplitude,
                      "intensity": calculate_r_gradient_intensity}
    gradient = calc_r_grad_dict[descent_info.r_gradient](signal_f, measured_trace, weights, sk, rn)

    return gradient



def calculate_S_prime_iterative_step(signal_t, measurement_info, descent_info):
    sk, rn, measured_trace = measurement_info.sk, measurement_info.rn, measurement_info.measured_trace
    xi = descent_info.xi

    signal_f=do_fft(signal_t.signal_t, sk, rn)
    trace=calculate_trace(signal_f)

    gradient = calculate_r_gradient(signal_f, measurement_info, descent_info)

    

    descent_direction = -1*gradient

    r_error = jax.vmap(calculate_r_error, in_axes=(0, None))(trace, measured_trace)
    eta = r_error/(jnp.sum(jnp.abs(descent_direction)**2, axis=(1, 2)) + xi)


    if descent_info.use_linesearch=="backtracking" or descent_info.use_linesearch=="wolfe":
        pk_dot_gradient = jax.vmap(lambda x,y: jnp.real(jnp.vdot(x,y)), in_axes=(0,0))(descent_direction, gradient)        
        linesearch_info=MyNamespace(signal_t=signal_t, descent_direction=descent_direction, error=r_error, 
                                    pk_dot_gradient=pk_dot_gradient, pk=descent_direction, eta=eta)
        
        gamma = jax.vmap(do_linesearch, in_axes=(0,None,None,None,None))(linesearch_info, measurement_info, descent_info, 
                                                                            Partial(calc_r_error_for_linesearch, descent_info=descent_info),
                                                                            Partial(calc_r_grad_for_linesearch, descent_info=descent_info))
        
    signal_t_new = signal_t.signal_t + gamma[:, jnp.newaxis, jnp.newaxis]*eta[:, jnp.newaxis, jnp.newaxis]*descent_direction

    return signal_t_new






def calculate_S_prime(signal_t, measured_trace, mu, measurement_info, descent_info, method="projection"):

    if method=="projection":
        # this one needs to be vmapped over
        signal_t_new = calculate_S_prime_projection(signal_t, measured_trace, mu, measurement_info)

    elif method=="iteratively":
        # this one is not made to be vmapped over 
        signal_t_new = calculate_S_prime_iterative_step(signal_t, measurement_info, descent_info)

    else:
        print("something is wrong")

    return signal_t_new