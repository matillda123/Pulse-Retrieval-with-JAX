import jax.numpy as jnp
import jax

from jax.tree_util import Partial


from pulsedjax.utilities import scan_helper, MyNamespace, do_fft, do_ifft, project_onto_intensity, calculate_trace
from .stepsize import adaptive_step_size






def calculate_S_prime_projection(signal_f, measured_trace, mu, sk, rn):
    """
    Calculates signal_t_new/S_prime via a projection onto the measured intensity.

    Args:
        signal_f (jnp.array): the complex signal field in the frequency domain of the current guess
        measured_trace (jnp.array): the measured intensity
        mu (float): the scaling factor between the measured intensity and the intensity of the current guess
        measurement_info (Pytree): contains measurement data and information
    
    Returns:
        jnp.array, the complex signal field in the time domain projected onto the measured intensity
    """
    signal_f_new = project_onto_intensity(signal_f, measured_trace)
    signal_t_new = do_ifft(signal_f_new, sk, rn)*1/(jnp.sqrt(mu)+1e-12)
    return signal_t_new











def calculate_r_newton_diagonal_intensity(trace, measured_trace):
    H_zz_diag = jnp.sum(2*trace - measured_trace, axis=-1) # in local stages this becomes a scalar, which is correct
    return H_zz_diag


def calculate_r_newton_diagonal_amplitude(trace, measured_trace):
    H_zz_diag = jnp.ones(jnp.shape(trace)[0])
    return H_zz_diag


def calculate_r_newton_diagonal(signal_f, measured_trace, sk, rn, descent_info):
    trace = calculate_trace(signal_f)

    calc_r_newton_diag_dict={"amplitude": calculate_r_newton_diagonal_amplitude,
                              "intensity": calculate_r_newton_diagonal_intensity}
    
    # r_gradient is correct here, no need for extra r_newton with amp/int
    hessian = calc_r_newton_diag_dict[descent_info.s_prime_params.r_gradient](trace, measured_trace)
    return hessian



def calculate_r_gradient_intensity(signal_f, mu, measured_trace, weights, sk, rn):
    trace = calculate_trace(signal_f)
    grad_r = -4*mu*do_ifft(signal_f*(measured_trace - mu*trace)*weights**2, sk, rn)
    return grad_r 


def calculate_r_gradient_amplitude(signal_f, mu, measured_trace, weights, sk, rn):
    grad_r = -4*mu*do_ifft((jnp.sqrt(measured_trace)*jnp.exp(1j*jnp.angle(signal_f))-mu*signal_f)*weights, sk, rn)
    return grad_r



def calculate_r_gradient(signal_f, mu, measured_trace, sk, rn, descent_info):
    weights = descent_info.s_prime_params.weights

    calc_r_grad_dict={"amplitude": calculate_r_gradient_amplitude,
                      "intensity": calculate_r_gradient_intensity}
    
    gradient = calc_r_grad_dict[descent_info.s_prime_params.r_gradient](signal_f, mu, measured_trace, weights, sk, rn)
    return gradient


def calculate_r_descent_direction(signal_f, mu, measured_trace, sk, rn, descent_info):
    """
    Calculates descent direction of the iterative calculation of signal_t_new/S_prime. 
    Uses either gradient descent or newtons method with the diagonal approximation. 
    The error-functions can be based on intensity or amplitude based residuals. 
    """
    gradient = calculate_r_gradient(signal_f, mu, measured_trace, sk, rn, descent_info)

    if descent_info.s_prime_params.r_newton!=False:
        hessian = calculate_r_newton_diagonal(signal_f, measured_trace, sk, rn, descent_info)
        descent_direction = -1*gradient/(hessian[:,jnp.newaxis] + 1e-12)
    else:
        descent_direction = -1*gradient
        
    return descent_direction, gradient






def calculate_r_error_intensity(trace, measured_trace, mu):
    return jnp.sum(jnp.abs(measured_trace - mu*trace)**2)


def calculate_r_error_amplitude(trace, measured_trace, mu):
    return jnp.sum(jnp.abs(jnp.sign(measured_trace)*jnp.sqrt(jnp.abs(measured_trace)) - mu*jnp.sqrt(trace))**2)


def calculate_r_error(trace, measured_trace, mu, descent_info):
    r_error_dict={"intensity": calculate_r_error_intensity,
                  "amplitude": calculate_r_error_amplitude}
    r_error=r_error_dict[descent_info.s_prime_params.r_gradient](trace, measured_trace, mu)
    return r_error



# def calc_r_error_for_linesearch(gamma, linesearch_info, measurement_info, descent_info):
#     measured_trace, sk, rn = measurement_info.measured_trace, measurement_info.sk, measurement_info.rn 
    
#     descent_direction, mu = linesearch_info.descent_direction, linesearch_info.mu
#     signal_t = linesearch_info.signal_t

#     signal_t_new = signal_t + gamma*descent_direction
#     trace = calculate_trace(do_fft(signal_t_new, sk, rn))
#     error = calculate_r_error(trace, measured_trace, mu, descent_info)
#     return error


# def calc_r_grad_for_linesearch(gamma, linesearch_info, measurement_info, descent_info):
#     measured_trace, sk, rn = measurement_info.measured_trace, measurement_info.sk, measurement_info.rn 
#     weights = descent_info.s_prime_params.weights
    
#     descent_direction, mu = linesearch_info.descent_direction, linesearch_info.mu
#     signal_t = linesearch_info.signal_t

#     signal_t_new = signal_t + gamma*descent_direction
#     signal_f = do_fft(signal_t_new, sk, rn)

#     calc_r_grad_dict={"amplitude": calculate_r_gradient_amplitude,
#                       "intensity": calculate_r_gradient_intensity}
    
#     gradient = calc_r_grad_dict[descent_info.s_prime_params.r_gradient](signal_f, mu, measured_trace, weights, sk, rn)
#     return gradient



def calculate_S_prime_iterative_step(signal_t, signal_f, measured_trace, mu, sk, rn, descent_info, local_or_global):
    """ One iteration of the iterative descent based calculation of signal_t_new/S_prime. """
    gamma = getattr(descent_info.gamma, local_or_global)

    trace = calculate_trace(signal_f)
    descent_direction, gradient = calculate_r_descent_direction(signal_f, mu, measured_trace, sk, rn, descent_info)
    r_error = calculate_r_error(trace, measured_trace, mu, descent_info)

    descent_direction, _ = adaptive_step_size(r_error, gradient, descent_direction, descent_info.xi, MyNamespace(), MyNamespace(order="pade_10", factor=-1), None, "_global")

    # Is removed because it makes usage more complicated. 
    # if (descent_info.linesearch_params.use_linesearch=="backtracking" or descent_info.linesearch_params.use_linesearch=="wolfe") and local_or_global=="_global":
    #     pk_dot_gradient = jnp.sum(jnp.real(jnp.vecdot(descent_direction, gradient)))
    #     linesearch_info = MyNamespace(signal_t=signal_t, descent_direction=descent_direction, error=r_error, 
    #                                 pk_dot_gradient=pk_dot_gradient, mu=mu)
        
    #     gamma = do_linesearch(linesearch_info, measurement_info, descent_info, 
    #                           Partial(calc_r_error_for_linesearch, descent_info=descent_info), 
    #                           Partial(calc_r_grad_for_linesearch, descent_info=descent_info), local_or_global)
    
    signal_t_new = signal_t + gamma*descent_direction
    return signal_t_new, None


def calculate_S_prime_iterative(signal_t, signal_f, measured_trace, mu, sk, rn, descent_info, local_or_global):
    """
    Calculates signal_t_new/S_prime via an iterative optimization of the least-squares error.

    Args:
        signal_t (jnp.array): the complex signal field in the time domain of the current guess
        signal_t (jnp.array): the complex signal field in the frequency domain of the current guess
        measured_trace (jnp.array): the measured intensity
        mu (float): the scaling factor between the measured intensity and the intensity of the current guess
        measurement_info (Pytree): contains measurement data and information
        descent_info (Pytree): contains information on the behaviour of the solver
        local_or_global (str): whether this is used in a local or global iteration
    
    Returns:
        jnp.array, the complex signal field in the time domain projected onto the measured intensity
    """


    number_of_iterations = descent_info.s_prime_params.number_of_iterations
    if number_of_iterations==1:
        signal_t_new, _ = calculate_S_prime_iterative_step(signal_t, signal_f, measured_trace, mu, sk, rn, descent_info, local_or_global)
    else:
        # this might be broken because of tuple input to scan?
        step = Partial(calculate_S_prime_iterative_step, measured_trace=measured_trace, mu=mu, sk=sk, rn=rn, descent_info=descent_info, 
                       local_or_global=local_or_global)
        do_step = Partial(scan_helper, actual_function=step, number_of_args=2, number_of_xs=0)
        signal_t_new, _ = jax.lax.scan(do_step, (signal_t, signal_f), length=number_of_iterations)
    return signal_t_new






def calculate_S_prime(signal_t, signal_f, measured_trace, mu, measurement_info, descent_info, local_or_global):
    """
    Calculates signal_t_new/S_prime via projection or iterative optimization

    Args:
        signal_t (jnp.array): the complex signal field in the time domain of the current guess
        signal_f (jnp.array): the complex signal field in the frequency domain of the current guess
        measured_trace (jnp.array): the measured intensity
        mu (float): the scaling factor between the measured intensity and the intensity of the current guess
        measurement_info (Pytree): contains measurement data and information
        descent_info (Pytree): contains information on the behaviour of the solver
        local_or_global (str): whether this is used in a local or global iteration
    
    Returns:
        jnp.array, the complex signal field in the time domain projected onto the measured intensity
    """
        
    method = getattr(descent_info.s_prime_params, local_or_global)

    if measurement_info.real_fields==True:
        sk, rn = measurement_info.sk_exp, measurement_info.rn_exp
    else:
        sk, rn = measurement_info.sk, measurement_info.rn
        

    if method=="projection":
        signal_t_new = calculate_S_prime_projection(signal_f, measured_trace, mu, sk, rn)

    elif method=="iteration":
        signal_t_new = calculate_S_prime_iterative(signal_t, signal_f, measured_trace, mu, sk, rn, descent_info, local_or_global)

    else:
         raise ValueError(f"method needs to be one of projection or iteration. Not {method}")

    return signal_t_new