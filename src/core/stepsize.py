import jax.numpy as jnp
import jax
from jax.tree_util import Partial
from equinox import tree_at

from src.utilities import while_loop_helper, MyNamespace



def backtracking_linesearch(linesearch_state, error_func, grad_func, linesearch_params, linesearch_info, measurement_info):
    """
    Perform one iteration of an Armijo-style linesearch.

    Args:
        linesearch_state: Pytree, contains the armijo condition, gamma and iteration
        error_func: Callable, calculates the error to be optimized, expects gamma, linesearch_info and measurement_info
        grad_func: Callable, unused
        linesearch_params: Pytree, contains parameters for the linesearch iteration
        linesearch_info: Pytree, contains variables related to the initial state of the linesearch
        measurement_info: Pytree, contains measurement data and parameters

    Returns:
        MyNamespace, the updated linesearch_state
    
    """
    gamma = linesearch_state.gamma
    c1, delta_gamma = linesearch_params.c1, linesearch_params.delta_gamma
    pk_dot_gradient, error = linesearch_info.pk_dot_gradient, linesearch_info.error

    assert delta_gamma < 1, "delta_gamma needs to be smaller than one for backtracking linesearch."


    error_new = error_func(gamma, linesearch_info, measurement_info)
    armijo_condition = ((error_new - error) <= gamma*c1*pk_dot_gradient).astype(jnp.int16)

    gamma = gamma*armijo_condition + gamma*delta_gamma*(1 - armijo_condition)
    return MyNamespace(condition=armijo_condition, gamma=gamma, iteration=linesearch_state.iteration+1)







def zoom_interpolation(low_vals, high_vals):
    gamma_low, phi_low, phi_prime_low = low_vals.gamma, low_vals.phi, low_vals.phi_prime
    gamma_high, phi_high, phi_prime_high = high_vals.gamma, high_vals.phi, high_vals.phi_prime

    d1 = phi_prime_low + phi_prime_high - 3*(phi_low - phi_high)/(gamma_low - gamma_high)
    diskriminante = d1**2 - phi_prime_low*phi_prime_high
    d2 = jnp.sign(gamma_high - gamma_low)*jnp.sqrt(diskriminante)
    gamma_cubic = gamma_high - (gamma_high - gamma_low)*(phi_prime_high + d2 - d1)/(phi_prime_high - phi_prime_low + 2*d2)

    gamma_bisection = 0.5*(gamma_low + gamma_high)

    too_much_out_of_range = ((gamma_cubic < gamma_low) | (gamma_cubic > 5*gamma_high))
    not_usable = ((jnp.sign(diskriminante) < 0) | jnp.isnan(gamma_cubic) | (1-jnp.isfinite(gamma_cubic)))
    use_bisection = (too_much_out_of_range | not_usable)

    gamma = jnp.where(use_bisection, gamma_bisection, gamma_cubic)
    return gamma



def finding_phase(current_vals, bracket, linesearch_info, linesearch_params):
    low_vals, high_vals = bracket.high, current_vals
    gamma = current_vals.gamma * linesearch_params.delta_gamma
    return gamma, low_vals, high_vals


def zoom_phase(current_vals, bracket, linesearch_info, linesearch_params):
    error, pk_dot_gradient = linesearch_info.error, linesearch_info.pk_dot_gradient
    c1 = linesearch_params.c1

    gamma, phi, phi_prime = current_vals.gamma, current_vals.phi, current_vals.phi_prime
    gamma_low, phi_low = bracket.low.gamma, bracket.low.phi
        

    psi = phi - error - c1*pk_dot_gradient*gamma
    psi_low = phi_low - error - c1*pk_dot_gradient*gamma_low
    psi_prime = phi_prime - c1*pk_dot_gradient

    case1 = (psi > psi_low)
    case2 = ((psi <= psi_low) & (psi_prime*(gamma_low-gamma) > 0))
    case3 = ((psi <= psi_low) & (psi_prime*(gamma_low-gamma) < 0))
    
    low_vals = case1 * bracket.low + case2 * current_vals + case3 * current_vals
    high_vals = case1 * current_vals + case2 * bracket.high + case3 * bracket.low
    
    gamma = zoom_interpolation(low_vals, high_vals)
    return gamma, low_vals, high_vals




def zoom_linesearch(linesearch_state, error_func, grad_func, linesearch_params, linesearch_info, measurement_info):
    """
    Perform one iteration of an zoom linesearch. Works in two phases. In the first phase an interval containing a minimum is 
    located by doubling of the step size and observation of the behavior of the error and error-gradient. 
    If a suitable interval has been located the minimum is approximated/found through successive cubic interpolation.
    The search terminates if the Armijo- and Wolfe conditions are met.

    Args:
        linesearch_state: Pytree, contains information on the current state of the linesearch
        error_func: Callable, calculates the error to be optimized, expects gamma, linesearch_info and measurement_info
        grad_func: Callable, calculates the error-gradient to be optimized, expects gamma, linesearch_info and measurement_info
        linesearch_params: Pytree, contains parameters for the linesearch iteration
        linesearch_info: Pytree, contains variables related to the initial state of the linesearch
        measurement_info: Pytree, contains measurement data and parameters

    Returns:
        MyNamespace, the updated linesearch_state
    
    """
        
    c1, c2, delta_gamma = linesearch_params.c1, linesearch_params.c2, linesearch_params.delta_gamma
    pk_dot_gradient, pk, error = linesearch_info.pk_dot_gradient, linesearch_info.descent_direction, linesearch_info.error

    assert delta_gamma > 1, "delta_gamma needs to be bigger than one for zoom linesearch."

    gamma, phi, phi_prime = linesearch_state.gamma, linesearch_state.phi, linesearch_state.phi_prime
    bracket = linesearch_state.bracket
    current_vals = MyNamespace(gamma=gamma, phi=phi, phi_prime=phi_prime)

    start_zoom_phase = ((jnp.sign(phi_prime)*jnp.sign(pk_dot_gradient) < 0) | ((phi > bracket.low.phi) & (linesearch_state.iteration > 0))) # overshot minimum
    in_zoom_phase = (linesearch_state.in_zoom_phase | start_zoom_phase)

    zoom = Partial(zoom_phase, linesearch_info=linesearch_info, linesearch_params=linesearch_params)
    finding = Partial(finding_phase, linesearch_info=linesearch_info, linesearch_params=linesearch_params)

    gamma_zoom, low_vals_zoom, high_vals_zoom = zoom(current_vals, bracket)
    gamma_find, low_vals_find, high_vals_find = finding(current_vals, bracket)

    gamma = in_zoom_phase*gamma_zoom + (1-in_zoom_phase)*gamma_find
    low_vals = in_zoom_phase*low_vals_zoom + (1-in_zoom_phase)*low_vals_find
    high_vals = in_zoom_phase*high_vals_zoom + (1-in_zoom_phase)*high_vals_find


    phi = error_func(gamma, linesearch_info, measurement_info)
    armijo_condition = ((phi-error) <= gamma*c1*pk_dot_gradient).astype(jnp.int32)

    grad = grad_func(gamma, linesearch_info, measurement_info)
    phi_prime = jnp.real(jnp.vdot(pk, grad))
    strong_wolfe_condition = (jnp.abs(phi_prime) <= c2*jnp.abs(pk_dot_gradient)).astype(jnp.int32)

    linesearch_done = (armijo_condition & strong_wolfe_condition)

    return MyNamespace(condition=linesearch_done, gamma=gamma, phi=phi, phi_prime=phi_prime, bracket=MyNamespace(low=low_vals, high=high_vals), 
                       in_zoom_phase=in_zoom_phase, iteration=linesearch_state.iteration+1)






def do_linesearch_step(linesearch_state, linesearch_info, measurement_info, linesearch_params, error_func, grad_func):
    linesearch_dict={"backtracking": backtracking_linesearch,
                     "zoom": zoom_linesearch}
                     #"more-thuente": more_thuente_linesearch}

    linesearch_state = linesearch_dict[linesearch_params.use_linesearch](linesearch_state, error_func, grad_func, linesearch_params, 
                                                                         linesearch_info, measurement_info)
    return linesearch_state
    


def end_linesearch(linesearch_state, max_steps_linesearch): 
    iteration_no, condition = linesearch_state.iteration, linesearch_state.condition
    run_out_of_steps = (max_steps_linesearch < iteration_no)
    is_linesearch_done = (condition | run_out_of_steps)
    return (1 - is_linesearch_done).astype(jnp.bool_)



def do_linesearch(linesearch_info, measurement_info, descent_info, error_func, grad_func, local_or_global):
    """
    Perform a linesearch to obtain an improved step size in a descent based optimization.

    Args:
        linesearch_info: Pytree, holds information on the initial state at gamma=0
        measurement_info: Pytree, holds measurement data and parameters
        descent_info: Pytree, holds parameters of the descent algorithm
        error_func: Callable, calculates the error to be optimized, expects gamma, linesearch_info, measurement_info
        grad_func: Callable, calculates the gradient of error_func, expects gamma, linesearch_info, measurement_info
        local_or_global: str, whether this is used in a local or global iteration

    Returns:
        float, the approximated optimal step size

    """
    assert 0 < descent_info.linesearch_params.c1 < descent_info.linesearch_params.c2 < 1, "Constants for linesearch c1 and c2 are invalid"

    gamma, max_steps_linesearch = jnp.float32(getattr(descent_info.gamma, local_or_global)), descent_info.linesearch_params.max_steps

    condition = 0
    iteration = 0

    linesearch_step = Partial(do_linesearch_step, linesearch_info=linesearch_info, measurement_info=measurement_info, 
                              linesearch_params=descent_info.linesearch_params, error_func=error_func, grad_func=grad_func)
    linesearch_step = Partial(while_loop_helper, actual_function=linesearch_step, number_of_args=1)

    linesearch_end = Partial(end_linesearch, max_steps_linesearch=max_steps_linesearch)
    linesearch_end = Partial(while_loop_helper, actual_function=linesearch_end, number_of_args=1)

    if descent_info.linesearch_params.use_linesearch=="zoom":
        gamma_low, phi_low, phi_prime_low = 0.0, linesearch_info.error, linesearch_info.pk_dot_gradient
        bracket = MyNamespace(low = MyNamespace(gamma=gamma_low, phi=phi_low, phi_prime=phi_prime_low), 
                              high = MyNamespace(gamma=gamma_low, phi=phi_low, phi_prime=phi_prime_low))

        phi = error_func(gamma, linesearch_info, measurement_info)
        grad = grad_func(gamma, linesearch_info, measurement_info)
        phi_prime = jnp.real(jnp.vdot(linesearch_info.descent_direction, grad))
        
        linesearch_state = MyNamespace(condition=condition, gamma=gamma, phi=phi, phi_prime=phi_prime, bracket=bracket, 
                                       in_zoom_phase=0, iteration=iteration)
    else:
        linesearch_state = MyNamespace(condition=condition, gamma=gamma, iteration=iteration)


    linesearch_state = jax.lax.while_loop(linesearch_end, linesearch_step, linesearch_state)

    return linesearch_state.gamma


















def get_scaling(gradient, descent_direction, xi, local_or_global_state, pulse_or_gate, local_or_global):

    scaling = jnp.real(jnp.sum(jnp.vecdot(descent_direction, gradient))) + xi

    if local_or_global=="_local":
        max_scaling = getattr(local_or_global_state.max_scaling, pulse_or_gate)
        scaling = jnp.greater(jnp.abs(scaling), jnp.abs(max_scaling))*scaling + jnp.greater(jnp.abs(max_scaling), jnp.abs(scaling))*max_scaling
        local_or_global_state = tree_at(lambda x: getattr(x.max_scaling, pulse_or_gate), local_or_global_state, scaling)

    elif local_or_global=="_global":
        pass

    else:
        raise ValueError(f"local_or_global needs to be _local or _global. Not {local_or_global}")
    

    return scaling, local_or_global_state




def get_step_size(error, gradient, descent_direction, local_or_global_state, xi, order, pulse_or_gate, local_or_global):
    scaling, local_or_global_state = get_scaling(gradient, descent_direction, xi, local_or_global_state, pulse_or_gate, local_or_global)


    L_prime = -1*error # this is the definition in copra paper, seems very aggressive

    if order=="linear" or order=="pade_10":
        eta = (L_prime-error)/(2*scaling)

    elif order=="nonlinear" or order=="pade_20":
        diskriminante = 1 - (L_prime-error)/(2*scaling)
        eta = 2*(1 - jnp.sign(diskriminante)*jnp.sqrt(jnp.abs(diskriminante)))

    elif order=="pade_01":
        eta = error/L_prime*(L_prime-error)/(2*scaling)

    elif order=="pade_11":
        eta =  2*(L_prime-error)/(4*scaling - (L_prime-error))

    elif order=="pade_02":
        diskriminante = 1 - 4*(1+error/(4*scaling))*(L_prime-error)/L_prime
        eta = error/(4*scaling+error)*(1+jnp.sign(diskriminante)*jnp.sqrt(jnp.abs(diskriminante)))
        
    else:
        raise NotImplementedError(f"order={order} is not available.")


    # actually i think having a negative eta is a good idea, since it will reverse descent_direction in the case that scaling is positive, 
    # which means that descent_direction is parallel to gradient -> although this logic is only trivially true for order=linear, idk about other cases
    
    # # if eta is negative then it is not used -> (e.g. if newton direction points opposite to gradient -> scaling is positive -> eta likely negative)
    # is_negative = (eta < 0)
    # eta = 1*is_negative + eta*(1-is_negative)

    return eta, local_or_global_state




def adaptive_step_size(error, gradient, descent_direction, local_or_global_state, xi, order, pulse_or_gate, local_or_global):
    """
    Calculate an improved step size based through a pade approximation of the error function at the current position.

    Args:
        error: float, the current error
        gradient: jnp.array, the current gradient
        descent_direction: jnp.array, the current descent direction
        local_or_global_state: Pytree, holds information of the current descent_state
        xi: float, a damping factor to avoid division by zero
        order: str, the pade-approximation to be used, can be one of pade_10 (linear), pade_20 (nonlinear), pade_01, pade_11 or pade_02
        pulse_or_gate: str, whether this is applied to pulse or gate
        local_or_global: str, whether this happens inside a local or global iteration

    Returns:
        tuple[jnp.array, Pytree], the scaled descent direction and the local_or_global_state
    """

    if order!=False:
        eta, local_or_global_state = get_step_size(error, gradient, descent_direction, local_or_global_state, xi, order, pulse_or_gate, local_or_global)
    else:
        eta = 1

    return eta*descent_direction, local_or_global_state

