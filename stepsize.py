import jax.numpy as jnp
import jax
from jax.tree_util import Partial
from utilities import while_loop_helper



# possible other linesearches
# bisection based on error-value -> e.g. like LSF algorihtm



def do_linesearch_step(condition, gamma, iteration, linesearch_info, measurement_info, linesearch_params, error_func, grad_func):
    c1, c2, delta_gamma = linesearch_params.c1, linesearch_params.c2, linesearch_params.delta_gamma
    pk_dot_gradient, pk, error = linesearch_info.pk_dot_gradient, linesearch_info.descent_direction, linesearch_info.error

    delta_gamma_1, delta_gamma_2 = delta_gamma

    error_new = error_func(gamma, linesearch_info, measurement_info)

    # Armijio Condition
    if linesearch_params.use_linesearch=="backtracking" or linesearch_params.use_linesearch=="wolfe":
        x = jnp.sign((error_new-error) - gamma*c1*pk_dot_gradient)
        condition_one = jnp.real(1-(x+1)/2).astype(jnp.int16)

    # Strong Wolfe Condition
    if linesearch_params.use_linesearch=="wolfe":
        grad = grad_func(gamma, linesearch_info, measurement_info)
        x = jnp.sign(jnp.abs(jnp.real(jnp.vdot(pk, grad))) - c2*jnp.abs(pk_dot_gradient)) # negative -> True
        condition_two = jnp.real(1-(x+1)/2).astype(jnp.int16)
    else:
        condition_two = 1

    gamma = gamma*condition_one*condition_two + gamma*delta_gamma_1*(1 - condition_one) + gamma*delta_gamma_2*(1 - condition_two)
    return condition_one*condition_two, gamma, iteration + 1
    


def end_linesearch(condition, gamma, iteration_no, max_steps_linesearch): 
    run_out_of_steps = 1 - jnp.sign(max_steps_linesearch - iteration_no)
    is_linesearch_done = condition + run_out_of_steps
    is_linesearch_done = -0.5*(is_linesearch_done - 1.5)**2 + 1.125
    return (1 - is_linesearch_done).astype(bool)



def do_linesearch(linesearch_info, measurement_info, descent_info, error_func, grad_func, local_or_global):
    assert 0 < descent_info.linesearch_params.c1 < descent_info.linesearch_params.c2 < 1, "Constants for linesearch ar invalid"

    gamma, max_steps_linesearch = getattr(descent_info.gamma, local_or_global), descent_info.linesearch_params.max_steps

    condition = 0
    current_step = 0

    linesearch_step=Partial(do_linesearch_step, linesearch_info=linesearch_info, measurement_info=measurement_info, linesearch_params=descent_info.linesearch_params, 
                            error_func=error_func, grad_func=grad_func)
    linesearch_step=Partial(while_loop_helper, actual_function=linesearch_step, number_of_args=3)

    linesearch_end=Partial(end_linesearch, max_steps_linesearch=max_steps_linesearch)
    linesearch_end=Partial(while_loop_helper, actual_function=linesearch_end, number_of_args=3)

    initial_vals=(condition, gamma, current_step)
    condition, gamma, iteration_no=jax.lax.while_loop(linesearch_end, linesearch_step, initial_vals)

    return gamma


















def calc_adaptive_step_size_global(error, gradient, gHg, xi, order, local_or_global_state, pulse_or_gate):
    if order=="linear":
        grad_norm2 = jnp.sum(jnp.abs(gradient)**2) + xi
        eta = error/grad_norm2

    elif order=="nonlinear":
        gHg = gHg + xi
        grad_norm2 = jnp.sum(jnp.abs(gradient)**2)
        diskriminante = (grad_norm2/gHg)**2 - error/gHg
        jax.debug.print("{error}", error=(error, diskriminante))
        diskriminante = jnp.maximum(diskriminante, 1e-12) # avoid sqrt of negative values
        eta = grad_norm2/gHg - jnp.sqrt(diskriminante)
        
    else:
        print("not available")

    return eta, local_or_global_state




def calc_adaptive_step_size_local(error, gradient, gHg, xi, order, local_or_global_state, pulse_or_gate):
    grad_norm2 = jnp.sum(jnp.abs(gradient)**2)

    if order=="linear":
        max_grad_norm2 = getattr(local_or_global_state.max_grad_norm2, pulse_or_gate)
        max_grad_norm2 = jnp.greater(grad_norm2, max_grad_norm2)*grad_norm2 + jnp.greater(max_grad_norm2, grad_norm2)*max_grad_norm2
        eta = error/max_grad_norm2

    elif order=="nonlinear":
        max_gHg = getattr(local_or_global_state.max_gHg, pulse_or_gate)
        max_gHg = jnp.greater(gHg, max_gHg)*gHg + jnp.greater(max_gHg, gHg)*max_gHg
        diskriminante = (grad_norm2/max_gHg)**2 - error/max_gHg
        diskriminante = jnp.maximum(diskriminante, 1e-12) # avoid sqrt(negative)
        eta = grad_norm2/max_gHg - jnp.sqrt(diskriminante)

    else:
        print("notavailable")

    return eta, local_or_global_state




def adaptive_scaling_of_step(descent_direction, error, gradient, gHg, local_or_global_state, descent_info, local_or_global, pulse_or_gate):
    order = getattr(descent_info.adaptive_scaling, local_or_global)
    if order!=False:
        get_step_size={"_local": calc_adaptive_step_size_local,
                       "_global": calc_adaptive_step_size_global}
        
        eta, local_or_global_state = get_step_size[local_or_global](error, gradient, gHg, descent_info.xi, order, 
                                                                    local_or_global_state, pulse_or_gate)
    else:
        eta = 1

    return eta*descent_direction, local_or_global_state