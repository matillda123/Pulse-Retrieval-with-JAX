import jax.numpy as jnp
import jax
from jax.tree_util import Partial
from equinox import tree_at

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


















def get_scaling(gradient, descent_direction, xi, local_or_global_state, pulse_or_gate, local_or_global):
    
    scaling = jnp.sum(jnp.real(jnp.vecdot(descent_direction, gradient))) + xi

    if local_or_global=="_local":
        max_scaling = getattr(local_or_global_state.max_scaling, pulse_or_gate)
        scaling = jnp.greater(-1*scaling, max_scaling)*scaling + jnp.greater(max_scaling, -1*scaling)*max_scaling
        local_or_global_state = tree_at(lambda x: getattr(x.max_scaling, pulse_or_gate), local_or_global_state, -1*scaling)

    elif local_or_global=="_global":
        pass

    else:
        print("not available")
    

    return scaling, local_or_global_state




def get_step_size(error, gradient, descent_direction, local_or_global_state, xi, order, pulse_or_gate, local_or_global):
    scaling, local_or_global_state = get_scaling(gradient, descent_direction, xi, local_or_global_state, pulse_or_gate, local_or_global)

    if order=="linear":
        eta = -1*error/(2*scaling) # negative is needed because descent_direction also has -1.

    elif order=="nonlinear":
        diskriminante = 1 + error/scaling
        eta = 1 - jnp.sqrt(jnp.abs(diskriminante))*jnp.sign(diskriminante)

    # elif order=="nonlinear_optimistic":
    #     diskriminante = 1 + error/scaling
    #     eta = 1 + jnp.sqrt(jnp.abs(diskriminante))*jnp.sign(diskriminante)
        
    else:
        print("not available")


    return eta, local_or_global_state




def adaptive_scaling_of_step(error, gradient, descent_direction, local_or_global_state, xi, order, pulse_or_gate, local_or_global):

    if order!=False:
        eta, local_or_global_state = get_step_size(error, gradient, descent_direction, local_or_global_state, xi, order, pulse_or_gate, local_or_global)
    else:
        eta = 1

    return eta*descent_direction, local_or_global_state

