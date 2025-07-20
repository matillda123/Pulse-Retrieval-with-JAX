import jax.numpy as jnp
import jax
from jax.tree_util import Partial
from utilities import while_loop_helper



def do_linesearch_step(condition, gamma, iteration, linesearch_info, measurement_info, descent_info, error_func, grad_func):
    c1, c2, delta_gamma = descent_info.c1, descent_info.c2, descent_info.delta_gamma
    pk_dot_gradient, pk, error = linesearch_info.pk_dot_gradient, linesearch_info.pk, linesearch_info.error

    delta_gamma_1, delta_gamma_2 = delta_gamma

    error_new = error_func(gamma, linesearch_info, measurement_info)

    # Armijio Condition
    x = jnp.sign((error_new-error) - gamma*c1*pk_dot_gradient)
    condition_one = jnp.real(1-(x+1)/2).astype(jnp.int16)

    # Strong Wolfe Condition
    if descent_info.wolfe_linesearch==True:
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



def do_linesearch(linesearch_info, measurement_info, descent_info, error_func, grad_func):
    assert 0 < descent_info.c1 < descent_info.c2 < 1, "Constants for linesearch ar invalid"

    gamma, max_steps_linesearch = descent_info.gamma, descent_info.max_steps_linesearch

    condition = 0
    current_step = 0

    linesearch_step=Partial(do_linesearch_step, linesearch_info=linesearch_info, measurement_info=measurement_info, descent_info=descent_info, 
                            error_func=error_func, grad_func=grad_func)
    linesearch_step=Partial(while_loop_helper, actual_function=linesearch_step, number_of_args=3)

    linesearch_end=Partial(end_linesearch, max_steps_linesearch=max_steps_linesearch)
    linesearch_end=Partial(while_loop_helper, actual_function=linesearch_end, number_of_args=3)

    initial_vals=(condition, gamma, current_step)
    condition, gamma, iteration_no=jax.lax.while_loop(linesearch_end, linesearch_step, initial_vals)

    return gamma