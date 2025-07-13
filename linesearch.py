import jax.numpy as jnp
import jax
from jax.tree_util import Partial
from utilities import while_loop_helper



def do_linesearch_step(condition_one, gamma, iteration, linesearch_info, measurement_info, descent_info, error_func):
    c1, delta_gamma = descent_info.c1, descent_info.delta_gamma
    pk_dot_gradient, error = linesearch_info.pk_dot_gradient, linesearch_info.error

    error_new=error_func(gamma, linesearch_info, measurement_info)
    x=jnp.sign((error_new-error) - gamma*c1*pk_dot_gradient) # -> replaces jax.lax.cond

    #condition_one=jnp.real(-0.5*x**2-0.5*x+1) # maps the sign to the correct "boolean"
    condition_one=jnp.real(1-(x+1)/2).astype(jnp.int16)

    gamma_new = gamma*condition_one + gamma*delta_gamma*(1-condition_one)
    return condition_one, gamma_new, iteration + 1
    


def end_linesearch(condition, gamma_new, iteration_no, max_steps_linesearch): 
        
    # replace this jax.lax.cond also with some math tricks?
    # equal -> sign 
    # or -> map both to boolean -> sum of bools has to be bigger than zero -> another sign + mapping could be used afterwards

    # replaced return_true, return_false with lambda function
    #return jax.lax.cond(jnp.logical_or(jnp.equal(iteration_no, max_steps_linesearch), jnp.equal(condition,True)), lambda: False, lambda: True)

    run_out_of_steps = 1 - jnp.sign(max_steps_linesearch - iteration_no)
    is_linesearch_done = condition + run_out_of_steps
    is_linesearch_done = -0.5*(is_linesearch_done - 1.5)**2 + 1.125
    return (1 - is_linesearch_done).astype(bool)



def do_linesearch(linesearch_info, measurement_info, descent_info, error_func):
    gamma, max_steps_linesearch = descent_info.gamma, descent_info.max_steps_linesearch

    condition_one=0
    current_step=0

    linesearch_step=Partial(do_linesearch_step, linesearch_info=linesearch_info, measurement_info=measurement_info, descent_info=descent_info, error_func=error_func)
    linesearch_step=Partial(while_loop_helper, actual_function=linesearch_step, number_of_args=3)

    linesearch_end=Partial(end_linesearch, max_steps_linesearch=max_steps_linesearch)
    linesearch_end=Partial(while_loop_helper, actual_function=linesearch_end, number_of_args=3)

    initial_vals=(condition_one, gamma, current_step)
    condition_one, gamma_new, iteration_no=jax.lax.while_loop(linesearch_end, linesearch_step, initial_vals)

    return gamma_new