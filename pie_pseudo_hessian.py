import jax.numpy as jnp
import jax
from jax.tree_util import Partial
from utilities import scan_helper, solve_linear_system



def PIE_get_pseudo_hessian_subelement(dummy, signal_f, measured_trace, D_arr_kj):
    res=D_arr_kj*(2-jnp.sign(measured_trace)*jnp.sqrt(jnp.abs(measured_trace))/(jnp.abs(signal_f)+1e-9))
    return dummy + res, None


def PIE_get_pseudo_hessian_element(probe_k, probe_j, time_k, time_j, omega, signal_f, measured_trace): # tauarr is very likely time 
    D_arr_kj=jnp.exp(1j*omega*(time_k-time_j))

    get_subelement=Partial(scan_helper, actual_function=PIE_get_pseudo_hessian_subelement, number_of_args=1, number_of_xs=3)

    carry=0+0j
    xs=(signal_f, measured_trace, D_arr_kj)
    val_subelement,_=jax.lax.scan(get_subelement, carry, xs)

    hess_element=0.25*jnp.conjugate(probe_k)*probe_j*val_subelement
    return hess_element



def PIE_get_pseudo_hessian_one_m(probe, signal_f, measured_trace, measurement_info, use_hessian):
    time, frequency = measurement_info.time, measurement_info.frequency


    get_hessian=Partial(PIE_get_pseudo_hessian_element, omega=2*jnp.pi*frequency, signal_f=signal_f, measured_trace=measured_trace)

    if use_hessian=="full":
        hessian=jax.vmap(jax.vmap(get_hessian, in_axes=(None, 0, None, 0)), in_axes=(0, None, 0, None))(probe, probe, time, time)

    elif use_hessian=="diagonal":
        hessian=jax.vmap(get_hessian, in_axes=(0,0,0,0))(probe, probe, time, time)

    return hessian



def PIE_get_pseudo_hessian_all_m(probe_all_m, signal_f, measurement_info, use_hessian, pulse_or_gate):

    if pulse_or_gate=="pulse" or pulse_or_gate=="dscan_pie":
        in_axes=(0,0,0)

    elif pulse_or_gate=="gate":
        in_axes=(None,0,0)

    get_hessian=Partial(PIE_get_pseudo_hessian_one_m, measurement_info=measurement_info, use_hessian=use_hessian)
    hessian_all_m=jax.vmap(get_hessian, in_axes=in_axes)(probe_all_m, signal_f, measurement_info.measured_trace)

    return hessian_all_m
    







def PIE_get_pseudo_newton_direction(grad, probe, signal_f, newton_direction_prev, measurement_info, descent_info, pulse_or_gate):
    if pulse_or_gate=="pulse":
        descent_direction = PIE_get_pseudo_newton_direction_pulse(grad, probe, signal_f, newton_direction_prev, measurement_info, descent_info, pulse_or_gate)

    elif pulse_or_gate=="gate":
        descent_direction = PIE_get_pseudo_newton_direction_gate(grad, probe, signal_f, newton_direction_prev, measurement_info, descent_info, pulse_or_gate)
    else:
        print("somwthing wrong")

    return descent_direction







def PIE_get_pseudo_newton_direction_pulse(grad, probe, signal_f, newton_direction_prev, measurement_info, descent_info, pulse_or_gate):
    use_hessian, lambda_lm, solver = descent_info.use_hessian, descent_info.lambda_lm, descent_info.linalg_solver

    hessian_all_m=jax.vmap(PIE_get_pseudo_hessian_all_m, in_axes=(0,0,None,None,None))(probe, signal_f, measurement_info, use_hessian, pulse_or_gate)

    if use_hessian=="full":
        grad = jnp.sum(grad, axis=1)
        hessian = jnp.sum(hessian_all_m, axis=1)

        idx=jax.vmap(jnp.diag_indices_from)(hessian)
        hessian=jax.vmap(lambda x,y: x.at[y].add(lambda_lm*jnp.abs(x[y])))(hessian, idx)

        newton_direction=solve_linear_system(hessian, grad, newton_direction_prev, solver)

    elif use_hessian=="diagonal":
        newton_direction=grad/(hessian_all_m + lambda_lm*jnp.max(jnp.abs(hessian_all_m)**2, axis=1)[:, jnp.newaxis, :])
        newton_direction=jnp.sum(newton_direction, axis=1)

    else:
        print(f"{use_hessian} not available. needs to be diagonal or full")

    return -1*newton_direction







def PIE_get_pseudo_newton_direction_gate(grad, probe, signal_f, newton_direction_prev, measurement_info, descent_info, pulse_or_gate):
    use_hessian, lambda_lm, solver = descent_info.use_hessian, descent_info.lambda_lm, descent_info.linalg_solver

    hessian_all_m=jax.vmap(PIE_get_pseudo_hessian_all_m, in_axes=(0,0,None,None,None))(probe, signal_f, measurement_info, use_hessian, pulse_or_gate)

    if use_hessian=="full":

        idx=jax.vmap(jax.vmap(jnp.diag_indices_from))(hessian_all_m)
        hessian=jax.vmap(jax.vmap(lambda x,y: x.at[y].add(lambda_lm*jnp.abs(x[y]))))(hessian_all_m, idx)

        newton_direction=jax.vmap(solve_linear_system, in_axes=(1,1,1,None), out_axes=1)(hessian, grad, newton_direction_prev, solver)

    elif use_hessian=="diagonal":
        newton_direction=grad/(hessian_all_m + lambda_lm*jnp.max(jnp.abs(hessian_all_m)**2, axis=1)[:, jnp.newaxis, :])

    else:
        print(f"{use_hessian} not available. needs to be diagonal or full")

    return -1*newton_direction