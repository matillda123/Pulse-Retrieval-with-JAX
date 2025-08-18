import jax.numpy as jnp
import jax
from jax.tree_util import Partial

from utilities import scan_helper, solve_linear_system, MyNamespace




def calc_Z_error_pseudo_hessian_subelement_shg(dummy_element, pulse_t_dispersed, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=2*jnp.abs(pulse_t_dispersed)**2*exp_arr_mn*jnp.conjugate(exp_arr_mp)
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None


def calc_Z_error_pseudo_hessian_subelement_thg(dummy_element, pulse_t_dispersed, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=4.5*jnp.abs(pulse_t_dispersed)**4*exp_arr_mn*jnp.conjugate(exp_arr_mp)
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None



def calc_Z_error_pseudo_hessian_subelement_pg(dummy_element, pulse_t_dispersed, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=2.5*jnp.abs(pulse_t_dispersed)**4
    Vzz_k=2*jnp.real(jnp.conjugate(pulse_t_dispersed)*(signal_t_new_m-signal_t_m))
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k*exp_arr_mn*jnp.conjugate(exp_arr_mp)
    return dummy_element + res, None






def calc_Z_error_pseudo_hessian_element(exp_arr_mp, exp_arr_mn, omega_p, omega_n, time, pulse_t_dispersed, signal_t_m, signal_t_new_m, nonlinear_method):
    
    D_arr_pn=jnp.exp(1j*time*(omega_p-omega_n))

    calc_hessian_subelement={"shg": calc_Z_error_pseudo_hessian_subelement_shg,
                             "thg": calc_Z_error_pseudo_hessian_subelement_thg,
                             "pg": calc_Z_error_pseudo_hessian_subelement_pg,
                             "sd": calc_Z_error_pseudo_hessian_subelement_pg,
                             "tg": calc_Z_error_pseudo_hessian_subelement_pg}


    calc_subelement=Partial(calc_hessian_subelement[nonlinear_method], exp_arr_mn=exp_arr_mn, exp_arr_mp=exp_arr_mp)
    element_scan=Partial(scan_helper, actual_function=calc_subelement, number_of_args=1, number_of_xs=4)

    carry=0+0j
    xs=(pulse_t_dispersed, signal_t_m, signal_t_new_m, D_arr_pn)
    hessian_element, _ =jax.lax.scan(element_scan, carry, xs)

    return hessian_element



def calc_Z_error_pseudo_hessian_one_m(exp_arr_m, pulse_t_dispersed, signal_t_m, signal_t_new_m, time, omega, nonlinear_method, full_or_diagonal):

    calc_hessian_partial=Partial(calc_Z_error_pseudo_hessian_element, time=time, pulse_t_dispersed=pulse_t_dispersed, 
                                 signal_t_m=signal_t_m, signal_t_new_m=signal_t_new_m, nonlinear_method=nonlinear_method)

    if full_or_diagonal=="full":
        calc_hessian=jax.vmap(jax.vmap(calc_hessian_partial, in_axes=(0, None, 0, None)), in_axes=(None, 0, None, 0))

    elif full_or_diagonal=="diagonal":
        calc_hessian=jax.vmap(calc_hessian_partial, in_axes=(0, 0, 0, 0))
    
    hessian=calc_hessian(exp_arr_m, exp_arr_m, omega, omega)
    return hessian



def calc_Z_error_pseudo_hessian_all_m(pulse_t_dispersed, signal_t, signal_t_new, phase_matrix, measurement_info, full_or_diagonal):
    time, omega, nonlinear_method = measurement_info.time, 2*jnp.pi*measurement_info.frequency, measurement_info.nonlinear_method

    exp_arr=jnp.exp(-1j*phase_matrix)

    hessian_all_m=Partial(calc_Z_error_pseudo_hessian_one_m, time=time, omega=omega, nonlinear_method=nonlinear_method, full_or_diagonal=full_or_diagonal)
    h_all_m=jax.vmap(hessian_all_m, in_axes=(0,0,0,0))(exp_arr, pulse_t_dispersed, signal_t, signal_t_new)

    return h_all_m




def get_pseudo_newton_direction_Z_error(grad_m, pulse_t_dispersed, signal_t, signal_t_new, phase_matrix, measurement_info, hessian_state, hessian_info, 
                                        full_or_diagonal):

    lambda_lm = hessian_info.lambda_lm
    solver = hessian_info.linalg_solver

    newton_direction_prev = hessian_state.newton_direction_prev.pulse

    # vmap over population here -> only for small populations since memory will explode. 
    hessian_m=jax.vmap(calc_Z_error_pseudo_hessian_all_m, in_axes=(0,0,0,0,None,None))(pulse_t_dispersed, signal_t, signal_t_new, phase_matrix, measurement_info, 
                                                                                       full_or_diagonal)

    hessian=jnp.sum(hessian_m, axis=1)
    grad=jnp.sum(grad_m, axis=1)

    if full_or_diagonal=="full":
        idx=jax.vmap(jnp.diag_indices_from)(hessian)
        hessian=jax.vmap(lambda x,y: x.at[y].add(lambda_lm*jnp.abs(x[y])))(hessian, idx)

        newton_direction=solve_linear_system(hessian, grad, newton_direction_prev, solver)

    elif full_or_diagonal=="diagonal":
        hessian = hessian + lambda_lm*jnp.max(jnp.abs(hessian), axis=1)[:, jnp.newaxis]
        newton_direction = grad/hessian
        hessian = jax.vmap(jnp.diag)(hessian)

    else:
        print("something is wrong")

    hessian_state = MyNamespace(newton_direction_prev = newton_direction)
    return -1*newton_direction, hessian_state
