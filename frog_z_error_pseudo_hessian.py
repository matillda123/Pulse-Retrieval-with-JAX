import jax.numpy as jnp
import jax
from jax.tree_util import Partial

from utilities import solve_linear_system, scan_helper, MyNamespace



def calc_Z_error_pseudo_hessian_subelement_shg(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    term1=pulse_t_shifted_m+pulse_t*exp_arr_mn
    term2=pulse_t_shifted_m+pulse_t*exp_arr_mp
    Uzz_k=0.5*jnp.conjugate(term1)*term2
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None


def calc_Z_error_pseudo_hessian_subelement_thg(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    term1=pulse_t_shifted_m+2*pulse_t*exp_arr_mn
    term2=pulse_t_shifted_m+2*pulse_t*exp_arr_mp
    Uzz_k=0.5*jnp.abs(pulse_t_shifted_m)**2*jnp.conjugate(term1)*term2
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None



def calc_Z_error_pseudo_hessian_subelement_pg(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    term1=pulse_t_shifted_m+pulse_t*exp_arr_mn
    term2=pulse_t_shifted_m+pulse_t*exp_arr_mp
    Uzz_k=0.5*(jnp.abs(pulse_t_shifted_m)**2*jnp.conjugate(term1)*term2 + jnp.abs(pulse_t*pulse_t_shifted_m)*2*jnp.conjugate(exp_arr_mn)*exp_arr_mp)

    term3=pulse_t_shifted_m+pulse_t*exp_arr_mn
    term4=pulse_t_shifted_m+pulse_t*exp_arr_mp
    Vzz_k=0.5*(jnp.conjugate(term3)*(signal_t_new_m-signal_t_m)*exp_arr_mp + term4*jnp.conjugate((signal_t_new_m-signal_t_m)*exp_arr_mn))

    Hzz_k=Uzz_k-Vzz_k
    res=D_arr_pn*Hzz_k
    return dummy_element + res, None



def calc_Z_error_pseudo_hessian_subelement_sd(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=0.5*(jnp.abs(pulse_t_shifted_m)**4+4*jnp.abs(pulse_t*pulse_t_shifted_m)**2*jnp.conjugate(exp_arr_mn)*exp_arr_mp)
    Vzz_k=pulse_t_shifted_m*(signal_t_new_m-signal_t_m)*exp_arr_mp+jnp.conjugate(pulse_t_shifted_m*(signal_t_new_m-signal_t_m)*exp_arr_mn)
    Hzz_k=Uzz_k-Vzz_k
    
    res=D_arr_pn*Hzz_k
    return dummy_element + res, None







def calc_Z_error_pseudo_hessian_subelement_xfrog_pulse(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=0.5*jnp.abs(gate_shifted_m)**2
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k
    
    res=D_arr_pn*Hzz_k
    return dummy_element + res, None





def calc_Z_error_pseudo_hessian_subelement_shg_xfrog_gate(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=0.5*jnp.conjugate(exp_arr_mn)*exp_arr_mp*jnp.abs(pulse_t)**2
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None


def calc_Z_error_pseudo_hessian_subelement_thg_xfrog_gate(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=2*jnp.conjugate(exp_arr_mn)*exp_arr_mp*jnp.abs(pulse_t)**2*jnp.abs(pulse_t_shifted_m)**2
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None


def calc_Z_error_pseudo_hessian_subelement_pg_xfrog_gate(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=0.5*jnp.conjugate(exp_arr_mn)*exp_arr_mp*jnp.abs(pulse_t)**2*jnp.abs(pulse_t_shifted_m)**2
    Vzz_k=jnp.real((signal_t_new_m-signal_t_m)*jnp.conjugate(pulse_t))*jnp.conjugate(exp_arr_mn)*exp_arr_mp
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None


def calc_Z_error_pseudo_hessian_subelement_sd_xfrog_gate(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=2*jnp.conjugate(exp_arr_mn)*exp_arr_mp*jnp.abs(pulse_t)**2*jnp.abs(pulse_t_shifted_m)**2
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None








def calc_Z_error_pseudo_hessian_subelement_shg_ifrog(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=2*jnp.conjugate(1+exp_arr_mn)*(1+exp_arr_mp)*jnp.abs(pulse_t+pulse_t_shifted_m)**2
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None

def calc_Z_error_pseudo_hessian_subelement_thg_ifrog(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=4.5*jnp.conjugate(1+exp_arr_mn)*(1+exp_arr_mp)*jnp.abs(pulse_t+pulse_t_shifted_m)**4  
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None


def calc_Z_error_pseudo_hessian_subelement_pg_ifrog(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=2.5*jnp.conjugate(1+exp_arr_mn)*(1+exp_arr_mp)*jnp.abs(pulse_t+pulse_t_shifted_m)**4
    Vzz_k=2*jnp.conjugate(1+exp_arr_mn)*(1+exp_arr_mp)*jnp.real((pulse_t+pulse_t_shifted_m)*jnp.conjugate(signal_t_new_m-signal_t_m))
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None






def calc_Z_error_pseudo_hessian_subelement_shg_ifrog_xfrog_pulse(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=2*jnp.abs(pulse_t+pulse_t_shifted_m)**2
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None

def calc_Z_error_pseudo_hessian_subelement_thg_ifrog_xfrog_pulse(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=4.5*jnp.abs(pulse_t+pulse_t_shifted_m)**4
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None

def calc_Z_error_pseudo_hessian_subelement_pg_ifrog_xfrog_pulse(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=2.5*jnp.abs(pulse_t+pulse_t_shifted_m)**4
    Vzz_k=2*jnp.real(jnp.conjugate(pulse_t+pulse_t_shifted_m)*(signal_t_new_m-signal_t_m))
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None



def calc_Z_error_pseudo_hessian_subelement_shg_ifrog_xfrog_gate(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=2*jnp.abs(pulse_t+pulse_t_shifted_m)**2*jnp.conjugate(exp_arr_mn)*exp_arr_mp
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None

def calc_Z_error_pseudo_hessian_subelement_thg_ifrog_xfrog_gate(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=4.5*jnp.abs(pulse_t+pulse_t_shifted_m)**4*jnp.conjugate(exp_arr_mn)*exp_arr_mp
    Vzz_k=0
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None

def calc_Z_error_pseudo_hessian_subelement_pg_ifrog_xfrog_gate(dummy_element, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn, exp_arr_mn, exp_arr_mp):
    Uzz_k=2.5*jnp.abs(pulse_t+pulse_t_shifted_m)**4*jnp.conjugate(exp_arr_mn)*exp_arr_mp
    Vzz_k=2*jnp.real(jnp.conjugate(pulse_t+pulse_t_shifted_m)*(signal_t_new_m-signal_t_m))*jnp.conjugate(exp_arr_mn)*exp_arr_mp
    Hzz_k=Uzz_k-Vzz_k

    res=D_arr_pn*Hzz_k
    return dummy_element + res, None






def calc_Z_error_pseudo_hessian_element_pulse(exp_arr_mp, exp_arr_mn, omega_p, omega_n, time_k, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, 
                                              frogmethod, xfrog, ifrog):
    
    D_arr_pn=jnp.exp(1j*time_k*(omega_p-omega_n))


    if xfrog==True or xfrog=="doubleblind":
        xfrog=True
    else:
        xfrog=False



    hess_func_ifrog_False_xfrog_False={"shg": calc_Z_error_pseudo_hessian_subelement_shg, 
                                       "thg": calc_Z_error_pseudo_hessian_subelement_thg, 
                                       "pg": calc_Z_error_pseudo_hessian_subelement_pg, 
                                       "sd": calc_Z_error_pseudo_hessian_subelement_sd}
    
    hess_func_ifrog_False_xfrog_True={"shg": calc_Z_error_pseudo_hessian_subelement_xfrog_pulse, 
                                       "thg": calc_Z_error_pseudo_hessian_subelement_xfrog_pulse, 
                                       "pg": calc_Z_error_pseudo_hessian_subelement_xfrog_pulse, 
                                       "sd": calc_Z_error_pseudo_hessian_subelement_xfrog_pulse}

    hess_func_ifrog_True_xfrog_False={"shg": calc_Z_error_pseudo_hessian_subelement_shg_ifrog,
                                     "thg": calc_Z_error_pseudo_hessian_subelement_thg_ifrog,
                                     "pg": calc_Z_error_pseudo_hessian_subelement_pg_ifrog}
    
    hess_func_ifrog_True_xfrog_True={"shg": calc_Z_error_pseudo_hessian_subelement_shg_ifrog_xfrog_pulse,
                                     "thg": calc_Z_error_pseudo_hessian_subelement_thg_ifrog_xfrog_pulse,
                                     "pg": calc_Z_error_pseudo_hessian_subelement_pg_ifrog_xfrog_pulse}


    hess_func_ifrog_False={False: hess_func_ifrog_False_xfrog_False,
                           True: hess_func_ifrog_False_xfrog_True}
    

    hess_func_ifrog_True={False: hess_func_ifrog_True_xfrog_False,
                          True: hess_func_ifrog_True_xfrog_True}
    
    
    calc_hessian_subelement={False: hess_func_ifrog_False,
                             True: hess_func_ifrog_True}

    
    calc_subelement=Partial(calc_hessian_subelement[ifrog][xfrog][frogmethod], exp_arr_mn=exp_arr_mn, exp_arr_mp=exp_arr_mp)
    element_scan=Partial(scan_helper, actual_function=calc_subelement, number_of_args=1, number_of_xs=6)

    carry=0+0j
    xs=(pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn)
    hessian_element, _ =jax.lax.scan(element_scan, carry, xs)

    return hessian_element







def calc_Z_error_pseudo_hessian_element_gate(exp_arr_mp, exp_arr_mn, omega_p, omega_n, time_k, pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, 
                                              frogmethod, xfrog, ifrog):
    
    D_arr_pn=jnp.exp(1j*time_k*(omega_p-omega_n))

    
    hess_func_ifrog_False_xfrog_gate={"shg": calc_Z_error_pseudo_hessian_subelement_shg_xfrog_gate, 
                                       "thg": calc_Z_error_pseudo_hessian_subelement_thg_xfrog_gate, 
                                       "pg": calc_Z_error_pseudo_hessian_subelement_pg_xfrog_gate, 
                                       "sd": calc_Z_error_pseudo_hessian_subelement_sd_xfrog_gate}
    
    
    hess_func_ifrog_True_xfrog_gate={"shg": calc_Z_error_pseudo_hessian_subelement_shg_ifrog_xfrog_gate,
                                     "thg": calc_Z_error_pseudo_hessian_subelement_thg_ifrog_xfrog_gate,
                                     "pg": calc_Z_error_pseudo_hessian_subelement_pg_ifrog_xfrog_gate}
    
    
    calc_hessian_subelement={False: hess_func_ifrog_False_xfrog_gate,
                             True: hess_func_ifrog_True_xfrog_gate}

    
    calc_subelement=Partial(calc_hessian_subelement[ifrog][frogmethod], exp_arr_mn=exp_arr_mn, exp_arr_mp=exp_arr_mp)
    element_scan=Partial(scan_helper, actual_function=calc_subelement, number_of_args=1, number_of_xs=6)

    carry=0+0j
    xs=(pulse_t, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, D_arr_pn)
    hessian_element, _ =jax.lax.scan(element_scan, carry, xs)

    return hessian_element






def calc_Z_error_pseudo_hessian_one_m(exp_arr_m, pulse_t_shifted_m, gate_shifted_m, signal_t_m, signal_t_new_m, 
                                      pulse_t, time, omega, frogmethod, xfrog, ifrog, full_or_diagonal, pulse_or_gate):

    calc_Z_error_pseudo_hessian_element = {"pulse": calc_Z_error_pseudo_hessian_element_pulse,
                                           "gate": calc_Z_error_pseudo_hessian_element_gate}
    calc_hessian_partial=Partial(calc_Z_error_pseudo_hessian_element[pulse_or_gate], time_k=time, pulse_t=pulse_t, pulse_t_shifted_m=pulse_t_shifted_m, 
                                 gate_shifted_m=gate_shifted_m, signal_t_m=signal_t_m, signal_t_new_m=signal_t_new_m, frogmethod=frogmethod, xfrog=xfrog, ifrog=ifrog)

    if full_or_diagonal=="full":
        calc_hessian=jax.vmap(jax.vmap(calc_hessian_partial, in_axes=(0, None, 0, None)), in_axes=(None, 0, None, 0))

    elif full_or_diagonal=="diagonal":
        calc_hessian=jax.vmap(calc_hessian_partial, in_axes=(0, 0, 0, 0))
    
    hessian=calc_hessian(exp_arr_m, exp_arr_m, omega, omega)
    return hessian



def calc_Z_error_pseudo_hessian_all_m(pulse_t, pulse_t_shifted, gate_shifted, signal_t, signal_t_new, tau_arr, measurement_info, full_or_diagonal, pulse_or_gate):
    time, omega = measurement_info.time, 2*jnp.pi*measurement_info.frequency
    xfrog, ifrog, frogmethod = measurement_info.xfrog, measurement_info.ifrog, measurement_info.nonlinear_method

    exp_arr = jnp.exp(-1j*jnp.outer(tau_arr, omega))


    hessian_all_m=Partial(calc_Z_error_pseudo_hessian_one_m, pulse_t=pulse_t, time=time, omega=omega, frogmethod=frogmethod, xfrog=xfrog, ifrog=ifrog, 
                          full_or_diagonal=full_or_diagonal, pulse_or_gate=pulse_or_gate)
    h_all_m=jax.vmap(hessian_all_m, in_axes=(0,0,0,0,0))(exp_arr, pulse_t_shifted, gate_shifted, signal_t, signal_t_new)

    return h_all_m




def get_pseudo_newton_direction_Z_error(grad_m, pulse_t_shifted, gate_shifted, signal_t, signal_t_new, tau_arr, descent_state, measurement_info, hessian, 
                                        full_or_diagonal, pulse_or_gate, in_axes=None):
    lambda_lm = hessian.lambda_lm
    solver = hessian.linalg_solver

    pulse_t_arr = descent_state.population.pulse
    newton_direction_prev = getattr(descent_state.hessian, pulse_or_gate).newton_direction_prev

    # vmap over population here -> only for small populations since memory will explode. 
    hessian_m=jax.vmap(calc_Z_error_pseudo_hessian_all_m, in_axes=in_axes)(pulse_t_arr, pulse_t_shifted, gate_shifted, signal_t, signal_t_new, 
                                                                           tau_arr, measurement_info, full_or_diagonal, pulse_or_gate)
    
    hessian=jnp.sum(hessian_m, axis=1)
    grad=jnp.sum(grad_m, axis=1)
    
    if full_or_diagonal=="full":
        
        idx=jax.vmap(jnp.diag_indices_from)(hessian)
        hessian=jax.vmap(lambda x,y: x.at[y].add(lambda_lm*jnp.abs(x[y])))(hessian, idx)

        newton_direction=solve_linear_system(hessian, grad, newton_direction_prev, solver)

    elif full_or_diagonal=="diagonal":
        hessian = hessian + lambda_lm*jnp.max(jnp.abs(hessian), axis=1)[:, jnp.newaxis]
        newton_direction=grad/hessian
        hessian = jax.vmap(jnp.diag)(hessian)
        
    else:
        print("something is wrong")

    hessian = MyNamespace(hessian=hessian, newton_direction_prev = newton_direction)
    return -1*newton_direction, hessian
        











