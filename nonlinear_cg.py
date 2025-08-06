import jax.numpy as jnp
from utilities import MyNamespace



def get_nonlinear_CG_direction(descent_direction, cg, beta_parameter_version):
    descent_direction_prev, CG_direction_prev = cg.descent_direction_prev, cg.CG_direction_prev
    
    # negative one to convert descent_direction to grad or pseudo-newton direction
    beta=get_beta_parameter(-1*descent_direction, descent_direction_prev, CG_direction_prev, beta_parameter_version)
    CG_direction = descent_direction + beta*CG_direction_prev

    cg = MyNamespace(descent_direction_prev = -1*descent_direction, CG_direction_prev=CG_direction)
    return CG_direction, cg




def get_beta_parameter(grad, grad_prev, descent_direction_prev, beta_parameter_version):
    
    beta_param={"fletcher_reeves": beta_fletcher_reeves,
               "polak_ribiere": beta_fletcher_reeves,
               "hestenes_stiefel": beta_hestenes_stiefel,
               "dai_yuan": beta_dai_yuan,
               "average": beta_average}

    return beta_param[beta_parameter_version](grad, grad_prev, descent_direction_prev)


def beta_fletcher_reeves(grad, grad_prev, descent_direction_prev):
    return jnp.sum(jnp.abs(grad)**2)/(jnp.sum(jnp.abs(grad_prev))**2+1e-6)

def beta_polak_ribiere(grad, grad_prev, descent_direction_prev):
    return jnp.abs(jnp.sum(jnp.conjugate(grad-grad_prev)*grad))/(jnp.sum(jnp.abs(grad_prev))**2+1e-6)


def beta_hestenes_stiefel(grad, grad_prev, descent_direction_prev):
    delta_grad=grad-grad_prev
    return jnp.abs(jnp.sum(jnp.conjugate(delta_grad)*grad))/jnp.abs(jnp.sum(jnp.conjugate(delta_grad)*descent_direction_prev)+1e-6)


def beta_dai_yuan(grad, grad_prev, descent_direction_prev):
    delta_grad=grad-grad_prev
    return jnp.sum(jnp.abs(grad)**2)/jnp.abs(jnp.sum(jnp.conjugate(delta_grad)*descent_direction_prev)+1e-6)


def beta_average(grad, grad_prev, descent_direction_prev):
    beta1=beta_fletcher_reeves(grad, grad_prev, descent_direction_prev)
    beta2=beta_polak_ribiere(grad, grad_prev, descent_direction_prev)
    beta3=beta_hestenes_stiefel(grad, grad_prev, descent_direction_prev)
    beta4=beta_dai_yuan(grad, grad_prev, descent_direction_prev)
    return (beta1+beta2+beta3+beta4)/4