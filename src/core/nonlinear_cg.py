import jax.numpy as jnp
from src.utilities import MyNamespace



def get_nonlinear_CG_direction(descent_direction, cg, beta_parameter_version):
    """
    Calculates the descent direction using Nonlinear Conjugate Gradients. 

    Args:
        descent_direction: jnp.array, the descent direction from a descent solver
        cg: Pytree, the current conjugate gradient state, holds the previous descent direction etc.
        beta_parameter_version: str, the concrete NCG method to use
    
    Returns:
        tuple[jnp.array, Pytree], the descent direction based on NCG, the updated NCG state
    """
    descent_direction_prev, CG_direction_prev = cg.descent_direction_prev, cg.CG_direction_prev
    
    # negative one to convert descent_direction to grad or pseudo-newton direction
    beta=get_beta(-1*descent_direction, descent_direction_prev, CG_direction_prev, beta_parameter_version)
    CG_direction = descent_direction + beta*CG_direction_prev

    cg = MyNamespace(CG_direction_prev=CG_direction, descent_direction_prev = -1*descent_direction)
    return CG_direction, cg




def get_beta(grad, grad_prev, descent_direction_prev, beta_parameter_version):
    """ Calls the NCG-parameter functions. """
    beta_param={"fletcher_reeves": beta_fletcher_reeves,
               "polak_ribiere": beta_fletcher_reeves,
               "hestenes_stiefel": beta_hestenes_stiefel,
               "dai_yuan": beta_dai_yuan,
               "average": beta_average}

    return beta_param[beta_parameter_version](grad, grad_prev, descent_direction_prev)


def beta_fletcher_reeves(grad, grad_prev, descent_direction_prev):
    """ Compute beta based on Fletcher-Reeves. """
    return jnp.sum(jnp.abs(grad)**2)/(jnp.sum(jnp.abs(grad_prev))**2+1e-12)

def beta_polak_ribiere(grad, grad_prev, descent_direction_prev):
    """ Compute beta based one Polak-Ribiere. """
    return jnp.real(jnp.sum(jnp.conjugate(grad-grad_prev)*grad))/(jnp.sum(jnp.abs(grad_prev))**2+1e-12)


def beta_hestenes_stiefel(grad, grad_prev, descent_direction_prev):
    """ Compute beta based on Hestenes-Stiefel. """
    delta_grad=grad-grad_prev
    return jnp.real(jnp.sum(jnp.conjugate(delta_grad)*grad))/jnp.real(jnp.sum(jnp.conjugate(delta_grad)*descent_direction_prev)+1e-12)


def beta_dai_yuan(grad, grad_prev, descent_direction_prev):
    """ Compute beta based on Dai-Yuan. """
    delta_grad=grad-grad_prev
    return jnp.sum(jnp.abs(grad)**2)/jnp.real(jnp.sum(jnp.conjugate(delta_grad)*descent_direction_prev)+1e-12)


def beta_average(grad, grad_prev, descent_direction_prev):
    """ Compute beta as an average of the existing methods. """
    beta1 = beta_fletcher_reeves(grad, grad_prev, descent_direction_prev)
    beta2 = beta_polak_ribiere(grad, grad_prev, descent_direction_prev)
    beta3 = beta_hestenes_stiefel(grad, grad_prev, descent_direction_prev)
    beta4 = beta_dai_yuan(grad, grad_prev, descent_direction_prev)
    return (beta1 + beta2 + beta3 + beta4)/4