import jax.numpy as jnp
import jax
from functools import partial as Partial

from pulsedjax.utilities import MyNamespace


# can i replace the loops by some vector-operation/broadcasting operation?


# the two loops need to have differing counting directions
# because of this it seems easier to scan over index and not over element
def backward_loop(q, i, rho, s, y):
    """ Constructing the LBFGS direction without materializing the inverse hessian is done through nested vector-multiplications. """
    
    # rho[i] has shape () -> but (C,) in dtme
    # s[i] has shape (n,) -> but (C,n) in dtme
    # q has shape (n,) -> but (C,n) in dtme
    # y[i] has shape (n,) .> but (C,n) in dtme
    alpha = rho[i]*jnp.vecdot(s[i], q)
    q = q - alpha*y[i]
    return q, alpha

def forward_loop(r, i, alpha, rho, s, y):
    """ Constructing the LBFGS direction without materializing the inverse hessian is done through nested vector-multiplications. """
    beta = rho[i]*jnp.vecdot(y[i], r)
    r = r + s[i] * (alpha[i] - beta)
    return r, None


def calculate_quasi_newton_direction(grad_current, grad_prev, rho, s, y, newton_info):
    """ Does the actual LBFGS calculation. """
    m = newton_info.lbfgs_memory

    m_backward = jnp.arange(0,m,1)
    m_forward = jnp.arange(m,0,-1)
    alpha = jnp.zeros(m, dtype=jnp.complex64)
    do_backward = Partial(backward_loop, rho=rho, s=s, y=y)
    q, alpha = jax.lax.scan(do_backward, grad_current, m_backward)

    n = jnp.shape(grad_prev)[-1] # (n,)
    I = jnp.broadcast_to(jnp.eye(n), jnp.shape(grad_current) + (n,))
    r = jnp.einsum("...np, n -> ...p", I, q) # this should work with dtme in streaking

    do_forward = Partial(forward_loop, alpha=alpha, rho=rho, s=s, y=y)
    newton_direction, _ = jax.lax.scan(do_forward, r, m_forward)
    return newton_direction



def do_lbfgs(grad_current, lbfgs_state, descent_info):
    """ Prepares and calls the LBFGS calculation. """
    grad_prev, newton_direction_prev, step_size_prev = lbfgs_state.grad_prev, lbfgs_state.newton_direction_prev, lbfgs_state.step_size_prev

    s = -1*step_size_prev*newton_direction_prev
    y = grad_current - grad_prev

    ys = jnp.vecdot(y,s)
    ys_is_zero = (ys==0)
    sign_ys = jnp.sign(ys)*(1-ys_is_zero) + 1*ys_is_zero
    rho = 1/jnp.real(ys + sign_ys*1e-14)
    rho = jnp.maximum(rho, 0) # ignore iterations with negative curvature

    newton_direction = calculate_quasi_newton_direction(grad_current, grad_prev, rho, s, y, descent_info.newton)
    return newton_direction




def get_quasi_newton_direction(grad, lbfgs_state, descent_info):
    """
    Calculate the quasi-newton direciton using LBFGS.

    Args:
        grad (jnp.array): the current gradient
        lbfgs_stat (Pytree): the current lbfgs state
        descent_info (Pytree): holds information on the solver (e.g. memory size for LBFGS)

    Returns:
        tuple[jnp.array, Pytree], the quasi-newton direction and the unchanged lbfgs_state 
    """
    newton_direction = jax.vmap(do_lbfgs, in_axes=(0,0,None))(grad, lbfgs_state, descent_info)

    return -1*newton_direction, lbfgs_state