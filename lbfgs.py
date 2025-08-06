import jax.numpy as jnp
import jax
from jax.tree_util import Partial

from utilities import MyNamespace



# the two loops need to have differing counting directions
# because of this its easier to scan over index and not over element
def backward_loop(q, i, rho, s, y):
    alpha = rho[i]*jnp.vdot(s[i], q)
    q = q - alpha*y[i]
    return q, alpha

def forward_loop(r, i, alpha, rho, s, y):
    beta = rho[i]*jnp.vdot(y[i], r)
    r = r + s[i] * (alpha[i] - beta)
    return r, None


def do_lbfgs(grad_current, lbfgs_state, hessian):
    grad_prev, newton_direction_prev, step_size_prev = lbfgs_state.grad_prev, lbfgs_state.newton_direction_prev, lbfgs_state.step_size_prev

    m = hessian.lbfgs_memory
    m_backward = jnp.arange(0,m,1)
    m_forward = jnp.arange(m,0,-1)

    s = -1*step_size_prev*newton_direction_prev
    y = grad_current - grad_prev


    rho = 1/jnp.real(jnp.vecdot(y,s) + 1e-14)
    rho = jnp.maximum(rho, 0) # ignore iterations with negative curvature
    alpha = jnp.zeros(m, dtype=jnp.complex64)
    
    do_backward = Partial(backward_loop, rho=rho, s=s, y=y)
    q, alpha = jax.lax.scan(do_backward, grad_current, m_backward)

    n = jnp.shape(grad_prev)[-1]
    gamma = 1 
    r = gamma*jnp.eye(n) @ q
    do_forward = Partial(forward_loop, alpha=alpha, rho=rho, s=s, y=y)
    r, _ = jax.lax.scan(do_forward, r, m_forward)

    newton_direction = r
    return newton_direction



# lbfgs isnt sensible when doing alternating optimization 
def get_pseudo_newton_direction(grad, lbfgs_state, descent_info):
    newton_direction = jax.vmap(do_lbfgs, in_axes=(0,0,None))(grad, lbfgs_state, descent_info.hessian)

    grad_arr = lbfgs_state.grad_prev
    grad_arr = grad_arr.at[:,1:].set(grad_arr[:,:-1])
    grad_arr = grad_arr.at[:,0].set(grad)

    newton_arr = lbfgs_state.newton_direction_prev
    newton_arr = newton_arr.at[:,1:].set(newton_arr[:,:-1])
    newton_arr = newton_arr.at[:,0].set(newton_direction)

    lbfgs_state = MyNamespace(grad_prev = grad_arr, 
                              newton_direction_prev=newton_arr,
                              step_size_prev = lbfgs_state.step_size_prev)
    return -1*newton_direction, lbfgs_state