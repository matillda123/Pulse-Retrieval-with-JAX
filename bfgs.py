import jax.numpy as jnp
import jax


# B0_inv=jnp.ones()
# sk = gamma*descent_direction # with the minus x_(k+1)=xk + sk
# yk = grad_(k+1) - grad_k

# t1=(sk.H*yk + yk.H*B_inv*yk)*(sk*sk.H)
# t2=(sk.H*yk)**2  # this might be abs or real?

# t3=B_inv*yk*sk.H + sk*yk.H*B_inv
# t4=sk.H*yk # division by complex value? -> its a real value by defualt, use jnp.real to get rid of numerical noise

# B_inv = B_inv + t1/t2 - t3/t4



def update_B_inv(B_inv, current_grad, previous_grad, previous_update):
    sk = previous_update
    yk = current_grad - previous_grad

    t1 = jnp.conjugate(sk) @ yk
    t2 = jnp.conjugate(sk) @ B_inv @ yk
    t3 = jnp.outer(sk, jnp.conjugate(sk))  # this jnp.outer

    m1 = ((t1 + t2)/jnp.real(t1)**2)*t3



    t4 = jnp.outer(B_inv @ yk, jnp.conjugate(sk))
    t5 = jnp.outer(sk, jnp.conjugate(yk) @ B_inv) # this will probably wont work because of shape isses for @

    m2 = (t4 + t5)/jnp.real(t1)


    B_inv = B_inv + m1 - m2
    return B_inv




def update_B_inv_for_all_m(grad_all_m, bfgs_state):
    B_inv, previous_grad, previous_descent_direction, previous_step_size = bfgs_state.B_inv, bfgs_state.grad_prev, bfgs_state.descent_direction_prev, bfgs_state.step_size_prev
    previous_update = previous_step_size*previous_descent_direction

    B_inv = jax.vmap(update_B_inv, in_axes=(0,0,0,0))(B_inv, grad_all_m, previous_grad, previous_update)
    return B_inv



def get_pseudo_newton_direction(grad_all_m, bfgs_state):
    B_inv = update_B_inv_for_all_m(grad_all_m, bfgs_state)

    newton_direction = B_inv @ grad_all_m
    return newton_direction