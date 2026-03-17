import jax.numpy as jnp
import jax
from scipy.special import factorial # using jax.scipy limits accuracy



def get_C_in(i, n):
    """ Naive implementation of n choose i. """
    return factorial(n)/(factorial(i)*factorial(n-i))


def get_m_ij(i, j, k):
    """ Calculates matrix elements (i,j) for a uniform bspline weight matrix of order k. """
    temp = [(-1)**(s-j) * get_C_in(s-j, k) * (k-s-1)**(k-1-i) for s in range(j, k, 1)]
    temp = jnp.sum(jnp.asarray(temp))

    m = get_C_in(k-1-i, k-1) * temp
    return m

def get_M(k):
    """ Calculates the weight matrix for uniform bsplines of order k. """
    M = [[jnp.round(get_m_ij(i, j, k), 0).astype(int) for j in range(k)] for i in range(k)]
    return jnp.asarray(M).T

def get_prefactor(k):
    """ Calculates the prefactor for a weight matrix for uniform bsplines of order k. """
    return 1/factorial(k-1)



def make_bsplines(cpoints, k, M, f, Nx):
    """ Evaluate arbitrary order bsplines in 1D. """
    u = jnp.linspace(0, 1, Nx)
    arr = jnp.arange(k).reshape(-1,1)
    u = u**arr
    w = jnp.dot(M, u)

    p = jax.lax.conv_general_dilated(cpoints[None,None,:], w.T[:,None,:], 
                                     window_strides=(1,), 
                                     padding="VALID", dimension_numbers=("NCH","OIH","HCN"))

    s_arr = jnp.concatenate(jnp.squeeze(p)[:,:-1], axis=-1) # :-1 to get rid of shared points between patches
    return s_arr*f