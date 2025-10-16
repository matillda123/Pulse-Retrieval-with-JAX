import jax.numpy as jnp
import jax
from jax.tree_util import Partial

from scipy.special import factorial # using jax.scipy limits to k=8 because of accuracy



def get_C_in(i, n):
    """ Naive implementation of n choose i. """
    return factorial(n)/(factorial(i)*factorial(n-i))


def get_m_ij(i, j, k):
    """ Calculates matrix elements (i,j) for a uniform bspline matrix of order k. """
    temp = [(-1)**(s-j) * get_C_in(s-j, k) * (k-s-1)**(k-1-i) for s in range(j, k, 1)]
    temp = jnp.sum(jnp.asarray(temp))

    m = get_C_in(k-1-i, k-1) * temp
    return m

def get_M(k):
    """ Calculates a basis matrix for uniform bsplines of order k. """
    M = [[jnp.round(get_m_ij(i, j, k), 0).astype(int) for j in range(k)] for i in range(k)]
    return jnp.asarray(M).T

def get_prefactor(k):
    """ Calculates the prefactor for a basis matrix for uniform bsplines of order k. """
    return 1/factorial(k-1)



    
def _make_one_patch(cpoints, k, M, Nx):
    """ Calculate one spline of order k. """
    u = jnp.linspace(0, 1, Nx)
    arr = jnp.arange(k).reshape(-1,1)
    u = u**arr

    w = jnp.dot(M, u)
    c = jnp.einsum("i, ij -> j", cpoints, w)
    return c


def _make_all_patches(i, cpoints, k, M, Nx): # this name is actually wrong. The function y itself only wraps around _make_one_patch.
    c = jax.lax.dynamic_slice(cpoints, (i,), (k,))
    s = _make_one_patch(c, k, M, Nx)
    return s


def make_bsplines(cpoints, k, M, f, Nx):
    """ Evaluate arbitrary order bsplines in 1D. """
    s = jnp.shape(cpoints)
    tx = jnp.arange(s[0] - (k-1))

    patch_func = Partial(_make_all_patches, cpoints=cpoints, k=k, M=M, Nx=Nx)
    s_vals = jax.vmap(patch_func)(tx)
    s_arr = jnp.concatenate(s_vals[:,:-1], axis=-1) # :-1 to get rid of shared points between patches
    return s_arr*f