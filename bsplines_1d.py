import jax.numpy as jnp
import jax
from jax.tree_util import Partial

import equinox as eqx
from scipy.special import factorial # using jax.scipy limits everything to k=8 because of accuracy



def get_C_in(i, n):
    return factorial(n)/(factorial(i)*factorial(n-i))


def get_m_ij(i, j, k):
    temp = [(-1)**(s-j) * get_C_in(s-j, k) * (k-s-1)**(k-1-i) for s in range(j, k, 1)]
    temp = jnp.sum(jnp.asarray(temp))

    m = get_C_in(k-1-i, k-1) * temp
    return m

def get_M(k):
    M = [[jnp.round(get_m_ij(i, j, k), 0).astype(int) for j in range(k)] for i in range(k)]
    return jnp.asarray(M).T

def get_prefactor(k):
    return 1/factorial(k-1)



    
def _make_one_patch(cpoints, k, M, Nx):
    u = jnp.linspace(0, 1, Nx)
    arr = jnp.arange(k).reshape(-1,1)
    u = u**arr

    w = jnp.dot(M, u)
    c = jnp.einsum("i, ij -> j", cpoints, w)
    return c


def _make_all_patches(i, cpoints, k, M, Nx):
    c = jax.lax.dynamic_slice(cpoints, (i,), (k,))
    s = _make_one_patch(c, k, M, Nx)
    return s


def make_bsplines(cpoints, k, M, f, Nx):
    s = jnp.shape(cpoints)
    tx = jnp.arange(s[0] - (k-1))

    patch_func = Partial(_make_all_patches, cpoints=cpoints, k=k, M=M, Nx=Nx)
    s_vals = jax.vmap(patch_func)(tx)
    s_arr = jnp.concatenate(s_vals[:,:-1], axis=-1) # :-1 to get rid of shared points between patches
    return s_arr*f