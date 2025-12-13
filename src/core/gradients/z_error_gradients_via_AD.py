import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from src.utilities import MyNamespace



def split_population_real_imag(population, doubleblind):
    """ Splits population into real/imag. """
    pulse_real, pulse_imag = jnp.real(population.pulse), jnp.imag(population.pulse)
    pulse = MyNamespace(real=pulse_real, imag=pulse_imag)
    if doubleblind==True:
        gate_real, gate_imag = jnp.real(population.gate), jnp.imag(population.gate)
        gate = MyNamespace(real=gate_real, imag=gate_imag)
    else:
        gate = None
    return MyNamespace(pulse=pulse, gate=gate)


def merge_real_imag_population(population, doubleblind):
    """ Merges a population from real/imag into a complex valued one. """
    pulse_real, pulse_imag = population.pulse.real, population.pulse.imag
    pulse = pulse_real + 1j*pulse_imag
    if doubleblind==True:
        gate_real, gate_imag = population.gate.real, population.gate.imag
        gate = gate_real + 1j*gate_imag
    else:
        gate = None
    return MyNamespace(pulse=pulse, gate=gate)




def calc_z_error(individual, transform_arr, signal_t_new, measurement_info, descent_info, calc_signal_t):
    """ Calculates the Z-error for an individual. """
    individual = merge_real_imag_population(individual, measurement_info.doubleblind)
    signal_t = calc_signal_t(individual, transform_arr, measurement_info)
    error = jnp.sum(jnp.abs(signal_t_new-signal_t.signal_t)**2)
    return error





def calc_Z_grad_AD(population, transform_arr, signal_t_new, measurement_info, descent_info, calculate_signal_t):
    """
    Calculates the Z-error gradient with respect to pulse and gate using jax.grad. 
    To do this the complex valued population is split into real/imag. The resulting gradients are 
    merged again to form complex values

    Args:
        population (Pytree): the current population
        transform_arr (jnp.array): the applied transform to get signal_t_new
        signal_t_new (jnp.array): the current signal field projected onto the measured intensity
        measurement_info (Pytree): contains measurement data and parameters
        descent_info (Pytree): contains parameters for the retrieval
        calculate_signal_t (Callable): the method depended function to calculate the nonlinear signal
    
    Returns:
        tuple[jnp.array, jnp.array|None], the gradient with respect to the pulse and gate
    """
    population = split_population_real_imag(population, measurement_info.doubleblind)

    loss_func = Partial(calc_z_error, measurement_info=measurement_info, descent_info=descent_info, 
                        calc_signal_t=calculate_signal_t)
    calc_grad = jax.grad(loss_func, argnums=0)
    grad = jax.vmap(calc_grad, in_axes=(0,None,None))(population, transform_arr, signal_t_new)

    grad = merge_real_imag_population(grad, measurement_info.doubleblind)
    return grad.pulse, grad.gate