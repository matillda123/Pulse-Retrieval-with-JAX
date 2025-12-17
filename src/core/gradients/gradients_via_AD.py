import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from src.utilities import MyNamespace



def split_population_real_imag(population, pulse_or_gate):
    """ Splits population into real/imag. """
    if pulse_or_gate=="pulse":
        pulse_real, pulse_imag = jnp.real(population.pulse), jnp.imag(population.pulse)
        pulse = MyNamespace(real=pulse_real, imag=pulse_imag)
        gate = population.gate

    elif pulse_or_gate=="gate":
        gate_real, gate_imag = jnp.real(population.gate), jnp.imag(population.gate)
        gate = MyNamespace(real=gate_real, imag=gate_imag)
        pulse = population.pulse

    else:
        raise ValueError()
    return MyNamespace(pulse=pulse, gate=gate)



def merge_real_imag_population(population, pulse_or_gate):
    """ Merges a population from real/imag into a complex valued one. """
    if pulse_or_gate=="pulse":
        pulse_real, pulse_imag = population.pulse.real, population.pulse.imag
        pulse = pulse_real + 1j*pulse_imag
        gate = population.gate

    elif pulse_or_gate=="gate":
        gate_real, gate_imag = population.gate.real, population.gate.imag
        gate = gate_real + 1j*gate_imag
        pulse = population.pulse

    else:
        raise ValueError()
    return MyNamespace(pulse=pulse, gate=gate)

    


def calc_grad_AD(individual, transform_arr, signal_t_new, measured_trace, measurement_info, calculate_signal_t, pulse_or_gate, loss_func):
    """
    Calculates the gradient of a loss function with respect to pulse and gate using jax.grad. 
    To do this the complex valued population is split into real/imag. The resulting gradient are merged into a complex one.

    Args:
        individual (Pytree): an individual of the current population
        transform_arr (jnp.array): the applied transform to get signal_t_new
        signal_t_new (jnp.array): the current signal field projected onto the measured intensity
        measured_trace (jnp.array): the measured intensity
        measurement_info (Pytree): contains measurement data and parameters
        descent_info (Pytree): contains parameters for the retrieval
        calculate_signal_t (Callable): the method depended function to calculate the nonlinear signal
        pulse_or_gate (str): the variable of with which to differentiate with respect to
        loss_func (Callable): the loss function
    
    Returns:
        jnp.array, the gradient with respect to the pulse and gate
    """
    individual = split_population_real_imag(individual, pulse_or_gate)

    loss_func = Partial(loss_func, transform_arr=transform_arr, signal_t_new=signal_t_new, measured_trace=measured_trace, 
                        measurement_info=measurement_info, calc_signal_t=calculate_signal_t, pulse_or_gate=pulse_or_gate)
    argnum_dict = {"pulse": 0,"gate": 1}
    grad = jax.grad(loss_func, argnums=argnum_dict[pulse_or_gate])(individual.pulse, individual.gate)
    return grad.real + 1j*grad.imag





def calc_z_error(pulse, gate, transform_arr, signal_t_new, measured_trace, measurement_info, calc_signal_t, pulse_or_gate):
    """ Calculates the Z-error for an individual. """
    individual = merge_real_imag_population(MyNamespace(pulse=pulse, gate=gate), pulse_or_gate)
    signal_t = calc_signal_t(individual, transform_arr, measurement_info)
    error = jnp.sum(jnp.abs(signal_t_new - signal_t.signal_t)**2)
    return error



def calc_pie_error(pulse, gate, transform_arr, signal_t_new, measured_trace, measurement_info, calc_signal_t, pulse_or_gate):
    individual = merge_real_imag_population(MyNamespace(pulse=pulse, gate=gate), pulse_or_gate)
    signal_t = calc_signal_t(individual, transform_arr, measurement_info)
    error = jnp.sum(jnp.abs(jnp.sqrt(measured_trace) - jnp.abs(signal_t.signal_f))**2)
    return error



def calc_grad_AD_z_error(individual, transform_arr, signal_t_new, measured_trace, measurement_info, calculate_signal_t, pulse_or_gate):
    grad = calc_grad_AD(individual, transform_arr, signal_t_new, measured_trace, measurement_info, calculate_signal_t, pulse_or_gate, calc_z_error)
    return jnp.expand_dims(grad, 0)  # is needed since returned gradient does not explicitely depend on m


def calc_grad_AD_pie_error(individual, transform_arr, signal_t_new, measured_trace, measurement_info, calculate_signal_t, pulse_or_gate):
    grad = calc_grad_AD(individual, transform_arr, signal_t_new, measured_trace, measurement_info, calculate_signal_t, pulse_or_gate, calc_pie_error)
    return jnp.expand_dims(grad, 0)