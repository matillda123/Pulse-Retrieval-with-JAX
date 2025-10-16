import jax.numpy as jnp
import jax
from jax.tree_util import Partial

from equinox import tree_at

from src.utilities import MyNamespace, generate_random_continuous_function, do_interpolation_1d




def get_initial_amp(measurement_info):
    """
    Estimate spectral amplitude from the integrated measured intensity. For SHG/THG the amplitude is interpolated to lie in the correct frequency region.
    """
    frequency, measured_trace, nonlinear_method = measurement_info.frequency, measurement_info.measured_trace, measurement_info.nonlinear_method
    mean_trace = jnp.mean(measured_trace, axis=0)
    amp = jnp.sqrt(jnp.abs(mean_trace))*jnp.sign(mean_trace)

    if nonlinear_method=="shg" or nonlinear_method=="thg":
        if nonlinear_method=="shg":
            factor=2
        elif nonlinear_method=="thg":
            factor=3
        else:
            raise ValueError(f"nonlinear_method needs to be one of shg or thg. Not {nonlinear_method}")

        amp = do_interpolation_1d(frequency, frequency/factor, amp)

    else:
        pass

    return amp/jnp.linalg.norm(amp)






def random(key, shape):
    key1, key2 = jax.random.split(key, 2)
    signal = jax.random.uniform(key1, shape, minval=-1, maxval=1) + 1j*jax.random.uniform(key2, shape, minval=-1, maxval=1)
    return signal


def random_phase(key, shape, amp):
    key1, key2 = jax.random.split(key, 2)
    phase = jax.random.uniform(key1, shape, minval=-jnp.pi, maxval=jnp.pi)
    amp = amp + jax.random.uniform(key2, shape, minval=-0.05, maxval=0.05)
    signal = amp*jnp.exp(1j*phase)
    return signal


def constant(key, shape):
    key1, key2 = jax.random.split(key, 2)
    vals = jax.random.uniform(key1, (shape[0], ), minval=-1, maxval=1) + 1j*jax.random.uniform(key2, (shape[0], ), minval=-1, maxval=1)
    signal = jnp.outer(vals, jnp.ones(shape[1]))
    return signal


def constant_phase(key, shape, amp):
    key1, key2 = jax.random.split(key, 2)
    phase = jax.random.uniform(key1, (shape[0], ), minval=-jnp.pi, maxval=jnp.pi)
    amp = amp + jax.random.uniform(key2, shape, minval=-0.05, maxval=0.05)
    signal = amp*jnp.exp(1j*phase[:,jnp.newaxis])
    return signal



def create_population_classic(key, population_size, guess_type, measurement_info):
    """
    Creates a stack of initial guesses with the shape (population_size, jnp.size(frequency)). The guesses are all in the frequency domain.
    The created populations can be optimized by both general and classical solvers.

    Args:
        key: jnp.array, jax.random.PRNGKey
        population_size: int, the number of guesses to be optimized
        guess_type: str, the guess mode. Has to be one of random, random_phase, constant or constant_phase. doublepulse is moved to initial_guess_doublepulse.py
        measurement_info: Pytree, holds the measurement information, is filled during initialization of each solver

    Returns:
        jnp.array, stack of 1D-arrays
    """

    amp = get_initial_amp(measurement_info)
    shape=(population_size, jnp.size(measurement_info.frequency))

    if guess_type=="random":
        signal_f_arr = random(key, shape)

    elif guess_type=="random_phase":
        signal_f_arr = random_phase(key, shape, amp)

    elif guess_type=="constant":
        signal_f_arr = constant(key, shape)

    elif guess_type=="constant_phase":
        signal_f_arr = constant_phase(key, shape, amp)

    elif guess_type=="doublepulse":
        raise RuntimeError("guess_type=doublepulse is in initial_guess_doublepulse.py. Its only implemented for " \
        "AC-Frog and without jax.")
    
    else:
        raise ValueError(f"guess_type needs to be one of random, random_phase, constant or constant_phase. Not {guess_type}")

    return signal_f_arr











def polynomial_guess(key, population, shape, measurement_info):
    key, subkey = jax.random.split(key, 2)
    c = jax.random.uniform(subkey, shape, minval=-1e3, maxval=1e3)
    population = tree_at(lambda x: x.phase, population, c, is_leaf=lambda x: x is None)
    return key, population


def sinusoidal_guess(key, population, shape, measurement_info):
    frequency = measurement_info.frequency
    key, subkey = jax.random.split(key, 2)
    key1, key2, key3 = jax.random.split(subkey, 3)

    bmin=0.5/(2*jnp.max(frequency))
    bmax=1/(2*jnp.max(frequency))

    a = jax.random.uniform(key1, shape, minval=0, maxval=1)
    b = jax.random.uniform(key2, shape, minval=bmin, maxval=bmax)
    c = jax.random.uniform(key3, shape, minval=0, maxval=2*jnp.pi)

    population = tree_at(lambda x: x.phase, population, MyNamespace(a=a, b=b, c=c), is_leaf=lambda x: x is None)
    return key, population
        

def sigmoidal_guess(key, population, shape, measurement_info):
    frequency = measurement_info.frequency
    key, subkey = jax.random.split(key, 2)
    key1, key2, key3 = jax.random.split(subkey, 3)

    a=jax.random.uniform(key1, shape, minval=0, maxval=4*jnp.pi)
    c=jax.random.uniform(key2, shape, minval=jnp.min(frequency), maxval=jnp.max(frequency))
    k=jax.random.uniform(key3, shape, minval=-2, maxval=2)

    population = tree_at(lambda x: x.phase, population, MyNamespace(a=a, c=c, k=k), is_leaf=lambda x: x is None)
    return key, population


def bspline_guess_phase(key, population, shape, measurement_info):
    key, subkey = jax.random.split(key, 2)

    c = jax.random.uniform(subkey, shape, minval=-2*jnp.pi, maxval=2*jnp.pi)
    population = tree_at(lambda x: x.phase, population, MyNamespace(c=c), is_leaf=lambda x: x is None)
    return key, population
    

def discrete_guess_phase(key, population, shape, measurement_info):
    frequency = measurement_info.frequency
    key, subkey = jax.random.split(key, 2)
    keys = jax.random.split(subkey, shape[0])

    phase = jax.vmap(generate_random_continuous_function, in_axes=(0, None, None, None, None, None))(keys, shape[1], frequency, 
                                                                                                        -4*jnp.pi, 4*jnp.pi, jnp.ones(jnp.size(frequency)))
    population = tree_at(lambda x: x.phase, population, phase, is_leaf=lambda x: x is None)
    return key, population



def gaussian_or_lorentzian_guess(key, population, shape, measurement_info):
    frequency = measurement_info.frequency
    key, subkey = jax.random.split(key, 2)
    key1, key2, key3 = jax.random.split(subkey, 3)
    a = jax.random.uniform(key1, shape, minval=0, maxval=1)
    b = jax.random.uniform(key2, shape, minval=1e-3, maxval=(jnp.max(frequency)-jnp.min(frequency))/3)
    c = jax.random.uniform(key3, shape, minval=jnp.min(frequency), maxval=jnp.max(frequency))
    
    population = tree_at(lambda x: x.amp, population, MyNamespace(a=a, b=b, c=c), is_leaf=lambda x: x is None)
    return key, population



def discrete_guess_amp(key, population, shape, measurement_info):
    key, subkey = jax.random.split(key, 2)

    amp = get_initial_amp(measurement_info)

    noise = jax.random.uniform(subkey, (shape[0], jnp.size(measurement_info.frequency)), minval=-0.05, maxval=0.05)
    amp = amp + noise
    amp = amp/jnp.linalg.norm(amp)

    population = tree_at(lambda x: x.amp, population, amp, is_leaf=lambda x: x is None)
    return key, population



def bspline_guess_amp(key, population, shape, measurement_info):
    key, subkey = jax.random.split(key, 2)

    c = jax.random.uniform(subkey, shape, minval=0.0, maxval=2.0)
    population = tree_at(lambda x: x.amp, population, MyNamespace(c=c), is_leaf=lambda x: x is None)
    return key, population





def general_phase(key, population, shape, measurement_info, phase_type):
    key, subkey = jax.random.split(key, 2)
    signal_f = create_population_classic(subkey, shape[0], phase_type, measurement_info)
    population = tree_at(lambda x: x.phase, population, jnp.angle(signal_f), is_leaf=lambda x: x is None)
    return key, population


def general_amp(key, population, shape, measurement_info, amp_type):
    key, subkey = jax.random.split(key, 2)
    signal_f = create_population_classic(subkey, shape[0], amp_type, measurement_info)
    population = tree_at(lambda x: x.amp, population, jnp.abs(signal_f), is_leaf=lambda x: x is None)
    return key, population




def create_phase(key, phase_type, population, shape, measurement_info):
    phase_guess_func_dict = {"polynomial": polynomial_guess,
                                "sinusoidal": sinusoidal_guess,
                                "sigmoidal": sigmoidal_guess,
                                "bsplines": bspline_guess_phase,
                                "discrete": discrete_guess_phase,
                                "random": Partial(general_phase, phase_type="random"),
                                "random_phase": Partial(general_phase, phase_type="random_phase"),
                                "constant": Partial(general_phase, phase_type="constant"),
                                "constant_phase": Partial(general_phase, phase_type="constant_phase")}
    
    key, population = phase_guess_func_dict[phase_type](key, population, shape, measurement_info)
    return key, population
    

def create_amp(key, amp_type, population, shape, measurement_info):
    amp_guess_func_dict = {"gaussian": gaussian_or_lorentzian_guess,
                            "lorentzian": gaussian_or_lorentzian_guess,
                            "bsplines": bspline_guess_amp,
                            "discrete": discrete_guess_amp,
                            "random": Partial(general_amp, amp_type="random"),
                            "random_phase": Partial(general_amp, amp_type="random_phase"),
                            "constant": Partial(general_amp, amp_type="constant"),
                            "constant_phase": Partial(general_amp, amp_type="constant_phase")}
    
    key, population = amp_guess_func_dict[amp_type](key, population, shape, measurement_info)
    return key, population



def create_population_general(key, amp_type, phase_type, population, population_size, no_funcs_amp, no_funcs_phase, spectrum_provided, measurement_info):
    """
    Creates an initial guess population for general solvers. Since general solvers do not require a grid, different 
    representations for amplitude and phase can be used. The population is represented in the freqeuncy domain.

    Args:
        key: jnp.array, jax.random.PRNGKey
        amp_type: str, representation of the amplitude
        phase_type: str, representation of the phase
        population: Pytree, a MyNamespace object containing amp and phase
        population_size: int, the number of individuals to optimize
        no_funcs_amp: int, some representations can consist of multiple basis functions
        no_funcs_phase: int, some representations can consist of multiple basis functions
        spectrum_provided: bool, if a spectrum is provided then the guessed population will not include an amplitude
        measurement_info: Pytree, holds the measurement information, is filled during initialization of each solver
       
    Returns:
        tuple[jnp.array, Pytree]
    """

    shape = (population_size, no_funcs_phase)
    key, population = create_phase(key, phase_type, population, shape, measurement_info)

    if spectrum_provided==False:
        shape = (population_size, no_funcs_amp)
        key, population = create_amp(key, amp_type, population, shape, measurement_info)
        
    return key, population














