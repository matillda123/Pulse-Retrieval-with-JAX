import jax.numpy as jnp
import jax

from utilities import MyNamespace, generate_random_continuous_function, do_interpolation_1d




def get_initial_amp_for_shg_thg(frequency, measured_trace, nonlinear_method):
    mean_trace=jnp.mean(measured_trace, axis=0)
    amp=jnp.sqrt(jnp.abs(mean_trace))*jnp.sign(mean_trace)

    if nonlinear_method=="shg":
        factor=2
    elif nonlinear_method=="thg":
        factor=3
    else:
        print("something went wrong")

    amp=do_interpolation_1d(frequency, frequency/factor, amp)
    return amp






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
    signal = jnp.outer(jnp.ones(shape), vals)
    return signal


def constant_phase(key, shape, amp):
    key1, key2 = jax.random.split(key, 2)
    phase = jax.random.uniform(key1, (shape[0], ), minval=-jnp.pi, maxval=jnp.pi)
    amp = amp + jax.random.uniform(key2, shape, minval=-0.05, maxval=0.05)
    signal = amp*jnp.exp(1j*phase[:,jnp.newaxis])
    return signal



def create_population_classic(key, population_size, guess_type, measurement_info):
    if measurement_info.nonlinear_method=="shg" or measurement_info.nonlinear_method=="thg":
        amp=get_initial_amp_for_shg_thg(measurement_info.frequency, measurement_info.measured_trace, measurement_info.nonlinear_method)
    else:
        mean_trace=jnp.mean(measurement_info.measured_trace, axis=0)
        amp=jnp.sqrt(jnp.abs(mean_trace))*jnp.sign(mean_trace)
    amp=amp/jnp.linalg.norm(amp)

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
        print("Not implemented. Is availabel as extranonlinear_method in RetrievePulsesFROG. Because it only works for frog and is not jax compatible.")
    
    else:
        print("not available")

    return signal_f_arr











def polynomial_guess(key, population, shape, measurement_info):
    key, subkey = jax.random.split(key, 2)
    c = jax.random.uniform(subkey, shape, minval=-1e3, maxval=1e3)
    population.phase=c
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

    population.phase=MyNamespace(a=a, b=b, c=c)
    return key, population
        

def sigmoidal_guess(key, population, shape, measurement_info):
    frequency = measurement_info.frequency
    key, subkey = jax.random.split(key, 2)
    key1, key2, key3 = jax.random.split(subkey, 3)

    a=jax.random.uniform(key1, shape, minval=0, maxval=4*jnp.pi)
    c=jax.random.uniform(key2, shape, minval=jnp.min(frequency), maxval=jnp.max(frequency))
    k=jax.random.uniform(key3, shape, minval=-2, maxval=2)

    population.phase=MyNamespace(a=a, c=c, k=k)
    return key, population


def spline_guess(key, population, shape, measurement_info):
    key, subkey = jax.random.split(key, 2)

    N=jnp.size(measurement_info.frequency)
    n=shape[1]
    nn = jnp.divide(N, jnp.linspace(1, jnp.ceil(N/n), int(jnp.ceil(N/n))))
    n = int(nn[jnp.round(nn%1, 5)==0][-1])

    c = jax.random.uniform(subkey, (shape[0], n), minval=-2*jnp.pi, maxval=2*jnp.pi)
    population.phase=MyNamespace(c=c)
    return key, population
    

def discrete_guess_phase(key, population, shape, measurement_info):
    frequency = measurement_info.frequency
    key, subkey = jax.random.split(key, 2)
    keys = jax.random.split(subkey, shape[0])

    phase = jax.vmap(generate_random_continuous_function, in_axes=(0, None, None, None, None, None))(keys, shape[1], frequency, 
                                                                                                        -4*jnp.pi, 4*jnp.pi, jnp.ones(jnp.size(frequency)))
    population.phase = phase
    return key, population



def gaussian_or_lorentzian_guess(key, population, shape, measurement_info):
    frequency = measurement_info.frequency
    key, subkey = jax.random.split(key, 2)
    key1, key2, key3 = jax.random.split(subkey, 3)
    a = jax.random.uniform(key1, shape, minval=0, maxval=1)
    b = jax.random.uniform(key2, shape, minval=1e-3, maxval=(jnp.max(frequency)-jnp.min(frequency))/3)
    c = jax.random.uniform(key3, shape, minval=jnp.min(frequency), maxval=jnp.max(frequency))
    
    population.amp = MyNamespace(a=a, b=b, c=c)
    return key, population



def discrete_guess_amp(key, population, shape, measurement_info):
    key, subkey = jax.random.split(key, 2)
    
    if measurement_info.nonlinear_method=="shg" or measurement_info.nonlinear_method=="thg":
        amp=get_initial_amp_for_shg_thg(measurement_info.frequency, measurement_info.measured_trace, measurement_info.nonlinear_method)
    else:
        mean_trace=jnp.mean(measurement_info.measured_trace, axis=0)
        amp=jnp.sqrt(jnp.abs(mean_trace))*jnp.sign(mean_trace)

    noise = jax.random.uniform(subkey, (shape[0], jnp.size(measurement_info.frequency)), minval=-0.05, maxval=0.05)
    amp = amp + noise
    amp=amp/jnp.linalg.norm(amp)

    population.amp = amp
    return key, population



def create_phase(key, phase_type, population, shape, measurement_info):
    phase_guess_func_dict = {"polynomial": polynomial_guess,
                                "sinusoidal": sinusoidal_guess,
                                "sigmoidal": sigmoidal_guess,
                                "splines": spline_guess,
                                "discrete": discrete_guess_phase}
    
    key, population = phase_guess_func_dict[phase_type](key, population, shape, measurement_info)
    return key, population
    

def create_amp(key, amp_type, population, shape, measurement_info):
    amp_guess_func_dict = {"gaussian": gaussian_or_lorentzian_guess,
                            "lorentzian": gaussian_or_lorentzian_guess,
                            "splines": spline_guess,
                            "discrete": discrete_guess_amp}
    
    key, population = amp_guess_func_dict[amp_type](key, population, shape, measurement_info)
    return key, population



def create_population_general(key, amp_type, phase_type, population, population_size, no_funcs_amp, no_funcs_phase, spectrum_provided, measurement_info):
    shape = (population_size, no_funcs_phase)
    key, population = create_phase(key, phase_type, population, shape, measurement_info)

    if spectrum_provided==False:
        shape = (population_size, no_funcs_amp)
        key, population = create_amp(key, amp_type, population, shape, measurement_info)
        
    return key, population














