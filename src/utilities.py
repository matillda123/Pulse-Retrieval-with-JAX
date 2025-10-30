import jax.numpy as jnp
import jax
from jax.tree_util import Partial

import lineax as lx
import equinox

from interpax import interp1d





def flatten_MyNamespace(MyNamespace):
    class_dict = MyNamespace.__dict__
    data_keys = list(class_dict.keys())
    data_values = list(class_dict.values())
    return data_values, data_keys


def unflatten_MyNamespace(aux_data, leaves):
    return MyNamespace(**dict(zip(aux_data, leaves)))


class MyNamespace:
    """
    The central Pytree.
    Does not have a fixed shaped/structure at initialization. Because it would be tedious and cumbersome to keep track of this for all different pytree types used.
    However this is dangerous with jax. 
    To solver this MyNamespace.expand() can be used to add attributes by returning a new MyNamespace object. On top inside jax-transformations one should make sure
    to keep the structure static. (If not jax may hopefully catch it.)
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


    def expand(self, **kwargs):
        new_dict = {**self.__dict__, **kwargs}
        return MyNamespace(**new_dict)
    
    
    def __repr__(self):
        mydict = self.__dict__
        keys = mydict.keys()

        myoutput=[]
        for key in keys:
            value = mydict[key]

            if type(value)==MyNamespace:
                myoutput.append([key, value.__repr__()])
            else:
                try:
                    myoutput.append([key, jnp.shape(value), value.dtype])
                except Exception:
                    myoutput.append([key, jnp.shape(value), type(value).__name__])
                except Exception:
                    myoutput.append([key, value, type(value).__name__])
                    
        return f"{myoutput}".replace("\'","").replace("\"", "")
    
        

    def __add__(self, other):
        if isinstance(other, MyNamespace):
            tree_new = jax.tree.map(lambda x,y: x+y, self, other)
        else:
            y = other
            tree_new = jax.tree.map(lambda x: x+y, self)
        return tree_new
    

    def __radd__(self, other):
        return self.__add__(other)
    

    def __mul__(self, other):
        if isinstance(other, MyNamespace):
            tree_new = jax.tree.map(lambda x,y: x*y, self, other)
        else:
            y = other
            tree_new = jax.tree.map(lambda x: x*y, self)
        return tree_new
    

    def __rmul__(self, other):
        return self.__mul__(other)
    

    def __sub__(self, other):
        return self.__add__((-1)*other)
    

jax.tree_util.register_pytree_node(MyNamespace, flatten_MyNamespace, unflatten_MyNamespace)
    





def run_scan(do_scan, carry, no_iterations, use_jit):
    """
    Run a solver iteratively using lax.scan with or without jax.jit.

    Args:
        do_scan: Callable, the callable needs to take carry its argument
        carry: Pytree, the initial state of the iteration
        no_iterations: int, the number of iterations
        use_jit: bool, whether jax.jit is supposed to be used or not

    Returns:
        tuple[Carry, Y], the output of jax.lax.scan

    """
    def scan(carry):
        return jax.lax.scan(do_scan, carry, length=no_iterations)


    if use_jit==True:
        scan = jax.jit(scan)
    else:
        pass
    
    return scan(carry)




def scan_helper(carry, xs, actual_function, number_of_args, number_of_xs):
    """
    jax.lax.scan expects the provided callable to accept two arguments carry and possibly xs. 
    This wraps around the function to be iterated by lax.scan, such that its inputs do not have to conform to lax.scan's requirements. 
    The provided carry and xs are unpacked and provided to actual_function.
    All arguments except carry and xs have to be fixed via partial. The resulting callable is then provided to lax.scan. 
    The output of actual_function needs to be of the same structure as carry.

    Args:
        carry: any or tuple, the initial state of the iteration
        xs: any or tuple, the xs used by jax.lax.scan
        actual_function: Callable, the function that is to be iterated over
        number_of_args: int, the number of individual arguments in carry
        number_of_xs: int, the number of individual arguments in xs
    
    Return:
        Any, the output of actual_function

    """
    
    if number_of_args==1:
        input_args = (carry, )
    else:
        input_args = carry

    if number_of_xs==0:
        return actual_function(*input_args)
    
    elif number_of_xs==1:
        input_args = input_args + (xs,)
    else:
        input_args = input_args + xs
        
    return actual_function(*input_args)




def while_loop_helper(carry, actual_function, number_of_args):
    """
    Similar to scan_helper. Unpacks carry, such that the input of actual_function does not have to conform to lax.while_loop's requirements. 

    Args:
        carry: any or tuple, the initial state of the iteration
        actual_function: Callable, the function to be iterated over
        number_of_args: int, the number of individual arguments in carry

    Returns:
        Any, the output of actual_function

    """
    if number_of_args==1:
        input_args = (carry, )
    else:
        input_args = carry
    return actual_function(*input_args)








def optimistix_helper_loss_function(input, args, function, no_of_args):
    """
    Optimistix's interactive solver API expects loss-functions which take two variables and returns a tuple with the error 
    and auxilary information. This wraps around function to adhere to this. 
    function and no_of_args have to be fixed via partial.

    Args:
        input: any, the input the function
        args: any, the args of function
        function: Callable, the actual loss function 
        no_of_args: int, the number of extra arguments

    Returns:
        tuple, a tuple which contains the calculated error twice, since there is no auxilary information

    """

    if no_of_args==0:
        error = function(input)
    elif no_of_args==1:
        error = function(input, args)
    else:
        raise NotImplementedError(f"didnt take care of this case, no_of_args={no_of_args}")

    return error, error



def scan_helper_equinox(carry, xs, step, static):
    """
    This function wraps around step, which is to be iterated over via lax.scan. In some cases the carry contains static not jax compatible parts.
    (e.g. some of the optimistix solvers contain jaxpr). These need to be filtered out to be jax compatible which can be done through equinox. 
    The function takes carry merges the static part and removes the static part once the iteration is done.
    step and static have to be fixed via partial.

    Args:
        carry: any or tuple, the carry to be iterated over
        xs: any or tuple, unused but required by lax.scan
        step: Callable, the function to be iterated over 
        static: any, a static non-jax-compatible object which is to be merged before calling step and removed afterwards

    Returns:
        tuple, the output of step

    """

    state = equinox.combine(carry, static)
    state, error = step(state)
    carry, _ = equinox.partition(state, equinox.is_array)
    return  carry, error



    










def do_fft(signal, sk, rn, axis=-1):
    """
    Do a complex-valued 1D-FFT. Does not use fftshift. Instead sk and rn obtained from get_sk_rn are 
    applied which have the same effect and make the fft work any frequency range.

    Args:
        signal: jnp.array, the signal on which the fft is applied
        sk: jnp.array, corrective values which "shift" the signal to the correct frequencies
        rn: jnp.array, corrective values which "shift" the signal to the correct frequencies
        axis: int, the axis over which the fft is applied (Default is -1)

    Returns:
        jnp.array, the fourier transformed signal

    """
    # if axis=0 -> sk, rn need to be use with jnp.newaxis to map over axis=0
    # default is always axis=-1
    sk = jnp.conjugate(sk)
    rn = jnp.conjugate(rn)
    return jnp.fft.fft(signal*sk, axis=axis)*rn


def do_ifft(signal, sk, rn, axis=-1):
    """
    Do a complex-valued 1D-IFFT. Does not use fftshift. Instead sk and rn obtained from get_sk_rn are 
    applied which have the same effect and make the fft work any frequency range.

    Args:
        signal: jnp.array, the signal on which the ifft is applied
        sk: jnp.array, corrective values which "shift" the signal to the correct positions
        rn: jnp.array, corrective values which "shift" the signal to the correct positions
        axis: int, the axis over which the ifft is applied (Default is -1)

    Returns:
        jnp.array, the inverse fourier transformed signal

    """
    return jnp.fft.ifft(signal*rn, axis=axis)*sk
 
 

def get_sk_rn(time, frequency):
    """
    The definition of the FFT differs from the discrete fourier transform. In order to correct for this the input and result 
    of fft/ifft can be multiplied by the values calculated here. This essentially results in the fourier shift theorem. 
    time and frequency have to fullfill N=1/(df*dt).

    Args:
        time: jnp.array, the time axis
        frequency: jnp.array, the frequency axis

    Returns:
        tuple[jnp.array, jnp.array], the corrections used by do_fft/do_ifft

    """
    n = jnp.arange(jnp.size(frequency))
    df = jnp.mean(jnp.diff(frequency))
    rn = jnp.exp(1j*time[0]*2*jnp.pi*n*df)
    sk = jnp.exp(1j*2*jnp.pi*frequency[0]*time)
    return sk, rn





def do_interpolation_1d(x_new, x, y, method="cubic"):
    """
    Wraps around interpax.interp1d
    """
    y_new = interp1d(x_new, x, y, method=method, extrap=1e-12)
    return y_new




def calculate_gate(pulse_t, method):
    """
    Calculate the gate field/signal for the nonlinear process.
    """
    assert method!="tg", "For TG, depending on the definition either pg or sd needs to be used."

    if method == "shg":
        gate = pulse_t

    elif method == "thg":
        gate = pulse_t**2

    elif method =="pg":
        gate = jnp.abs(pulse_t)**2
        
    elif method == "sd":
        gate = jnp.conjugate(pulse_t)**2

    else:
        print("not_implemented")
    return gate



def calculate_gate_with_Real_Fields(pulse_t, method):
    """
    Calculate the gate field/signal for the nonlinear process using real input fields. 
    This allows for the description of difference frequency generation.
    """
    assert method!="tg", "For TG, depending on the definition either pg or sd needs to be used."

    if method=="shg":
        gate = jnp.real(pulse_t)

    elif method=="thg":
        gate = jnp.real(pulse_t)**2

    elif method=="pg":
        gate=jnp.abs(pulse_t)**2

    elif method=="sd":
        print("not sure if this can work")
    else:
        print("not_implemented")
    return gate




def project_onto_intensity(signal_f, measured_intensity):
    """ Project the current complex guess signal onto the measured intensity. """
    return jnp.sqrt(jnp.abs(measured_intensity))*jnp.sign(measured_intensity)*jnp.exp(1j*jnp.angle(signal_f))


def project_onto_amplitude(signal_f, measured_amplitude):
    """ Project the current complex guess signal onto the measured amplitude. """
    return measured_amplitude*jnp.exp(1j*jnp.angle(signal_f))



# def calculate_S_prime(signal_t, measured_trace, mu, measurement_info):
#     sk, rn = measurement_info.sk, measurement_info.rn
#     signal_f=do_fft(signal_t, sk, rn)

#     signal_f_new=project_onto_intensity(signal_f, measured_trace)

#     signal_t_new=do_ifft(signal_f_new, sk, rn)*1/(jnp.sqrt(mu)+1e-12)
#     return signal_t_new








def calculate_trace(signal_f):
    """ Calculates intensity from a complex signal. """
    trace = jnp.abs(signal_f)**2
    return trace


def calculate_mu(trace, measured_trace):
    """ Calculates scaling factor between measured intensity and intensity of current guess. """
    return jnp.sum(trace*measured_trace)/(jnp.sum(trace**2) + 1e-12)


def calculate_trace_error(trace, measured_trace):
    """ 
    Calculates the mean of the squared L2-Norm between the measured intensity and intensity of the current guess.
    With the current guess being scaled by mu.
    """
    mu = calculate_mu(trace, measured_trace)
    return jnp.mean(jnp.abs(measured_trace - mu*trace)**2)



def calculate_Z_error(signal_t, signal_t_new):
    """ Calculates the squared L2-Norm between the complex signal fields in the time domain before and after projection onto the measured signal. """
    deltaS = signal_t_new-signal_t
    return jnp.sum(jnp.abs(deltaS)**2)






def generate_random_continuous_function(key, no_points, x, minval, maxval, distribution):
    """
    Generates a 1D-array with random but continuous values. Uses on cubic inter/extrapolation of random values.

    Args:
        key: jnp.array, a jax.random.PRNGKey
        no_points: int, the number of random points to use for the interpolation
        x: jnp.array, the x-values from which to choose the location of random values
        minval: int or float, the minimal random y-value, the interpolation may lead to lower values
        maxval: int or float, the maximal random y-value, the interpolation may lead to higher values
        distribution: jnp.array, a probability distribution for the x-location of the random values.

    Returns:
        jnp.array, the interpolated random y-values. 

    """

    key1, key2 = jax.random.split(key, 2)

    p_arr = distribution/jnp.sum(distribution)
    x_rand = jnp.sort(jax.random.choice(key1, x, (no_points, ), replace=False, p=p_arr))
    y_rand = jax.random.uniform(key2, (no_points, ), minval=minval, maxval=maxval)

    y = do_interpolation_1d(x, x_rand, y_rand)
    return y






import lineax

def _solve_system_using_lineax_iteratively(A, b, x_prev, solver):
    """ Wraps around lineax.linear_solve. Supplies lineax with a preconditioner and an approximate solution in case the solver may use those. """
    
    if isinstance(solver, lineax.CG): 
        # appyling this tag might be a lie. But lineax will throw an error otherwise
        operator = lx.MatrixLinearOperator(A, lineax.positive_semidefinite_tag)
    else:
        operator = lx.MatrixLinearOperator(A)

    A_precond = jnp.diag(jnp.diag(A)).astype(jnp.complex64)
    preconditioner = lx.MatrixLinearOperator(A_precond, lx.positive_semidefinite_tag)
    solution = lx.linear_solve(operator, b, 
                                solver=solver, 
                                throw=False, 
                                options={"y0": x_prev, "preconditioner": preconditioner})
    
    return solution


def solve_linear_system(A, b, x_prev, solver):
    """
    Solve a stack of linear equation Ax=b using scipy or lineax.

    Args:
        A: jnp.array, stack of 2D-arrays
        b: jnp.array, stack of 1D-arrays
        x_prev: jnp.array, stack of 1D-arrays with approximate solutions.
        solver: str or lineax-solver, which library/method to use

    Returns:
        jnp.array, stack of 1D-arrays with the solution to Ax=b

    """

    if solver=="scipy":
        newton_direction = jax.scipy.linalg.solve(A, b[..., None], assume_a="her").squeeze(-1)
        return newton_direction
    
    elif solver=="lineax":
        solution = jax.vmap(lambda M,x: lx.linear_solve(lx.MatrixLinearOperator(M), x, throw=False))(A, b)

    else:
        solution = jax.vmap(_solve_system_using_lineax_iteratively, in_axes=(0,0,0,None))(A, b, x_prev, solver)
        
    return solution.value





def calculate_newton_direction(grad_m, hessian_m, lambda_lm, newton_direction_prev, solver, full_or_diagonal):
    """
    Calculates the newton-direction give a gradient and a hessian. 

    Args:
        grad_m: jnp.array,
        hessian_m: jnp.array,
        lambda_lm: float,
        newton_direction_prev: jnp.array,
        solver: str or lineax-solver,
        full_or_diagonal: str.

    Returns:
        tuple[jnp.array, Pytree]
        
    """
    hessian = jnp.sum(hessian_m, axis=1)
    grad = jnp.sum(grad_m, axis=1)

    if full_or_diagonal=="full":
        idx = jax.vmap(jnp.diag_indices_from)(hessian)
        hessian = jax.vmap(lambda x,y: x.at[y].add(lambda_lm*jnp.abs(x[y])))(hessian, idx)

        newton_direction = solve_linear_system(hessian, grad, newton_direction_prev, solver)

    elif full_or_diagonal=="diagonal":
        hessian = hessian + lambda_lm*jnp.max(jnp.abs(hessian), axis=1)[:, jnp.newaxis]
        newton_direction = grad/hessian

    else:
        raise ValueError(f"full_or_diagonal needs to be full or diagonal. Not {full_or_diagonal}")

    hessian_state = MyNamespace(newton_direction_prev = newton_direction)
    return -1*newton_direction, hessian_state



















    

def get_idx_arr(N, M, key):
    """
    Create a stack of size M with randomized arrays with indices with range 0, N.

    Args:
        N: int, the maximum index
        M: int, the number of randomizations
        key: jnp.array, a jax.random.PRNGKey

    Returns:
        jnp.array, a stack of 1D-arrays with randomized indices
    
    """
    idx = jnp.arange(N, dtype=jnp.int32)
    idx_arr = jnp.zeros((M,N), dtype=jnp.int32)
    for i in range(M):
        idx_arr = idx_arr.at[i].set(jax.random.permutation(key, idx))
        key = jax.random.split(key, 1)[0]
    return jnp.array(idx_arr, dtype=jnp.int32)


def get_com(signal, idx_arr):
    """
    Calculate the center of mass of a signal.
    """
    com = jnp.sum(signal*idx_arr)/jnp.sum(signal)
    return com


def center_signal(signal):
    """
    Center a signal to the middle of an array via its center of mass. 
    Is done in two stages since periodic boundaries distort the actual center of mass.

    Args:
        signal: jnp.array, the signal to be centered.

    Returns:
        jnp.array, the signal with its center of mass located at index N/2
    
    """

    # center using argmax
    N = jnp.shape(signal)[0]
    idx_arr = jnp.arange(N)
    max_idx = jnp.argmax(jnp.abs(signal))
    signal = jnp.roll(signal, -(max_idx-N//2))

    # center using com
    com = get_com(jnp.abs(signal), idx_arr)
    signal = jnp.roll(signal, -(com-N//2))
    return signal


def center_signal_to_max(signal):
    """ Center a signal to the middle of an array via jnp.argmax. """
    N = jnp.shape(signal)[0]
    max_idx = jnp.argmax(jnp.abs(signal))
    signal = jnp.roll(signal, -(max_idx-N//2))
    return signal





























def loss_function_modifications(trace, measured_trace, tau_or_zarr, frequency, amplitude_or_intensity, use_fd_grad):
    """
    General optimization algorithms are not limited to a specific loss function. This function modifies the given trace and measured_trace
    such that residuals based on the amplitude instead of the intensity are optimized. Alternatively finite difference derivatives of the trace 
    can be optimized instead.
    
    """
    measured_trace = measured_trace/jnp.max(jnp.abs(measured_trace))
    trace = trace/jnp.max(jnp.abs(trace))

    if amplitude_or_intensity=="amplitude":
        # add small value since auto-diff of sqrt(0) is problematic
        measured_trace = jnp.sqrt(jnp.abs(measured_trace) + 1e-9)*jnp.sign(measured_trace)
        trace = jnp.sqrt(jnp.abs(trace) + 1e-9)*jnp.sign(trace)
    elif amplitude_or_intensity=="intensity":
        pass
    elif type(amplitude_or_intensity)==int or type(amplitude_or_intensity)==float:
        exp_val = amplitude_or_intensity
        measured_trace = (jnp.abs(measured_trace) + 1e-9)**exp_val*jnp.sign(measured_trace)
        trace = (jnp.abs(trace) + 1e-9)**exp_val*jnp.sign(trace)
    else:
        raise ValueError(f"amplitude_or_intensity needs to be amplitude, intensity or an int/float. Not {amplitude_or_intensity}")


    if use_fd_grad!=False:
        raise ValueError("gradient id missing dx, dy")
        measured_trace_0, _ = jax.lax.scan(lambda x,y: (Partial(jnp.gradient, axis=0)(x), None), measured_trace, length=use_fd_grad)
        measured_trace_1, _ = jax.lax.scan(lambda x,y: (Partial(jnp.gradient, axis=1)(x), None), measured_trace, length=use_fd_grad)

        trace_0, _ = jax.lax.scan(lambda x,y: (Partial(jnp.gradient, axis=0)(x), None), trace, length=use_fd_grad)
        trace_1, _ = jax.lax.scan(lambda x,y: (Partial(jnp.gradient, axis=1)(x), None), trace, length=use_fd_grad)

        measured_trace = measured_trace_0 + measured_trace_1
        trace = trace_0 + trace_1

    return trace, measured_trace











def get_score_values(final_result, input_pulses):
    """
    Computes different error-metrics for a reconstructed pulse given the exact pulse is known and provided. 
    The error metrics are the maximum cross-correlation between reconstructed and exact pulse in the time and frequency domain. 
    The cross-correlation in the frequency domain without any shifts. This evaluates the efficacy of the retrived central freqeuncy. 
    A weighted and normalized L2-Norm of the GDD difference of reconstructed and exact pulse.

    """

    pulse_t, pulse_f = final_result.pulse_t, final_result.pulse_f
    if hasattr(final_result, "time"):
        time=final_result.time
    elif hasattr(final_result, "tau_arr"):
        time=final_result.tau_arr
    else:
       raise ValueError("final_result need to contain either time or tau_arr. It contains neither.")

    frequency = final_result.frequency

    time_inp, frequency_inp = input_pulses.time, input_pulses.frequency
    pulse_t_inp, pulse_f_inp = input_pulses.pulse_t, input_pulses.pulse_f


    pulse_t_inp_interp = do_interpolation_1d(time, time_inp, pulse_t_inp)
    pulse_f_inp_interp = do_interpolation_1d(frequency, frequency_inp, pulse_f_inp)
    
    pulse_t = pulse_t/jnp.linalg.norm(pulse_t)
    pulse_f = pulse_f/jnp.linalg.norm(pulse_f)
    pulse_t_inp_interp = pulse_t_inp_interp/jnp.linalg.norm(pulse_t_inp_interp)
    pulse_f_inp_interp = pulse_f_inp_interp/jnp.linalg.norm(pulse_f_inp_interp)

    amp_t_conv = jax.scipy.signal.correlate(jnp.abs(pulse_t), jnp.abs(pulse_t_inp_interp))
    amp_t_score = jnp.max(amp_t_conv)
    
    amp_f_conv = jax.scipy.signal.correlate(jnp.abs(pulse_f), jnp.abs(pulse_f_inp_interp))
    amp_f_score_1 = jnp.max(amp_f_conv)
    amp_f_score_2 = jnp.sum(jnp.abs(pulse_f)*jnp.abs(pulse_f_inp_interp)) # cross correlation without shift -> how good is central_f ?



    phase_f = jnp.unwrap(jnp.angle(pulse_f))
    phase_f_inp_interp = jnp.unwrap(jnp.angle(pulse_f_inp_interp))

    phase_f_grad = jnp.gradient(phase_f, frequency)
    phase_f_inp_interp_grad = jnp.gradient(phase_f_inp_interp, frequency)

    phase_f_grad_grad = jnp.gradient(phase_f_grad, frequency)
    phase_f_inp_interp_grad_grad = jnp.gradient(phase_f_inp_interp_grad, frequency)


    spectrum_norm = (jnp.abs(pulse_f)/jnp.max(jnp.abs(pulse_f)))**2
    mask = jnp.zeros(jnp.size(spectrum_norm))
    mask = jnp.where(spectrum_norm<0.1, mask, 1)
    residual = phase_f_grad_grad-phase_f_inp_interp_grad_grad
    residual = residual/jnp.std(residual) # -> a "normalization" to treat different chirps the same? Should maybe be done after mask is applied?
    error_phase_f = jnp.mean(mask*jnp.abs(residual)**2)
    
    return 1-amp_t_score, 1-amp_f_score_1, 1-amp_f_score_2, error_phase_f