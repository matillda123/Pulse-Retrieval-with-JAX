import jax.numpy as jnp
import lineax as lx
import jax
import equinox
from interpax import interp1d





@jax.tree_util.register_pytree_node_class
class MyNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def tree_flatten(self):
        class_dict=self.__dict__
        data_keys=list(class_dict.keys())
        data_values=list(class_dict.values())
        
        return data_values, data_keys


    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        custom_namespace_instance = cls()
        custom_namespace_instance.__dict__ = dict(zip(aux_data, leaves))
        return custom_namespace_instance
    

    def __add__(self, other):
        if isinstance(other, MyNamespace):
            leaves1, treedef = jax.tree.flatten(self)
            leaves2, treedef = jax.tree.flatten(other)

             # adding len() makes sure that both have trees have the same number of leaves
            leaves_new = [leaves1[i] + leaves2[i] for i in range((len(leaves1) + len(leaves2))//2)]
            tree_new = jax.tree.unflatten(treedef, leaves_new)
        else:
            leaves, treedef = jax.tree.flatten(self)
            leaves_new = [leaves[i] + other for i in range(len(leaves))]
            tree_new = jax.tree.unflatten(treedef, leaves_new)

        return tree_new
    

    def __radd__(self, other):
        return self.__add__(other)
    

    def __mul__(self, other):
        if isinstance(other, MyNamespace):
            leaves1, treedef = jax.tree.flatten(self)
            leaves2, treedef = jax.tree.flatten(other)

            leaves_new = [leaves1[i]*leaves2[i] for i in range((len(leaves1)+len(leaves2))//2)]
            tree_new = jax.tree.unflatten(treedef, leaves_new)
        else:
            leaves, treedef = jax.tree.flatten(self)
            leaves_new = [leaves[i]*other for i in range(len(leaves))]
            tree_new = jax.tree.unflatten(treedef, leaves_new)

        return tree_new
    

    def __rmul__(self, other):
        return self.__mul__(other)
    

    def __sub__(self, other):
        return self.__add__((-1)*other)
    






def scan_helper(carry, xs, actual_function, number_of_args, number_of_xs):
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
    if number_of_args==1:
        input_args = (carry, )
    else:
        input_args = carry
    return actual_function(*input_args)








def optimistix_helper_loss_function(input, dummy, function):
    error=function(input)
    return error, error


def optimistix_helper_alternating_loss_function(input, dummy, function):
    error=function(input)
    return error, error


def scan_helper_equinox(descent_state, xs, step, static):
    state = equinox.combine(descent_state, static)
    state, error = step(state)
    descent_state, _ = equinox.partition(state, equinox.is_array)
    return  descent_state, error



    










def do_fft(signal, sk, rn, axis=-1):
    # if axis=0 -> sk, rn need to be use with jnp.newaxis to map over axis=0
    # default is always axis=-1
    sk=jnp.conjugate(sk)
    rn=jnp.conjugate(rn)
    return jnp.fft.fft(signal*sk, axis=axis)*rn

def do_ifft(signal, sk, rn, axis=-1):
    return jnp.fft.ifft(signal*rn, axis=axis)*sk
 
 

def get_sk_rn(time, frequency):
    n=jnp.arange(jnp.size(frequency))
    df=jnp.mean(jnp.diff(frequency))
    rn=jnp.exp(1j*time[0]*2*jnp.pi*n*df)
    sk=jnp.exp(1j*2*jnp.pi*frequency[0]*time)
    return sk, rn



def do_interpolation_1d(x_new, x, y):
    # this exists to possibly switch to a jax.scipy based interpolation
    y_new = interp1d(x_new, x, y, method="cubic", extrap=1e-12)
    return y_new




def calculate_gate(pulse_t, method):
    assert method!="tg", "For TG, depending on the definition either pg or sd needs to be used."

    if method=="shg":
        gate=pulse_t
    elif method=="thg":
        gate=pulse_t**2
    elif method=="pg":
        gate=jnp.abs(pulse_t)**2
    elif method=="sd":
        gate=jnp.conjugate(pulse_t)**2

    return gate




def project_onto_intensity(signal_f, measured_intensity):
    return jnp.sqrt(jnp.abs(measured_intensity))*jnp.sign(measured_intensity)*jnp.exp(1j*jnp.angle(signal_f))



def project_onto_amplitude(signal_f, measured_amplitude):
    return measured_amplitude*jnp.exp(1j*jnp.angle(signal_f))




def calculate_S_prime(signal_t, measured_trace, mu, measurement_info):
    sk, rn = measurement_info.sk, measurement_info.rn
    signal_f=do_fft(signal_t, sk, rn)

    signal_f_new=project_onto_intensity(signal_f, measured_trace)

    signal_t_new=do_ifft(signal_f_new, sk, rn)*1/(jnp.sqrt(mu)+1e-12)
    return signal_t_new





def calculate_trace(signal_f):
    trace=jnp.abs(signal_f)**2
    return trace


def calculate_mu(trace, measured_trace):
    return jnp.sum(trace*measured_trace)/(jnp.sum(trace**2) + 1e-12)


def calculate_trace_error(trace, measured_trace):
    mu = calculate_mu(trace, measured_trace)
    return jnp.mean(jnp.abs(mu*trace-measured_trace)**2)



def calculate_Z_error(signal_t, signal_t_new):
    deltaS=signal_t_new-signal_t
    return jnp.sum(jnp.abs(deltaS)**2)






def generate_random_continuous_function(key, no_points, x, minval, maxval, distribution):
    key1, key2 = jax.random.split(key, 2)

    p_arr = distribution/jnp.sum(distribution)
    x_rand = jnp.sort(jax.random.choice(key1, x, (no_points, ), replace=False, p=p_arr))
    y_rand = jax.random.uniform(key2, (no_points, ), minval=minval, maxval=maxval)

    y = do_interpolation_1d(x, x_rand, y_rand)
    return y






def solve_system_using_lineax_iteratively(A, b, x_prev, solver):
    operator=lx.MatrixLinearOperator(A)
    A_precond=jnp.abs(jnp.diag(jnp.diag(A))).astype(jnp.complex64)
    preconditioner = lx.MatrixLinearOperator(A_precond, lx.positive_semidefinite_tag)
    solution = lx.linear_solve(operator, b, 
                                solver=solver, 
                                throw=False, 
                                options={"y0": x_prev, "preconditioner": preconditioner})
    
    return solution


def solve_linear_system(A, b, x_prev, solver):

    if solver=="scipy":
        newton_direction=jax.scipy.linalg.solve(A, b[..., None], assume_a="her").squeeze(-1)
        return newton_direction
    
    elif solver=="lineax":
        solution = jax.vmap(lambda A,b: lx.linear_solve(lx.MatrixLinearOperator(A), b, throw=False))(A, b)

    else:
        solution = jax.vmap(solve_system_using_lineax_iteratively, in_axes=(0,0,0,None))(A,b,x_prev,solver)
        
        
    return solution.value






    

def get_idx_arr(N, M, key):
    idx=jnp.arange(N, dtype=jnp.int32)
    idx_arr=jnp.zeros((M,N), dtype=jnp.int32)
    for i in range(M):
        idx_arr=idx_arr.at[i].set(jax.random.permutation(key, idx))
        key=jax.random.split(key, 1)[0]
    return jnp.array(idx_arr, dtype=jnp.int32)


def get_com(signal, idx_arr):
    com=jnp.sum(signal*idx_arr)/jnp.sum(signal)
    return com


def center_signal(signal):
    # center twice to hopefully actually center, one time is not enough because of periodic boundaries

    N=jnp.shape(signal)[0]
    idx_arr=jnp.arange(N)
    max_idx=jnp.argmax(jnp.abs(signal))
    signal=jnp.roll(signal, -(max_idx-N//2))

    com=get_com(jnp.abs(signal), idx_arr)
    signal=jnp.roll(signal, -(com-N//2))
    return signal    





























def loss_function_modifications(trace_or_signal_t, measured_trace_or_signal_t_new, time_or_zarr, frequency, amplitude_or_intensity, use_fd_grad):
    measured_trace_or_signal_t_new=measured_trace_or_signal_t_new/jnp.max(jnp.abs(measured_trace_or_signal_t_new))
    trace_or_signal_t=trace_or_signal_t/jnp.max(jnp.abs(trace_or_signal_t))

    if amplitude_or_intensity=="amplitude":
        # add small value since auto-diff of sqrt(0) is problematic
        measured_trace_or_signal_t_new=jnp.sqrt(jnp.abs(measured_trace_or_signal_t_new) + 1e-9)*jnp.sign(measured_trace_or_signal_t_new)
        trace_or_signal_t=jnp.sqrt(jnp.abs(trace_or_signal_t) + 1e-9)*jnp.sign(trace_or_signal_t)
    elif amplitude_or_intensity=="intensity":
        pass
    elif type(amplitude_or_intensity)==int or type(amplitude_or_intensity)==float:
        exp_val=amplitude_or_intensity
        measured_trace_or_signal_t_new=(jnp.abs(measured_trace_or_signal_t_new) + 1e-9)**exp_val*jnp.sign(measured_trace_or_signal_t_new)
        trace_or_signal_t=(jnp.abs(trace_or_signal_t) + 1e-9)**exp_val*jnp.sign(trace_or_signal_t)
    else:
        print("something is wrong")

    if use_fd_grad!=False:
        print("generalize me using scan")
        # measured_0, _ = jax.lax.scan(lambda x: (Partial(jnp.gradient, axis=0)(x),None), measured_0, length=use_fd_grad)
        # measured_1, _ = jax.lax.scan(lambda x: (Partial(jnp.gradient, axis=1)(x),None), measured_1, length=use_fd_grad)

        # reconstructed_0, _ = jax.lax.scan(lambda x: (Partial(jnp.gradient, axis=0)(x),None), reconstructed_0, length=use_fd_grad)
        # reconstructed_1, _ = jax.lax.scan(lambda x: (Partial(jnp.gradient, axis=1)(x),None), reconstructed_1, length=use_fd_grad)

        # measured = measured_0 + measured_1
        # reconstructed = reconstructed_0 + reconstructed_1

        measured_trace_or_signal_t_new_0=jnp.gradient(measured_trace_or_signal_t_new, time_or_zarr, axis=0)
        measured_trace_or_signal_t_new_1=jnp.gradient(measured_trace_or_signal_t_new, frequency, axis=1)
        
        trace_or_signal_t_0=jnp.gradient(trace_or_signal_t, time_or_zarr, axis=0)
        trace_or_signal_t_1=jnp.gradient(trace_or_signal_t, frequency, axis=1)

        if use_fd_grad==2:
            measured_trace_or_signal_t_new_0=jnp.gradient(measured_trace_or_signal_t_new_0, time_or_zarr, axis=0)
            measured_trace_or_signal_t_new_1=jnp.gradient(measured_trace_or_signal_t_new_1, frequency, axis=1)
            
            trace_or_signal_t_0=jnp.gradient(trace_or_signal_t_0, time_or_zarr, axis=0)
            trace_or_signal_t_1=jnp.gradient(trace_or_signal_t_1, frequency, axis=1)

        measured_trace_or_signal_t_new = measured_trace_or_signal_t_new_0 + measured_trace_or_signal_t_new_1
        trace_or_signal_t = trace_or_signal_t_0 + trace_or_signal_t_1
        

    return trace_or_signal_t, measured_trace_or_signal_t_new











def get_score_values(final_result, input_pulses):

    pulse_t, pulse_f = final_result.pulse_t, final_result.pulse_f
    if hasattr(final_result, "time"):
        time=final_result.time
    elif hasattr(final_result, "tau_arr"):
        time=final_result.tau_arr
    else:
        print("somethings wrong :/")

    frequency = final_result.frequency

    time_inp, frequency_inp = input_pulses.time, input_pulses.frequency
    pulse_t_inp, pulse_f_inp = input_pulses.pulse_t, input_pulses.pulse_f


    pulse_t_inp_interp=do_interpolation_1d(time, time_inp, pulse_t_inp)
    pulse_f_inp_interp=do_interpolation_1d(frequency, frequency_inp, pulse_f_inp)
    
    pulse_t=pulse_t/jnp.linalg.norm(pulse_t)
    pulse_f=pulse_f/jnp.linalg.norm(pulse_f)
    pulse_t_inp_interp=pulse_t_inp_interp/jnp.linalg.norm(pulse_t_inp_interp)
    pulse_f_inp_interp=pulse_f_inp_interp/jnp.linalg.norm(pulse_f_inp_interp)

    amp_t_conv=jax.scipy.signal.correlate(jnp.abs(pulse_t), jnp.abs(pulse_t_inp_interp))
    amp_t_score=jnp.max(amp_t_conv)
    
    amp_f_conv=jax.scipy.signal.correlate(jnp.abs(pulse_f), jnp.abs(pulse_f_inp_interp))
    amp_f_score_1=jnp.max(amp_f_conv)
    amp_f_score_2=jnp.sum(jnp.abs(pulse_f)*jnp.abs(pulse_f_inp_interp)) # cross correlation for tau=0 -> no shift correction



    phase_f=jnp.unwrap(jnp.angle(pulse_f))
    phase_f_inp_interp=jnp.unwrap(jnp.angle(pulse_f_inp_interp))

    phase_f_grad=jnp.gradient(phase_f, frequency)
    phase_f_inp_interp_grad=jnp.gradient(phase_f_inp_interp, frequency)

    phase_f_grad_grad=jnp.gradient(phase_f_grad, frequency)
    phase_f_inp_interp_grad_grad=jnp.gradient(phase_f_inp_interp_grad, frequency)


    spectrum_norm=(jnp.abs(pulse_f)/jnp.max(jnp.abs(pulse_f)))**2
    mask=jnp.zeros(jnp.size(spectrum_norm))
    mask=jnp.where(spectrum_norm<0.1, mask, 1)
    residual=phase_f_grad_grad-phase_f_inp_interp_grad_grad
    residual=residual/jnp.std(residual)
    error_phase_f=jnp.mean(mask*jnp.abs(residual)**2)
    
    return 1-amp_t_score, 1-amp_f_score_1, 1-amp_f_score_2, error_phase_f