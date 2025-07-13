import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

import jax.numpy as jnp
import jax
from jax.tree_util import Partial

from utilities import scan_helper, center_signal, do_fft, do_ifft, project_onto_intensity, do_interpolation_1d



def gerchberg_saxton_step(pulse_f, amp_f, amp_t):
    pulse_t=do_ifft(pulse_f)
    pulse_t=amp_t*jnp.exp(1j*jnp.angle(pulse_t))

    pulse_f=do_fft(pulse_t)
    pulse_f=amp_f*jnp.exp(1j*jnp.angle(pulse_f))
    return pulse_f, None


def gerchberg_saxton(trace, number_of_iterations):
    amp_f=jnp.sqrt(jnp.sum(trace,axis=0))
    amp_t=jnp.sqrt(jnp.sum(trace,axis=1))

    pulse_f=amp_f*jnp.exp(1j)

    do_step=Partial(gerchberg_saxton_step, amp_f=amp_f, amp_t=amp_t)
    do_scan=Partial(scan_helper, actual_function=do_step, number_of_args=1, number_of_xs=0)

    pulse_f, _=jax.lax.scan(do_scan, pulse_f, length=number_of_iterations)

    return pulse_f








def gaussian(x,a,b,c):
    return a*np.exp(-0.5*(x-c)**2/b**2)
 
def get_double_pulse_amps(trace,sigma=10, init_std=0.001):
    m=np.mean(trace, axis=1)

    x1=find_peaks(gaussian_filter1d(m,sigma=sigma))[0]
    a0=m[x1]
    x0=x1/len(m)

    if len(x1)>3:
        x1=find_peaks(gaussian_filter1d(m, sigma=2*sigma))[0]
        a0=m[x1]
        x0=x1/len(m)
    elif len(x1)<=1:
        print("reduce sigma")

    x=np.linspace(0,1,len(m))
    y=m
    
    dx0=x0[2]-x0[0]

    amp=[]
    for i in [0,2]:
        sol=curve_fit(gaussian, x,y, p0=[a0[i], init_std, x0[i]])

        if i==0:
            sol[0][2]=sol[0][2]+dx0/4
        elif i==2:
            sol[0][2]=sol[0][2]-dx0/4
        
        amp.append(gaussian(x,sol[0][0], sol[0][1], sol[0][2]))
    return np.array(amp)



def get_central_frequencies(time, freq, trace, monochromatic=True, sigma=10):
    m=np.mean(trace, axis=0)

    if monochromatic==True:
        x1=np.argmax(gaussian_filter1d(m,sigma=sigma))
        f1=freq[x1]
        return np.array([2*np.pi*f1*time])

    elif monochromatic==False:
        x1=find_peaks(gaussian_filter1d(m,sigma=sigma))[0]
        if len(x1)!=2:
            max1=np.max(x1)
            max2=np.max(x1[x1!=max1])
            print(f"Didnt find exactly two central frequencies. Found {len(x1)}. Picked the strongest two.")

            f1=freq[max1]
            f2=freq[max2]

        else:
            f1=freq[x1[0]]
            f2=freq[x1[1]]

        return np.array([2*np.pi*f1*time, 2*np.pi*f2*time])
    


def get_double_pulse_initial_guess(tau_arr, frequency, measured_trace, monochromatic_double_pulse, sigma=3, init_std=0.001):

    amp=get_double_pulse_amps(measured_trace, sigma=sigma, init_std=init_std)
    phase=get_central_frequencies(tau_arr, frequency, measured_trace, monochromatic=monochromatic_double_pulse, sigma=sigma)

    if len(phase)==1:
        amp_time=np.sum(amp, axis=0)
        pulse_t=amp_time*np.exp(1j*phase[0])
    elif len(phase)==2:
        pulse_t=0
        for i in range(2):
            p=amp[i]*np.exp(1j*phase[i])
            pulse_t=pulse_t+p
    else:
        print("something went wrong")

    pulse_t=center_signal(pulse_t)

    return do_fft(pulse_t)









def split_data_for_rana_method(N, tau_arr, frequency, measured_trace):

    mini_trace_arr=[]
    mini_tau_arr=[]
    mini_freq_arr=[]
    for i in range(N):
        for j in range(N):
            mini_trace=measured_trace[i::N,j::N]
            mini_freq=frequency[i::N]
            mini_tau=tau_arr[i::N]

            mini_trace_arr.append(mini_trace)
            mini_tau_arr.append(mini_tau)
            mini_freq_arr.append(mini_freq)

    return jnp.array(mini_tau_arr), jnp.array(mini_freq_arr), jnp.array(mini_trace_arr)


def rana_method_helper(tau_arr, frequency, trace, custom_inital_guess, initial_guess_type, frogmethod, RetrievalClass, number_of_iterations, **kwargs):
    retrieval=RetrievalClass(tau_arr, frequency, trace, frogmethod, **kwargs)
    pulse_t=retrieval.create_initial_guess(guess_type=initial_guess_type, custom_guess=custom_inital_guess)
    pulse_t, error_arr=retrieval.run(pulse_t, number_of_iterations)
    return pulse_t, error_arr



def do_one_round_of_rana_method(N, M, tau_arr, frequency, measured_trace, spectral_intensity, 
                                frogmethod, RetrievalClass, initial_guess_type, custom_initial_guess, number_of_iterations, **kwargs):
    
    # there may be an issue at some point between the jit-compiler through vmap and functionality in the __init__ function of the retrieval class.
    # idk why I wrote this comment. :/

    
    mini_tau_arr, mini_freq_arr, mini_trace_arr = split_data_for_rana_method(N, tau_arr, frequency, measured_trace)
    
    if custom_initial_guess==None:
        custom_initial_guess=jnp.zeros(len(mini_tau_arr))

    rana_method=Partial(rana_method_helper, initial_guess_type=initial_guess_type, frogmethod=frogmethod, RetrievalClass=RetrievalClass,
                        number_of_iterations=number_of_iterations, **kwargs)
    rana_method=jax.vmap(rana_method, in_axes=(0,0,0,0))

    pulse_t_all, error_arr=rana_method(mini_tau_arr, mini_freq_arr, mini_trace_arr, custom_initial_guess)
    pulse_t_all=np.array(pulse_t_all)
    error_arr=np.array(error_arr)

    final_error=[]
    for i in range(len(mini_tau_arr)):
        pulse_t_all[i]=center_signal(pulse_t_all[i])
        final_error.append(np.mean(error_arr[i,-3:]))

    idx_best=np.argsort(final_error)[:M]

    pulse_t_best=np.array(pulse_t_all)[idx_best]
    mini_tau_best=np.array(mini_tau_arr)[idx_best]

    pulse_f_arr=[]
    for i in range(len(pulse_t_best)):
        pulse_t=do_interpolation_1d(tau_arr, mini_tau_best[i], pulse_t_best[i])
        pulse_t=gaussian_filter1d(pulse_t, sigma=3)
        pulse_t=center_signal(pulse_t)
        pulse_f=do_fft(pulse_t)
        pulse_f=project_onto_intensity(pulse_f, spectral_intensity)
        pulse_f_arr.append(pulse_f)

    shape=np.shape(pulse_f_arr)
    temp=np.zeros((2,shape[0],shape[1]))+0j
    temp[0,:,:]=frequency
    temp[1,:,:]=pulse_f_arr

    return temp


def rana_method_for_initial_guess(tau_arr, frequency, measured_trace, spectral_intensity, N, RetrievalClass, frogmethod, number_of_iterations):
    M=N**2//4
    pulse_f_arr=do_one_round_of_rana_method(N, M, tau_arr, frequency, measured_trace, spectral_intensity, frogmethod=frogmethod, RetrievalClass=RetrievalClass, 
                                            initial_guess_type="random", custom_initial_guess=None, number_of_iterations=number_of_iterations)
    N=N//2

    while N>=2:
        M=N**2//4
        if N==2:
            M=2*N
            
        pulse_f_arr=do_one_round_of_rana_method(N, M, tau_arr, frequency, measured_trace, spectral_intensity, frogmethod=frogmethod, RetrievalClass=RetrievalClass, 
                                            initial_guess_type="custom", custom_initial_guess=list(pulse_f_arr), number_of_iterations=number_of_iterations)
        N=N//2
    
    return np.mean(pulse_f_arr[:,1],axis=0), pulse_f_arr