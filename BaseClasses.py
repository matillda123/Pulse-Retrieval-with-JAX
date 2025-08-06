import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as c0
import refractiveindex

import jax
import jax.numpy as jnp

from jax.tree_util import Partial
from equinox import tree_at

from utilities import MyNamespace, get_com, center_signal, do_fft, do_ifft, get_sk_rn, do_interpolation_1d, calculate_gate, calculate_gate_with_Real_Fields, calculate_trace, calculate_trace_error, project_onto_amplitude, run_scan
from create_population import create_population_classic



class AlgorithmsBASE:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.adaptive_scaling = False
        self.use_jit = False



    def shuffle_data_along_m(self, descent_state, measurement_info, descent_info):
        descent_state.key, subkey=jax.random.split(descent_state.key, 2)
        keys = jax.random.split(subkey, descent_info.population_size)

        idx_arr=jax.vmap(jax.random.permutation, in_axes=(0,None))(keys, descent_info.idx_arr)

        transform_arr, measured_trace = measurement_info.transform_arr, measurement_info.measured_trace
        
        transform_arr = jax.vmap(Partial(jnp.take, axis=0), in_axes=(None, 0))(transform_arr, idx_arr)
        measured_trace = jax.vmap(Partial(jnp.take, axis=0), in_axes=(None, 0))(measured_trace, idx_arr)

        transform_arr=jnp.swapaxes(transform_arr, 0, 1)
        if len(jnp.shape(transform_arr))==2:
            transform_arr=jnp.expand_dims(transform_arr, axis=2)

        measured_trace=jnp.transpose(measured_trace, axes=(1,0,2))
        return transform_arr, measured_trace, descent_state


    def run(self, init_vals, no_iterations=100):
        if self.spectrum_is_being_used==True:
            assert self.descent_info.measured_spectrum_is_provided.pulse==True or self.descent_info.measured_spectrum_is_provided.gate==True, "you need to provide a spectrum"

        carry, do_scan = self.initialize_run(init_vals)

        if self.use_jit==True:
            scan = jax.jit(run_scan, static_argnames=("do_scan", "no_iterations"))
            carry, error_arr = scan(do_scan, carry, no_iterations)
        else:
            carry, error_arr = jax.lax.scan(do_scan, carry, length=no_iterations)

        error_arr = jnp.squeeze(error_arr)
        final_result = self.post_process(carry, error_arr)
        return final_result
    



    def do_step_and_apply_spectrum(self, descent_state, measurement_info, descent_info):
        population = descent_state.population
        
        if descent_info.measured_spectrum_is_provided.pulse==True:
            pulse = jax.vmap(self.apply_spectrum, in_axes=(0,None,None,None))(population.pulse, measurement_info.spectral_amplitude.pulse, 
                                                                              measurement_info.sk, measurement_info.rn)
            population = tree_at(lambda x: x.pulse, population, pulse)

        if descent_info.measured_spectrum_is_provided.gate==True:
            gate = jax.vmap(self.apply_spectrum, in_axes=(0,None,None,None))(population.gate, measurement_info.spectral_amplitude.gate, 
                                                                             measurement_info.sk, measurement_info.rn)
            population = tree_at(lambda x: x.gate, population, gate)
            
        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state, trace_error = self.step_pre_chaining__apply_spectrum(descent_state, measurement_info, descent_info)
        return descent_state, trace_error
    



    def do_global_iteration_and_apply_spectrum_COPRA(self, descent_state, measurement_info, descent_info):
        population = descent_state.population

        if descent_info.measured_spectrum_is_provided.pulse==True:
            pulse=jax.vmap(self.apply_spectrum, in_axes=(0,None,None,None))(population.pulse, measurement_info.spectral_amplitude.pulse, 
                                                                                       measurement_info.sk, measurement_info.rn)
            population = tree_at(lambda x: x.pulse, population, pulse)

        if descent_info.measured_spectrum_is_provided.gate==True:
            gate=jax.vmap(self.apply_spectrum, in_axes=(0,None,None,None))(population.gate, measurement_info.spectral_amplitude.gate, 
                                                                                      measurement_info.sk, measurement_info.rn)
            population = tree_at(lambda x: x.gate, population, gate)

        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state, trace_error=self.global_iteration_pre_chaining__apply_spectrum(descent_state, measurement_info, descent_info)
        return descent_state, trace_error




    def use_measured_spectrum(self):
        # if-else is needed to avoid recursion
        if self.spectrum_is_being_used==True:
            return self
        else:
            names_list = ["DifferentialEvolution", "Evosax", "AutoDiff"]
            if self.name=="COPRA":
                # applying_spectrum on local stage in COPRA doesnt seem to work nicely -> maybe this conclusion is wrong, maybe test again some time
                self.global_iteration_pre_chaining__apply_spectrum = self.global_iteration
                self.global_iteration = self.do_global_iteration_and_apply_spectrum_COPRA
            elif any([self.name==name for name in names_list])==True:
                # in these classes the spectrum is applied directly
                pass
            else:
                self.step_pre_chaining__apply_spectrum = self.step
                self.step = self.do_step_and_apply_spectrum

            self.spectrum_is_being_used = True
            return self
        



    def do_step_and_apply_momentum(self, descent_state, measurement_info, descent_info):
        population = descent_state.population
        momentum = descent_state.momentum

        population_pulse, momentum_pulse = self.apply_momentum(population.pulse, momentum.pulse, descent_info.eta)
        population = tree_at(lambda x: x.pulse, population, population_pulse)
        momentum = tree_at(lambda x: x.pulse, momentum, momentum_pulse)

        if measurement_info.doubleblind==True:
            population_gate, momentum_gate = self.apply_momentum(population.gate, momentum.gate, descent_info.eta)
            population = tree_at(lambda x: x.gate, population, population_gate)
            momentum = tree_at(lambda x: x.gate, momentum, momentum_gate)

        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x.momentum, descent_state, momentum)

        descent_state, trace_error = self.step_pre_chaining__momentum(descent_state, measurement_info, descent_info)
        return descent_state, trace_error
    

    
    def do_local_iteration_and_apply_momentum_COPRA(self, descent_state, measurement_info, descent_info):
        population = descent_state.population
        momentum = descent_state.momentum
        
        population_pulse, momentum_pulse = self.apply_momentum(population.pulse, momentum.pulse, descent_info.eta)
        population = tree_at(lambda x: x.pulse, population, population_pulse)
        momentum = tree_at(lambda x: x.pulse, momentum, momentum_pulse)

        if measurement_info.doubleblind==True:
            population_gate, momentum_gate = self.apply_momentum(population.gate, momentum.gate, descent_info.eta)
            population = tree_at(lambda x: x.gate, population, population_gate)
            momentum = tree_at(lambda x: x.gate, momentum, momentum_gate)

        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x.momentum, descent_state, momentum)

        descent_state = self.local_iteration_pre_chaining__apply_momentum(descent_state, measurement_info, descent_info)
        return descent_state
    

    
    def do_global_iteration_and_apply_momentum_COPRA(self, descent_state, measurement_info, descent_info):
        population = descent_state.population
        momentum = descent_state.momentum
        
        population_pulse, momentum_pulse = self.apply_momentum(population.pulse, momentum.pulse, descent_info.eta)
        population = tree_at(lambda x: x.pulse, population, population_pulse)
        momentum = tree_at(lambda x: x.pulse, momentum, momentum_pulse)

        if measurement_info.doubleblind==True:
            population_gate, momentum_gate = self.apply_momentum(population.gate, momentum.gate, descent_info.eta)
            population = tree_at(lambda x: x.gate, population, population_gate)
            momentum = tree_at(lambda x: x.gate, momentum, momentum_gate)

        descent_state = tree_at(lambda x: x.population, descent_state, population)
        descent_state = tree_at(lambda x: x.momentum, descent_state, momentum)

        descent_state = self.global_iteration_pre_chaining__apply_momentum(descent_state, measurement_info, descent_info)
        return descent_state
    


    def momentum_acceleration(self, population_size, eta):
        if self.momentum_is_being_used==True:
            return self
        else:
            shape=(population_size, jnp.size(self.frequency))
            init_arr = jnp.zeros(shape, dtype=jnp.complex64)
            
            self.descent_info = self.descent_info.expand(eta=eta)
            self.descent_state = self.descent_state.expand(momentum = MyNamespace(pulse = MyNamespace(update_for_velocity_map=init_arr, velocity_map=init_arr), 
                                                                                  gate = MyNamespace(update_for_velocity_map=init_arr, velocity_map=init_arr)))
            
            names_list = ["DifferentialEvolution", "Evosax", "LSF", "AutoDiff"]
            if self.name=="COPRA":
                self.local_iteration_pre_chaining__apply_momentum = self.local_iteration
                self.global_iteration_pre_chaining__apply_momentum = self.global_iteration
                self.local_iteration = self.do_local_iteration_and_apply_momentum_COPRA
                self.global_iteration = self.do_global_iteration_and_apply_momentum_COPRA
            elif any([self.name==name for name in names_list])==True:
                pass
            else:
                self.step_pre_chaining__momentum = self.step
                self.step = self.do_step_and_apply_momentum
            
            self.momentum_is_being_used = True
            return self
    


    def apply_momentum(self, signal, momentum, eta):
        update_for_velocity_map, velocity_map = momentum.update_for_velocity_map, momentum.velocity_map

        velocity_map = eta*velocity_map + (signal - update_for_velocity_map)
        signal = signal + eta*velocity_map

        momentum = MyNamespace(update_for_velocity_map=signal, velocity_map=velocity_map)
        return signal, momentum

    











class RetrievePulses:
    def __init__(self, nonlinear_method, *args, seed=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.child_class=None
        self.nonlinear_method=nonlinear_method
        self.f0=0
        self.doubleblind=False

        self.spectrum_is_being_used=False
        self.momentum_is_being_used=False

        self.measurement_info=MyNamespace(nonlinear_method = self.nonlinear_method,
                                          spectral_amplitude = MyNamespace(pulse=None, gate=None),
                                          central_f = MyNamespace(pulse=None, gate=None))
        self.descent_info=MyNamespace(measured_spectrum_is_provided = MyNamespace(pulse=False, gate=False))
        self.descent_state=MyNamespace()


        if seed==None:
            self.prng_seed = np.random.randint(0, 1e9)
        else:
            self.prng_seed = int(seed)

        self.update_PRNG_key(self.prng_seed)

        if nonlinear_method=="shg":
            self.factor=2
        elif nonlinear_method=="thg":
            self.factor=3
        else:
            self.factor=1


    def update_PRNG_key(self, seed):
        self.prng_seed=seed
        self.key=jax.random.PRNGKey(seed)



    def get_data(self, x_arr, frequency, measured_trace):
        measured_trace=measured_trace/jnp.linalg.norm(measured_trace)

        self.x_arr=jnp.array(x_arr)
        self.frequency=jnp.array(frequency)
        self.time=jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(frequency), jnp.mean(jnp.diff(self.frequency))))
        self.measured_trace=jnp.array(measured_trace)

        return self.x_arr, self.time, self.frequency, self.measured_trace



    def get_spectral_amplitude(self, measured_frequency, measured_spectrum, pulse_or_gate="pulse"):
        frequency = self.frequency

        spectral_intensity=do_interpolation_1d(frequency, measured_frequency-self.f0/self.factor, measured_spectrum)
        spectral_amplitude = jnp.sqrt(jnp.abs(spectral_intensity))*jnp.sign(spectral_intensity)
        
        if pulse_or_gate=="pulse":
            self.measurement_info.spectral_amplitude.pulse = spectral_amplitude
            self.descent_info.measured_spectrum_is_provided.pulse=True

        elif pulse_or_gate=="gate":
            self.measurement_info.spectral_amplitude.gate=spectral_amplitude
            self.descent_info.measured_spectrum_is_provided.gate=True

        else:
            print("wrong :/")

        return spectral_amplitude
    

    def get_gate_pulse(self, frequency, gate_f):
        gate_f=do_interpolation_1d(self.frequency, frequency, gate_f)
        self.gate=do_ifft(gate_f, self.sk, self.rn)
        self.measurement_info.gate=self.gate
        return self.gate
    

    


    def create_initial_population(self, population_size=1, guess_type="random"):
        self.key, subkey = jax.random.split(self.key, 2)
        pulse_f_arr = create_population_classic(subkey, population_size, guess_type, self.measurement_info)

        if self.doubleblind==True:
            self.key, subkey = jax.random.split(self.key, 2)
            gate_f_arr = create_population_classic(subkey, population_size, guess_type, self.measurement_info)
        else:
            gate_f_arr=None

        self.descent_info = self.descent_info.expand(population_size=population_size)
        return pulse_f_arr, gate_f_arr




    def get_individual_from_idx(self, idx, population):
        # idx can also be an array (i think, didnt test yet)
        leaves, treedef = jax.tree.flatten(population)
        leaves_individual = [leaves[i][idx] for i in range(len(leaves))]
        individual = jax.tree.unflatten(treedef, leaves_individual)
        return individual
    

    


    def plot_results(self, final_result, exact_pulse=None):
        pulse_t, pulse_f, trace = final_result.pulse_t, final_result.pulse_f, final_result.trace
        error_arr=final_result.error_arr

        x_arr, time, frequency, measured_trace = final_result.x_arr, final_result.time, final_result.frequency, final_result.measured_trace
        
        trace=trace/jnp.max(trace)
        measured_trace=measured_trace/jnp.max(measured_trace)
        trace_difference=measured_trace-trace

        fig=plt.figure(figsize=(22,14))
        ax1=plt.subplot(2,3,1)
        ax1.plot(time, np.abs(pulse_t), label="Amplitude")
        ax1.set_xlabel(r"Time [arb. u.]")
        ax1.legend(loc=2)

        ax2 = ax1.twinx()
        ax2.plot(time, np.unwrap(np.angle(pulse_t))*1/np.pi, c="tab:orange", label="Phase")
        ax2.set_ylabel(r"Phase [$\pi$]")
        ax2.legend(loc=1)

        if exact_pulse!=None:
            ax1.plot(exact_pulse.time, np.abs(exact_pulse.pulse_t)*np.max(np.abs(pulse_t))/np.max(np.abs(exact_pulse.pulse_t)), 
                     "--", c="black", label="Exact Amplitude")
            ax2.plot(exact_pulse.time, np.unwrap(np.angle(exact_pulse.pulse_t)), "--", c="black", label="Exact Phase", alpha=0.5)

        ax1=plt.subplot(2,3,2)
        ax1.plot(frequency,jnp.abs(pulse_f), label="Amplitude")
        ax1.set_xlabel(r"Frequency [arb. u.]")
        ax1.legend(loc=2)

        ax2 = ax1.twinx()
        ax2.plot(frequency, jnp.unwrap(jnp.angle(pulse_f))*1/np.pi, c="tab:orange", label="Phase")
        ax2.set_ylabel(r"Phase [$\pi$]")
        ax2.legend(loc=1)

        if exact_pulse!=None:
            ax1.plot(exact_pulse.frequency, np.abs(exact_pulse.pulse_f)*np.max(np.abs(pulse_f))/np.max(np.abs(exact_pulse.pulse_f)),
                     "--", c="black", label="Exact Amplitude")
            ax2.plot(exact_pulse.frequency, np.unwrap(np.angle(exact_pulse.pulse_f)), "--", c="black", label="Exact Phase", alpha=0.5)

        plt.subplot(2,3,3)
        plt.plot(error_arr)
        plt.yscale("log")
        plt.title("Trace Error")
        plt.xlabel("Iteration No.")

        plt.subplot(2,3,4)
        plt.pcolormesh(x_arr, frequency, measured_trace.T)
        plt.xlabel("Shift [arb. u.]")
        plt.ylabel("Frequency [arb. u.]")

        plt.subplot(2,3,5)
        plt.pcolormesh(x_arr, frequency, trace.T)
        plt.xlabel("Shift [arb. u.]")
        plt.ylabel("Frequency [arb. u.]")

        plt.subplot(2,3,6)
        plt.pcolormesh(x_arr, frequency, trace_difference.T)
        plt.xlabel("Shift [arb. u.]")
        plt.ylabel("Frequency [arb. u.]")
        plt.colorbar()







    def get_idx_best_individual(self, descent_state):
        measurement_info, descent_info = self.measurement_info, self.descent_info 

        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)        
        trace=calculate_trace(do_fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn))
        trace_error=jax.vmap(calculate_trace_error, in_axes=(0,None))(trace, measurement_info.measured_trace)
        idx=jnp.argmin(trace_error)
        return idx
    




    def post_process_center_pulse_and_gate(self, pulse_t, gate_t):
        sk, rn = self.measurement_info.sk, self.measurement_info.rn

        N=jnp.size(pulse_t)
        idx_arr=jnp.arange(N)
        com_1=jnp.argmax(jnp.abs(pulse_t))
        pulse_t=jnp.roll(pulse_t, -(com_1-N//2))

        com_2=get_com(jnp.abs(pulse_t), idx_arr)
        pulse_t=jnp.roll(pulse_t, -(com_2-N//2))

        shift_total=-(com_1+com_2-N)
        gate_t=jnp.roll(gate_t, shift_total)

        pulse_f=do_fft(pulse_t, sk, rn)
        gate_f=do_fft(gate_t, sk, rn)

        return pulse_t, gate_t, pulse_f, gate_f




    def post_process(self, descent_state, error_arr):
        self.descent_state=descent_state

        pulse_t, gate_t, pulse_f, gate_f = self.post_process_get_pulse_and_gate(descent_state, self.measurement_info, self.descent_info)
        pulse_t, gate_t, pulse_f, gate_f = self.post_process_center_pulse_and_gate(pulse_t, gate_t)

        measured_trace = self.measurement_info.measured_trace
        measured_trace = measured_trace/jnp.linalg.norm(measured_trace)
        
        trace = self.post_process_create_trace(pulse_t, gate_t)
        trace=trace/jnp.linalg.norm(trace)

        x_arr = self.get_x_arr()
        time, frequency = self.measurement_info.time, self.measurement_info.frequency + self.f0

        final_result=MyNamespace(x_arr=x_arr, time=time, frequency=frequency, 
                                 pulse_t=pulse_t, pulse_f=pulse_f, gate_t=gate_t, gate_f=gate_f,
                                 trace=trace, measured_trace=measured_trace,
                                 error_arr=error_arr)
        return final_result




    
    
    































class RetrievePulsesFROG(RetrievePulses):
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, xfrog=False, ifrog=False, **kwargs):
        
        super().__init__(nonlinear_method, **kwargs)

        self.tau_arr, self.time, self.frequency, self.measured_trace = self.get_data(delay, frequency, measured_trace)
        self.gate = jnp.zeros(jnp.size(self.time))

        self.transform_arr = self.tau_arr
        self.idx_arr = jnp.arange(jnp.shape(self.transform_arr)[0])   
        

        self.dt = jnp.mean(jnp.diff(self.time))
        self.df = jnp.mean(jnp.diff(self.frequency))
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        self.xfrog = xfrog
        self.ifrog = ifrog

        if self.xfrog=="doubleblind":
            self.doubleblind = True
            self.xfrog = False

        self.measurement_info = self.measurement_info.expand(tau_arr = self.tau_arr,
                                                             frequency = self.frequency,
                                                             time = self.time,
                                                             measured_trace = self.measured_trace,
                                                             xfrog = self.xfrog,
                                                             doubleblind = self.doubleblind,
                                                             ifrog = self.ifrog,
                                                             dt = self.dt,
                                                             df = self.df,
                                                             sk = self.sk,
                                                             rn = self.rn,
                                                             gate = self.gate,
                                                             transform_arr = self.transform_arr,
                                                             x_arr = self.x_arr)
        


    def create_initial_population(self, population_size=1, guess_type="random"):
        pulse_f_arr, gate_f_arr = super().create_initial_population(population_size, guess_type)

        sk, rn = self.sk, self.rn
        pulse_t_arr = do_ifft(pulse_f_arr, sk, rn)

        if self.measurement_info.doubleblind==True:
            gate_t_arr = do_ifft(gate_f_arr, sk, rn)
        else:
            gate_t_arr = None

        population = MyNamespace(pulse=pulse_t_arr, gate=gate_t_arr)
        return population



    def create_initial_population_for_doublepulse(self, monochromatic_double_pulse=False, sigma=3, init_std=0.001):
        print("make me a population and put me in create_population.py")
        pulse_f=get_double_pulse_initial_guess(self.tau_arr, self.frequency, self.measured_trace, monochromatic_double_pulse=monochromatic_double_pulse, 
                                                   sigma=sigma, init_std=init_std)
        
        pulse_t=do_ifft(pulse_f, self.sk, self.rn)
        return pulse_t
    





    def shift_signal_in_time(self, signal, tau, frequency, sk, rn):
        signal_f=do_fft(signal, sk, rn)
        signal_f=signal_f*jnp.exp(-1j*2*jnp.pi*frequency*tau)
        signal=do_ifft(signal_f, sk, rn)
        return signal


    def calculate_shifted_signal(self, signal, frequency, tau_arr, time, in_axes=(None, 0, None, None, None)):

        # im really unhappy with this, but this re-definition/calculation of sk, rn is necessary
        # in the original case a global phase shift dependent on tau and f[0] occured, which i couldnt figure out
        # the reason for this issue could be the padding pattern? Im padding before and afterwards, maybe one should only pad afterwards ?
        frequency = frequency - (frequency[-1] + frequency[0])/2

        N = jnp.size(frequency)
        pad_arr = [(0,0)]*(signal.ndim-1) + [(N,N)]
        signal = jnp.pad(signal, pad_arr) # this is annoying, id like to only pad axis=-1 but thats only availble througnt he shape of the padwidth, but that can be different for different cases

        frequency = jnp.linspace(jnp.min(frequency), jnp.max(frequency), 3*N)
        time = jnp.fft.fftshift(jnp.fft.fftfreq(3*N, jnp.mean(jnp.diff(frequency))))

        sk, rn = get_sk_rn(time, frequency)

        signal_shifted = jax.vmap(self.shift_signal_in_time, in_axes=in_axes)(signal, tau_arr, frequency, sk, rn)
        return signal_shifted[ ... , N:2*N]




    def calculate_signal_t(self, individual, tau_arr, measurement_info):
        time, frequency = measurement_info.time, measurement_info.frequency
        xfrog, doubleblind, ifrog = measurement_info.xfrog, measurement_info.doubleblind, measurement_info.ifrog
        frogmethod = measurement_info.nonlinear_method

        pulse, gate = individual.pulse, individual.gate


        pulse_t_shifted=self.calculate_shifted_signal(pulse, frequency, tau_arr, time)

        if xfrog==True:
            gate_pulse_shifted =self.calculate_shifted_signal(measurement_info.xfrog_gate, frequency, tau_arr, time)
            gate_shifted = calculate_gate(gate_pulse_shifted, frogmethod)

        elif doubleblind==True:
            gate_pulse_shifted = self.calculate_shifted_signal(gate, frequency, tau_arr, time)
            gate_shifted = calculate_gate(gate_pulse_shifted, frogmethod)

        else:
            gate_pulse_shifted=None
            gate_shifted=calculate_gate(pulse_t_shifted, frogmethod)


        if ifrog==True and xfrog==False and doubleblind==False:
            signal_t=(pulse + pulse_t_shifted)*calculate_gate(pulse + pulse_t_shifted, frogmethod)
        elif ifrog==True:
            signal_t=(pulse + gate_pulse_shifted)*calculate_gate(pulse + gate_pulse_shifted, frogmethod)
        else:
            signal_t=pulse*gate_shifted
            

        signal_t = MyNamespace(signal_t=signal_t, pulse_t_shifted=pulse_t_shifted, gate_shifted=gate_shifted, gate_pulse_shifted=gate_pulse_shifted)
        return signal_t




    def generate_signal_t(self, descent_state, measurement_info, descent_info):
        tau_arr = measurement_info.tau_arr
        population = descent_state.population
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,None,None))(population, tau_arr, measurement_info)
        return signal_t
    




    def post_process_get_pulse_and_gate(self, descent_state, measurement_info, descent_info):
        sk, rn = measurement_info.sk, measurement_info.rn
        idx = self.get_idx_best_individual(descent_state)

        individual = self.get_individual_from_idx(idx, descent_state.population)
        pulse_t = individual.pulse
        pulse_f = do_fft(pulse_t, sk, rn)

        if measurement_info.doubleblind==True:
            gate_t = individual.gate
            gate_f = do_fft(gate_t, sk, rn)
        else:
            gate_t, gate_f = pulse_t, pulse_f

        return pulse_t, gate_t, pulse_f, gate_f
    


    def post_process_create_trace(self, pulse_t, gate_t):
        sk, rn = self.measurement_info.sk, self.measurement_info.rn
        tau_arr = self.measurement_info.tau_arr
    
        signal_t = self.calculate_signal_t(MyNamespace(pulse=pulse_t, gate=gate_t), tau_arr, self.measurement_info)
        signal_f = do_fft(signal_t.signal_t, sk, rn)
        trace = calculate_trace(signal_f)
        return trace
    

    
    def get_x_arr(self):
        return self.tau_arr

    
        



    def apply_spectrum(self, pulse_t, spectrum, sk, rn):
        pulse_f = do_fft(pulse_t, sk, rn)
        pulse_f_new = project_onto_amplitude(pulse_f, spectrum)
        pulse_t = do_ifft(pulse_f_new, sk, rn)
        return pulse_t
    























class RetrievePulsesDSCAN(RetrievePulses):
    def __init__(self, z_arr, frequency, measured_trace, nonlinear_method, 
                 refractive_index = refractiveindex.RefractiveIndexMaterial(shelf="main", book="SiO2", page="Malitson"), **kwargs):
        super().__init__(nonlinear_method, **kwargs)

        self.z_arr, self.time, self.frequency, self.measured_trace = self.get_data(z_arr, frequency, measured_trace)

        self.dt = jnp.mean(jnp.diff(self.time))
        self.df = jnp.mean(jnp.diff(self.frequency))
        self.sk, self.rn = get_sk_rn(self.time, self.frequency)

        self.refractive_index = refractive_index
        self.c0 = c0


        self.measurement_info = self.measurement_info.expand(z_arr = self.z_arr, # needs to be in mm
                                                             frequency = self.frequency,
                                                             time = self.time,
                                                             measured_trace = self.measured_trace,
                                                             doubleblind = self.doubleblind,
                                                             dt = self.dt,
                                                             df = self.df,
                                                             sk = self.sk,
                                                             rn = self.rn,
                                                             c0 = self.c0)


        self.phase_matrix = self.get_phase_matrix(self.refractive_index, self.z_arr, self.measurement_info)
        self.transform_arr = self.phase_matrix
        self.idx_arr = jnp.arange(jnp.shape(self.transform_arr)[0])   

        self.measurement_info = self.measurement_info.expand(phase_matrix = self.phase_matrix,
                                                             transform_arr = self.transform_arr,
                                                             x_arr = self.x_arr)




    def create_initial_population(self, population_size=1, guess_type="random"):
        pulse_f_arr, gate_f_arr = super().create_initial_population(population_size, guess_type)

        if self.doubleblind==True:
            gate_arr = gate_f_arr
        else:
            gate_arr = None

        population = MyNamespace(pulse=pulse_f_arr, gate=gate_arr)
        return population
    
    



    def get_phase_matrix(self, refractive_index, z_arr, measurement_info):
        frequency, c0 = measurement_info.frequency, measurement_info.c0
        c0 = c0*1e-12 # speed of light in mm/fs
        wavelength = c0/frequency
        n_arr = refractive_index.material.getRefractiveIndex(jnp.abs(wavelength)*1e6 + 1e-9, bounds_error=False) # wavelength needs to be in nm
        n_arr = jnp.where(jnp.isnan(n_arr)==False, n_arr, 1.0)
        k_arr = 2*jnp.pi/(wavelength + 1e-9)*n_arr
        phase_matrix = jnp.outer(z_arr, k_arr)
        return phase_matrix



    def get_dispersed_pulse_t(self, pulse_f, phase_matrix, measurement_info):
        sk, rn = measurement_info.sk, measurement_info.rn
        
        pulse_f=pulse_f*jnp.exp(1j*phase_matrix)
        pulse_t_disp=do_ifft(pulse_f, sk, rn)
        return pulse_t_disp, phase_matrix


    def calculate_signal_t(self, individual, phase_matrix, measurement_info):
        pulse = individual.pulse

        pulse_t_disp, phase_matrix = self.get_dispersed_pulse_t(pulse, phase_matrix, measurement_info)
        #pulse_t_disp = jax.vmap(center_signal)(pulse_t_disp)
        gate_disp = calculate_gate(pulse_t_disp, measurement_info.nonlinear_method)
        signal_t = pulse_t_disp*gate_disp

        signal_t = MyNamespace(signal_t=signal_t, pulse_t_disp=pulse_t_disp, gate_disp=gate_disp)
        return signal_t
    

    def generate_signal_t(self, descent_state, measurement_info, descent_info):
        phase_matrix = measurement_info.phase_matrix
        population = descent_state.population
        signal_t = jax.vmap(self.calculate_signal_t, in_axes=(0,None,None))(population, phase_matrix, measurement_info)
        return signal_t





    


    def post_process_get_pulse_and_gate(self, descent_state, measurement_info, descent_info):
        sk, rn =  measurement_info.sk, measurement_info.rn
        idx = self.get_idx_best_individual(descent_state)

        individual = self.get_individual_from_idx(idx, descent_state.population)
        pulse_f = individual.pulse
        pulse_t = do_ifft(pulse_f, sk, rn)

        if measurement_info.doubleblind==True:
            gate_f = individual.gate
            gate_t = do_ifft(gate_f, sk, rn)
        else:
            gate_f, gate_t = pulse_f, pulse_t

        return pulse_t, gate_t, pulse_f, gate_f
    



    def post_process_create_trace(self, pulse_t, gate_t):
        sk, rn = self.measurement_info.sk, self.measurement_info.rn
        pulse_f = do_fft(pulse_t, sk, rn)
        signal_t = self.calculate_signal_t(MyNamespace(pulse=pulse_f, gate=None), self.measurement_info.phase_matrix, self.measurement_info)
        trace = calculate_trace(do_fft(signal_t.signal_t, sk, rn))
        return trace
    

    def get_x_arr(self):
        return self.z_arr





    def apply_spectrum(self, pulse, spectrum, sk, rn):
        pulse = project_onto_amplitude(pulse, spectrum)
        return pulse
    










class RetrievePulsesFROGwithRealFields(RetrievePulsesFROG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def calculate_signal_t(self, individual, tau_arr, measurement_info):
        time, frequency = measurement_info.time, measurement_info.frequency
        xfrog, doubleblind, ifrog = measurement_info.xfrog, measurement_info.doubleblind, measurement_info.ifrog
        frogmethod = measurement_info.nonlinear_method

        pulse, gate = individual.pulse, individual.gate


        pulse_t_shifted = self.calculate_shifted_signal(pulse, frequency, tau_arr, time)

        if xfrog==True:
            gate_pulse_shifted = self.calculate_shifted_signal(measurement_info.xfrog_gate, frequency, tau_arr, time)
            gate_shifted = calculate_gate_with_Real_Fields(gate_pulse_shifted, frogmethod)

        elif doubleblind==True:
            gate_pulse_shifted = self.calculate_shifted_signal(gate, frequency, tau_arr, time)
            gate_shifted = calculate_gate_with_Real_Fields(gate_pulse_shifted, frogmethod)

        else:
            gate_pulse_shifted = None
            gate_shifted = calculate_gate_with_Real_Fields(pulse_t_shifted, frogmethod)


        if ifrog==True and xfrog==False and doubleblind==False:
            signal_t = jnp.real(pulse + pulse_t_shifted)*calculate_gate_with_Real_Fields(pulse + pulse_t_shifted, frogmethod)
        elif ifrog==True:
            signal_t = jnp.real(pulse + gate_pulse_shifted)*calculate_gate_with_Real_Fields(pulse + gate_pulse_shifted, frogmethod)
        else:
            signal_t = jnp.real(pulse)*gate_shifted
            

        signal_t = MyNamespace(signal_t = signal_t, 
                               pulse_t_shifted = pulse_t_shifted, 
                               gate_shifted = gate_shifted, 
                               gate_pulse_shifted = gate_pulse_shifted)
        return signal_t





class RetrievePulsesDSCANwithRealFields(RetrievePulsesDSCAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def calculate_signal_t(self, individual, phase_matrix, measurement_info):
        pulse = individual.pulse

        pulse_t_disp, phase_matrix = self.get_dispersed_pulse_t(pulse, phase_matrix, measurement_info)
        gate_disp = calculate_gate_with_Real_Fields(pulse_t_disp, measurement_info.nonlinear_method)
        signal_t = jnp.real(pulse_t_disp)*gate_disp

        signal_t = MyNamespace(signal_t = signal_t, 
                               pulse_t_disp = pulse_t_disp, 
                               gate_disp = gate_disp)
        return signal_t







class RetrievePulsesMIIPS(RetrievePulses):
    # this method is essentially a generalized version of dscan -> make dscan inherit from this -> classical algorithms dscan should work 
    # only difference is generation of phase matrix -> at least i think so
    pass




class RetrievePulses2DSI(RetrievePulses):
    pass