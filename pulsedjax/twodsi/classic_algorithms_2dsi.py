import jax.numpy as jnp
import jax
from jax.tree_util import Partial

from equinox import tree_at

from pulsedjax.core.base_classes_methods import RetrievePulses2DSI
from pulsedjax.core.base_classes_algorithms import ClassicAlgorithmsBASE
from pulsedjax.core.base_classic_algorithms import GeneralizedProjectionBASE, PtychographicIterativeEngineBASE, COPRABASE
from pulsedjax.utilities import scan_helper, center_signal, do_interpolation_1d, integrate_signal_1D, calculate_trace, calculate_trace_error

from pulsedjax.core.gradients.twodsi_z_error_gradients import calculate_Z_gradient
from pulsedjax.core.hessians.twodsi_z_error_pseudo_hessian import get_pseudo_newton_direction_Z_error
from pulsedjax.core.hessians.pie_pseudo_hessian import PIE_get_pseudo_newton_direction



class DirectReconstruction(ClassicAlgorithmsBASE, RetrievePulses2DSI):
    """ 
    Reconstructs a 2DSI trace non-iteratively by extracting the relative phase of the fringes for each frequency.

    [1] J. R. Birge et al., Opt. Lett. 31, 2063-2065, 10.1364/OL.31.002063 (2006)

    Attributes:
        integration_method (str): the integration method for the group delay. Has to be one of cumsum or euler_maclaurin_k
        integration_order (None, int): the euler_maclaurin order, infered from integration_method
        windowing (bool, str): windowing type for the FFT. Can be one of False, hamming or hann.
        cut_off_intensity_for_GD (float): a cut-off which sets the GD to zero for smaller intensities. (Needed for stable numerics)
        
    """

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, spectral_filter1=None, spectral_filter2=None, **kwargs):

        if "cross_correlation" in kwargs:
            if kwargs["cross_correlation"]=="doubleblind":
                raise KeyError("DirectReconstruction cannot work for Doubleblind-like methods")
            
        assert spectral_filter1!=None and spectral_filter2!=None, "For DirectReconstruction you need to provide spectral_filters, to get a ancillae-frequency."

        super().__init__(delay, frequency, measured_trace, nonlinear_method, spectral_filter1=spectral_filter1, spectral_filter2=spectral_filter2, **kwargs)

        self._name = "DirectReconstruction"

        self.integration_method = "euler_maclaurin_3"
        self.integration_order = None

        self.windowing = "hamming"
        self.cut_off_intensity_for_GD = 0.0001


    def apply_windowing(self, a0, signal, axis=-1):
        """ Applies a windowing on a signal along a given axis. """
        N = jnp.shape(signal)[axis]
        n = jnp.arange(N)
        window = a0 - (1-a0)*jnp.cos(2*jnp.pi*n/N)
        return jnp.swapaxes(jnp.swapaxes(signal, -1, axis)*window, axis, -1)
    

    def interpolate_group_delay_onto_spectral_amplitude(self, spectral_phase, measurement_info):
        """ The group delay is obtained on the wrong position on the frequency axis. This is solved by shifted/interpolation. """
        frequency = measurement_info.frequency

        idx_trace = jnp.argmax(jnp.mean(measurement_info.measured_trace, axis=0))
        idx_spectrum = jnp.argmax(measurement_info.spectral_amplitude.pulse)
        f_shift = frequency[idx_trace] - frequency[idx_spectrum]

        spectral_phase = do_interpolation_1d(frequency, frequency - f_shift, spectral_phase)
        return spectral_phase
    


    def reconstruct_2dsi_1dfft(self, descent_state, measurement_info, descent_info):
        """ Performs the standard 2DSI reconstruction by integrating over the group delay. 
        Which is obtained as the phase of an fft of the measured trace along the delay axis. """
        tau_arr, frequency, trace = measurement_info.tau_arr, measurement_info.frequency, measurement_info.measured_trace
        pulse_spectral_amplitude, anc1_frequency, anc2_frequency = measurement_info.spectral_amplitude.pulse, measurement_info.anc1_frequency, measurement_info.anc2_frequency

        use_windowing = descent_info.windowing
        if use_windowing!=False:
            trace_wind = self.apply_windowing(use_windowing, trace, axis=0)
        else:
            trace_wind = trace
        trace_f = jnp.fft.fftshift(jnp.fft.fft(trace_wind, axis=0), axes=0)

        frequency_tau = jnp.fft.fftshift(jnp.fft.fftfreq(jnp.size(tau_arr), jnp.mean(jnp.diff(tau_arr))))
        shear_frequency_mean = (anc1_frequency + anc2_frequency)/2
        idx = jnp.argmin(jnp.abs(frequency_tau - shear_frequency_mean))

        signal_abs = jnp.abs(trace_f)[idx]
        signal_angle = jnp.angle(trace_f)[idx]

        group_delay = jnp.unwrap(signal_angle)/(anc2_frequency - anc1_frequency)
        group_delay = jnp.where(signal_abs < measurement_info.cut_off_intensity*jnp.max(signal_abs), 0, group_delay)
        group_delay = group_delay - jnp.mean(group_delay)

        group_delay = self.interpolate_group_delay_onto_spectral_amplitude(group_delay, measurement_info)
        spectral_phase = integrate_signal_1D(group_delay, frequency, descent_info.integration_method, descent_info.integration_order)

        pulse_f = pulse_spectral_amplitude*jnp.exp(1j*spectral_phase)
        pulse_t = self.ifft(pulse_f, measurement_info.sk, measurement_info.rn)
        pulse_t = center_signal(pulse_t).reshape(1,-1)

        descent_state = tree_at(lambda x: x.group_delay, descent_state, group_delay)
        descent_state = tree_at(lambda x: x.spectral_phase, descent_state, spectral_phase)
        descent_state = tree_at(lambda x: x.population.pulse, descent_state, pulse_t)

        return descent_state
    
    

    def calc_error_of_reconstruction(self, descent_state, measurement_info, descent_info):
        """ Calculates the error of the reconstruction. """
        signal_t = self.generate_signal_t(descent_state, measurement_info, descent_info)
        trace = calculate_trace(signal_t.signal_f)
        # if population is larger than one this may cause an error or bug ? 
        trace_error = calculate_trace_error(trace, measurement_info.measured_trace)
        return trace_error
    


    def step(self, descent_state, measurement_info, descent_info):
        """ 
        Performs the reconstruction in one interation.

        Args:
            descent_state: Pytree,
            measurement_info: Pytree,
            descent_info: Pytree,
        
        Returns:
            tuple[Pytree, jnp.array], the updated descent state and the current errors

        """
        descent_state = self.reconstruct_2dsi_1dfft(descent_state, measurement_info, descent_info)
        trace_error = self.calc_error_of_reconstruction(descent_state, measurement_info, descent_info)
        return descent_state, jnp.asarray([trace_error, trace_error])

    

    def initialize_run(self, population):
        """
        Prepares all provided data and parameters for the reconstruction. 
        Here the final shape/structure of descent_state, measurement_info and descent_info are determined. 

        Args:
            population: Pytree, the initial guess as created by self.create_initial_population()
        
        Returns:
            tuple[Pytree, Callable], the initial descent state and the step-function of the algorithm.

        """
        assert self.descent_info.measured_spectrum_is_provided.pulse==True, "You need to provide a spectrum for the pulse."
        assert self.descent_info.population_size==1, "DirectReconstruction has no inherent randomness, so its not sensible to use or expect more than one result."

        self.measurement_info = self.measurement_info.expand(cut_off_intensity = self.cut_off_intensity_for_GD)
        measurement_info = self.measurement_info

        if self.integration_method[:-2]=="euler_maclaurin":
            self.integration_order = int(self.integration_method[-1])
            self.integration_method = "euler_maclaurin"

        a0_dict = {"hamming": 0.54, 
                   "hann": 0.5,
                   False: False}
            
        self.descent_info = self.descent_info.expand(integration_method = self.integration_method, 
                                                     integration_order = self.integration_order,
                                                     windowing = a0_dict[self.windowing])
        
        descent_info = self.descent_info
        
        init_arr = jnp.zeros(jnp.size(self.measurement_info.frequency))
        self.descent_state = self.descent_state.expand(population = population, 
                                                       group_delay = init_arr, 
                                                       spectral_phase = init_arr)

        do_scan = Partial(self.step, measurement_info=measurement_info, descent_info=descent_info)
        do_scan = Partial(scan_helper, actual_function=do_scan, number_of_args=1, number_of_xs=0)
        return self.descent_state, do_scan










class GeneralizedProjection(GeneralizedProjectionBASE, RetrievePulses2DSI):
    """
    The Generalized Projection Algorithm for 2DSI.

    """

    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)


    def calculate_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, population.pulse, signal_t.gate_pulses, signal_t.gate, tau_arr, measurement_info, pulse_or_gate)
        return grad


    def calculate_Z_newton_direction(self, grad, signal_t_new, signal_t, tau_arr, descent_state, measurement_info, descent_info, full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """
        descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, descent_state.population.pulse, signal_t.gate_pulses, signal_t.gate, 
                                                                         signal_t.signal_t, signal_t_new, tau_arr, measurement_info, 
                                                                         descent_state.newton, descent_info.newton, full_or_diagonal, pulse_or_gate)
        return descent_direction, newton_state


    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Updates an individual based on a descent direction and step size."""
        sk, rn = measurement_info.sk, measurement_info.rn

        pulse_f = self.fft(getattr(individual, pulse_or_gate), sk, rn)
        pulse_f = pulse_f + gamma*descent_direction
        pulse = self.ifft(pulse_f, sk, rn)

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, pulse)
        return individual









class PtychographicIterativeEngine(PtychographicIterativeEngineBASE, RetrievePulses2DSI):
    """
    The Ptychographic Iterative Engine (PIE) for 2DSI.
    Is not set up to be used for doubleblind. The PIE was not invented for reconstruction of interferometric signals.

    Attributes:
        pie_method (None, str): specifies the PIE variant. Can be one of None, PIE, ePIE, rPIE. Where None indicates that the pure gradient is used.

    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, pie_method="rPIE", **kwargs):
        assert cross_correlation!="doubleblind", "Doubleblind is not implemented for 2DSI-PtychographicIterativeEngine."
        
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)

        self.pie_method = pie_method


    # def reverse_transform_grad(self, signal, tau_arr, measurement_info):
    #     frequency, time = measurement_info.frequency, measurement_info.time
    #     signal = self.calculate_shifted_signal(signal, frequency, -1*tau_arr, time, in_axes=(0, 0, None, None, None))
    #     return signal

    # def modify_grad_for_gate_pulse(self, grad_all_m, gate_pulse_shifted, nonlinear_method):
    #     pass


    def calculate_PIE_descent_direction_m(self, signal_t, signal_t_new, tau, measured_trace, population, pie_method, measurement_info, descent_info, pulse_or_gate):
        """ Calculates the PIE direction for a given shift. """
        alpha = descent_info.alpha
        difference_signal_t = signal_t_new - signal_t.signal_t

        probe = signal_t.gate
        grad = -1*jnp.conjugate(probe)*difference_signal_t
        U = jax.vmap(self.get_PIE_weights, in_axes=(0,None,None))(probe, alpha, pie_method)
        
        return grad*U
    

    def update_individual(self, individual, gamma, descent_direction, measurement_info, pulse_or_gate):
        """ Updates an individual based on a descent direction and step size. """
        signal = getattr(individual, pulse_or_gate)
        signal = signal + gamma*descent_direction

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal)
        return individual


    # def get_gate_probe_for_hessian(self, pulse_t, gate_pulse_shifted, nonlinear_method):
    #     pass


    def calculate_PIE_newton_direction(self, grad, signal_t, tau_arr, measured_trace, population, local_or_global_state, measurement_info, descent_info, 
                                       pulse_or_gate, local_or_global):
        
        """ Calculates the PIE newton direction for a population. """
        
        newton_direction_prev = getattr(local_or_global_state.newton, pulse_or_gate).newton_direction_prev
        probe = signal_t.gate

        reverse_transform = None
        # signal_f = self.fft(signal_t.signal_t, measurement_info.sk, measurement_info.rn)
        descent_direction, newton_state = PIE_get_pseudo_newton_direction(grad, probe, signal_t.signal_f, tau_arr, measured_trace, reverse_transform, newton_direction_prev, 
                                                                     measurement_info, descent_info, pulse_or_gate, local_or_global)
        return descent_direction, newton_state
    
    










class COPRA(COPRABASE, RetrievePulses2DSI):
    """
    The Common Pulse Retrieval Algorithm (COPRA) for 2DSI.
    
    """
    def __init__(self, delay, frequency, measured_trace, nonlinear_method, cross_correlation=False, **kwargs):
        super().__init__(delay, frequency, measured_trace, nonlinear_method, cross_correlation=cross_correlation, **kwargs)


    def update_individual(self, individual, gamma, descent_direction, measurement_info, descent_info, pulse_or_gate):
        """ Updates an individual via a descent direction and a step size. """
        sk, rn = measurement_info.sk, measurement_info.rn

        signal = getattr(individual, pulse_or_gate)
        signal_f = self.fft(signal, sk, rn)
        signal_f = signal_f + gamma*descent_direction
        signal = self.ifft(signal_f, sk, rn)

        individual = tree_at(lambda x: getattr(x, pulse_or_gate), individual, signal)
        return individual



    def get_Z_gradient_individual(self, signal_t, signal_t_new, population, tau_arr, measurement_info, pulse_or_gate):
        """ Calculates the Z-error gradient for an individual. """
        grad = calculate_Z_gradient(signal_t.signal_t, signal_t_new, population.pulse, signal_t.gate_pulses, signal_t.gate, tau_arr, measurement_info, pulse_or_gate)
        return grad



    def get_Z_newton_direction(self, grad, signal_t, signal_t_new, tau_arr, population, local_or_global_state, measurement_info, descent_info, 
                                           full_or_diagonal, pulse_or_gate):
        """ Calculates the Z-error newton direction for a population. """
        
        newton_state = local_or_global_state.newton
        descent_direction, newton_state = get_pseudo_newton_direction_Z_error(grad, population.pulse, signal_t.gate_pulses, signal_t.gate, 
                                                                         signal_t.signal_t, signal_t_new, tau_arr, measurement_info, 
                                                                         newton_state, descent_info.newton, full_or_diagonal, pulse_or_gate)
        return descent_direction, newton_state