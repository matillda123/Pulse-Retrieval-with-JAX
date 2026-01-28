from pulsedjax.utilities import get_score_values
from pulsedjax.frog import COPRA
from pulsedjax.simulate_trace import MakeTrace
from pulsedjax.simulate_trace import GaussianAmplitude, PolynomialPhase
import jax.numpy as jnp

from bayesian_optimization import initialize_client, bayes_opt_run


def make_trace(gdd, tod):
    amp0 = GaussianAmplitude((1,1,1), (0.175,0.2,0.25), (0.01,0.05,0.1), (1,2,1))
    phase0 = PolynomialPhase(None, (0,0,gdd,tod))

    mp = MakeTrace(N=128*20, f_max=2)
    time, frequency, pulse_t, pulse_f = mp.generate_pulse((amp0,phase0))
    input_pulses = mp.pulses

    delay = jnp.linspace(-100,100,128)
    delay, frequency_trace, trace, spectra = mp.generate_frog(time, frequency, pulse_t, pulse_f, "shg", delay,
                                                            N=128, frequency_range=(0.1,0.5), interpolate_fft_conform=True,
                                                            plot_stuff=False)
    return delay, frequency_trace, trace, input_pulses
        



def run_retrieval(delay, frequency_trace, trace, local_gamma, global_gamma):
    copra = COPRA(delay, frequency_trace, trace, "shg")

    population = copra.create_initial_population(5, "random")
    copra.local_gamma = local_gamma
    copra.global_gamma = global_gamma

    final_result = copra.run(population, 50, 150)
    return final_result.population



def eval_retrieval(output_pulses, input_pulses):
    temp = []
    for i in range(5):
        # test both temporal directions because of shg-ambiguity
        op = output_pulses[i]
        score_val_0 = get_score_values(op, input_pulses)[0]
        op.pulse_t = jnp.flip(op.pulse_t)
        score_val_1 = get_score_values(op, input_pulses)[0]

        if jnp.isnan(score_val_0)==True:
            score_val_0 = 1
        if jnp.isnan(score_val_1)==True:
            score_val_1 = 1
        
        score_val = jnp.minimum(score_val_0, score_val_1)
        temp.append(score_val)

    return jnp.mean(jnp.asarray(temp))




def run_and_eval_once(local_gamma, global_gamma, gdd, tod):
    delay, frequency_trace, trace, input_pulses = make_trace(gdd, tod)
    output_pulses = run_retrieval(delay, frequency_trace, trace, local_gamma, global_gamma)
    score_average = eval_retrieval(output_pulses, input_pulses)
    return score_average




def run_and_eval_all_parameters(local_gamma, global_gamma):
    gdd_arr = jnp.linspace(0,50,10)
    tod_arr = jnp.linspace(0,50,10)
    temp = []
    for gdd in gdd_arr:
        for tod in tod_arr:
            score_average = run_and_eval_once(local_gamma, global_gamma, gdd, tod)
            temp.append(score_average)

    return jnp.mean(jnp.asarray(temp)).item()
    



if __name__ == "__main__":
    client = initialize_client(["local_gamma", "global_gamma"], ["float","float"], [(0,2),(0,2)], run_and_eval_all_parameters)
    client, best_params = bayes_opt_run(client, myfunc=run_and_eval_all_parameters, max_trials=3, max_iterations=20)
    client.save_to_json_file("final_client.json")