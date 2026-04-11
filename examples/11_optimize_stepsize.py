"""
This example shows how the local and global stepsize of the COPRA implementation can be optimized 
using optax/optimistix interactively.
In principle this approach should work with any float variable of any algorithm. But thats untested. 
In addition the blind optimization of variables without knowledge on their function, properties and sensible ranges is risky.
"""



# Generate SHG-FROG trace
from pulsedjax.simulate_trace import MakeTrace
from pulsedjax.simulate_trace import GaussianAmplitude, PolynomialPhase
import jax.numpy as jnp

amp0 = GaussianAmplitude((1,0.9), (0.2,0.25), (0.05,0.075), (1,1))
phase0 = PolynomialPhase(None, (0,0,50,5))

mp = MakeTrace(N=128*10, f_max=1)
time, frequency, pulse_t, pulse_f = mp.generate_pulse((amp0,phase0))

delay = jnp.linspace(-100,100,128)
delay, frequency_trace, trace, spectra = mp.generate_frog(time, frequency, pulse_t, pulse_f, "shg", delay,
                                                        N=128, frequency_range=(0.1,0.5), interpolate_fft_conform=True)






from pulsedjax.frog import COPRA
import jax
import equinox as eqx
import optimistix
import optax

# create optimistix interactive-solve compatible loss-function
@eqx.filter_jit
def run_copra(gamma, args):
    local_gamma, global_gamma = gamma
    copra = COPRA(delay, frequency_trace, trace, "shg", seed=0) # set a seed for reproducability
    copra.local_gamma = 10**local_gamma # uses exponent since local gamma usually needs to be quite large
    copra.global_gamma = global_gamma
    population = copra.create_initial_population(5, "random")
    final_result = copra.run(population, 15, 15) # only for a small number of stpes since this is expensive

    error = jnp.mean(final_result.error_arr[-1])
    return error, error # return the error twice, once for the optimization, once as aux-data




# set up the interactive solve
solver = optimistix.OptaxMinimiser(optax.adam(learning_rate=0.01), rtol=1e-6, atol=1e-6)
y0 = jnp.array([1.0, 0.1])
args = None
options = dict()
f_struct = jax.ShapeDtypeStruct((), jnp.float32)
aux_struct = jax.ShapeDtypeStruct((), jnp.float32)
tags = frozenset()


# set up a function for an interactive solve
def solve(y, solver, fn, max_steps):
    step = eqx.filter_jit(eqx.Partial(solver.step, fn=fn, args=args, options=options, tags=tags))
    terminate = eqx.filter_jit(eqx.Partial(solver.terminate, fn=fn, args=args, options=options, tags=tags))

    state = solver.init(fn, y, args, options, f_struct, aux_struct, tags)
    aux_arr = []
    for _ in range(max_steps):
        y, state, aux = step(y=y, state=state)
        aux_arr.append(aux)

    _, result = terminate(y=y, state=state)
    y, fn_y, _ = solver.postprocess(fn, y, aux, args, options, state, tags, result)
    return y, fn_y, aux_arr


# run the optimization of gamma
y, fn_y, aux_arr = solve(y0, solver, run_copra, 25)



import matplotlib.pyplot as plt
plt.title("Average Error each iteration")
plt.plot(aux_arr)
plt.show()

print("Final stepsizes:", y)