import numpy as np
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

import torch


def hartmann6(x1, x2, x3, x4, x5, x6):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = 10**-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])

    outer = 0.0
    for i in range(4):
        inner = 0.0
        for j, x in enumerate([x1, x2, x3, x4, x5, x6]):
            inner += A[i, j] * (x - P[i, j])**2
        outer += alpha[i] * np.exp(-inner)
    return -outer



def initialize_client(name, parameter_type, bounds, myfunc, scaling=None):
    client = Client()
    parameters = [RangeParameterConfig(name=name[i], parameter_type=parameter_type[i], bounds=bounds[i], scaling=scaling) 
                  for i in range(len(name))]
    client.configure_experiment(parameters=parameters)

    objective = f"-{str(myfunc.__name__)}"
    client.configure_optimization(objective=objective)
    return client




def bayes_opt_step(client, max_trials, myfunc, observation_noise):
    trials = client.get_next_trials(max_trials=max_trials)

    for trial_index, parameters in trials.items():
        kwargs = parameters
        result = myfunc(**kwargs)
        result = np.asarray(result).item()
        noise = observation_noise*np.abs(result)
        result = (result + np.random.normal(0, noise, size=np.shape(result)), noise)
        raw_data = {str(myfunc.__name__): result}
        client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        
    return client



def bayes_opt_run(client, max_iterations=10, max_trials=5, myfunc=None, observation_noise=0):
    for _ in range(max_iterations):
        client = bayes_opt_step(client, max_trials, myfunc, observation_noise)
    return client, client.get_best_parameterization()





def get_observed_data(client):
    params = np.asarray([list(completed_trials.arm.parameters.values()) for completed_trials in client._experiment.completed_trials])
    observed = client._experiment.lookup_data().df["mean"].to_numpy()
    return params, observed


def probe_model(Xs, client):
    posterior = client._generation_strategy.model.model.surrogate.model.posterior

    Xs = torch.tensor(np.asarray(Xs))
    Ys = posterior(Xs)
    return Ys.mean.detach().numpy().squeeze(), Ys.variance.detach().numpy().squeeze()




if __name__=="__main__":
    # hartmann6(0.1, 0.45, 0.8, 0.25, 0.552, 1.0)

    name = ["x1","x2","x3","x4","x5","x6"]
    parameter_type = ["float","float","float","float","float","float"]
    bounds = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]

    client = initialize_client(name, parameter_type, bounds, hartmann6)
    res = bayes_opt_run(client, myfunc=hartmann6)
    print(res)