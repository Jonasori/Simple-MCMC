"""
Basic MCMC pt. 2: Fit the fake data.

Develop a simple MCMC code to fit to data generated in utils.py
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import parabola, generate_fake_data, chi2
from utils import truncated_random_normal

# Choose a starting point
true_a = 10.1
true_b = 5.1
true_sigma = 100
# Generate some data
xs = np.arange(-20, 20)
ys = generate_fake_data(xs, true_a, true_b, true_sigma)

# Specify priors
priors_a = [-50, 50]
priors_b = [-50, 50]


def mcmc(xs, ys, priors_a, priors_b, sigma=1, nsteps=1000):
    # nsteps: how many steps the walkers should take
    # xs, ys: input data (lists/numpy arrays)
    # priors_a, priors_b: bounds on where the walkers can go
    # sigma: noise level

    # Test
    
    # Initialize the arrays
    a_vals = np.zeros(nsteps)
    b_vals = np.zeros(nsteps)
    chisqs = np.zeros(nsteps)

    # Give a starting point and calculate that initial chi2
    initial_a = np.random.uniform(priors_a[0], priors_a[1])
    initial_b = np.random.uniform(priors_b[0], priors_b[1])
    a_vals[0], b_vals[0] = initial_a, initial_b

    first_model = parabola(xs, initial_a, initial_b)
    chisqs[0] = chi2(ys, first_model, sigma)

    # Start the loop!
    i = 1
    while i < nsteps:
        print i
        # Propose a new step
        choice = np.random.choice(['a', 'b'], 1)
        print choice
        if choice == 'a':
            # a_new = np.random.normal(loc=a_vals[i-1], scale=sigma)
            a_new = truncated_random_normal(a_vals[i-1],
                                            sigma, priors_a[0],
                                            priors_a[1])[0]

            new_step = np.array([a_new, b_vals[i-1]])
        else:
            # b_new = np.random.normal(loc=b_vals[i-1], scale=sigma)
            b_new = truncated_random_normal(b_vals[i-1],
                                            sigma, priors_b[0],
                                            priors_b[1])[0]

            new_step = np.array([a_vals[i-1], b_new])

        print new_step
        # Calculate Chi-Squared for that new step
        model_new = parabola(xs, new_step[0], new_step[1])
        chisq_new = chi2(ys, model_new, sigma)

        # Make sure the new step doesn't violate the priors
        # Note that if this fails, the loop starts over the with same i value
        # if is_valid_step(new_step, priors_a) and is_valid_step(new_step, priors_b):

        # If the new one is an improvement, take it.
        if chisq_new < chisqs[i-1]:
            a_vals[i], b_vals[i] = new_step[0], new_step[1]
            chisqs[i] = chisq_new

        # Otherwise, generate a random number in [0,1]
        else:
            r_num = np.random.random()
            # delta_chisq = chisq_old - chisq_new from eqn(13) of Ford 2005
            delta_chisq = (chisqs[i-1] - chisq_new)
            alpha = np.exp(delta_chisq/2)

            # If alpha < random, reject this step.
            # It feels weird that we reject if alpha is smaller.
            if alpha < r_num:
                a_vals[i], b_vals[i] = a_vals[i-1], b_vals[i-1]
                chisqs[i] = chisqs[i-1]

            # If alpha > random, accept the new step.
            else:
                a_vals[i], b_vals[i] = new_step[0], new_step[1]
                chisqs[i] = chisq_new

        # Bump the counter
        i += 1

        print '\n\n'

    # Final outputs
    # best_fit_index = chisqs.index(min(chisqs))
    # best_fit_vals = [a_vals[best_fit_index], b_vals[best_fit_index]]
    final_output = {
                    'a_vals_visited': a_vals,
                    'b_vals_visited': b_vals,
                    'chi2_vals': chisqs,
                    }

    # 'best_fit_vals': best_fit_vals

    return final_output


# A generic plotter
def plot_whatever(xs, ys):
    plt.plot(xs, ys, '-og')
    plt.show(block=False)


# Plot it in parameter space
def plot_param_walk(a_vals, b_vals):
    plt.plot(a_vals, b_vals, '.k', alpha=0.1)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.show(block=False)


# Plot the resulting parabola
def plot_best_fit_model(chisqs):
    best_fit_index = chisqs.index(min(chisqs))
    ys = parabola(xs, coeffs[-1][0], coeffs[-1][1])
    plt.plot(xs, ys)
    plt.show(block=False)


















# The End
