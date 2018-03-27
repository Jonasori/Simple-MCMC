"""
Basic MCMC pt. 2: Fit the fake data with emcee

Develop a simple MCMC code to fit to data generated in utils.py
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import parabola, generate_fake_data, chi2
from utils import truncated_random_normal

# Acceptance fractions independently


# Calculate Chi-Squared
def chi2(data, model, sigma):
    # Takes two lists.
    if len(data) != len(model):
        return "Bad! No Good!"

    c = 0
    for i in range(len(data)):
        c += (data[i]-model[i])**2 * sigma**(-2)
        # print data[i], model[i], '\t', (data[i]-model[i])**2/sigma
    return c




# Choose a starting point
true_a = 1
true_b = 50.1
true_sigma = 10
# Generate some data
xs = np.arange(-20, 20)
ys = generate_fake_data(xs, true_a, true_b, true_sigma)

# Specify priors
priors_a = [-100, 100]
priors_b = [-100, 100]


def run_emcee():
    







# A generic plotter
def plot_whatever(xs, ys):
    plt.plot(xs, ys, '-og')
    plt.show(block=False)


# Plot it in parameter space
def plot_param_walk(run_output):
    nsteps = run_output['nsteps']
    burn_in_len = int(0.2 * nsteps)
    burn_in_len = 5000
    print burn_in_len
    a_vals = run_output['a_vals_visited'][burn_in_len:]
    b_vals = run_output['b_vals_visited'][burn_in_len:]
    plt.plot(a_vals, b_vals, '.k', alpha=0.01)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.show(block=False)


# Plot the resulting parabola
def plot_best_fit_model(run_output):
    a_vals = run_output['a_vals_visited']
    b_vals = run_output['b_vals_visited']
    chisqs = run_output['chi2_vals']

    min_vals = np.where(run_output['chi2_vals'] == np.min(run_output['chi2_vals']))

    #model_ys = parabola(xs, a_vals[min_vals[0,0]], b_vals[min_vals[0,1]])
    model_ys = parabola(xs, a_vals[min_vals[0][0]], b_vals[min_vals[0][1]])
    plt.plot(xs, model_ys, '-og')
    plt.plot(xs, ys, '-or')
    plt.show(block=False)


















# The End
