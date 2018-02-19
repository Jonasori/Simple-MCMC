"""
Basic MCMC pt. 2: Fit the fake data.

Develop a simple MCMC code to fit to data generated in utils.py
"""

import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt
import utils

# Choose a starting point
a_init = 10
b_init = 5

# Add some noise
sigma = 100

# Generate some data
xs = np.arange(-20, 20)
ds = utils.generate_fake_data(xs, a_init, b_init, sigma)

# Specify priors
priors_a = range
priors_b = [-20, 20]


# Start the actual MCMC

nsteps = 10000
# The a and b values [a, b]
a_vals = np.zeros(nsteps)
b_vals = np.zeros(nsteps)
chisqs = np.zeros(nsteps)

# Give a starting point and calculate that initial chi2
a_vals[0], b_vals[0] = a_init, b_init
first_model = utils.parabola(xs, a_init, b_init)
chisqs[0] = utils.chi2(ds, first_model, sigma)


for i in range(1, nsteps):

    # Randomly choose which parameter to move
    # Propose a new step: choose a dimension to move in
    choice = r.choice(['a', 'b'], 1)
    if choice == 'a':
        a_new = np.random.normal(loc=a_vals[i-1], scale=sigma)
        new_step = np.array([a_new, b_vals[i-1]])
    else:
        b_new = np.random.normal(loc=b_vals[i-1], scale=sigma)
        new_step = np.array([a_vals[i-1], b_new])

    # Calculate Chi-Squared for new step, store it
    model_new = utils.parabola(xs, new_step[0], new_step[1])
    chisq_new = utils.chi2(ds, model_new, sigma)

    # Make sure the new step doesn't violate the priors:
    if utils.is_valid_step(new_step, priors_a, priors_b):

        # If the new one is an improvement, take it.
        if chisq_new > chisqs[i-1]:
            a_vals[i], b_vals[i] = new_step[0], new_step[1]
            chisqs[i] = chisq_new

        # Otherwise, generate a random number in [0,1]
        else:
            r_num = r.random()
            # delta_chisq = chisq_old - chisq_new from eqn(13) of Ford 2005
            delta_chisq = (chisqs[i-1] - chisq_new)
            alpha = np.exp(delta_chisq/2)

            # If exp(-dChi2/2) < random, reject.
            # It feels weird that we reject if ex is smaller.
            if alpha < r_num:
                a_vals[i], b_vals[i] = a_vals[i-1], b_vals[i-1]
                chisqs[i] = chisqs[i-1]

            # Else, accept
            else:
                a_vals[i], b_vals[i] = new_step[0], new_step[1]
                chisqs[i] = chisq_new


# Plot it in parameter space
plt.plot(a_vals, b_vals, '.k', alpha=0.1)
plt.xlabel('a')
plt.ylabel('b')
plt.show(block=False)

# Plot the resulting parabola
"""
best_fit_index =
ys = utils.parabola(xs, coeffs[-1][0], coeffs[-1][1])
plt.plot(xs, ys)
plt.show(block=False)
"""

















# The End
