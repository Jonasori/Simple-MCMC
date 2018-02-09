"""
Basic MCMC pt. 2: Fit the fake data.

Develop a simple MCMC code to fit to data generated in utils.py
"""

import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt

import utils

a = 10
b = 5
# Leave C at 0 for the moment so I only have to MCMC over two params
c = 0
noise = 100

# Generate some data
xs = np.arange(-20, 20)
ds = utils.generate_fake_data(xs, a, b, noise)

# Test plot if you want
"""
plt.plot(xs, ds)
plt.show(block=False)
#"""


# Start the actual MCMC
# Set initial walker positions:
a_init, b_init = 1, 2
sig_a, sig_b = 3, 3
proposed_new_step_a = np.random.normal(loc=a_init, scale=sig_a)
proposed_new_step_b = np.random.normal(loc=b_init, scale=sig_b)


chisq_a_old = utils.chi2(utils.parabola(
                        xs,
                        proposed_new_step_a,
                        b_init),
                        ds)

chisq_a_new = utils.chi2(utils.parabola(
                        xs,
                        proposed_new_step_a,
                        b_init),
                        ds)


if chisq_a_new < chisq_a_old:
    #accept
    blah = 1

else:
    accept_metric_a = np.exp(-chisq_a)
    random_test_range = np.random.random()
    if accept_metric_a < random_test_range:
        # reject
    else:
        # accept

nsteps = 100
position = [[1,1]]
for i in range(nsteps):

    # Randomly choose which parameter to move
    # Use this as an index for the positions list
    p = r.randint(2)
    p_other = 1 if p==0 else 0

    # Propose a new step
    new_pos = np.random.normal(loc=position[-1][p], scale=sig[p])




    # Calcualte Chi-Squared for new step, store it

    chisq_a_old = utils.chi2(
                    utils.parabola(xs, new_pos, pos[-1][p_other]), ds)

    # If chi2_new < chi2_old, accept.
    # Else, generate a random number in [0,1]

    # If exp(-chi2_new/2) < random, reject
    # Else, accept
















# The End
