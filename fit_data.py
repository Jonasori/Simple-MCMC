"""
Basic MCMC pt. 2: Fit the fake data.

Develop a simple MCMC code to fit to data generated in utils.py
"""

import numpy as np
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

chisq_a = utils.chi2(utils.parabola(
                        xs,
                        proposed_new_step_a,
                        b_init),
                        ds)

chisq_b = utils.chi2([proposed_new_step_b], [b_init])
chisq_a
