"""
Basic MCMC pt. 2: Fit the fake data.

Develop a simple MCMC code to fit to data generated in generate_fake_data.py
"""

import numpy as np
import matplotlib.pyplot as plt

from generate_fake_data import generate_fake_data

a = 10
b = 5
# Leave C at 0 for the moment so I only have to MCMC over two params
c = 0
noise = 100

# Generate some data
xs = np.arange(-20, 20)
d = generate_fake_data(xs, a, b, c, noise)

# Test plot if you want
"""
plt.plot(xs, d)
plt.show(block=False)
"""


# Start the actual MCMC
# Set initial walker positions:
a_init, b_init = 1, 2
