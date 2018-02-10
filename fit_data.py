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
noise = 1

# Generate some data
xs = np.arange(-20, 20)
ds = utils.generate_fake_data(xs, a, b, noise)

# Test plot if you want
"""
plt.plot(xs, ds)
plt.show(block=False)
#"""


# Start the actual MCMC


sig = [1, 1]
nsteps = 10000
# The a and b values [a, b]
coeffs = [[10, 5]]

for i in range(nsteps):

    # Randomly choose which parameter to move
    # Use this as an index for the coeffss list
    # Note that p and p_other will always be
    p = r.randint(2)
    p_other = 1 if p == 0 else 0
    p
    # Propose a new step in PARAMETER SPACE
    new_step = np.random.normal(loc=coeffs[-1][p], scale=sig[p])

    # Quickly figure out which is which
    if p < p_other:
        a_old = coeffs[-1][p]
        b_old = coeffs[-1][p_other]
        a_new = new_step
        b_new = coeffs[-1][p_other]
    else:
        a_old, b_old = coeffs[-1][p_other], coeffs[-1][p]
        a_new, b_new = coeffs[-1][p_other], new_step

    b_new
    # Calcualte Chi-Squared for new step, store it
    model_old = utils.parabola(xs, a_old, b_old)
    chisq_old = utils.chi2(ds, model_old)

    chisq_old
    # Calcualte Chi-Squared for new step, store it
    model_new = utils.parabola(xs, a_new, b_new)
    chisq_new = utils.chi2(ds, model_new)


    chisq_new
    # If chi2_new < chi2_old, accept.
    if chisq_new > chisq_old:
        # This can probably be made shorter.
        new_coeffs = [0, 0]
        new_coeffs[p] = [a, b]
        new_coeffs[p_other] = coeffs[-1][p_other]
        coeffs.append(new_coeffs)

    # Else, generate a random number in [0,1]
    else:
        r_num = r.random()
        # Is this order right? delta_chisq = chisq_old - chisq_new
        delta_chisq = (chisq_new - chisq_old)

        ex = np.exp(-(delta_chisq)/2)

        # If exp(-dChi2/2) < random, reject
        if ex < r_num:
            coeffs.append(coeffs[-1])

        # Else, accept
        else:
            new_coeffs = [0, 0]
            new_coeffs[p] = [a, b]
            new_coeffs[p_other] = coeffs[-1][p_other]
            coeffs.append(new_coeffs)


# Plot it in parameter space
plt.plot(coeffs, '.k', alpha=0.1)
plt.xlabel('a')
plt.ylabel('b')
plt.show(block=False)

# Plot the resulting parabola
"""
ys = utils.parabola(xs, coeffs[-1][0], coeffs[-1][1])
plt.plot(xs, ys)
plt.show(block=False)
"""


















# The End
