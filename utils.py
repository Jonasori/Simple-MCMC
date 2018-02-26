"""
Basic MCMC Pt. 1: Some Tools.

Generate a noisy parabola, and other tools to be used in fit_data.py
"""

# Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


# Make a Parabola from a list of xs
def parabola(xs, a, b):
    xs_np = np.array(xs)
    ys = a*xs_np**2 + b
    return ys


parabola([1, 2, 3, 4, 5, 6], 1, 0)


# Generate some a noisy parabola with parameters a, b, and c
def generate_fake_data(xs, a, b, noise):
    ys_clean = parabola(xs, a, b)
    ys_dirty = [np.random.normal(loc=y, scale=noise, size=1) for y in ys_clean]

    return ys_dirty


def generate_fake_data_2(xs, a, b, noise):
    dirty_a = np.random.normal(loc=a, scale=noise, size=1)
    dirty_b = np.random.normal(loc=b, scale=noise, size=1)
    ys_dirty = parabola(xs, dirty_a, dirty_b)
    return ys_dirty


# Calculate Chi-Squared
def chi2(data, model, sigma):
    # Takes two lists.
    if len(data) != len(model):
        return "Bad! No Good!"

    c = 0
    for i in range(len(data)):
        c += (data[i]-model[i])**2 * sigma**(-2)
        # print data[i], model[i], '\t', (data[i]-model[i])**2/sigma
        print c

    return c


# Check if a step is within the priors
# This is now redundant thanks to the truncated_random_normal function
def is_valid_step(step, priors):
    # Each argument is a tuple
    step_ok = priors[0] < step[0] and step[0] < priors[1]
    # Do the logics
    if step_ok:
        return True
    else:
        return False


# Calculate acceptance fraction:
def calculate_acceptance_fraction():
    blah = 'blah'
    return blah


# Truncated Normal Distribution
# Pulled from:
# stackoverflow -> how-to-get-a-normal-distribution-within-a-range-in-numpy
def truncated_random_normal(mean, sd, low, upp, N=1):
    """
    - To be used instead of np.random.normal and then binding that.
    - N defaults to 1 to only return one value.
        - Note that this means its a list, so be sure to drop an [0] on the end
    To verify that this yields a normal distribution:
        ys = truncated_random_normal(0, 1, -10, 10, 1000)
        plt.hist(ys); plt.show()
    """
    X = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    return X.rvs(N)





# TESTING AREA
def test_plots(xs, ys):
    plt.plot(xs, ys, '-o')
    plt.show()


ms = np.arange(1, 10)
ds = [np.random.normal(loc=m) for m in ms]



# More testing Stuff
def more_testing_plots():
    c2 = chi2(ms, ds)
    plt.plot(ms, ms, '-og')
    plt.plot(ms, ds, '-or')
    plt.plot(ms, c2, 'ob')
    plt.show()


# The End
