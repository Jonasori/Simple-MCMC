"""
Basic MCMC Pt. 1: Some Tools.

Generate a noisy parabola
"""

# Packages
import numpy as np
import matplotlib.pyplot as plt


# Parabola parameters
a = 10
b = 30


# Make a Parabola from a list of xs
def parabola(xs, a, b):
    xs_np = np.array(xs)
    ys = a*xs_np**2 + b
    return ys


parabola([1, 2, 3, 4, 5, 6], 1, 0)


# Generate some a noisy parabola with parameters a, b, and c
def generate_fake_data(xs, a, b, noisiness):
    ys_clean = parabola(xs, a, b)
    # Should probably be able to do this with list ops
    ys_noisy = [np.random.normal(loc=y, scale=noisiness, size=1) for y in ys_clean]
    return ys_noisy


# Check if a step is within the priors
def is_valid_step(step, priors_a, priors_b):
    # Each argument is a tuple
    if not priors_a[0] < step[0] < priors_a[1]:
        return False
    if not priors_b[0] < step[1] < priors_b[1]:
        return False
    else:
        return True

# Testing Stuff
"""
xs = np.arange(-10, 10)
ys = generate_fake_data(xs, a, b, 30.)

print len(xs)
print len(ys)

plt.plot(xs, ys, '-o')
plt.show()
"""


ms = np.arange(1, 10)
ds = [np.random.normal(loc=m) for m in ms]


# Calculate Chi-Squared
def chi2(data, model, sigma):
    # Takes two lists.
    if len(data) != len(model):
        return "Bad! No Good!"

    val = np.sum((np.array(data) - np.array(model))**2 / sigma**2)
    return val

chi2([1,2,3,4], [1.1, 2.1, 3.1, 4.1], 0.1)

# More testing Stuff
"""
c2 = chi2(ms, ds)
plt.plot(ms, ms, '-og')
plt.plot(ms, ds, '-or')
plt.plot(ms, c2, 'ob')
plt.show()
"""

# The End
