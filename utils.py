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
    ys = []
    [ys.append(b + a * x**2) for x in xs]
    return ys

parabola([1,3,4,5,6], 1, 10)

# Generate some a noisy parabola with parameters a, b, and c
def generate_fake_data(xs, a, b, noisiness):
    ys_clean = parabola(xs, a, b)
    ys_noisy = [np.random.normal(loc=y, scale=noisiness, size=1) for y in ys_clean]
    return ys_noisy


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
    # Want numpy arrays not tuples
    # val = np.sum((data - model)**2 / sigma**2)

    if len(data) != len(model):
        return "Bad! No Good!"

    val = 0
    # Bad Jonas
    for i in range(len(data)):
        val += (data[i] - model[i])**2 / sigma**2


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
