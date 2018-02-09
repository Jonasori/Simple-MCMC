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
    ys = [a*(x**2) + b for x in xs]
    return ys


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


ms = np.arange(1, 10); len(ms)
ds = [np.random.normal(loc=m) for m in ms]; len(ds)


# Calculate Chi-Squared
def chi2(model, data):
    # Takes two lists
    chisq = 0
    #[chisq += (d - m)**2 / m for m, d in zip(model, data)]

    for m, d in zip(model, data):
        chisq += (d-m)**2 / m

    return chisq

# More testing Stuff
"""
c2 = chi2(ms, ds)
plt.plot(ms, ms, '-og')
plt.plot(ms, ds, '-or')
plt.plot(ms, c2, 'ob')
plt.show()
"""

# The End
