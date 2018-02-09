"""
Basic MCMC Pt. 1: Generating Fake Data

Generate a noisy parabola
"""

# Packages
import numpy as np
import matplotlib.pyplot as plt


# Parabola parameters
a = 1
b = 30
c = 1


# Generate some a noisy parabola with parameters a, b, and c
def generate_fake_data(xs, a, b, c, noisiness):
    # Initialize the final array
    ys = []
    for x in xs:
        y_clean = a*(x - b)**2 + c
        y = np.random.normal(loc=y_clean, scale=noisiness, size=1)
        ys.append(y)
    return ys


# Testing Stuff
"""
xs = np.arange(-10, 100)
ys = generate_fake_data(xs, a, b, c)

print len(xs)
print len(ys)

plt.plot(xs, ys, '-o')
plt.show()
"""
# The End
