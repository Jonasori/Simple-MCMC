"""
Experimenting with emcee, the MCMC Hammer
March 2018

EnsembleSampler docs: http://dfm.io/emcee/current/api/#emcee.EnsembleSampler.run_mcmc
"""

#Note: learn correct docstring stuff


import numpy as np
import emcee


# From the docs:
def lnprob(x, ivar):
    return -0.5 * np.sum(ivar * x ** 2)


ndim, nwalkers = 10, 100
ivar = 1. / np.random.rand(ndim)
p0 = [np.random.rand(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[ivar])
pos, prob, state = sampler.run_mcmc(p0, 1000)
