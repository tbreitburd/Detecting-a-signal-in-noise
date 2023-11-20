# Code these expressions for the probability density in python, in any way you so
# desire. Plugging in some different values for the parameters 𝜽, use a numerical
# integration to convince yourself that the total probability density integrates to unity
# in the range 𝑀 ∈ [5, 5.6]. [2]
# "
# You may use predefined methods in libraries such as scipy.stats and
# scipy.integrate to help you. 

import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt


# First, we define the expression for the probability density function derived in (b)

def N(alpha, beta, mu, sigma, lam, f):
    return f*((1/2)*integrate.quad(lambda x: np.exp(-x**2), (alpha-mu)/(sigma*np.sqrt(2)), (beta-mu)/(sigma*np.sqrt(2)))) + (1-f)*(np.exp(-lam*alpha)-np.exp(-lam*beta))


def pdf(M, mu, sigma, lam):
    return N*(M**(-2.35)) * (1 + (M / ))**(-2.35)