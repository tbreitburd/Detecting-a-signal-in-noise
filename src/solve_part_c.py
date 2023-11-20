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


def pdf(M,mu, sigma, lam, f, alpha, beta):
    """
    This function defines the normalised probability density function of M,
    using the normalisation factor derived in (b)
    within the limits alpha and beta.
    ----------------------------
    Inputs:
    M: r.v. M, float (when calculating a probability for a specific value of M)
    alpha: lower limit of M, float
    beta: upper limit of M, float
    mu: mean of signal (normal) component, float, must be within alpha and beta
    sigma: width of signal (normal) component, float, must be positive
    lam: decay parameter of background (exponential) component, float
    f: ratio of signal/background components, float
    """

    
    lim_l = (alpha-mu)/(sigma*np.sqrt(2))
    lim_h = (beta-mu)/(sigma*np.sqrt(2))
    I,_ = integrate.fixed_quad(lambda x: np.exp(-x**2), lim_l, lim_h, n=10) # Difference of erf at limits, see (b)

    N = 1 / ( f * ((1/np.sqrt(np.pi)) * I)  +   (1-f) * ( np.exp(-lam*alpha) - np.exp(-lam*beta) )) # Normalisation factor
    
    # define the pdf
    pdf = N * ( f * stats.norm.pdf(M, mu, sigma)
                +
                (1-f) * stats.expon.pdf(M, scale=1/lam)
                )

    return pdf




# 
