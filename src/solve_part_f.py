from scipy.stats import chi2, binom
from iminuit import cost, Minuit
import numpy as np
import matplotlib.pyplot as plt
from funcs import accept_reject, pdf_norm_efg, plot_discovery_rates
import os


np.random.seed(75016)

mu = 5.28
sigma = 0.018
lam = 0.5
f = 0.1

alpha = 5
beta = 5.6

# Define the pdf and the sample sizes
pdf_true = lambda x: pdf_norm_efg(x, mu, sigma, lam, f)
sample_sizes = [100, 200, 300, 400, 500, 550, 600, 650, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]


discovery_rates = []
stdevs = []

M = accept_reject(pdf_true, alpha, beta, 100000)

for sample_size in sample_sizes:

    discovery = []
    p_vals = []
    #M_bootstrap = np.zeros((10, sample_size))

    for i in range(1000):

        M_bootstrap = np.random.choice(M, size=sample_size, replace=True)


        nll = cost.UnbinnedNLL(M_bootstrap, pdf_norm_efg)
        
        # Run the fit for the null hypothesis
        mi_null = Minuit(nll,  f = 0.2,  lam=0.4, mu=5.3, sigma = 0.02)
        mi_null.limits['f'] = (0,1)
        mi_null.limits['lam'] = (0.01,1)
        mi_null.limits['sigma'] = (0,20)
        mi_null.limits['mu'] = (5,5.6)
        mi_null.values['f'] = 0
        mi_null.fixed['f'] = True
        H_null = mi_null.migrad()

        null_params = list(mi_null.values) 
        null_min = mi_null.fval
    
        # Run the fit for the alternate hypothesis
        mi_alt = Minuit(nll,  f = 0.2,  lam=0.4, mu=5.3, sigma = 0.02)
        mi_alt.limits['f'] = (0,1)
        mi_alt.limits['lam'] = (0.01,1)
        mi_alt.limits['sigma'] = (0.0001,20)
        mi_alt.limits['mu'] = (5,5.6)
        H_alt = mi_alt.migrad()

        alt_params = list(mi_alt.values)
        alt_min = mi_alt.fval


        # Calculate the test statistic
        T = null_min - alt_min
        alt_ndof = 1.76
        alt_pval = 1 - chi2.cdf(T, alt_ndof)

        p_vals.append(alt_pval)

        if alt_pval < 2.9e-7:
            discovery.append(1)
        else:
            discovery.append(0)

    # Calculate the discovery rate
    discovery_rate = np.mean(discovery) * 100
    discovery_rates.append(discovery_rate)
    
    # We bootstrap the discovery rate to get an estimate of the standard deviation
    discovery_rates_bootstraps = []
    for i in range(1000):
        discovery_bootstrap = np.random.choice(discovery, size=1000, replace=True)
        discovery_rate_bootstrap = np.mean(discovery_bootstrap) * 100
        discovery_rates_bootstraps.append(discovery_rate_bootstrap)
    
    Stdev = np.std(discovery_rates_bootstraps, ddof=1)
    
    # We want to plot error bars of 3 standard deviations.
    stdevs.append(3 * Stdev)
    
    # Print the results
    print("Sample size: " + "{:e}".format(sample_size))
    print("Discovery rate: " + str(discovery_rate) + "%")
    print("Standard deviation of d.r.: {} %".format(Stdev))
    print("---------------------------------------")
    print("---------------------------------------")
    
    # If the last 3 discovery rates are all above 90%, we stop the loop
    if all(rate > 90 for rate in discovery_rates[-3:]):
        break


# Plot the results
plot_discovery_rates(sample_sizes[:len(discovery_rates)], discovery_rates, stdevs, 'f')

