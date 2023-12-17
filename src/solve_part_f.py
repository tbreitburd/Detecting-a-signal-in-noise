"""In this file, we generate a large pretend dataset from our defined p.d.f. 
using an accept-reject method. We then bootstrap a sample of the desired size
from the large sample, and fit the pdf model and its parameters to that dataset.
We then calculate the test statistic, T, and the p-value i.e. the probability of
rejecting the null hypothesis when it is true.
We repeat this process 1000 times for each sample size, and calculate the 
discovery rate, i.e. the fraction of times the p-value is below the 5 sigma threshold.
We plot the discovery rate as a function of sample size, and stop the loop when
the discovery rate is above 90% for the last 3 sample sizes. We also plot the
discovery rate with error bars of 3 standard deviations, calculated by bootstrapping
the discovery rate 1000 times for each sample size."""

from scipy.stats import chi2, binom
from iminuit import cost, Minuit
import numpy as np
import matplotlib.pyplot as plt
from funcs import accept_reject, pdf_norm_efg, plot_discovery_rates
import os

# Set the random seed
np.random.seed(75016)

# Define true parameters
mu = 5.28
sigma = 0.018
lam = 0.5
f = 0.1

# Define the r.v. domain limits
alpha = 5
beta = 5.6

# Define the pdf and the sample sizes
pdf_true = lambda x: pdf_norm_efg(x, mu, sigma, lam, f)
sample_sizes = [100, 200, 300, 400, 500, 550, 600, 650, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]

# Initialise the list of discovery rates to store the results
discovery_rates = []
stdevs = []

M = accept_reject(pdf_true, alpha, beta, 100000)

for sample_size in sample_sizes:

    discovery = []
    p_vals = []

    for i in range(1000):  # Run 1000 toys

        # Bootstrap the sample of desired size from the large sample
        M_bootstrap = np.random.choice(M, size=sample_size, replace=True)

        # Define the cost function, here the unbinned Negative Log Likelihood
        nll = cost.UnbinnedNLL(M_bootstrap, pdf_norm_efg)
        
        # Run the fit for the null hypothesis
        mi_null = Minuit(nll,  f = 0.2,  lam=0.4, mu=5.3, sigma = 0.02)
        mi_null.limits['f'] = (0,1)
        mi_null.limits['lam'] = (0.01,1)
        mi_null.limits['sigma'] = (0,20)
        mi_null.limits['mu'] = (5,5.6) # This doesn't matter, since f is fixed to 0
                                       # but it causes invalid operations if it is allowed 
                                        # to float completely freely
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
        alt_ndof = 1.9151 # from Wilk's theorem
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
    
    # Calculate the standard deviation of the discovery rate,
    # correcting for bias
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

