"""In this file we generate a large pretend dataset from a new p.d.f. with 2 signal components
using an accept-reject method. We then bootstrap a sample of the desired size
from the large sample, and fit the pdf model and its parameters to that dataset.
We then calculate the test statistic, T, and the p-value, i.e. the probability of
rejecting the null hypothesis when it is true.
We repeat this process 1000 times for each sample size, and calculate the
discovery rate, i.e. the fraction of times the p-value is below the 5 sigma threshold.
We plot the discovery rate as a function of sample size, and stop the loop when
the discovery rate is above 90% for the last 3 sample sizes. We also plot the
discovery rate with error bars of 3 standard deviations, calculated by bootstrapping
the discovery rate 1000 times for each sample size."""

from scipy.stats import chi2
from iminuit import cost, Minuit
import numpy as np
import os
import matplotlib.pyplot as plt
from funcs import accept_reject, signal_norm, background_norm, pdf_norm_g, plot_discovery_rates


# Define the plotting function
def plot_pdf_g(pdf, mu_1, mu_2, sigma, lam, f1, f2, alpha, beta):
    """
    This function plots the 2 signal components,
    the background component,
    the pdf of M, using the normalisation factors derived in (b),
    all overlaid on the same plot,
    using component-wise normaliation.
    ----------------------------
    Inputs:
    pdf: pdf of M, callable
    alpha: lower limit of the domain of M, float
    beta: upper limit of the domain of M, float
    mu_1: mean of first signal (normal) component, float, must be within alpha and beta
    mu_2: mean of second signal (normal) component, float, must be within alpha and beta
    sigma: width of signal (normal) component, float, must be positive
    lam: decay parameter of background (exponential) component, float
    f1: weight of first signal, float between 0 and 1
    f2: weight of second signal, float between 0 and 1
    ----------------------------
    Outputs:
    plot of signal components, background component, normalised pdf of M
    """

    # Define x-axis
    x = np.linspace(alpha, beta-0.001, 1000)

    # Define signal component
    signal_1 = signal_norm(x, mu_1, sigma, alpha, beta)
    signal_2 = signal_norm(x, mu_2, sigma, alpha, beta)

    # Define background component
    background_ = background_norm(x, lam, alpha, beta)

    # Define pdf
    pdf_true = pdf(x, mu_1, mu_2, sigma, lam, f1, f2)

    plt.figure(figsize=(10,8))

    # Plot
    plt.plot(x, f1*signal_1,'--',color = 'r', label='Scaled First Signal, s(M; $\u03BC_{1}$, \u03C3)')
    plt.plot(x, f2*signal_2,'--',color = 'b', label='Scaled Second Signal, s(M; $\u03BC_{2}$, \u03C3)')
    plt.plot(x, (1-f1-f2)*background_,'--', color = 'g', label='Scaled Background, b(M; \u03BB)')
    plt.plot(x, pdf_true, color = 'k', label='PDF')
    plt.title('PDF of M, for the 2 signal model, for the true parameters')
    plt.xlim(alpha, beta)
    plt.xlabel('M')
    plt.ylabel('Probability density')
    plt.legend()
    plt.grid()
    proj_dir = os.getcwd()
    plots_dir = os.path.join(proj_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_dir = os.path.join(plots_dir, 'plot_pdf_g.png')
    plt.savefig(plot_dir)

    return None

np.random.seed(75016)

# Define true parameters
mu_1 = 5.28
mu_2 = 5.35
sigma = 0.018
lam = 0.5
f1 = 0.1
f2 = 0.05

# Define the r.v. domain limits
alpha = 5
beta = 5.6


# Define the pdf for the true parameters
pdf_true = lambda x: pdf_norm_g(x, mu_1, mu_2, sigma, lam, f1, f2)

# Plot this pdf, with the true parameters
plot_pdf_g(pdf_norm_g, mu_1, mu_2, sigma, lam, f1, f2, alpha, beta)

# List of sample sizes to test
sample_sizes = [100, 200, 300, 400, 500, 550, 600, 650, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000]

# Initialise the list of discovery rates to store the results
discovery_rates = []

# Generate a large sample of M values to bootstrap from
M = accept_reject(pdf_true, alpha, beta, 100000)


nll = cost.UnbinnedNLL(M, pdf_norm_g)
        
# Run the fit for the null hypothesis, just 1 signal component
mi_null = Minuit(nll,  f1 = 0.2, f2 = 0.1, lam=0.4, mu_1=5.3, mu_2=5.4, sigma = 0.02)
mi_null.limits['f1'] = (0.00001,1)
mi_null.limits['lam'] = (0.00001, None)
# Fix the value of f2 to 0, essentially creating the null hypothesis model
mi_null.values['f2'] = 0
mi_null.fixed['f2'] = True
# Fix mu_2 as well, otherwise it floats and causes problems, but since it is
# irrelevant for the null hypothesis, we can fix it
mi_null.values['mu_2'] = 5.4
mi_null.fixed['mu_2'] = True
H_null = mi_null.migrad()

null_min = mi_null.fval # Store the minimum of the fit
print(H_null)

# Run the fit for the alternate hypothesis, 2 signal components
mi_alt = Minuit(nll,  f1 = 0.2, f2 = 0.1,  lam=0.4, mu_1=5.3, mu_2=5.4, sigma = 0.02)
mi_alt.limits['f1'] = (0.00001,1)
mi_alt.limits['f2'] = (0.00001,1)
mi_alt.limits['lam'] = (0.00001, None)
mi_alt.limits['sigma'] = (0.00001, None)
mi_alt.limits['mu_1'] = (5,5.6)
mi_alt.limits['mu_2'] = (5,5.6)
H_alt = mi_alt.migrad()

alt_min = mi_alt.fval # Store the minimum of the fit
print(H_alt)

# Calculate the test statistic
T = null_min - alt_min
# Set the number of degrees of freedom
alt_ndof = 2.06 # see ndof_for_part_f_g.py

# Calculate the p-value
alt_pval = 1 - chi2.cdf(T, alt_ndof)
print(alt_pval)

# Initialise the list of discovery rates to store the results
discovery_rates = []
stdevs = []

# Loop over the sample sizes
for sample_size in sample_sizes:

    discovery = [] # Initialise the list that will keep track of discoveries for this sample size

    for i in range(1000): # Run 1000 toys for each sample size

        # Bootstrap the sample of desired size from the large sample
        M_bootstrap = np.random.choice(M, size=sample_size, replace=True) 

        # Define the cost function, here the unbinned Negative Log Likelihood
        nll = cost.UnbinnedNLL(M_bootstrap, pdf_norm_g)
        
        # Run the fit for the null hypothesis
        mi_null = Minuit(nll,  f1 = 0.2, f2 = 0.1, lam=0.4, mu_1=5.3, mu_2=5.4, sigma = 0.02)

        # We can set some physical limits
        mi_null.limits['f1'] = (0.00001,1)
        mi_null.limits['lam'] = (0.00001, None)

        # Fix the value of f2 to 0, essentially creating the null hypothesis model
        mi_null.values['f2'] = 0
        mi_null.fixed['f2'] = True
        mi_null.values['mu_2'] = 5.4
        mi_null.fixed['mu_2'] = True

        H_null = mi_null.migrad()

        null_min = mi_null.fval # Store the minimum of the fit
    


        # Run the fit for the alternate hypothesis
        mi_alt = Minuit(nll,  f1 = 0.2, f2 = 0.1,  lam=0.4, mu_1=5.3, mu_2=5.4, sigma = 0.02)
        mi_alt.limits['f1'] = (0.0001,1)
        mi_alt.limits['f2'] = (0.0001,1)
        mi_alt.limits['lam'] = (0.00001, None)
        mi_alt.limits['sigma'] = (0.00001, None)
        mi_alt.limits['mu_1'] = (5,5.6)
        mi_alt.limits['mu_2'] = (5,5.6)

        H_alt = mi_alt.migrad()

        alt_min = mi_alt.fval # Store the minimum of the fit

        # Calculate the test statistic
        T = null_min - alt_min

        # Set the number of degrees of freedom
        alt_ndof = 1.5885 # the number of degrees of freedom for the test statistic 
                        # distribution under the null hypothesis
                        # see ndof_for_part_f_g.py

        # Calculate the p-value
        alt_pval = 1 - chi2.cdf(T, alt_ndof)

        # If the p-value is less than the threshold, we have a discovery
        if alt_pval < 2.9e-7:
            discovery.append(1)
        else:
            discovery.append(0)

    discovery_rate = np.mean(discovery) * 100
    discovery_rates.append(discovery_rate)
    
    # We bootstrap the discovery rate to get an estimate of the standard deviation
    discovery_rates_bootstraps = []
    for i in range(1000):
        discovery_bootstrap = np.random.choice(discovery, size=1000, replace=True)
        discovery_rate_bootstrap = np.mean(discovery_bootstrap) * 100
        discovery_rates_bootstraps.append(discovery_rate_bootstrap)
    
    # Calculate the standard deviation of the discovery rate from its bootstrapped distribution
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
plot_discovery_rates(sample_sizes[:len(discovery_rates)], discovery_rates, stdevs, 'g')