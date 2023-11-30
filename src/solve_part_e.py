from funcs import plot_pdf_e, accept_reject, pdf_norm_e, cdf_e
import numpy as np
from iminuit import cost, Minuit

np.random.seed(75016)

# Define true parameters
mu_true = 5.28
sigma_true = 0.018
lam_true = 0.5
f_true = 0.1

# Define limits
alpha = 5
beta = 5.6

# Define the pdf
pdf_true = lambda x: pdf_norm_e(x, mu_true, sigma_true, lam_true, f_true)

# Generate the sample from this pdf
M_pdf = accept_reject(pdf_true, alpha, beta, 100000)

# We want to estimate the parameters of the pdf from our generated sample. 
# First, we use an unbinned negative log likelihood method to estimate the parameters.

# Define the negative log likelihood function
nll = cost.UnbinnedNLL(M_pdf, pdf_norm_e)

mi_unbin = Minuit(nll,  f = 0.15,  lam=0.4, mu=5.3, sigma = 0.020)
mi_unbin.limits['f'] = (0,1)
mi_unbin.limits['lam'] = (0.01,1)
mi_unbin.limits['sigma'] = (0,0.05)
mi_unbin.limits['mu'] = (5,5.6)
mi_unbin.migrad()

print(mi_unbin)

# Here, we use a binned negative log likelihood method to estimate the parameters.
# So, we want to minimize that negative log likelihood function.

# First, we bin the data to get the counts in each bin, and the bin edges.
bins = np.linspace(5,5.6,1000)
bin_counts, bin_edges = np.histogram(M_pdf, bins=bins)

# Now, we define the negative log likelihood function which we will minimize using iminuit.
# This function takes in the bin counts, bin edges, and the cdf of the pdf we want to fit to.
binned_nll = cost.BinnedNLL(bin_counts, bin_edges, cdf_e)

# Now, we minimize that function using iminuit.
mi_bin = Minuit(binned_nll,  f = 0.2,  lam=0.4, mu=5.3, sigma = 0.02)
mi_bin.limits['f'] = (0,0.5)
mi_bin.limits['lam'] = (0.01,1)
mi_bin.limits['sigma'] = (0,0.05)
mi_bin.limits['mu'] = (5,5.6)
mi_bin.migrad()

print(mi_bin)

# Get returned parameter estimates
mu_est = mi_bin.values['mu']
sigma_est = mi_bin.values['sigma']
lam_est = mi_bin.values['lam']
f_est = mi_bin.values['f']

# Plot the pdf with the estimated parameters
plot_pdf_e(pdf_norm_e, M_pdf, mu_est, sigma_est, lam_est, f_est, alpha, beta)

