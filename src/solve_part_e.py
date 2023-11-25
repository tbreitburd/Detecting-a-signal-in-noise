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
M_pdf = accept_reject(pdf_true,100000)

# We want to estimate the parameters of the pdf from our generated sample. 
# Here, we use a binned negative log likelihood method to estimate the parameters.
# So, we want to minimize that negative log likelihood function.

# First, we bin the data to get the counts in each bin, and the bin edges.
bins = np.linspace(5,5.6,1000)
bin_counts, bin_edges = np.histogram(M_pdf, bins=bins)

# Now, we define the negative log likelihood function which we will minimize using iminuit.
# This function takes in the bin counts, bin edges, and the cdf of the pdf we want to fit to.
binned_nll = cost.BinnedNLL(bin_counts, bin_edges, cdf_e)

# Now, we minimize that function using iminuit.
mi = Minuit(binned_nll,  f = 0.4,  lam=0.4, mu=5.1, sigma = 0.01)
mi.limits['f'] = (0,0.5)
mi.limits['lam'] = (0.01,1)
mi.limits['sigma'] = (0,0.05)
mi.limits['mu'] = (5,5.6)
mi.migrad()

