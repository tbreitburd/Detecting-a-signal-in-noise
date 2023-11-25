from funcs import plot_pdf_e, accept_reject, signal_e, background_e, pdf_norm_e, signal_norm_e, background_norm_e
import numpy as np

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

