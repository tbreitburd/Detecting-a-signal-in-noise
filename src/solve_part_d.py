from funcs import plot_pdf_d
import matplotlib.pyplot as plt

# Define true parameters
mu_true = 5.28
sigma_true = 0.018
lam_true = 0.5
f_true = 0.1

# Define limits
alpha = 5
beta = 5.6

plot_pdf_d(mu_true, sigma_true, lam_true, f_true, alpha, beta)
