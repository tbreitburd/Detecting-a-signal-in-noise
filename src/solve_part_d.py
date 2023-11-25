from funcs import plot_pdf_d_comp

# Define true parameters
mu_true = 5.28
sigma_true = 0.018
lam_true = 0.5
f_true = 0.1

# Define limits
alpha = 5
beta = 5.6

# Plot the signal, background and total pdf
# This function takes a linspace of x values, 
# and plots the pdfs for the signal, background and total pdf   
plot_pdf_d_comp(mu_true, sigma_true, lam_true, f_true, alpha, beta)