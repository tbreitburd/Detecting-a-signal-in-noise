"""In this file we perform a fit of the number of degrees of freedom for the
 T-statistic distribution, assuming it is chi2 distributed, for part f and g"""

import numpy as np
from scipy.stats import chi2
from iminuit import cost, Minuit
import matplotlib.pyplot as plt
import os
from funcs import accept_reject, pdf_norm_efg, background_norm_efg, pdf_norm_g


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

# Define the chi2 pdf
def chi2_pdf(x, df):
    return chi2.pdf(x, df)


# --------------------------------------- PART F ---------------------------------------



# Define the pdf for the true parameters, here the null hypothesis, background only
pdf_true = lambda x: (1-f) * background_norm_efg(x,lam)

# Generate a large sample of M values to bootstrap from
M = accept_reject(pdf_true, alpha, beta-0.001, 100000)

# Define the sample size
sample_size = 3000

# Initialise the list of T statistics
T_ = []

for i in range(1000): # Run 1000 toys

    # Bootstrap the sample of desired size from the large sample
    M_bootstrap = np.random.choice(M, size=sample_size, replace=True)

    # Define the cost function, here the unbinned Negative Log Likelihood
    nll = cost.UnbinnedNLL(M_bootstrap, pdf_norm_efg)
    
    # Run the fit for the null hypothesis
    mi_null = Minuit(nll,  f = 0.2,  lam=0.4, mu=5.2, sigma = 0.02)
    mi_null.limits['f'] = (0,1)
    mi_null.limits['lam'] = (0, None)
    mi_null.values['f'] = 0 
    mi_null.fixed['f'] = True
    mi_null.values['mu'] = 5.2  # This doesn't matter, since f is fixed to 0
    mi_null.fixed['mu'] = True  # but it causes invalid operations if it is not fixed
    H_null = mi_null.migrad()

    null_min = mi_null.fval # Store the minimum of the fit

    # Run the fit for the alternate hypothesis
    mi_alt = Minuit(nll,  f = 0.2,  lam=0.4, mu=5.2, sigma = 0.02)
    mi_alt.limits['f'] = (0,1)
    mi_alt.limits['lam'] = (0, None)
    mi_alt.limits['sigma'] = (0, None)
    H_alt = mi_alt.migrad()

    alt_min = mi_alt.fval # Store the minimum of the fit

    # Calculate the test statistic, as the log likelihood ratio
    T = null_min - alt_min
    T_.append(T)


# Define the cost function
nll = cost.UnbinnedNLL(T_, chi2_pdf)

# Use iminuit to minimise the cost function, with an initial guess of 0.5 degrees of freedom
m = Minuit(nll, df=0.5)
ndof_fit = m.migrad()

print("The number of degrees of freedom for part f (T-statistic distribution under null hypothesis):"
       + "{:.4f} ± {:.4f}".format(ndof_fit.values['df'], ndof_fit.errors['df']))


# We can plot the T-statistic distribution, and compare it to the chi2 distribution with 3 degrees of freedom from Wilk's
# and the chi2 distribution fitted to the T-statistic distribution
# We plot from just below 0 so that the peak at 0 can be observed
# This is because Wilk's thrm is not truly valid for our case,
# since the null hypothesis is on the boundary of the parameter space
plt.figure()
plt.hist(T_, bins=50, density=True, label='T-statistic distribution', 
         color='grey', range=(-0.001,10))
plt.plot(np.linspace(-0.001,10,1000), chi2.pdf(np.linspace(0,10,1000), ndof_fit.values['df']),
             color = 'red', label=r"$\chi^2$"+' distribution fitted (ndof = {:.4f})'.format(ndof_fit.values['df']))
plt.plot(np.linspace(-0.001,10,1000), chi2.pdf(np.linspace(0,10,1000), 3),
          color = 'green', label=r"$\chi^2$"+" distribution from Wilk's (ndof = 3)")
plt.xlabel('T')
plt.ylabel('Normalised Counts')
plt.legend()
proj_dir = os.getcwd()
plots_dir = os.path.join(proj_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)
plot_dir = os.path.join(plots_dir, 'ndof_for_part_f.png')
plt.savefig(plot_dir)

# --------------------------------------- PART G ---------------------------------------



# Define the pdf for the true parameters, here the null hypothesis, background only
pdf_true = lambda x: pdf_norm_efg(x, mu, sigma, lam, f)

# Define the sample size
sample_size = 3000

# Generate a large sample of M values to bootstrap from
M = accept_reject(pdf_true, alpha, beta-0.001, 100000)

# Initialise the list of T statistics
T_ = []

for i in range(1000): # Run 1000 toys

    # Bootstrap the sample of desired size from the large sample
    M_bootstrap = np.random.choice(M, size=sample_size, replace=True)

    # Define the cost function, here the unbinned Negative Log Likelihood
    nll = cost.UnbinnedNLL(M_bootstrap, pdf_norm_g)
        
    # Run the fit for the null hypothesis, just 1 signal component
    mi_null = Minuit(nll,  f1 = 0.2, f2 = 0.1, lam=0.4, mu_1=5.3, mu_2=5.4, sigma = 0.02)
    mi_null.limits['f1'] = (0,1)
    mi_null.limits['lam'] = (0, None)
    mi_null.limits['sigma'] = (0, None)
    mi_null.values['f2'] = 0
    mi_null.fixed['f2'] = True
    mi_null.values['mu_2'] = 5.4  # This doesn't matter, since f2 is fixed to 0
    mi_null.fixed['mu_2'] = True  # but it causes invalid operations if it is not fixed
    H_null = mi_null.migrad()

    null_min = mi_null.fval # Store the minimum of the fit

    # Run the fit for the alternate hypothesis, 2 signal components
    mi_alt = Minuit(nll,  f1 = 0.2, f2 = 0.1,  lam=0.4, mu_1=5.3, mu_2=5.4, sigma = 0.02)
    mi_alt.limits['f1'] = (0,1)
    mi_alt.limits['f2'] = (0,1)
    mi_alt.limits['lam'] = (0, None)
    mi_alt.limits['sigma'] = (0, None)
    mi_alt.limits['mu_1'] = (5,5.6)
    mi_alt.limits['mu_2'] = (5,5.6)
    H_alt = mi_alt.migrad()

    alt_min = mi_alt.fval # Store the minimum of the fit

    # Calculate the test statistic
    T = null_min - alt_min
    T_.append(T)

# Define the cost function
nll = cost.UnbinnedNLL(T_, chi2_pdf)

# Use iminuit to minimise the cost function, with an initial guess of 1 degree of freedom
m = Minuit(nll, df=1)
ndof_fit = m.migrad()

print("The number of degrees of freedom for part g (T-statistic distribution under null hypothesis):"
       + "{:.4f} ± {:.4f}".format(ndof_fit.values['df'], ndof_fit.errors['df']))


# Same as part f, this time Wilk's thrm states that the T-statistic distribution 
# should follow a chi2 distribution with 2 degrees of freedom
plt.figure()
plt.hist(T_, bins=50, density=True, label='T-statistic distribution', 
         color='grey', range=(-0.001,10))
plt.plot(np.linspace(-0.001,10,1000), chi2.pdf(np.linspace(0,10,1000), ndof_fit.values['df']),
          color = 'red', label=r"$\chi^2$" + " distribution fitted (ndof = {:.4f})".format(ndof_fit.values['df']))
plt.plot(np.linspace(-0.001,10,1000), chi2.pdf(np.linspace(0,10,1000), 2),
          color = 'green', label=r"$\chi^2$"+" distribution from Wilk's (ndof = 2)")
plt.xlabel('T')
plt.ylabel('Normalised Counts')
plt.legend()
proj_dir = os.getcwd()
plots_dir = os.path.join(proj_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)
plot_dir = os.path.join(plots_dir, 'ndof_for_part_g.png')
plt.savefig(plot_dir)