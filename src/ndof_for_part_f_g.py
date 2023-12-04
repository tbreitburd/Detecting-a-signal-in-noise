import numpy as np
from scipy.stats import chi2
from iminuit import cost, Minuit
from funcs import accept_reject, pdf_norm_efg, background_norm_efg, pdf_norm_g

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

# Define the sample size
sample_size = 3000

# Generate a large sample of M values to bootstrap from
M = accept_reject(pdf_true, alpha, beta, 100000)

# Initialise the list of T statistics
T_ = []

for i in range(1000): # Run 1000 toys

    # Bootstrap the sample of desired size from the large sample
    M_bootstrap = np.random.choice(M, size=sample_size, replace=True)

    # Define the cost function, here the unbinned Negative Log Likelihood
    nll = cost.UnbinnedNLL(M_bootstrap, pdf_norm_efg)
    
    # Run the fit for the null hypothesis
    mi_null = Minuit(nll,  f = 0.2,  lam=0.4, mu=5.2, sigma = 0.02)
    mi_null.limits['f'] = (0.01,1)
    mi_null.limits['lam'] = (0.00001, None)
    mi_null.values['f'] = 0
    mi_null.fixed['f'] = True
    H_null = mi_null.migrad()

    null_min = mi_null.fval # Store the minimum of the fit

    # Run the fit for the alternate hypothesis
    mi_alt = Minuit(nll,  f = 0.2,  lam=0.4, mu=5.2, sigma = 0.02)
    mi_alt.limits['f'] = (0.01,1)
    mi_alt.limits['lam'] = (0.00001, None)
    mi_alt.limits['sigma'] = (0.00001, None)
    mi_alt.limits['mu'] = (5,5.6)
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

print("The number of degrees of freedom for part f (T-statistic distribution under null hypothesis):"
       + "{:.4f} ± {:.4f}".format(ndof_fit.values['df'], ndof_fit.errors['df']))



# --------------------------------------- PART G ---------------------------------------



# Define the pdf for the true parameters, here the null hypothesis, background only
pdf_true = lambda x: pdf_norm_efg(x, mu, sigma, lam, f)

# Define the sample size
sample_size = 3000

# Generate a large sample of M values to bootstrap from
M = accept_reject(pdf_true, alpha, beta, 100000)

# Initialise the list of T statistics
T_ = []

for i in range(1000): # Run 1000 toys

    # Bootstrap the sample of desired size from the large sample
    M_bootstrap = np.random.choice(M, size=sample_size, replace=True)

    # Define the cost function, here the unbinned Negative Log Likelihood
    nll = cost.UnbinnedNLL(M_bootstrap, pdf_norm_g)
        
    # Run the fit for the null hypothesis, just 1 signal component
    mi_null = Minuit(nll,  f1 = 0.2, f2 = 0.1, lam=0.4, mu_1=5.3, mu_2=5.4, sigma = 0.02)
    mi_null.limits['f1'] = (0.01,1)
    mi_null.limits['lam'] = (0.00001, None)
    mi_null.values['f2'] = 0
    mi_null.fixed['f2'] = True
    H_null = mi_null.migrad()

    null_min = mi_null.fval # Store the minimum of the fit

    # Run the fit for the alternate hypothesis, 2 signal components
    mi_alt = Minuit(nll,  f1 = 0.2, f2 = 0.1,  lam=0.4, mu_1=5.3, mu_2=5.4, sigma = 0.02)
    mi_null.limits['f1'] = (0.01,1)
    mi_null.limits['f2'] = (0.01,1)
    mi_alt.limits['lam'] = (0.00001, None)
    mi_alt.limits['sigma'] = (0.00001, None)
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
