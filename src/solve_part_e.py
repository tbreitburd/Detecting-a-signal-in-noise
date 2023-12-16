from funcs import accept_reject, pdf_norm_efg, cdf_efg, signal_norm_efg, background_norm_efg
import numpy as np
from iminuit import cost, Minuit
from scipy import stats
import os
import matplotlib.pyplot as plt


# Define the plotting function
def plot_e(pdf, gen_sample, mu_hat, sigma_hat, lam_hat, f_hat, alpha, beta):
    """
    This function plots the generated sample,
    estimates of the signal component,
    the background component,
    and total probability,
    all overlaid on the same plot,
    ----------------------------
    Inputs:
    alpha: lower limit of M, float
    beta: upper limit of M, float
    mu: mean of signal (normal) component, float, must be within alpha and beta
    sigma: width of signal (normal) component, float, must be positive
    lam: decay parameter of background (exponential) component, float
    f: ratio of signal/background components, float
    ----------------------------
    Outputs:
    plot of signal component, background component, normalised pdf of M
    """

    # Define x-axis
    x = np.linspace(alpha, beta, 200)

    # Define signal component
    signal_ = signal_norm_efg(x, mu_hat, sigma_hat)

    # Define background component
    background_ = background_norm_efg(x, lam_hat)

    # Define pdf
    pdf_pred = lambda x: pdf(x, mu_hat, sigma_hat, lam_hat, f_hat)

    # Calculate some values for plotting
    pdf_vals = pdf_pred(x)

    #Bin the generated sample
    bin_counts, bin_widths = np.histogram(gen_sample, bins=100, range=(5,5.6))

    # Get predicted values
    N = len(gen_sample)
    bin_centres = (bin_widths[:-1] + bin_widths[1:]) * 0.5
    bin_width = bin_widths[1:] - bin_widths[:-1]
    pred_vals = N * bin_width * pdf_pred(bin_centres)

    # Get residuals, errors and pull 
    residuals = bin_counts - pred_vals
    errors = np.sqrt(bin_counts)
    pull = residuals / errors

    # Plot all of this
    fig, ax = plt.subplots(2, 2, figsize=(6.4,6.4), sharex='col', sharey='row',
                            gridspec_kw=dict(hspace=0, wspace=0, height_ratios=(3,1), width_ratios=(7,1)))

    # Top figure generated samples
    ax[0,0].errorbar( bin_centres, bin_counts, errors, fmt='ko', label = 'Generated Sample')
    
    # Top figure true pdf, background and signal lines
    x = np.linspace(bin_widths[0], bin_widths[-1], 200)
    ax[0,0].plot(x, f_hat * signal_ * N * bin_width[0],'--',color = 'r', label='f x Signal, s(M; \u03BC, \u03C3)')
    ax[0,0].plot(x, (1 - f_hat) * background_ * N * bin_width[0],'--', color = 'g', label='Background, b(M; \u03BB)')
    ax[0,0].plot(x, pdf_vals * N * bin_width[0],'--', color = 'k', label='PDF')
    ax[0,0].legend(loc='upper right')
    ax[0,0].grid()
    ax[0,0].set_ylabel('Counts')

    # Bottom figure pull plot
    ax[1,0].errorbar( bin_centres, pull, np.ones_like(bin_centres), fmt='ko')
    
    # Add a flat line at 0
    ax[1,0].plot(x, np.zeros_like(x))
    ax[1,0].set_xlabel('$M$')
    ax[1,0].set_ylabel('Pull')

    # Pull distribution figure
    ax[0,1].set_visible(False)
    ax[1,1].hist(pull, bins=10, range=(-3,3), density=True, alpha=0.5, orientation='horizontal')
    ax[1,1].xaxis.set_visible(False)
    ax[1,1].spines[['top','bottom','right']].set_visible(False)
    ax[1,1].tick_params( which='both', direction='in', axis='y', right=False, labelcolor='none')
    
    # Overlay a normal distribution
    xp = np.linspace(-3,3,100)
    ax[1,1].plot(stats.norm.pdf(xp), xp, 'r-', alpha=0.5 )

    # Format the plot
    ax[0,0].autoscale(enable=True, tight=True, axis='x')
    fig.align_ylabels()
    
    proj_dir = os.getcwd()
    plots_dir = os.path.join(proj_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_dir = os.path.join(plots_dir, 'plot_pdf_e.png')
    plt.savefig(plot_dir)
    plt.show()

    return None


# Set the seed
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
pdf_true = lambda x: pdf_norm_efg(x, mu_true, sigma_true, lam_true, f_true)

# Generate the sample from this pdf
M_pdf = accept_reject(pdf_true, alpha, beta, 100000)

# Here, we use a binned negative log likelihood method to estimate the parameters.
# So, we want to minimize that negative log likelihood function.

# First, we bin the data to get the counts in each bin, and the bin edges.
bins = np.linspace(5,5.6,1000)
bin_counts, bin_edges = np.histogram(M_pdf, bins=bins)

# Now, we define the negative log likelihood function which we will minimize using iminuit.
# This function takes in the bin counts, bin edges, and the cdf of the pdf we want to fit to.
binned_nll = cost.BinnedNLL(bin_counts, bin_edges, cdf_efg)

# Now, we minimize that function using iminuit.
mi_bin = Minuit(binned_nll,  f = 0.2,  lam=0.4, mu=5.3, sigma = 0.02)
mi_bin.limits['f'] = (0,0.5)
mi_bin.limits['lam'] = (0.01,1)
mi_bin.limits['sigma'] = (0,0.05)
mi_bin.limits['mu'] = (5,5.6)
mi_bin.migrad()
mi_bin.hesse()

print(mi_bin)

# Get returned parameter estimates
mu_est = mi_bin.values['mu']
sigma_est = mi_bin.values['sigma']
lam_est = mi_bin.values['lam']
f_est = mi_bin.values['f']

# From iminuit, we can plot the profile likelihoods for each parameter, with their 1-stdev errors.

fig, ax = plt.subplots(2, 2, figsize=(8,8))
subplot_titles = ['\u03BC', '\u03C3', '\u03BB', 'f']

plt.suptitle('Profile Likelihoods of Parameters, with 1-stdev Errors')

plt.subplot(2,2,1)
mi_bin.draw_profile('mu',  band = True)
plt.ylabel('Profile Negative Log-Likelihood')
plt.xticks([5.2785, 5.279, 5.2795, 5.28, 5.2805])
plt.title(subplot_titles[0])
plt.grid()

plt.subplot(2,2,2)
mi_bin.draw_profile('sigma', band = True)
plt.ylabel(None)
plt.xticks([0.0175, 0.018, 0.0185, 0.019])
plt.title(subplot_titles[1])
plt.grid()

plt.subplot(2,2,3)
mi_bin.draw_profile('lam', band = True)
plt.ylabel('Profile Negative Log-Likelihood')
plt.title(subplot_titles[2])
plt.grid()

plt.subplot(2,2,4)
mi_bin.draw_profile('f', band = True)
plt.ylabel(None)
plt.title(subplot_titles[3])
plt.grid()

plt.tight_layout()

proj_dir = os.getcwd()
plots_dir = os.path.join(proj_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)
plot_dir = os.path.join(plots_dir, 'plot_profile_likelihood_e.png')
plt.savefig(plot_dir)

plt.show()


# Plot the pdf with the estimated parameters
plot_e(pdf_norm_efg, M_pdf, mu_est, sigma_est, lam_est, f_est, alpha, beta)

