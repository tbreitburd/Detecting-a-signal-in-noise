from funcs import pdf_norm, signal, background, signal_norm, background_norm
import numpy as np
import os
import matplotlib.pyplot as plt

# Define true parameters
mu_true = 5.28
sigma_true = 0.018
lam_true = 0.5
f_true = 0.1

# Define limits
alpha = 5
beta = 5.6

# Define the plotting functions

def plot_pdf_d(pdf, mu, sigma, lam, f, alpha, beta):
    """
    This function plots the signal component,
    the background component,
    the pdf of M, using the normalisation factor derived in (b),
    all overlaid on the same plot.
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
    x = np.linspace(alpha, beta, 1000)

    # Define signal component
    signal_ = signal(x, mu, sigma)

    # Define background component
    background_ = background(x, lam)

    # Define pdf
    pdf_true = pdf(x, mu, sigma, lam, f, alpha, beta)

    # Plot
    plt.plot(x, signal_,'--',color = 'r', label='Signal, s(M; \u03BC, \u03C3)')
    plt.plot(x, background_,'--', color = 'g', label='Background, b(M; \u03BB)')
    plt.plot(x, pdf_true, color = 'k', label='PDF')
    plt.xlabel('M')
    plt.ylabel('Probability density')
    plt.legend()
    proj_dir = os.getcwd()
    plots_dir = os.path.join(proj_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_dir = os.path.join(plots_dir, 'plot_pdf_d.png')
    plt.savefig(plot_dir)
    plt.show()

    return None


def plot_pdf_d_comp(pdf, mu, sigma, lam, f, alpha, beta):
    """
    This function plots the signal component,
    the background component,
    the pdf of M, using the normalisation factor derived in (b),
    all overlaid on the same plot,
    using component-wise normaliation.
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
    x = np.linspace(alpha, beta, 1000)

    # Define signal component
    signal_ = signal_norm(x, mu, sigma, alpha, beta)

    # Define background component
    background_ = background_norm(x, lam, alpha, beta)

    # Define pdf
    pdf_true = pdf(x, mu, sigma, lam, f, alpha, beta)

    # Plot
    plt.plot(x, signal_,'--',color = 'r', label='Signal, s(M; \u03BC, \u03C3)')
    plt.plot(x, background_,'--', color = 'g', label='Background, b(M; \u03BB)')
    plt.plot(x, pdf_true, color = 'k', label='PDF')
    plt.xlabel('M')
    plt.ylabel('Probability density')
    plt.legend()
    proj_dir = os.getcwd()
    plots_dir = os.path.join(proj_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_dir = os.path.join(plots_dir, 'plot_pdf_d_comp.png')
    plt.savefig(plot_dir)
    plt.show()

    return None

# Plot the signal, background and total pdf
# This function takes a linspace of x values, 
# and plots the pdfs for the signal, background and total pdf   
plot_pdf_d_comp(pdf_norm ,mu_true, sigma_true, lam_true, f_true, alpha, beta)