
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import os

def signal(M, mu, sigma):
    """ 
    This function defines the signal component of the pdf.
    ----------------------------
    Inputs:
    M: r.v. M, float (when calculating a probability for a specific value of M)
    mu: mean of signal (normal) component, float, must be within alpha and beta
    sigma: width of signal (normal) component, float, must be positive
    ----------------------------
    Outputs:
    signal: signal component of pdf, float
    """

    signal = stats.norm.pdf(M, mu, sigma)

    return signal


def background(M, lam):
    """ 
    This function defines the background component of the pdf.
    ----------------------------
    Inputs:
    M: r.v. M, float (when calculating a probability for a specific value of M)
    lam: decay parameter of background (exponential) component, float
    ----------------------------
    Outputs:
    background: background component of pdf, float
    """

    background = stats.expon.pdf(M, 0, scale=1/lam)

    return background


def signal_norm(M, mu, sigma, alpha, beta):
    """ 
    This function defines the normalised signal component of the pdf.
    ----------------------------
    Inputs:
    M: r.v. M, float (when calculating a probability for a specific value of M)
    mu: mean of signal (normal) component, float, must be within alpha and beta
    sigma: width of signal (normal) component, float, must be positive
    alpha: lower limit of M, float
    beta: upper limit of M, float
    ----------------------------
    Outputs:
    signal: signal component of pdf, float
    """

    I,_ = integrate.fixed_quad(lambda x: signal(x,mu, sigma), alpha, beta, n=10) # Difference of erf at limits, see (b)

    N = 1 / I # Normalisation factor

    signal_ = N * stats.norm.pdf(M, mu, sigma)

    return signal_


def background_norm(M, lam, alpha, beta):
    """ 
    This function defines the normalised background component of the pdf.
    ----------------------------
    Inputs:
    M: r.v. M, float (when calculating a probability for a specific value of M)
    lam: decay parameter of background (exponential) component, float
    alpha: lower limit of M, float
    beta: upper limit of M, float
    ----------------------------
    Outputs:
    background: background component of pdf, float
    """

    I,_ = integrate.fixed_quad(lambda x: background(x,lam), alpha, beta, n=10) # Difference of erf at limits, see (b)

    N = 1 / I # Normalisation factor

    background_ = N * stats.expon.pdf(M, 0, scale=1/lam)

    return background_


def pdf_(M,mu, sigma, lam, f):
    """
    This function defines the probability density function of M,
    within the limits alpha and beta.
    ----------------------------
    Inputs:
    M: r.v. M, float (when calculating a probability for a specific value of M)
    alpha: lower limit of M, float
    beta: upper limit of M, float
    mu: mean of signal (normal) component, float, must be within alpha and beta
    sigma: width of signal (normal) component, float, must be positive
    lam: decay parameter of background (exponential) component, float
    f: ratio of signal/background components, float
    ----------------------------
    Outputs:
    pdf: pdf of M, float
    """

    # define the pdf
    pdf = ( f * stats.norm.pdf(M, mu, sigma)
            +
            (1-f) * stats.expon.pdf(M, 0, scale=1/lam)
            )

    return pdf


def pdf_norm(M,mu, sigma, lam, f, alpha, beta):
    """
    This function defines the probability density function of M,
    normalised within the limits alpha and beta.
    It is component-wise normalised.
    ----------------------------
    Inputs:
    M: r.v. M, float (when calculating a probability for a specific value of M)
    alpha: lower limit of M, float
    beta: upper limit of M, float
    mu: mean of signal (normal) component, float, must be within alpha and beta
    sigma: width of signal (normal) component, float, must be positive
    lam: decay parameter of background (exponential) component, float
    f: ratio of signal/background components, float
    ----------------------------
    Outputs:
    pdf: pdf of M, float
    """

    I_s,_ = integrate.fixed_quad(lambda x: signal(x,mu, sigma), alpha, beta, n=10) # Difference of erf at limits, see (b)

    N_s = 1 / I_s # Normalisation factor of signal component

    I_b,_ = integrate.fixed_quad(lambda x: background(x,lam), alpha, beta, n=10) # Difference of erf at limits, see (b)
    
    N_b = 1 / I_b # Normalisation factor of background component

    # define the pdf
    pdf = ( N_s * f * stats.norm.pdf(M, mu, sigma)
            +
            N_b * (1-f) * stats.expon.pdf(M, 0, scale=1/lam)
            )

    return pdf


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
    proj_dir = os.path.dirname(os.getcwd())
    plot_dir = os.path.join(proj_dir, 'plot_pdf_d.png')
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
    proj_dir = os.path.dirname(os.getcwd())
    plot_dir = os.path.join(proj_dir, 'plot_pdf_d_comp.png')
    plt.savefig(plot_dir)
    plt.show()

    return None


#----------------------------------------------------------------
#------------------------ PART E --------------------------------
#----------------------------------------------------------------
from numba_stats import truncnorm, truncexpon, norm, expon


def accept_reject(f,alpha, beta, sample_size):
    """
    This function implements the accept-reject method to generate samples from a given pdf.
    ----------------------------
    Inputs:
    f: pdf, function
    alpha: lower limit of domain, float
    beta: upper limit of domain, float
    sample_size: number of samples to generate, int
    ----------------------------
    Outputs:
    M_accepted: samples from pdf, array
    """
    
    X = np.random.uniform(alpha, beta, sample_size)
    y_max = f(X).max()
    X = None
    M_accepted = []
    while len(M_accepted) < sample_size:
        M = np.random.uniform(alpha, beta)
        y = np.random.uniform(0, y_max)
        if y < f(M):
            M_accepted.append(M)
    return M_accepted


def pdf_e(M, mu, sigma, lam, f):
    """
    This function defines the probability density function of M,
    within the limits alpha and beta.
    It is a truncated normal distribution for the signal component,
    and a truncated exponential distribution for the background component.
    Thus, it doesnt reauire normalisation. And is component-wise normalised.
    ----------------------------
    Inputs:
    M: r.v. M, float (when calculating a probability for a specific value of M)
    mu: mean of signal (normal) component, float, must be within alpha and beta
    sigma: width of signal (normal) component, float, must be positive
    lam: decay parameter of background (exponential) component, float
    f: ratio of signal/background components, float
    """

    # define the pdf
    pdf = ( f * norm.pdf(M, loc = mu, scale = sigma)
            +
            (1-f) * expon.pdf(M, 0, 1/lam)
            )

    return pdf


def pdf_norm_e(M, mu, sigma, lam, f):
    """
    This function defines the probability density function of M,
    within the limits alpha and beta.
    It is a truncated normal distribution for the signal component,
    and a truncated exponential distribution for the background component.
    Thus, it doesnt reauire normalisation. And is component-wise normalised.
    ----------------------------
    Inputs:
    M: r.v. M, float (when calculating a probability for a specific value of M)
    mu: mean of signal (normal) component, float, must be within alpha and beta
    sigma: width of signal (normal) component, float, must be positive
    lam: decay parameter of background (exponential) component, float
    f: ratio of signal/background components, float
    """
    alpha = 5
    beta = 5.6

    # define the pdf
    pdf = ( f * truncnorm.pdf(M, alpha, beta, loc = mu, scale = sigma)
            +
            (1-f) * truncexpon.pdf(M, alpha, beta, 0, 1/lam)
            )

    return pdf


def signal_e(M, mu, sigma):
    """ 
    This function defines the signal component of the pdf.
    ----------------------------
    Inputs:
    M: r.v. M, float (when calculating a probability for a specific value of M)
    mu: mean of signal (normal) component, float, must be within alpha and beta
    sigma: width of signal (normal) component, float, must be positive
    ----------------------------
    Outputs:
    signal: signal component of pdf, float
    """
    alpha = 5
    beta = 5.6

    signal = truncnorm.pdf(M, alpha, beta, loc = mu, scale = sigma)

    return signal


def signal_norm_e(M, mu, sigma):
    """ 
    This function defines the signal component of the pdf.
    ----------------------------
    Inputs:
    M: r.v. M, float (when calculating a probability for a specific value of M)
    mu: mean of signal (normal) component, float, must be within alpha and beta
    sigma: width of signal (normal) component, float, must be positive
    ----------------------------
    Outputs:
    signal: signal component of pdf, float
    """
    
    signal = norm.pdf(M, loc = mu, scale = sigma)

    return signal


def background_e(M, lam):
    """ 
    This function defines the background component of the pdf.
    ----------------------------
    Inputs:
    M: r.v. M, float (when calculating a probability for a specific value of M)
    lam: decay parameter of background (exponential) component, float
    ----------------------------
    Outputs:
    background: background component of pdf, float
    """

    background = expon.pdf(M, 0, 1/lam)

    return background


def background_norm_e(M, lam):
    """ 
    This function defines the background component of the pdf.
    ----------------------------
    Inputs:
    M: r.v. M, float (when calculating a probability for a specific value of M)
    lam: decay parameter of background (exponential) component, float
    ----------------------------
    Outputs:
    background: background component of pdf, float
    """
    alpha = 5
    beta = 5.6

    background = truncexpon.pdf(M, alpha, beta, 0, 1/lam)

    return background


def cdf_e(M, mu, sigma, lam, f):
    """
    This function defines the cumulative distribution function of M,
    within the limits alpha and beta.
    ----------------------------
    Inputs:
    M: r.v. M, float (when calculating a probability for a specific value of M)
    mu: mean of signal (normal) component, float, must be within alpha and beta
    sigma: width of signal (normal) component, float, must be positive
    lam: decay parameter of background (exponential) component, float
    f: ratio of signal/background components, float
    """
    alpha = 5
    beta = 5.6

    # define the pdf
    cdf = ( f * truncnorm.cdf(M, alpha, beta, loc = mu, scale = sigma)
            +
            (1-f) * truncexpon.cdf(M, alpha, beta, 0, 1/lam)
            )

    return cdf

def plot_pdf_e(pdf, gen_sample, mu_hat, sigma_hat, lam_hat, f_hat, alpha, beta):
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
    x = np.linspace(alpha, beta, 1000)

    # Define signal component
    signal_ = signal_norm_e(x, mu_hat, sigma_hat)

    # Define background component
    background_ = background_norm_e(x, lam_hat)

    # Define pdf
    pdf_true = pdf(x, mu_hat, sigma_hat, lam_hat, f_hat)

    # Plot
    plt.hist(gen_sample, color = 'b', alpha = 0.7, bins = 100, density = True, label = 'Generated sample') 
    plt.plot(x, signal_,'--',color = 'r', label='Signal, s(M; \u03BC, \u03C3)')
    plt.plot(x, background_,'--', color = 'g', label='Background, b(M; \u03BB)')
    plt.plot(x, pdf_true,'--', color = 'k', label='PDF')
    plt.xlabel('M')
    plt.ylabel('Probability density')
    plt.legend()
    proj_dir = os.path.dirname(os.getcwd())
    plot_dir = os.path.join(proj_dir, 'plot_pdf_e.png')
    plt.savefig(plot_dir)
    plt.show()

    return None


#----------------------------------------------------------------
#------------------------ PART F --------------------------------
#----------------------------------------------------------------


def plot_f(sample_sizes, discovery_rates):
    """
    This function plots the discovery rate against the sample size.
    ----------------------------
    Inputs:
    sample_sizes: sample sizes, array
    discovery_rates: discovery rates, array
    ----------------------------
    Outputs:
    plot of discovery rate against sample size
    """

    plt.plot(sample_sizes, discovery_rates, color = '--', marker = 'x', label='Discovery rate')
    plt.xlabel('Sample size')
    plt.ylabel('Discovery rate (%)')
    plt.legend()
    proj_dir = os.path.dirname(os.getcwd())
    plot_dir = os.path.join(proj_dir, 'plot_e.png')
    plt.savefig(plot_dir)
    plt.show()

    return None