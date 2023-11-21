
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt


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

    background = stats.expon.pdf(M, scale=1/lam)

    return background

def pdf(M,mu, sigma, lam, f, alpha, beta):
    """
    This function defines the normalised probability density function of M,
    using the normalisation factor derived in (b)
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
    """

    
    lim_l = (alpha-mu)/(sigma*np.sqrt(2))
    lim_h = (beta-mu)/(sigma*np.sqrt(2))
    I,_ = integrate.fixed_quad(lambda x: np.exp(-x**2), lim_l, lim_h, n=10) # Difference of erf at limits, see (b)

    N = 1 / ( f * ((1/np.sqrt(np.pi)) * I)  +   (1-f) * ( np.exp(-lam*alpha) - np.exp(-lam*beta) )) # Normalisation factor
    
    # define the pdf
    pdf = N * ( f * stats.norm.pdf(M, mu, sigma)
                +
                (1-f) * stats.expon.pdf(M, scale=1/lam)
                )

    return pdf


def plot_pdf_d(mu, sigma, lam, f, alpha, beta):
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
    plt.show()

    return None