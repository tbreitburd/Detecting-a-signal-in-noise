# Code these expressions for the probability density in python, in any way you so
# desire. Plugging in some different values for the parameters 𝜽, use a numerical
# integration to convince yourself that the total probability density integrates to unity
# in the range 𝑀 ∈ [5, 5.6]. [2]
# "
# You may use predefined methods in libraries such as scipy.stats and
# scipy.integrate to help you. 

import scipy.integrate as integrate
from funcs import pdf_norm

# Define parameters
#mu = [5, 5.3, 5.6]; sigma = [0.02, 0.01, 0.03]; lam = [0.5, 0.6, 0.4]; f = [0.9, 0.5, 0.1]; alpha = 5; beta = 5.6

Int_1,_ = integrate.fixed_quad(lambda x: pdf_norm(x, 5, 0.02, 0.5, 0.9, 5, 5.6), 5, 5.6, n=10)
Int_2,_ = integrate.fixed_quad(lambda x: pdf_norm(x, 5.3, 0.01, 0.6, 0.5, 5, 5.6), 5, 5.6, n=10)
Int_3,_ = integrate.fixed_quad(lambda x: pdf_norm(x, 5.6, 0.03, 0.4, 0.1, 5, 5.6), 5, 5.6, n=10)

print("The integral of the pdf, for the given parameters is: \n" +
      "\u03BC = 5, \u03C3 = 0.02, \u03BB = 0.5, f = 0.9: " + str(Int_1) + "\n" +
      "\u03BC = 5.3, \u03C3 = 0.01, \u03BB = 0.6, f = 0.5: " + str(Int_2) + "\n" +
      "\u03BC = 5.6, \u03C3 = 0.03, \u03BB = 0.4, f = 0.1: " + str(Int_3))
