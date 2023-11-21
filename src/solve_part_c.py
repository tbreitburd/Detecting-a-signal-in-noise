# Code these expressions for the probability density in python, in any way you so
# desire. Plugging in some different values for the parameters 𝜽, use a numerical
# integration to convince yourself that the total probability density integrates to unity
# in the range 𝑀 ∈ [5, 5.6]. [2]
# "
# You may use predefined methods in libraries such as scipy.stats and
# scipy.integrate to help you. 

import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from funcs.funcs import pdf

# Define parameters
#mu = [5, 5.3, 5.6]; sigma = [0.02, 0.01, 0.03]; lam = [0.5, 0.6, 0.4]; f = [0.9, 0.5, 0.1]; alpha = 5; beta = 5.6

Int_1,_ = integrate.fixed_quad(lambda x: pdf(x, 5, 0.02, 0.5, 0.9, 5, 5.6), 5, 5.6, n=10)
Int_2,_ = integrate.fixed_quad(lambda x: pdf(x, 5.3, 0.01, 0.6, 0.5, 5, 5.6), 5, 5.6, n=10)
Int_3,_ = integrate.fixed_quad(lambda x: pdf(x, 5.6, 0.03, 0.4, 0.1, 5, 5.6), 5, 5.6, n=10)

print(Int_1, Int_2, Int_3)

