import scipy.integrate as integrate
import numpy as np
from funcs import pdf_norm

# Set the number of tests we want to run to test
num_tests = 3

print("The integral of the pdf, for the given parameters is:")

# Loop over the number of tests
for i in range(num_tests):
      mu = np.random.uniform(5, 5.6) # Draw a random value for each parameter from a uniform distribution
      sigma = np.random.uniform(0.01, 0.03) 
      lam = np.random.uniform(0.3, 0.7)
      f = np.random.uniform(0.1, 0.9)

      Integral,_ = integrate.fixed_quad(lambda x: pdf_norm(x, mu, sigma, lam, f, 5, 5.6), 5, 5.6, n = 10)
      print("\u03BC = {:.4f}, \u03C3 = {:.4f}, \u03BB = {:.4f}, f = {:.4f}: {:.4f}".format(mu, sigma, lam, f, Integral))

