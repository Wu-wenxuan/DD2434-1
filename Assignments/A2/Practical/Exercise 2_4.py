# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm

# This function computes the value of tau_N for a normal distribution
def compute_tauN(tau_0, mu_0, vector, a_N, b_N):
    
    tau_N = (tau_0 + len(vector)) * (a_N / b_N)
    return tau_N

# This function computes the values of a_N and b_N for a gamma distribution
def compute_bN(a_0, b_0, vector, mu_N, mu_0, tau_N):
    
    term1 = (1 / tau_N) + pow(mu_N,2) + pow(mu_0,2) - 2 * mu_N * mu_0
    term2 = 0
    for x in vector:
        term2 += pow(x,2) + (1 / tau_N) + pow(mu_N,2) - 2 * mu_N * x
    b_N = b_0 + 0.5 * tau_0 * term1 + 0.5 * term2
    
    return b_N

if __name__ == "__main__":

    # VARIABLES
    iterations = 10                 # Number of iterations
    n = 100                         # Number of samples
    mu_0 = 0                        # Initial value mu_0
    mu_N = 0                        # Initial value mu_N
    tau_0 = 0                       # Initial value tau_0
    tau_N = 0                       # Initial value mu_N
    a_0 = 0                         # Initial value a_0
    a_N = 1                         # Initial value a_N
    b_0 = 1                         # Initial value b_0
    b_N = 1                         # Initial value b_N
    x_axis = np.arange(-1,1,0.02)   # X axis grid
    y_axis = np.arange(0,2,0.02)    # Y axis grid

    # SAMPLES GENERATION
    vector = np.random.normal(0,1,n)

    mu_N = (tau_0 * mu_0 + np.sum(vector)) / (tau_0 + len(vector))  # mu_N value computation (it won't update)
    a_N = a_0 + (len(vector)) / 2                                   # a_N value computation (it won't update)

    for i in range(iterations):
        tau_N = compute_tauN(tau_0, mu_0, vector, a_N, b_N)
        b_N = compute_bN(a_0, b_0, vector, mu_N, mu_0, tau_N)

    q_mu_tau = np.zeros((len(x_axis), len(y_axis)))

    for i in tqdm(range(len(x_axis))):
        for j in range(len(y_axis)):
            q_mu = stats.norm(mu_N, 1/tau_N).pdf(x_axis[i])
            q_tau = stats.gamma.pdf(y_axis[j], a_N, loc=0, scale=(1/b_N))
            q_mu_tau[i][j] = q_mu * q_tau
            
    plt.contour(x_axis, y_axis, q_mu_tau)
    plt.show()