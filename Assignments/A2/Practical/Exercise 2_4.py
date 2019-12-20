import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm

# This function computes the value of tau_N
def compute_tauN(tau_0, mu_0, vector, a_N, b_N):
    
    result = (tau_0 + len(vector)) * (a_N / b_N)
    return result

# This function computes the values of b_N
def compute_bN(a_0, b_0, vector, mu_0, tau_N):
    
    term1 = (1 / tau_N) + pow(np.mean(vector),2) + pow(mu_0,2) - 2 * np.mean(vector) * mu_0
    term2 = 0
    for x in vector:
        term2 += pow(x,2) + (1 / tau_N) + pow(np.mean(vector),2) - 2 * np.mean(vector) * x
    result = b_0 + 0.5 * tau_0 * term1 + 0.5 * term2
    
    return result

#This function computes the approximated distribution
def getQ(mu_axis, tau_axis, mu_N, tau_N, a_N, b_N):

    q_mu_tau = np.zeros((len(mu_axis), len(tau_axis)))

    for i in tqdm(range(len(mu_axis))):
        for j in range(len(tau_axis)):
            q_mu = stats.norm(mu_N, 1/(tau_N)).pdf(mu_axis[i])
            q_tau = stats.gamma.pdf(tau_axis[j], a_N, loc=0, scale=(1/b_N))
            q_mu_tau[j][i] = q_mu * q_tau

    return q_mu_tau

#This function computes the original distribution
def getP(mu_axis, tau_axis, mu_N, tau_N, a_N, b_N):

    p_mu_tau = np.zeros((len(mu_axis), len(tau_axis)))

    for i in tqdm(range(len(mu_axis))):
        for j in range(len(tau_axis)):
            if(tau_axis[j] == 0):
                tau_axis[j] = 0.00001 # Correction for 0 denominator
            p_mu = stats.norm(mu_N, 1/(tau_N * tau_axis[j])).pdf(mu_axis[i])
            p_tau = stats.gamma.pdf(tau_axis[j], a_N, loc=0, scale=(1/b_N))
            p_mu_tau[j][i] = p_mu * p_tau

    return p_mu_tau

if __name__ == "__main__":

    # VARIABLES
    iterations = 10                 # Number of iterations
    n = 10                          # Number of samples
    mu_0 = 0                        # Initial value mu_0
    mu_N = 1                        # Initial value mu_N
    tau_0 = 1                       # Initial value tau_0
    tau_N = 1                       # Initial value mu_N
    a_0 = 0                         # Initial value a_0
    a_N = 1                         # Initial value a_N
    b_0 = 1                         # Initial value b_0
    b_N = 1                         # Initial value b_N
    mu_axis = np.linspace(-5,5,100)  # X axis grid
    tau_axis = np.linspace(0,5,100)  # Y axis grid

    # SAMPLES GENERATION
    vector = np.random.normal(0,1,n)

    mu_N = (tau_0 * mu_0 + np.sum(vector)) / (tau_0 + len(vector))  # mu_N value computation (it won't update)
    a_N = a_0 + (len(vector)) / 2                                   # a_N value computation (it won't update)

    q_mu_tau = np.zeros((len(mu_axis), len(tau_axis)))
    p_mu_tau = np.zeros((len(mu_axis), len(tau_axis)))

    for i in range(iterations):
        tau_N = compute_tauN(tau_0, mu_0, vector, a_N, b_N)
        b_N = compute_bN(a_0, b_0, vector, mu_N, tau_N)
            
    p_mu_tau = getP(mu_axis, tau_axis, mu_N, tau_N, a_N, b_N)
    q_mu_tau = getQ(mu_axis, tau_axis, mu_N, tau_N, a_N, b_N)
    #plt.axis([-0.2,0.2, 0, 2])                                     # Uncomment this line to modify the grid area
    plt.contour(mu_axis, tau_axis, q_mu_tau, colors='red')
    plt.contour(mu_axis, tau_axis, p_mu_tau, colors='green')
    plt.show()