import pylab as pb
import numpy as np
from math import pi, exp, sin
from scipy.stats import multivariate_normal as mv_norm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from random import sample 

# ------------------------------ Exercises Switch ------------------------------ #

ex9 = False
ex10 = False
ex11 = True

# ------------------------------ Functions ------------------------------ #

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

def calculatePosterior(mean, variance, point_x, point_t, sigma):

    beta = 1/sigma
    phi = np.array([np.repeat(1, len(point_x)),point_x])
    phi = np.transpose(phi)
    reshaped_t = np.reshape(point_t, (len(point_t), 1)) # Set vector as N rows 1 column
    mean = np.reshape(mean, (len(mean), 1)) # Set vector as 2 rows 1 column

    new_variance = np.linalg.inv(np.linalg.inv(variance) + beta * np.dot(np.transpose(phi), phi))
    new_mean = np.dot(new_variance, (np.dot(np.linalg.inv(variance), mean) + beta * np.dot(np.transpose(phi), reshaped_t)))

    new_mean = np.reshape(new_mean, (2,))

    f = mv_norm(new_mean,new_variance)
    plt.contourf(w_0, w_1, f.pdf(pos))
    if len(point_x) == 1:
        plt.title("For %d point" %len(point_x))
    else:
        plt.title("For %d points" %len(point_x))
    plt.xlabel('$w_0$', fontsize=16)
    plt.ylabel('$w_1$', fontsize=16)
    plt.show()

    return new_mean, new_variance

def calcKernel(sigma, xi, xj, l):

    matrix = [[0 for x in range(len(xj))] for y in range(len(xi))] 

    for i in range(len(xi)):
        for j in range(len(xj)):
            matrix[i][j] = pow(sigma,2) * exp(- (np.dot(np.transpose(xi[i] - xj[j]), (xi[i] - xj[j])) / pow(l,2)))

    return np.array(matrix)

def predPosterior(xi, X, f, sigma, sigmaf, l, noise):

    if noise:
        pred_mean = np.dot(np.dot(calcKernel(sigmaf, xi, X, l), np.linalg.inv(calcKernel(sigmaf, X, X, l) + sigma * np.identity(len(X)))), f)
        pred_variance = calcKernel(sigmaf, xi, xi, l) - np.dot(np.dot(calcKernel(sigmaf, xi, X, l), np.linalg.inv(calcKernel(sigmaf, X, X, l) + sigma * np.identity(len(X)))), calcKernel(sigmaf, X, xi, l))
    else:
        pred_mean = np.dot(np.dot(calcKernel(sigmaf, xi, X, l), np.linalg.inv(calcKernel(sigmaf, X, X, l))), f)
        pred_variance = calcKernel(sigmaf, xi, xi, l) - np.dot(np.dot(calcKernel(sigmaf, xi, X, l), np.linalg.inv(calcKernel(sigmaf, X, X, l))), calcKernel(sigmaf, X, xi, l))

    return pred_mean, pred_variance

# ------------------------------ Question 9 ------------------------------ #

if ex9:

    # ti =w0*xi + w1 + ε = 0.5*xi − 1.5 + ε
    # x = [−1,−0.99,...,0.99,1]
    # ε ∼ N (0, 0.2)

    a1 = 0.5 # final value for w0
    a0 = -1.5 # final value for w1
    sigma =  0.2

    x = np.arange(-1,1.01,0.01)

    # Avoid strange decimals

    for i in range(len(x)):
        x[i] = round(x[i],2)

    # X already filled from [-1,1] step 0.01

    t = np.zeros(len(x))

    t = a1 * x + a0

    alpha = 2.0 # Arbitrary number --> Bishop
    mean = np.array([0.,0.])
    variance = np.array([[alpha,0.],[0.,alpha]])

    # ------------------------------------------------------------ #

    # First iteration

    f = mv_norm(mean,variance)

    w_0, w_1 = np.mgrid[-2.0:2.0:.01, -2.0:2.0:.01]
    pos = np.empty(w_0.shape + (2,))
    pos[:, :,0] = w_0
    pos[:, :,1] = w_1
    f = mv_norm(mean,variance)
    plt.contourf(w_0, w_1, f.pdf(pos))
    plt.xlabel('$w_0$', fontsize=16)
    plt.ylabel('$w_1$', fontsize=16)
    plt.show()

    # Points calculation

    #TODO: update or not mean and variance

    for i in range(1, 8):
        #auxmean = mean
        #auxvar = variance
        point_x = sample(x.tolist(), i)
        point_t = a1 * np.asarray(point_x) + a0 + np.random.normal(0, sigma, len(point_x))
        mean, variance = calculatePosterior(mean, variance, point_x, point_t, sigma)
        for j in range(5):
            smp_0, smp_1 = np.random.multivariate_normal(mean, variance)
            t_i = smp_1 * x + smp_0
            plt.plot(x,t_i, color='black')
        if(i == 1):
            plt.title("Plot for %d point with sigma = %.2f" %(i, sigma))
        else:
            plt.title("Plot for %d points with sigma = %.2f" %(i, sigma))
        plt.plot(x,t, color='red')
        plt.show()

if ex10:

    sigma = 1
    l = 1

    x = np.arange(-1,1.01,0.01)
    mean = np.zeros(len(x))

    # Avoid strange decimals

    for i in range(len(x)):
        x[i] = round(x[i],2)

    covariance = calcKernel(sigma, x, x, l)

    for i in range(10):
        prior = np.random.multivariate_normal(mean,covariance)
        plt.plot(x,prior)

    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', fontsize=16)
    plt.show()
    
if ex11:

    sigma = 3
    sigmaf = 1
    l = 1
    xi = np.array(np.arange(-10,10,0.1))
    xi = np.transpose(xi)
    X = [-4,-3,-2,-1,0,2,3,5]
    X = np.transpose(X)

    f_Noise = np.zeros(len(X))
    t = np.zeros(len(xi))
    t_noNoise = np.zeros(len(X))

    for i in range(len(f_Noise)): # Points generated with noise
        f_Noise[i] = (2 + pow((0.5 * X[i] - 1), 2)) * sin(3 * X[i]) + np.random.normal(0, sigma, 1)

    for i in range(len(t)): # Line generated without noise
        t[i] = (2 + pow((0.5 * xi[i] - 1), 2)) * sin(3 * xi[i])

    for i in range(len(X)): # Points generated without noise
        t_noNoise[i] = (2 + pow((0.5 * X[i] - 1), 2)) * sin(3 * X[i])

    f_Noise = np.transpose(f_Noise)

    noise = False

    pred_mean, pred_variance = predPosterior(xi, X, t_noNoise, sigma, sigmaf, l, noise)

    for i in range(10):
        posterior = np.random.multivariate_normal(pred_mean,pred_variance)
        plt.plot(xi,posterior)
    plt.plot(xi, t)
    plt.plot(X, t_noNoise, 'o', color='black')
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', fontsize=16)
    plt.title("Without noise", fontsize=24)
    plt.show()

    noise = True

    pred_mean, pred_variance = predPosterior(xi, X, f_Noise, sigma, sigmaf, l, noise)

    for i in range(10):
        posterior = np.random.multivariate_normal(pred_mean,pred_variance)
        plt.plot(xi,posterior)
    plt.plot(xi, t)
    plt.plot(X, f_Noise, 'o', color='black')
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', fontsize=16)
    plt.title("With noise", fontsize=24)
    plt.show()

# ------------------------------ Question 10 ------------------------------ #

# ------------------------------ Question 11 ------------------------------ #