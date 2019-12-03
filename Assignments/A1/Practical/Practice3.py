import itertools as it
from math import exp, sqrt, pi
import scipy.stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

# -------- Constants --------

ITERATIONS = pow(10,4)

def displayGrid(gridnum, title):

    fig, ax = plt.subplots()

    min_val, max_val = -1.5, 1.5
    ind_array = np.arange(min_val + 0.5, max_val + 0.5, 1.0)
    x, y = np.meshgrid(ind_array, ind_array)

    for i in range(dataset[gridnum].shape[0]):
        for j in range(dataset[gridnum].shape[1]):
            if dataset[gridnum][i][j] == -1:
                c = 'x'
                ax.text(i-1, j-1, c, va='center', ha='center', fontsize=50, color='r')
            else:
                c = 'o'
                ax.text(i-1, j-1, c, va='center', ha='center', fontsize=50, color='b')            
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xticks(ind_array)
    ax.set_yticks(ind_array)
    ax.grid()
    plt.title(title, fontsize=18)
    plt.show()

def M0():
    return 1/512

def M1(dataset, param):
    result = 1

    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            
            x1 = 0
            if j == 0:
                x1 = -1
            elif j == 2:
                x1 = 1

            result *= (1 / (1 + exp(-dataset[i][j] * param * x1)))
            
    return result

def M2(dataset, param1, param2):
    result = 1

    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):

            x1 = 0
            x2 = 0
            if j == 0:
                x1 = -1
            elif j == 2:
                x1 = 1
            if i == 0:
                x2 = 1
            elif i == 2:
                x2 = -1

            result *= (1 / (1 + exp(-dataset[i][j] * (param1 * x1 + param2 * x2))))
            
    return result

def M3(dataset, param1, param2, param3):
    result = 1

    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):

            x1 = 0
            x2 = 0
            if j == 0:
                x1 = -1
            elif j == 2:
                x1 = 1
            if i == 0:
                x2 = 1
            elif i == 2:
                x2 = -1

            result *= (1 / (1 + exp(-dataset[i][j] * (param1 * x1 + param2 * x2 + param3))))
            
    return result

def getDistribution():
    mean = np.zeros(3)
    #mean = 5 * np.ones(3) #Uncoment for mean 5
    #variance = pow(10,2) * np.identity(3)
    variance = np.random.rand(3,3) #Uncoment for semi-definite random covariance
    variance = pow(10,2) * np.dot(variance, np.transpose(variance)) #Uncoment for semi-definite random covariance
    
    return mean, variance

def computeModels(results, dataset, i, mean, variance):

    results[i][0] = M0()

    for k in (range(ITERATIONS)):          
        theta = np.random.multivariate_normal(mean,variance)
        results[i][1] += M1(dataset, theta[0])
        results[i][2] += M2(dataset, theta[0], theta[1])
        results[i][3] += M3(dataset, theta[0], theta[1], theta[2])

    results[i][1] = results[i][1] / ITERATIONS
    results[i][2] = results[i][2] / ITERATIONS
    results[i][3] = results[i][3] / ITERATIONS

    return results[i]

# Define x belongs to {-1,1}

length = 9
values = (-1,1)

dataset = it.product(values,repeat=length)
dataset = list(dataset)


for i in range(len(dataset)):
    dataset[i] = np.reshape(np.array(dataset[i]), (3,3))

results = np.zeros((len(dataset), 4))
mean, variance = getDistribution()
num_cores = multiprocessing.cpu_count() # Parallel code for computing models avg. probabilities
results = Parallel(n_jobs=num_cores)(delayed(computeModels)(results, dataset[i], i, mean, variance) for i in tqdm(range(results.shape[0])))

results = np.transpose(results)
evidence_sum = np.zeros(shape=(results.shape[1]))

for i in range(results.shape[1]):
    for j in range(results.shape[0]):
        evidence_sum[i] += results[j][i]

evidence_sum = np.reshape(evidence_sum, (len(evidence_sum), 1))

distances = scipy.spatial.distance.cdist(evidence_sum, evidence_sum, metric="euclidean")
D = list(range(evidence_sum.shape[0]))
L = []
L.append(evidence_sum.argmin())
D.remove(L[-1])

while len(D) > 0:
        N = [d for d in D if distances[d, D].min() > distances[d, L[-1]]]
        if len(N) == 0:
            L.append(D[distances[ L[ -1 ], D ].argmin()])
        else:
            L.append(N[distances[L[-1],N].argmax()])
        D.remove(L[-1])

finalresult = np.array(L)[::-1]

results = np.transpose(results)

for i in range(results.shape[1]):
    plt.plot(results[finalresult,i], drawstyle="steps", label=f"Model {i}")
plt.xlabel('$Sample$', fontsize=12)
plt.ylabel('$Probability$', fontsize=12)
plt.legend(loc=1)
plt.show()

for i in range(results.shape[1]):
    plt.plot(results[finalresult,i], drawstyle="steps", label=f"Model {i}")
plt.xlim(right=100)
plt.xlabel('$Sample$', fontsize=12)
plt.ylabel('$Probability$', fontsize=12)
plt.legend(loc=1)
plt.show()

results = np.transpose(results)

for i in range(results.shape[0]):
    title = "Maximum sample in model M{}".format(i)
    displayGrid(results[i].argmax(), title)
    title = "Minimum sample in model M{}".format(i)
    displayGrid(results[i].argmin(), title)


