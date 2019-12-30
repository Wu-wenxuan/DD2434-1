import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import networkx as nx
import dendropy
import Kruskal_v1 as kruskal
from Tree import Tree, TreeMixture
from tqdm import tqdm

def save_results(loglikelihood, topology_array, theta_array, filename):
    """ This function saves the log-likelihood vs iteration values,
        the final tree structure and theta array to corresponding numpy arrays. """

    likelihood_filename = filename + "_em_loglikelihood.npy"
    topology_array_filename = filename + "_em_topology.npy"
    theta_array_filename = filename + "_em_theta.npy"
    print("Saving log-likelihood to ", likelihood_filename, ", topology_array to: ", topology_array_filename,
          ", theta_array to: ", theta_array_filename, "...")
    np.save(likelihood_filename, loglikelihood)
    np.save(topology_array_filename, topology_array)
    np.save(theta_array_filename, theta_array)

def computeLikelihood(samples, topology, theta):
    result = theta[0][samples[0]]
    for i in range(1, len(topology)):
        result *= theta[i][samples[int(topology[i])]][samples[i]]

    return (result + sys.float_info.epsilon)

def computeLogLikelihood(pi, likelihood):
    result = 0
    for i in range(likelihood.shape[0]):
        aux = 0
        for k in range(likelihood.shape[1]):
            aux += pi[k] * likelihood[i,k]
        result += np.log(aux)
    
    return result

def computeCondQ(responsibility, samples, node1, node2, value1, value2):
    num = 0
    denom = 0
    for i in range(samples.shape[0]):
        if (samples[i,node1] == value1):
            denom += responsibility[i]
            if (samples[i,node2] == value2):
                num += responsibility[i]

    return (num / (denom + sys.float_info.epsilon))

def computeQ(responsibility, samples, node, val):
    num = 0
    for i in range(samples.shape[0]):
        if (samples[i,node] == val):
            num += responsibility[i]      
    denom = np.sum(responsibility) + sys.float_info.epsilon

    return (num / denom)

def computeQJoint(responsibility, samples, node1, node2, val1, val2):
    num = 0
    for i in range(samples.shape[0]):
        if (samples[i,node1] == val1) and (samples[i,node2] == val2):
            num += responsibility[i]      
    denom = np.sum(responsibility) + sys.float_info.epsilon

    return (num / denom)

def computeResponsibility(num_clusters, samples, pi, likelihood):
    result = np.zeros((samples.shape[0], num_clusters))
    for i in range(samples.shape[0]):
        for k in range(num_clusters):
            result[i,k] = pi[k] * likelihood[i,k]
        result[i] = (result[i] + sys.float_info.epsilon) / (np.sum(result[i]) + num_clusters * sys.float_info.epsilon)

    return result

def calculateI(responsibility, samples, node1, node2):
    result = 0
    for i in range(2):
        for j in range(2):
            q_node1 = computeQ(responsibility, samples, node1, i)
            q_node2 = computeQ(responsibility, samples, node2, j)
            q_joint = computeQJoint(responsibility, samples, node1, node2, i, j)
            if (q_joint != 0):
                if (q_node1 == 0):
                    q_node1 = sys.float_info.epsilon
                if (q_node2 == 0):
                    q_node2 = sys.float_info.epsilon
                if ((q_node1 * q_node2) == 0):
                    result += q_joint * np.log(q_joint / (sys.float_info.min))
                else:
                    result += q_joint * np.log(q_joint / (q_node1 * q_node2))

    return result

def computeTheta(theta_list, responsibility, samples, topology, num_nodes, num_clusters):
    result = theta_list
    for k in range(num_clusters):
        for i in range(num_nodes):
            if (i == 0):
                result[k][0,0] = computeQ(responsibility[:,k], samples, 0, 0)
                result[k][0,1] = computeQ(responsibility[:,k], samples, 0, 1)
            else:
                result[k][i,0][0]= computeCondQ(responsibility[:,k], samples, int(topology[k][i]), i, 0, 0)
                result[k][i,0][1]= computeCondQ(responsibility[:,k], samples, int(topology[k][i]), i, 0, 1)
                result[k][i,1][0]= computeCondQ(responsibility[:,k], samples, int(topology[k][i]), i, 1, 0)
                result[k][i,1][1]= computeCondQ(responsibility[:,k], samples, int(topology[k][i]), i, 1, 1)

    return result

def computationsEM(iterations, samples, num_clusters, tm, topology_list, theta_list):
    pi = tm.pi
    loglikelihood = np.zeros(iterations)

    for it in range(iterations):
        num_samples = samples.shape[0]
        num_nodes = samples.shape[1]
        likelihood = np.zeros((num_samples, num_clusters)) # Probability of having this sample per tree

        # Compute likelihood per sample
        for i in range(num_samples):
            for k in range(num_clusters):
                likelihood[i,k] = computeLikelihood(samples[i,:], topology_list[k], theta_list[k])

        # Computation of responsibilities
        responsibility = computeResponsibility(num_clusters, samples, pi, likelihood)

        # Computation of pi'
        res_sum = np.sum(responsibility, axis=0)
        total_sum = np.sum(res_sum)
        pi = np.zeros(len(res_sum))
        for i in range(len(res_sum)):
            pi[i] = res_sum[i] / total_sum

        # Get the IQ for using as weights
        IQ = np.zeros((num_nodes, num_nodes, num_clusters))
        for k in range(num_clusters):
            for i in range(len(topology_list[k])):
                for j in range(len(topology_list[k])):
                    if (i != j):
                        IQ[i,j,k] = calculateI(responsibility[:,k], samples, i, j)     

        # Create the graphs
        graphs = list()
        for k in range(num_clusters):
            graphs.append(kruskal.Graph(num_nodes))
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    graphs[-1].addEdge(i, j, IQ[i,j,k])

        # Get the Maximum Spanning Tree from each graph
        tree = np.zeros((num_nodes-1, 3, num_clusters))
        for k in range(num_clusters):
            result = graphs[k].maximum_spanning_tree()
            cnt = 0
            for u_aux, v_aux, weight_aux in result:
                tree[cnt,0,k] = u_aux
                tree[cnt,1,k] = v_aux
                tree[cnt,2,k] = weight_aux
                cnt += 1

        # Creation of the tree
        topology_list = list()
        for k in range(num_clusters):
            topology_list.append(np.zeros(num_nodes))
            topology_list[-1][0] = np.nan
            max_tree = nx.Graph()
            for i in range(tree.shape[0]):
                max_tree.add_edge(tree[i,0,k], tree[i,1,k])
            finaltree = list(nx.bfs_edges(max_tree, 0))
            for i in range(num_nodes - 1):
                topology_list[-1][int(finaltree[i][1])] = finaltree[i][0]

        # Computation of theta'
        theta_list = computeTheta(theta_list, responsibility, samples, topology_list, num_nodes, num_clusters)

        loglikelihood[it] = computeLogLikelihood(pi, likelihood)

    return loglikelihood   


def em_algorithm(seed_val, samples, num_clusters, max_num_iter):

    # Initialize the needed variables    
    sieving = 100
    max_log = float("-inf")
    best_seed = 0

    # Get the best seed for likelihood
    for siev in tqdm(range(sieving)):
        # Set the seed
        aux_seed = seed_val + siev # Try with all seeds from @param:seed_val to @param:seed_val + sieving
        np.random.seed(aux_seed)

        # Generate tree mixture
        tm = TreeMixture(num_clusters=num_clusters, num_nodes=samples.shape[1])
        tm.simulate_pi(seed_val=aux_seed)
        tm.simulate_trees(seed_val=aux_seed)
        topology_list = []
        theta_list = []

        for i in range(num_clusters):
            topology_list.append(tm.clusters[i].get_topology_array())
            theta_list.append(tm.clusters[i].get_theta_array())

        # Run 10 iterations according to this mixture
        loglikelihood = computationsEM(10, samples, num_clusters, tm, topology_list, theta_list)          

        aux = loglikelihood[-1]
        if (aux > max_log):
            max_log = aux
            best_seed = aux_seed

    # -------------------- End of sieving -------------------- #

    # Variable initialization
    np.random.seed(best_seed)
    topology_list = [] # Dimensions: (num_clusters, num_nodes)
    theta_list = [] # Dimensions: (num_clusters, num_nodes, 2)
    tm = TreeMixture(num_clusters = num_clusters, num_nodes = samples.shape[1])
    tm.simulate_pi(seed_val = best_seed)
    tm.simulate_trees(seed_val = best_seed)

    for k in range(num_clusters):
        topology_list.append(tm.clusters[k].get_topology_array())
        theta_list.append(tm.clusters[k].get_theta_array())

    # Beginning of iterations
    pi = tm.pi
    loglikelihood = computationsEM(max_num_iter, samples, num_clusters, tm, topology_list, theta_list)

    return loglikelihood, topology_list, theta_list


def main():
    # Code to process command line arguments
    parser = argparse.ArgumentParser(description='EM algorithm for likelihood of a tree GM.')
    parser.add_argument('sample_filename', type=str,
                        help='Specify the name of the sample file (i.e data/example_samples.txt)')
    parser.add_argument('output_filename', type=str,
                        help='Specify the name of the output file (i.e data/example_results.txt)')
    parser.add_argument('num_clusters', type=int, help='Specify the number of clusters (i.e 3)')
    parser.add_argument('--seed_val', type=int, default=42, help='Specify the seed value for reproducibility (i.e 42)')
    parser.add_argument('--real_values_filename', type=str, default="",
                        help='Specify the name of the real values file (i.e data/example_tree_mixture.pkl)')
    # You can add more default parameters if you want.

    print("This file demonstrates the flow of function templates of question 2.5.")

    print("\n0. Load the parameters from command line.\n")

    args = parser.parse_args()
    print("\tArguments are: ", args)

    print("\n1. Load samples from txt file.\n")

    samples = np.loadtxt(args.sample_filename, delimiter="\t", dtype=np.int32)
    num_samples, num_nodes = samples.shape
    print("\tnum_samples: ", num_samples, "\tnum_nodes: ", num_nodes)
    print("\tSamples: \n", samples)

    print("\n2. Run EM Algorithm.\n")
    max_iterations = 100 # Maximum number of iterations for the EM algorithm
    loglikelihood, topology_array, theta_array = em_algorithm(args.seed_val, samples, args.num_clusters, max_iterations)

    print("\n3. Save, print and plot the results.\n")

    save_results(loglikelihood, topology_array, theta_array, args.output_filename)

    for i in range(args.num_clusters):
        print("\n\tCluster: ", i)
        print("\tTopology: \t", topology_array[i])
        print("\tTheta: \t", theta_array[i])

    print("\n4. Retrieve real results and compare.\n")
    if args.real_values_filename != "":
        print("\tComparing the results with real values...")

        print("\t4.1. Make the Robinson-Foulds distance analysis.\n")

        tns = dendropy.TaxonNamespace()
        original_tree = list()
        original_topology = list()

        for k in range(args.num_clusters):
            filename = args.real_values_filename + "_tree_" + str(k) + "_topology.npy"
            original_topology.append(np.load(filename))
            original_tree.append(Tree())
            original_tree[-1].load_tree_from_direct_arrays(original_topology[-1])
            original_tree[-1] = dendropy.Tree.get(data = original_tree[-1].newick, schema = "newick", taxon_namespace = tns)

        generated_tree = list()

        for k in range(args.num_clusters):
            generated_tree.append(Tree())
            generated_tree[-1].load_tree_from_direct_arrays(topology_array[k])
            generated_tree[-1] = dendropy.Tree.get(data = generated_tree[-1].newick, schema = "newick", taxon_namespace = tns)
            print("Generated tree ", k, " ",generated_tree[-1].as_string("newick"))
            generated_tree[-1].print_plot()

        print("\tDistances of trees:\n")
        for k in range(args.num_clusters):
            for i in range(args.num_clusters):
                print("\tOriginal tree",k,"compared to generated tree",i)
                print("\t\tRobinson-Foulds distance:", dendropy.calculate.treecompare.symmetric_difference(original_tree[k], generated_tree[i]))     


        print("\n\t4.2. Make the likelihood comparison.\n")

        original_theta = list()
        for k in range(args.num_clusters):
            filename = args.real_values_filename + "_tree_" + str(k) + "_theta.npy"
            original_theta.append(np.load(filename, allow_pickle = True))
        
        filename = args.real_values_filename + "_pi.npy"
        original_pi = np.load(filename)
        original_likelihood = np.zeros((num_samples, args.num_clusters))
        for i in range(num_samples):
            for k in range(args.num_clusters):
                original_likelihood[i,k] = computeLikelihood(samples[i,:], original_topology[k], original_theta[k])

        original_log_likelihood = computeLogLikelihood(original_pi, original_likelihood)
        original_log_likelihood_array = [original_log_likelihood for i in range(max_iterations)]

        plt.figure(figsize=(16, 7))
        plt.subplot(121)
        plt.plot(np.exp(loglikelihood), label='Estimated')
        plt.plot(np.exp(original_log_likelihood_array), label='Real',color = 'r')
        plt.ylabel("Likelihood of Mixture")
        plt.xlabel("Iterations")
        plt.subplot(122)
        plt.plot(loglikelihood, label='Estimated')
        plt.plot(original_log_likelihood_array, label='Original',color = 'r')
        plt.ylabel("Log-Likelihood of Mixture")
        plt.xlabel("Iterations")
        plt.legend(loc=(1.04, 0))
        plt.show()

        print("End of execution.\n")

if __name__ == "__main__":
    main()