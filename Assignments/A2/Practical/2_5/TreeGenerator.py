from Tree import TreeMixture
import Exercise2_5
import argparse

# This file has the pupose of creating new samples for testing exercise 5

parser = argparse.ArgumentParser()
parser.add_argument("seed", help="Introduce the seed to generate trees", type=int)
parser.add_argument("samples", help="Introduce the number of samples", type=int)
parser.add_argument("nodes", help="Introduce the number of nodes", type=int)
parser.add_argument("clusters", help="Introduce the number of clusters", type=int)
args = parser.parse_args()
print("Generating tree with seed:", args.seed, "\tsamples:", args.samples, 
        "\tnodes:", args.nodes, "\tclusters:", args.clusters)
tm = TreeMixture(num_clusters=args.clusters, num_nodes=args.nodes)
tm.simulate_pi(seed_val=args.seed)
tm.simulate_trees(seed_val=args.seed)
tm.sample_mixtures(num_samples=args.samples, seed_val=args.seed)
path = 'data/q_2_5_tm_'+str(args.nodes)+'node_'+str(args.samples)+'sample_'+str(args.clusters)+'clusters.pkl'
tm.save_mixture(path, True)