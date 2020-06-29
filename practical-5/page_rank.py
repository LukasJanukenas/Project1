import os
import time
import random
import networkx as nx
from progress import Progress

WEB_DATA = os.path.join(os.path.dirname(__file__), 'school_web.txt')

def load_graph(fd):
    """Load graph from text file

    Parameters:
    fd -- a file like object that contains lines of URL pairs

    Returns:
    A representation of the graph.

    Called for example with

    >>> graph = load_graph(open("web.txt"))

    the function parses the input file and returns a graph representation.
    Each line in the file contains two white space seperated URLs and
    denotes a directed edge (link) from the first URL to the second.
    """
    graph = nx.DiGraph()

    # Iterate through the file line by line
    for line in fd:
        # And split each line into two URLs
        line = line.split()
        graph.add_edge(line[0], line[1])
    return graph


def print_stats(graph):
    """Print number of nodes and edges in the given graph"""
    print("The number of nodes is:", graph.number_of_nodes())
    print("The number of edges is:", graph.number_of_edges())

def stochastic_page_rank(graph, n_iter=1_000_000, n_steps=100):
    """Stochastic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    n_iter (int) -- number of random walks performed
    n_steps (int) -- number of followed links before random walk is stopped

    Returns:
    A dict that assigns each page its hit frequency

    This function estimates the Page Rank by counting how frequently
    a random walk that starts on a random node will after n_steps end
    on each node of the given graph.
    """

    # using networkx get an iterator of all nodes in the graph
    nodes = graph.nodes()
    # initialize hit_count that will track each nodes hit_count
    hit_count = {node: 0 for node in graph.nodes}
    for x in range(0, n_iter):
        # choose a random node from the list of nodes
        current_node = random.choice(list(nodes))
        for z in range(0, n_steps):
            # choose a random node from the current nodes out edges
            current_node = random.choice(list(graph.neighbors(current_node)))
        # increase the current nodes hit count
        hit_count[current_node] += 1/n_iter
    return hit_count

def distribution_page_rank(graph, n_iter=100):
    """Probabilistic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    n_iter (int) -- number of probability distribution updates

    Returns:
    A dict that assigns each page its probability to be reached

    This function estimates the Page Rank by iteratively calculating
    the probability that a random walker is currently on any node."""

    # using networkx get an iterator of all nodes in the graph
    nodes = graph.nodes()
    # save the amount of nodes in the graph for later calculations
    amount = graph.number_of_nodes()
    # initialize both node_prob and next_prob dictionaries
    node_prob = {node: 1/amount for node in nodes}
    next_prob = {node: 0 for node in nodes}
    for x in range(0, n_iter):
        # iterate through each node in the graph
        for node in nodes:
            # initialize p
            p = node_prob[node]/graph.out_degree(node)
            # iterate through every out edge of the node
            for edge in graph.neighbors(node):
                # increase the current selected out edges probability
                next_prob[edge] += p
        # iterate through each node again to update node_prob
        for node in nodes:
            # node_prob now is equal to next_prob
            node_prob[node] = next_prob[node]
            # reset next_prob value to 0
            next_prob[node] = 0
    return node_prob

def main():
    # Load the web structure from file
    web = load_graph(open(WEB_DATA))

    # print information about the website
    print_stats(web)

    # The graph diameter is the length of the longest shortest path
    # between any two nodes. The number of random steps of walkers
    # should be a small multiple of the graph diameter.
    diameter = 3

    # Measure how long it takes to estimate PageRank through randomnodes = nx.nodes(graph)
    print("Estimate PageRank through random walks:")
    n_iter = len(web)**2
    n_steps = 2*diameter
    start = time.time()
    ranking = stochastic_page_rank(web, n_iter, n_steps)
    stop = time.time()
    time_stochastic = stop - start

    # Show top 20 pages with their page rank and time it took to compute
    top = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    print('\n'.join(f'{100*v:.2f}\t{k}' for k, v in top[:20]))
    print(f'Calculation took {time_stochastic:.2f} seconds.\n')

    # Measure how long it takes to estimate PageRank through probabilities
    print("Estimate PageRank through probability distributions:")
    n_iter = 2*diameter
    start = time.time()
    ranking = distribution_page_rank(web, n_iter)
    stop = time.time()
    time_probabilistic = stop - start

    # Show top 20 pages with their page rank and time it took to compute
    top = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    print('\n'.join(f'{100*v:.2f}\t{k}' for k,v in top[:20]))
    print(f'Calculation took {time_probabilistic:.2f} seconds.\n')

    # Compare the compute time of the two methods
    speedup = time_stochastic/time_probabilistic
    print(f'The probabilitic method was {speedup:.0f} times faster.')


if __name__ == '__main__':
    main()
