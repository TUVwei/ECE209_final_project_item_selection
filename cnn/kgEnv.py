import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from networkx.generators.random_graphs import random_powerlaw_tree
from networkx.linalg.graphmatrix import adjacency_matrix


class kgEnv():
    def __init__(self, n_nodes):
        # generate a tree
        np.random.seed(42)
        G = random_powerlaw_tree(n_nodes, seed = 42, gamma=np.random.rand() * 2 + 2, tries=1000000)
        A = adjacency_matrix(G).todense()
        A = np.array(A)



        # random change the direction
        for i in range(n_nodes):
            for j in range(i, n_nodes):
                if A[i, j] == 1:
                    if np.random.rand() < 0.85:
                        A[i, j] = 0
                    else:
                        A[j, i] = 0

        # save this but not used: the original tree
        self.origin_A = A.copy()

        # save the layout for further use
        G = nx.convert_matrix.from_numpy_array(A, create_using=nx.DiGraph())
        self.origin_layout = nx.planar_layout(G)

        # randomly generate the probability
        prob = np.random.rand(n_nodes)
        self.prob = prob

        # save the graph
        self.n_nodes = n_nodes
        self.nodes = list(range(n_nodes))
        self.A = A.copy()

    def get_adj(self):
        return self.nodes.copy(), self.A.copy()

    def remove_node(self, node):
        if node not in self.nodes:
            return [], -5, None

        # correct or not
        correct = False
        if np.random.rand() <= self.prob[node]:
            correct = True

        # initialize
        tmp1 = [node]
        tmp2 = []
        rm_list = set()
        # if not correct
        if not correct:
            while len(tmp1) > 0:
                for n in tmp1:
                    tmp2 = tmp2 + list(np.where(self.A[n, :] == 1)[0])
                    self.A[n, :] = 0
                    self.A[:, n] = 0
                    rm_list.add(n)
                tmp1 = list(set(tmp2))
                tmp2 = []

        # if forward
        if correct:
            while len(tmp1) > 0:
                for n in tmp1:
                    tmp2 = tmp2 + list(np.where(self.A[:, n] == 1)[0])
                    self.A[n, :] = 0
                    self.A[:, n] = 0
                    rm_list.add(n)
                tmp1 = list(set(tmp2))
                tmp2 = []

        for n in rm_list:
            self.nodes.remove(n)
        return rm_list, (len(self.nodes) == 0) * 11 - 1, correct

    def reset_all(self):
        self.A = self.origin_A.copy()
        self.nodes = list(range(self.n_nodes))

    def plot_graph(self, plot_type=0):
        # plot_type = 0/1: origin graph, current graph
        if plot_type == 0:
            i_edge, o_edge = np.where(self.origin_A == 1)
            labels = {i: '{}: {:.2f}'.format(i, self.prob[i]) for i in range(self.n_nodes)}
        if plot_type == 1:
            i_edge, o_edge = np.where(self.A == 1)

        # create a graph
        df = pd.DataFrame({'from': i_edge, 'to': o_edge})
        G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
        for n in self.nodes:
            G.add_node(n)

        # plot in a planar method
        if plot_type < 0.5:
            nx.draw(G, pos=self.origin_layout, labels=labels,
                    with_labels=True, node_size=250, arrows=True,
                    node_color='whitesmoke', font_color='red')
        else:
            nx.draw(G, with_labels=True, node_size=250, alpha=0.8,
                    arrows=True, pos=self.origin_layout)
        plt.title("Graph: " + str(plot_type))
        plt.show()


