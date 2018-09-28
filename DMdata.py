import numpy as np
import networkx as nx

class DMinstance:
    def __init__(self, G, T, alpha, beta, s0, r):
        '''
        G: undirected NetworkX graph
        T: time periods
        alpha, beta: system dynamics parameters: E[s'_i|s_i] = alpha_i s_i + beta_i
        '''
        self.Graph = G
        self.Nodes = G.nodes
        self.V = len(list(G.nodes))
        self.E = len(list(G.edges))
        self.r = r # edge weight, len |E|
        self.Edges = []
        for e in G.edges:
            self.Edges.append(frozenset(e))
        self.Horizon = range(1,T+1,1)
        self.alpha = alpha # len |V| array
        self.beta = beta # len |V| array
        self.s = s0

    def generateArrivalDeparture(self):
        ''' generate arriving and departing pairs according to alpha and beta
            Possion Arrival and Binomial Departure'''
        for i in self.Nodes:
            self.s[i] = np.random.binomial(self.s[i], self.alpha[i])
            self.s[i] += np.random.poisson(self.beta[i])
