import networkx as nx
import random
from DMdata import DMinstance
import numpy as np
from AffineModel import affine_model

class InstanceGenerator:
    def __init__(self, graphSize, graphDensity, arrival, sojourn, horizon):
        self.size = graphSize
        self.density = graphDensity
        self.arr = arrival
        self.l = sojourn
        self.t = horizon


    def generateInstance(self):
        self.G = nx.gnp_random_graph(self.size, self.density)
        self.initState = {}
        for i in self.G.nodes:
            self.initState[i] = np.random.randint(5)

        self.alpha, self.beta = {}, {}
        for i in self.G.nodes:
            self.alpha[i] = 1 - np.exp(-self.l)
        self.vertexArrivalProb = {}
        for i in self.G.nodes:
            self.vertexArrivalProb[i] = np.random.rand()
        s = sum(self.vertexArrivalProb.values())
        for i in self.G.nodes:
            self.beta[i] = self.arr * self.vertexArrivalProb[i] / s

        Edges = [ frozenset(e) for e in self.G.edges  ]
        self.reward = {}
        for e in Edges:
            self.reward[e] = np.random.rand()

        thisDM = DMinstance(self.G, self.t, self.alpha, self.beta, self.initState, self.reward)
        return thisDM

def test(DM):
    am = affine_model(DM)
    am.buildALP()
    UB = am.getUpperBound()
    return UB

if __name__ == '__main__':
    from joblib import Parallel, delayed
    c
    DMs = []
    for i in range(100):
        DMs.append(gen.generateInstance())
    res = Parallel(n_jobs=-1,verbose=0)(delayed(test)(DM) for DM in DMs)
    print(sum(res))
