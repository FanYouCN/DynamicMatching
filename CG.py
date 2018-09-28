import networkx as nx
import numpy as np
from gurobipy import *
def CG(G, c, d, D):
    IP = Model()
    IP.setParam('OutputFlag', 0)
    V = G.nodes
    E = [frozenset(e) for e in G.edges]
    a = IP.addVars(E, name = "a", vtype=GRB.INTEGER)
    b = IP.addVars(V, name = "b", vtype=GRB.INTEGER)

    obj = 0
    for i in V:
        obj += d[i] * b[i]
    for e in E:
        obj += c[e] * a[e]
    IP.setObjective(obj, GRB.MAXIMIZE)
    for i in V:
        Ei = [frozenset(e) for e in G.edges(i)]
        # IP.addConstr(b[i] <= D)
        IP.addConstr(quicksum([a[e] for e in Ei]) <= b[i])
    IP.optimize()
    aa = []
    for e in E:
        aa.append(a[e].getAttr("x"))
    print aa
    return IP.objVal



def CG_LR(G, c, d, D):
    LP = Model()
    LP.setParam('OutputFlag', 1)
    V = G.nodes
    E = [frozenset(e) for e in G.edges]
    a = LP.addVars(E, name = "a")
    b = LP.addVars(V, name = "b")

    obj = 0
    for i in V:
        obj += d[i] * b[i]
    for e in E:
        obj += c[e] * a[e]
    LP.setObjective(obj, GRB.MAXIMIZE)
    for i in V:
        Ei = [frozenset(e) for e in G.edges(i)]
        # LP.addConstr(b[i] <= D)
        LP.addConstr(quicksum([a[e] for e in Ei]) <= b[i])
    LP.optimize()
    aa = []
    for e in E:
        aa.append(a[e].getAttr("x"))
    print aa
    return LP.objVal

if __name__ == '__main__':
    # M, N = 0, 0
    # for i in range(1):
    #     G = nx.gnp_random_graph(np.random.randint(low=15,high=27),np.random.uniform(0.3,0.9))
    #     c = {}
    #     d = {}
    #     D = 9
    #     DD = 10
    #     V = G.nodes
    #     E = [frozenset(e) for e in G.edges]
    #     for i in V:
    #         d[i] = np.random.uniform(-1,1)
    #     for e in E:
    #         c[e] = np.random.uniform(-1,1)
    #
    #     res1 = CG(G, c, d, D)
    #     res2 = CG_LR(G, c, d, D)
    #     if (res1 != res2):
    #         N +=1
    #     res1 = CG(G, c, d, DD)
    #     res2 = CG_LR(G, c, d, DD)
    #     if (res1 != res2):
    #         M +=1
    for i in range(1):
        G = nx.gnp_random_graph(np.random.randint(low=15,high=27),np.random.uniform(0.3,0.9))
        c, d = {}, {}
        V = G.nodes
        E = [frozenset(e) for e in G.edges]
        for i in V:
            d[i] = np.random.uniform(-1,1)
        for e in E:
            c[e] = np.random.uniform(-1,1)

        CG(G, c, d, 0)
        CG_LR(G, c, d, 0)
