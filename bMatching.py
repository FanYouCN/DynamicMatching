import networkx as nx
from gomory_hu import gomory_hu_tree, min_odd_cut
import numpy as np
from itertools import combinations
from gurobipy import *
import time

from blossom_separation import blossom_separation

def all_subsets(arr):
    """returns a list of all subsets of a list"""
    combs = []
    for i in xrange(0, len(arr)+1):
        listing = [list(x) for x in combinations(arr, i)]
        combs.extend(listing)
    return combs

def check_blossom(model, G, b, a):
    X = {}
    for key in a:
        X[key] = a[key].getAttr("x")
    thisG = G.copy()
    cut = blossom_separation(thisG, b, X)
    if cut == None:
        return False
    else:
        Ew = []
        bw = sum([b[i] for i in cut])
        for e in G.edges:
            if e[0] in cut and e[1] in cut:
                Ew.append(frozenset(e))
        model.addConstr(quicksum([a[e] for e in Ew]) <= (bw-1)/2 )
        print("===========CUT!=============")
        return True

def bMatchingInteger(G, b, r):
    '''
    input:  G = (V, E)
            b = integer capacity vector
    output: max cardinality matching
    '''

    IP = Model()
    IP.setParam('OutputFlag', 0)
    V = G.nodes
    E = [frozenset(e) for e in G.edges]

    a = IP.addVars(E, name = "a", vtype=GRB.INTEGER)

    obj = 0
    for e in E:
        obj += r[e] * a[e]

    IP.setObjective(obj, GRB.MAXIMIZE)

    for i in V:
        Ei = [frozenset(e) for e in G.edges(i)]
        IP.addConstr(quicksum([a[e] for e in Ei]) <= b[i])
    IP.optimize()
    action = {}
    for e in E:
        action[e] = abs(a[e].X)
    return action

def bMatching(G, b, r):
    '''
    input:  G = (V, E)
            b = integer capacity vector
    output: max cardinality matching
    '''

    LP = Model()
    LP.setParam('OutputFlag', 0)
    V = G.nodes
    E = [frozenset(e) for e in G.edges]

    a = LP.addVars(E, name = "a")

    obj = 0
    for e in E:
        obj += r[e] * a[e]
    LP.setObjective(obj, GRB.MAXIMIZE)

    for i in V:
        Ei = [frozenset(e) for e in G.edges(i)]
        LP.addConstr(quicksum([a[e] for e in Ei]) <= b[i])
    LP.optimize()
    while check_blossom(LP, G, b, a):
        LP.setParam('Method', 1)
        LP.optimize()
    action = {}
    for e in E:
        action[e] = abs(a[e].X)
    return action.values(), LP.objVal

def bMatchingComplete(G, b):
    '''
    input:  G = (V, E)
            b = integer capacity vector
    output: max cardinality matching
    '''

    LP = Model()
    LP.setParam('OutputFlag', 0)
    V = G.nodes
    E = [frozenset(e) for e in G.edges]

    a = LP.addVars(E, name = "a")

    obj = quicksum(a)
    LP.setObjective(obj, GRB.MAXIMIZE)

    for i in V:
        Ei = [frozenset(e) for e in G.edges(i)]
        LP.addConstr(quicksum([a[e] for e in Ei]) <= b[i])

    for W in all_subsets(list(V)):
        Ew = []
        bw = sum([b[i] for i in W])
        for e in G.edges:
            if e[0] in W and e[1] in W:
                Ew.append(frozenset(e))
        if len(Ew)>0:
            LP.addConstr(quicksum([a[e] for e in Ew]) <= (bw-1)/2 )
    LP.optimize()

    print(LP.objVal)


def mycallback(model, where):
    thisG = G.copy()
    if where == GRB.Callback.MIPNODE:
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL:
            rel = model.cbGetNodeRel(model._vars)
            cut = blossom_separation(thisG, b, rel)
            Ew = []
            bw = sum([b[i] for i in cut])
            for e in thisG.edges:
                if e[0] in cut and e[1] in cut:
                    Ew.append(frozenset(e))
            model.cbCut(quicksum([a[e] for e in Ew]) <= (bw-1)/2 )


def bMatchingUserCut(G, b):
    '''
    input:  G = (V, E)
            b = integer capacity vector
    output: max cardinality matching
    '''

    IP = Model()
    IP.setParam('OutputFlag', 0)
    V = G.nodes
    E = [frozenset(e) for e in G.edges]

    a = IP.addVars(E, name = "a", vtype=GRB.INTEGER)

    obj = quicksum(a)
    IP.setObjective(obj, GRB.MAXIMIZE)
    IP.setParam('Method', 1)
    for i in V:
        Ei = [frozenset(e) for e in G.edges(i)]
        IP.addConstr(quicksum([a[e] for e in Ei]) <= b[i])

    IP._vars = IP.getVars()
    IP.optimize(mycallback)

    print(IP.objVal)

if __name__ == '__main__':
    # while True:
    #     G = nx.gnp_random_graph(18,0.5)
    #     b = {}
    #     E = [frozenset(e) for e in G.edges]
    #     for i in G.nodes():
    #         b[i] = np.random.randint(low=5,high=10)
    #     r = {}
    #     for e in E:
    #         r[e] = np.random.rand()
    #
    #     a1, obj1 = bMatching(G,b,r)
    #     a2, obj2 = bMatchingInteger(G,b,r)
    #     if abs(obj1 - obj2) > 1e-10:
    #         print a1, obj1
    #         print a2, obj2
    #         break
    print(0)
