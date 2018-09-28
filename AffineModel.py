from gurobipy import *
import numpy as np
import networkx as nx
from DMdata import DMinstance
from bMatching import *
from blossom_separation import blossom_separation
import copy

class affine_model:
    def __init__(self, aDMinstance):
        self.DM = aDMinstance

    def buildALP(self):
        self.reduced_ALP = Model()
        self.reduced_ALP.setParam('OutputFlag', 0)
        self.reduced_ALP.setParam('Method', 2)
        self.reduced_ALP.setParam('BarConvTol', 0.0)
        self.a = self.reduced_ALP.addVars(self.DM.Horizon, self.DM.Edges, name="a")
        self.s = self.reduced_ALP.addVars(self.DM.Horizon, self.DM.Nodes, name="s")
        obj = 0
        for t in self.DM.Horizon:
            for e in self.DM.Edges:
                obj += self.DM.r[e] * self.a[t,e]
        self.reduced_ALP.setObjective(obj, GRB.MAXIMIZE)

        for t in self.DM.Horizon:
            for i in self.DM.Nodes:
                if t == 1:
                    self.reduced_ALP.addConstr(self.s[(t,i)] == self.DM.s[i])
                else:
                    self.reduced_ALP.addConstr(self.s[(t,i)] == self.DM.alpha[i] * (self.s[(t-1,i)] - sum([self.a[t-1,frozenset(e)]*self.DM.r[frozenset(e)] for e in self.DM.Graph.edges(i)])) + self.DM.beta[i] )

        for t in self.DM.Horizon:
            for i in self.DM.Nodes:
                self.reduced_ALP.addConstr(sum([self.a[t,frozenset(e)] for e in self.DM.Graph.edges(i)]) <= self.s[(t,i)])

    def UB1(self):
        self.reduced_ALP.optimize()
        obv = self.reduced_ALP.objVal
        a = {}
        for e in self.DM.Edges:
            a[e] =abs(self.a[1,e].X)
        # print a.values(), obv
        return obv, a

    def UB2(self):
        for W in all_subsets(list(self.DM.Graph.nodes)):
            Ew = []
            bw = sum([self.DM.s[i] for i in W])
            if bw % 2 == 1:
                for e in self.DM.Graph.edges:
                    if e[0] in W and e[1] in W:
                        Ew.append(frozenset(e))
                if len(Ew)>0:
                    self.reduced_ALP.addConstr(quicksum([self.a[1,e] for e in Ew]) <= (bw-1)/2 )
        self.reduced_ALP.optimize()
        obv = self.reduced_ALP.objVal
        a = {}
        for e in self.DM.Edges:
            a[e] =abs(self.a[1,e].X)
        # print a.values(), obv
        return obv, a

    def UB3(self):
        self.reduced_ALP.optimize()
        while self.check_blossom():
            self.reduced_ALP.optimize()
        obv = self.reduced_ALP.objVal
        a = {}
        for e in self.DM.Edges:
            a[e] =abs(self.a[1,e].X)
        # print a.values(), obv
        return obv, a

    def getUpperBound(self):
        self.reduced_ALP.optimize()
        while self.check_blossom():
            self.reduced_ALP.optimize()
        obv = self.reduced_ALP.objVal
        return obv

    def solve(self):
        self.reduced_ALP.optimize()
        while self.check_blossom():
            self.reduced_ALP.optimize()
        reward = 0
        action = {}
        for e in self.DM.Edges:
            ae = self.a[1,e].X
            if abs(ae - round(ae)) < 1e-5:
                ae = round(ae)
            if ae % 1 > 0:
                ae = max(ae/1, 0)
            action[e] = np.random.binomial(ae, self.DM.r[e])
            reward += action[e]
            for i in list(e):
                self.DM.s[i] -= action[e]
        self.DM.Horizon = self.DM.Horizon[:-1]
        return reward

    def check_blossom(self):
        X = {}
        for e in self.DM.Edges:
            X[e] = self.a[1,e].X
        localG = copy.deepcopy(self.DM.Graph)
        b = self.DM.s
        cut = blossom_separation(localG, b, X)
        if cut == None:
            return False
        else:
            Ew = []
            bw = sum([b[i] for i in cut])
            for e in localG.edges:
                if e[0] in cut and e[1] in cut:
                    Ew.append(frozenset(e))
            if sum([X[e] for e in Ew]) > ((bw-1)//2+1e-10):
                self.reduced_ALP.addConstr(quicksum([self.a[1,e] for e in Ew]) <= (bw-1)/2)
                self.reduced_ALP.update()
                return True
            else:
                return False


    def bMatchingInt(self):
        action = bMatchingInteger(self.DM.Graph, self.DM.s, self.DM.r)
        reward = 0
        for e in self.DM.Edges:
            action[e] = np.random.binomial(action[e], self.DM.r[e])
            reward += action[e]
            for i in list(e):
                self.DM.s[i] -= action[e]
        self.DM.Horizon = self.DM.Horizon[:-1]
        return reward


def simALP(DM):
    aAM = affine_model(DM)
    totalRewardALP = 0
    while(len(aAM.DM.Horizon)>0):
        aAM.buildALP()
        totalRewardALP += aAM.solve()
        aAM.DM.generateArrivalDeparture()
    return totalRewardALP

def simGreedy(DM):
    aAM = affine_model(DM)
    totalRewardGreedy = 0
    while(len(aAM.DM.Horizon)>0):
        totalRewardGreedy += aAM.bMatchingInt()
        aAM.DM.generateArrivalDeparture()
    return totalRewardGreedy

if __name__ == '__main__':
    from SaidmanGenerator import *
    from bMatching import *
    from joblib import Parallel, delayed
    from InstanceGenerator import InstanceGenerator
    from scipy import stats
    import pickle

    # gen = InstanceGenerator(10, 0.3, 10, 10, 20)
    # DMs = []
    #
    # def eq(a1, a2):
    #     a1 = a1.values()
    #     a2 = a2.values()
    #     l = len(a1)
    #     for i in range(l):
    #         if abs(a1[i] - a2[i]) > 1e-5:
    #             return False
    #     return True
    # cnt = 1
    #
    # def isfraction(a):
    #     l = len(a)
    #     for i in range(l):
    #         if abs(a[i] - round(a[i])) > 1e-5:
    #             return True
    #     return False
    #
    # before, after = 0, 0
    #
    # while True:
    #     cnt += 1
    #     # print cnt
    #     thisDM = gen.generateInstance()
    #     dm1, dm2, dm3 = copy.deepcopy(thisDM), copy.deepcopy(thisDM), copy.deepcopy(thisDM)
    #     am1, am2, am3 = affine_model(dm1), affine_model(dm2), affine_model(dm3)
    #
    #     am1.buildALP()
    #     am2.buildALP()
    #     am3.buildALP()
    #     ubComplete ,aComplete = am1.UB2()
    #     ubSep ,aSep = am2.UB3()
    #     ubRelax, aRelax = am3.UB1()
    #
    #     if isfraction(aSep.values()):
    #         print aSep
    #         break
    #
    #     # if isfraction(a):
    #     #     before += 1
    #     #     print "--------------------------------------------------------------------------------"
    #     #     print a, ub
    #     #     print a1, ub1
    #     #     print a2, ub2
    #
    #     # if isfraction(a1):
    #     #     after += 1
    #     #     print "Total: ", cnt, " Before: ", before, " After: ", after, eq(a,a1)
    #     #     print "--------------------------------------------------------------------------------"
    #     #     print a, ub
    #     #     print a1, ub1
    #     #     print a2, ub2
    #     #     if not eq(a,a1):
    #     #         output = open('instances/dm.pkl', 'wb')
    #     #         pickle.dump(thisDM, output)
    #     #         output.close()
    #
    #     # if not eq(aComplete, aSep):
    #     #     # print'=============================================================================='
    #     #     print cnt
    #     #     E = thisDM.Edges
    #     #     vio = 0
    #     #     for W in all_subsets(list(thisDM.Graph.nodes)):
    #     #         Ew = []
    #     #         bw = sum([thisDM.s[i] for i in W])
    #     #         if bw % 2 == 1:
    #     #             for e in thisDM.Graph.edges:
    #     #                 if e[0] in W and e[1] in W:
    #     #                     Ew.append(frozenset(e))
    #     #             if len(Ew)>0:
    #     #                 if sum([aSep[e] for e in Ew]) > (bw-1)/2:
    #     #                     vio += 1
    #     #     print aComplete.values(), ubComplete
    #     #     print aSep.values(), ubSep
    #         # print vio
    #         # break
    #
    #         # if vio>0:
    #         #     print aComplete.values(), ubComplete
    #         #     print aSep.values(), ubSep
    #         #     print vio
    #         #     break
    #
    #     #     output = open('instances/dm.pkl', 'wb')
    #     #     pickle.dump(thisDM, output)
    #     #     output.close()
    #     #     break

    # pkl_file = open('instances/dm_`4.pkl', 'rb')
    # testDM = pickle.load(pkl_file)
    # dm1, dm2, dm3 = copy.deepcopy(testDM), copy.deepcopy(testDM), copy.deepcopy(testDM)
    # am1, am2, am3 = affine_model(dm1), affine_model(dm2), affine_model(dm3)
    # am1.buildALP()
    # am2.buildALP()
    # am3.buildALP()
    # print am1.UB1()
    # print am2.UB2()
    # print am3.UB3()
