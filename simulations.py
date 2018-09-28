from InstanceGenerator import InstanceGenerator
from AffineModel import affine_model
from bMatching import *
from joblib import Parallel, delayed
from scipy import stats
from datetime import datetime
import copy

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

def sim(DM):
    rALP, rGreedy, UB = [],[],0
    sameDM = copy.deepcopy(DM)
    othersameDM = copy.deepcopy(DM)
    am = affine_model(DM)
    am.buildALP()
    UB = am.getUpperBound()
    rALP = simALP(sameDM)
    rGreedy = simGreedy(othersameDM)
    return rGreedy, rALP, UB

if __name__ == '__main__':
    now = str(datetime.now())
    outfile = open("res/"+now+".txt", "w")

    import time
    t = time.time()

    # gen = InstanceGenerator(10, 0.9, 100, 10, 20)
    # DMs = []
    # N = 1000
    # for i in range(N):
    #     DMs.append(gen.generateInstance())
    # res = Parallel(n_jobs=-1,verbose=0)(delayed(sim)(DM) for DM in DMs)

    for size in [100]:
        for density in [0.5]:
            for arrival in [20]:
                for sojourn in [20]:
                    for horizon in [30]:
                        gen = InstanceGenerator(size, density, arrival*size, sojourn, horizon)
                        DMs = []
                        N = 100
                        for i in range(N):
                            DMs.append(gen.generateInstance())
                        res = Parallel(n_jobs=-1,verbose=0)(delayed(sim)(DM) for DM in DMs)
                        grd = np.mean([i[0] for i in res])
                        prm = np.mean([i[1] for i in res])
                        s = ("Graph Size: "+str(size) + ", Graph Density: "+str(density)+", Arrival Rate: "+str(arrival) + ", Sojourn: "+ str(sojourn)+", Horizon: " + str(horizon) + ", Greedy Reward: " + str(grd)+", Primal Reward: "
                             +str(prm)+", Upper Bound: " + str(np.mean([i[2] for i in res])) + ", Improvement: " + str( (prm-grd)/grd * 100 ))
                        print(s)
                        print(s, file=outfile)
    print(time.time()-t)
