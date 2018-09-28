import numpy as np
import pdb

class DistributionGenerator:

    def gen_donor_age(self, m=1):
        x = np.random.normal(48.22, 12.68, m)
        x = x[0]
        if x < 20:
            x = 20
        if x > 74:
            x = 74
        return int(x)

    def gen_donor_bmi(self, weight, m=1):
        # x = np.random.normal(27.78, 4.46, m)
        # x = x[0]
        # if x < 12.4:
        #     x = 12.4
        # if x > 37.8:
        #     x = 37.8
        x = 0.0948 * weight + 11.387
        return x

    def gen_donor_egfr(self, age, m=1):
        # x = np.random.normal(98.11, 15.08, m)
        # x = x[0]
        # if x < 64 :
        #     x = 64
        # if x > 141 :
        #     x = 141
        # return x
        if age <= 29:
            return 116
        elif 30 <= age <= 39:
            return 107
        elif 40 <= age <= 49:
            return 99
        elif 50 <= age <= 59:
            return 93
        elif 60 <= age <= 69:
            return 85
        else:
            return 75

    def gen_donor_sbp(self, donor_egfr, m=1):
        # x = np.random.normal(124.14, 13.11, m)
        if donor_egfr <= 80:
            x = np.random.normal(128.23, 12.947, m)
        elif 80 < donor_egfr <= 100:
            x = np.random.normal(126.027, 12.60, m)
        else:
            x = np.random.normal(120.2, 12.2, m)
        x = x[0]
        if x < 96 : x = 96
        if x > 170 : x = 170
        return x

    def gen_donor_sex(self, m=1):
        x = np.random.choice([0, 1], p=[116.0 / 166.0, 50.0 / 166.0], size=m)
        return x.astype(int)

    def gen_donor_afam(self, m=1):
        x = np.random.choice([0, 1], p=[158.0 / 166.0, 8.0 / 166.0], size=m)

        # x = np.random.choice([0, 1], p=[4946.0/5538.0, 592/5538.0], size=m)
        return x.astype(int)

    def gen_donor_cig_use(self, m=1):
        x = np.random.choice([0, 1], p=[113.0 / 166.0, 53.0 / 166.0], size=m)
        return x.astype(int)

    def gen_donor_rec_abo_comp(self, m=1):
        x = np.random.choice([0, 1], p=[20.0 / 166.0, 146.0 / 166.0], size=m)
        return x.astype(int)

    def gen_donor_rec_HLA_B_mis(self, related, m=1):
        if related == 0:
            x = np.random.choice([0, 1, 2], p=[1.0 / 84.0, 8.0 / 84.0, 75.0 / 84.0], size=m)
        else:
            x = np.random.choice([0, 1, 2], p=[15.0 / 82.0, 26.0 / 82.0, 41.0 / 82.0], size=m)
        # x = np.random.choice([0, 1, 2], p=[16.0 / 166.0, 34.0 / 166.0, 116.0 / 166.0], size=m)
        return x.astype(int)

    def gen_donor_rec_HLA_DR_mis(self, related, m=1):
        if related == 0:
            x = np.random.choice([0, 1, 2], p=[1.0 / 84.0, 5.0 / 84.0, 78.0 / 84.0], size=m)
        else:
            x = np.random.choice([0, 1, 2], p=[11.0 / 82.0, 5.0 / 82.0, 66.0 / 82.0], size=m)
        # x = np.random.choice([0, 1, 2], p=[12.0 / 166.0, 10.0 / 166.0, 144.0 / 166.0], size=m)
        return x.astype(int)

    def gen_donor_rec_HLA_mis(self, related, m=1):
        mis_match_dict = dict()
        mis_match_dict[0] = [0, 0]
        mis_match_dict[1] = [0, 1]
        mis_match_dict[2] = [0, 2]
        mis_match_dict[3] = [1, 0]
        mis_match_dict[4] = [1, 1]
        mis_match_dict[5] = [1, 2]
        mis_match_dict[6] = [2, 0]
        mis_match_dict[7] = [2, 1]
        mis_match_dict[8] = [2, 2]

        if related == 0:
            k = np.random.choice(range(9), p=[0.0 / 84.0, 0.0 / 84.0, 1.0 / 84.0,
                                              1.0 / 84.0, 0.0 / 84.0, 7.0 / 84.0,
                                              0.0 / 84.0, 5.0 / 84.0, 70.0 / 84.0], size=m)
        else:
            k = np.random.choice(range(9), p=[9.0 / 82.0, 1.0 / 82.0, 5.0 / 82.0,
                                              0.0 / 82.0, 0.0 / 82.0, 26.0 / 82.0,
                                              2.0 / 82.0, 4.0 / 82.0, 35.0 / 82.0], size=m)
        return mis_match_dict[k[0]]


    def gen_rec_weight(self, sex, m=1):
        if sex == 0:
            x = np.random.normal(180.698, 42.255, m)
            x = x[0]
            if x < 100:
                x = 100
            if x > 303:
                x = 303
        else:
            x = np.random.normal(190.34, 39.897, m)
            x = x[0]
            if x < 97:
                x = 97
            if x > 272:
                x = 272
        return x

    def gen_donor_weight(self, sex, m=1):
        if sex == 0:
            x = np.random.normal(160.75, 30.06, m)
            x = x[0]
            if x < 102:
                x = 102
            if x > 254:
                x = 254
        else:
            x = np.random.normal(200.796, 32.797, m)
            x = x[0]
            if x < 150:
                x = 150
            if x > 265:
                x = 265
        return x


    # def gen_donor_rec_weight_ratio(self, m=1):
    #     x = np.random.choice([0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #                          p=[1.0 / 166.0, 4.0 / 166.0, 14.0 / 166.0, 26.0 / 166.0, 31.0 / 166.0, 90 / 166.0],
    #                          size=m)
    #     return x

    def gen_donor_rec_related(self):
        x = np.random.choice([0, 1], p=[0.5, 0.5])
        return x.astype(int)

    def gen_rec_sex(self, m=1):
        x = np.random.choice([0, 1], p=[58.0 / 166.0, 108.0 / 166.0], size=m)
        return x.astype(int)

    def gen_not_donor_rec_abo_comp(self, m=1):
        x = np.random.choice([0, 1], p=[7360.0 / 27390.0, 20030.0 / 27390.0], size=m)
        return x.astype(int)

    def gen_not_donor_rec_HLA_B_mis(self, m=1):
        x = np.random.choice([0, 1, 2], p=[237.0 / 27390.0, 2241.0 / 27390.0, 24912.0 / 27390.0], size=m)
        return x.astype(int)

    def gen_not_donor_rec_HLA_DR_mis(self, m=1):
        x = np.random.choice([0, 1, 2], p=[593.0 / 27390.0, 1004.0 / 27390.0, 25793.0 / 27390.0], size=m)
        return x.astype(int)

    def gen_Possion():
        return np.random.poisson(lam=100)

    def gen_Binomial(n, p):
        return np.random.binomial(n, p)

    # def gen_not_donor_rec_weight_ratio(self, m=1):
    #     x = np.random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #                          p=[41.0 / 27390.0, 411.0 / 27390.0, 1495.0 / 27390.0, 2727.0 / 27390.0,
    #                             3589.0 / 27390.0, 3895 / 27390.0, 15232.0 / 27390.0],
    #                          size=m)
    #     return x
