import numpy as np
import pandas as pd
import scipy as scp
from scipy import stats

class evaluator():

    def estimate(self, endtimes, tail = "exponential", timer = None, returnParams = False, returnTable = False, minSamples = 5):
        
        endtimes = np.array(endtimes)
        timer = endtimes.max() if timer is None else timer
        fpts = endtimes[endtimes<timer]
        fpts.sort()
        nFPT = len(fpts)
        nSimulations = len(endtimes)
        survival = np.linspace(nSimulations-1,nSimulations-nFPT,nFPT)/nSimulations
        limits = [i for i in range(nFPT + 1 - minSamples)]
        params, table = getattr(self, f"_{tail}")(fpts = fpts, survival = survival, limits = limits,
                                                  nFPT = nFPT, nSimulations = nSimulations, timer = timer)

        if returnParams:
            ret = params
        elif returnTable:
            ret = table
        else:
            ret = params["MFPT"]
        
        return ret
    
    def _exponential(self, fpts, survival, limits, nFPT, nSimulations, timer):

        Rs = []
        slopes = []

        for limit in limits:
            fit = stats.linregress(fpts[limit:], np.log(survival[limit:]))
            Rs.append(fit[2]**2)
            slopes.append(fit[0])
        table = pd.DataFrame({"tPrime":fpts[:limits[-1] + 1], "slope":slopes, "R":Rs})
        
        k = -float(table.loc[table.R == table.R.max()].slope)
        MFPT = nFPT / nSimulations * fpts.mean() + (1 - nFPT / nSimulations) * (timer + 1 / k)
        tPrime = float(table.loc[table.R == table.R.max()].tPrime.min())

        return {"MFPT": MFPT, "k": k, "tPrime": tPrime}, table
    
    def _power(self, fpts, survival, limits, nFPT, nSimulations, timer):

        Rs = []
        slopes = []

        for limit in limits:
            fit = stats.linregress(np.log(fpts[limit:]), np.log(survival[limit:]))
            Rs.append(fit[2]**2)
            slopes.append(fit[0])

        table = pd.DataFrame({"tPrime":fpts[:limits[-1] + 1], "slope":slopes, "R":Rs})
        table = table.loc[table.slope < -1]

        if 0 == len(table):
            raise ValueError(r"Could not found $\alpha$ with finite mean!")
        
        alpha = -float(table.loc[table.R == table.R.max()].slope)
        MFPT = nFPT / nSimulations * fpts.mean() + (1 - nFPT / nSimulations) * alpha * timer / (alpha - 1)
        tPrime = float(table.loc[table.R == table.R.max()].tPrime.min())

        return {"MFPT": MFPT, "alpha": alpha, "tPrime": tPrime}, table