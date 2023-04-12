from scipy.stats import norm
import math
import numpy as np
from Geometric_Asian_option_pricer import geometric_asia_option

# arithmatic Asian option

def arithmetic_asia_option (S0,sigma,r,T,K,n,option_type,M,cv):
    # the input cv should be 'control variate' meaning using control variate, or others meaning not using the general Monte Carlo
    value_geo = geometric_asia_option (S0,sigma,r,T,K,n,option_type)
    Spath = np.zeros((M,n))
    Dt = T/n
    np.random.seed(1)
    Z = np.random.normal(loc=0, scale=1, size=(M,n))
    # i for diferent paths, j for different time spot
    for i in range(M):
        Spath[i,0] = S0*math.exp((r-0.5*sigma**2)*Dt + sigma*math.sqrt(Dt)*Z[i,0])
        for j in range(1,n):
            Spath[i,j] = Spath[i,j-1]*math.exp((r-0.5*sigma**2)*Dt + sigma*math.sqrt(Dt)*Z[i,j])
    
    arithpayoff = np.zeros(M)
    geopayoff = np.zeros(M)
    arithmean = np.mean(Spath,axis=1)
    geomean = np.exp((1/n)*np.sum(np.log(Spath), axis=1))
    
    if (option_type == 'call') or (option_type == 'Call') or (option_type == 'CALL'):
        payoff1 = arithmean-K
        payoff1[payoff1 < 0] = 0
        arithpayoff = np.exp(-r*T)*payoff1
        payoff2 = geomean-K
        payoff2[payoff2 < 0] = 0
        geopayoff = np.exp(-r*T)*payoff2
    if (option_type == 'put') or (option_type == 'Put') or (option_type == 'PUT'):
        payoff1 = K-arithmean
        payoff1[payoff1 < 0] = 0
        arithpayoff = np.exp(-r*T)*payoff1
        payoff2 = K-geomean
        payoff2[payoff2 < 0] = 0
        geopayoff = np.exp(-r*T)*payoff2
    
    if cv == 'control variate': # monte carlo with control variate
        cov = np.mean(arithpayoff*geopayoff) - np.mean(arithpayoff)*np.mean(geopayoff)
        theta = cov/np.var(geopayoff)
        
        Z = arithpayoff + theta*(value_geo-geopayoff)
        Zmean = np.mean(Z)
        Zstd = np.std(Z)
        conf = [Zmean-1.96*Zstd/math.sqrt(M), Zmean+1.96*Zstd/math.sqrt(M)]
    else: # general monte carlo 
        Pmean = np.mean(arithpayoff)
        Pstd = np.std(arithpayoff)
        conf = [Pmean-1.96*Pstd/math.sqrt(M), Pmean+1.96*Pstd/math.sqrt(M)]
    
    return conf

