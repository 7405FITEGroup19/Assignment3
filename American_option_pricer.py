from scipy.stats import norm
import math
import numpy as np

# American option

def american_option (S0,sigma,r,T,K,N,option_type):
    dt = T/N
    u = math.exp(sigma*math.sqrt(dt))
    d = 1/u
    p = (math.exp(r*dt)-d)/(u-d)
    
    tree = np.zeros((N+1, N+1))
    tree[0,0] = S0
    
    for j in range(1, N+1):
        for i in range(j+1):
            tree[i,j] = S0*u**(j-i)*d**i
    
    if (option_type == 'call') or (option_type == 'Call') or (option_type == 'CALL'):
        tree[:,N] -= K
        tree[:,N][tree[:,N]<0] = 0
    if (option_type == 'put') or (option_type == 'Put') or (option_type == 'PUT'):
        tree[:,N] = K - tree[:,N]
        tree[:,N][tree[:,N]<0] = 0
    
    for j in range(N-1, -1, -1):
        for i in range(j+1):
            if tree[i,j+1] == 0 and tree[i+1, j+1] == 0:
                tree[i,j] = 0
            else:
                tree[i,j] = math.exp(-r*dt)*(p*tree[i,j+1]+(1-p)*tree[i+1,j+1])
    
    return tree[0,0]
