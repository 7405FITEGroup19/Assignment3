import math
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

#================================


def Quasi_Monte_Carlo(S, K, T, r, sigma, N, R, barrier_lower, barrier_upper, is_reversed = False):
    #================================
    # delta t
    deltaT = T/N
    #================================
    # set the random seed
    seed = 1
    np.random.seed(seed)

    #================================
    # generate the paths of stock prices
    values = []
    M = int(1e3)
    sequencer = qmc.Sobol(d=N, seed=seed)
    # uniform samples
    X = np.array(sequencer.random(n=M))
    # standard normal samples
    Z = norm.ppf(X)
    # Z = np.array([np.random.standard_normal(N) for _ in range(M)])
    # scaled samples
    samples = (r - 0.5 * sigma * sigma) * deltaT \
              + sigma * math.sqrt(deltaT) * Z
    df_samples = pd.DataFrame(samples)
    df_samples_cumsum = df_samples.cumsum(axis=1)




    # the simulated stock prices, M rows, N columns
    df_stocks = s * np.exp(df_samples_cumsum)
    for ipath in df_stocks.index.to_list():
        ds_path_local = df_stocks.loc[ipath, :]
        price_max = ds_path_local.max()
        price_min = ds_path_local.min()
        if is_reversed:
            if price_max >= barrier_upper: # knock-in happend
                final_price = ds_path_local.iloc[-1]
                payoff = np.exp(- r * T) * max( K - final_price, 0)
                values.append(payoff)

            elif price_min <= barrier_lower: # knock-out happened
                knockout_time = ds_path_local[ds_path_local \
                                              <= barrier_upper].index.to_list()[0]
                # payoff = R * np.exp(-(T-knockout_time * deltaT) * r)
                payoff = R * np.exp(-knockout_time * r * deltaT)
                values.append(payoff)
            else: # no knock-out, no knock-in
                values.append(0)
        else:
            if price_max >= barrier_upper:  # knock-out happened
                knockout_time = ds_path_local[ds_path_local \
                                              >= barrier_upper].index.to_list()[0]
                # payoff = R * np.exp(-(T-knockout_time * deltaT) * r)
                payoff = R * np.exp(-knockout_time * r * deltaT)
                values.append(payoff)

            elif price_min <= barrier_lower:  # knock-in happend
                final_price = ds_path_local.iloc[-1]
                payoff = np.exp(- r * T) * max(K - final_price, 0)
                values.append(payoff)
            else:  # no knock-out, no knock-in
                values.append(0)

    value = np.mean(values)
    std = np.std(values)
    conf_interval_lower = value - 1.96 * std / math.sqrt(M)
    conf_interval_upper = value + 1.96 * std / math.sqrt(M)
    return value, conf_interval_lower, conf_interval_upper
# # option parameters
# r = 0.05; sigma = 0.20; T = 2.0; s = 100; K = 100;
# barrier_lower = 80; barrier_upper = 125; N = 24; R = 1.5
# value, conf_interval_lower, conf_interval_upper = Quasi_Monte_Carlo(s, K, T, r, sigma, N, R, barrier_lower, barrier_upper, is_reversed=False)
# print('the mc price and conf. interval: {:.4f}, [{:.4f}, {:.4f}]'.format(value, conf_interval_lower, conf_interval_upper))
#
# r = 0.05; sigma = 0.20; T = 2.0; s = 100; K = 120;
# barrier_lower = 80; barrier_upper = 125; N = 24; R = 1.5
# value, conf_interval_lower, conf_interval_upper = Quasi_Monte_Carlo(s, K, T, r, sigma, N, R, barrier_lower, barrier_upper, is_reversed=False)
# print('the mc price and conf. interval: {:.4f}, [{:.4f}, {:.4f}]'.format(value, conf_interval_lower, conf_interval_upper))
#
# r = 0.05; sigma = 0.20; T = 1.0; s = 100; K = 100;
# barrier_lower = 80; barrier_upper = 125; N = 24; R = 1.5
# value, conf_interval_lower, conf_interval_upper = Quasi_Monte_Carlo(s, K, T, r, sigma, N, R, barrier_lower, barrier_upper, is_reversed=False)
# print('the mc price and conf. interval: {:.4f}, [{:.4f}, {:.4f}]'.format(value, conf_interval_lower, conf_interval_upper))
#
# r = 0.05; sigma = 0.30; T = 2.0; s = 100; K = 100;
# barrier_lower = 80; barrier_upper = 125; N = 24; R = 1.5
# value, conf_interval_lower, conf_interval_upper = Quasi_Monte_Carlo(s, K, T, r, sigma, N, R, barrier_lower, barrier_upper, is_reversed=False)
# print('the mc price and conf. interval: {:.4f}, [{:.4f}, {:.4f}]'.format(value, conf_interval_lower, conf_interval_upper))
#
# r = 0.08; sigma = 0.20; T = 2.0; s = 100; K = 100;
# barrier_lower = 80; barrier_upper = 125; N = 24; R = 1.5
# value, conf_interval_lower, conf_interval_upper = Quasi_Monte_Carlo(s, K, T, r, sigma, N, R, barrier_lower, barrier_upper, is_reversed=False)
# print('the mc price and conf. interval: {:.4f}, [{:.4f}, {:.4f}]'.format(value, conf_interval_lower, conf_interval_upper))
#
# r = 0.05; sigma = 0.20; T = 2.0; s = 100; K = 100;
# barrier_lower = 90; barrier_upper = 125; N = 24; R = 1.5
# value, conf_interval_lower, conf_interval_upper = Quasi_Monte_Carlo(s, K, T, r, sigma, N, R, barrier_lower, barrier_upper, is_reversed=False)
# print('the mc price and conf. interval: {:.4f}, [{:.4f}, {:.4f}]'.format(value, conf_interval_lower, conf_interval_upper))
#
# r = 0.05; sigma = 0.20; T = 2.0; s = 100; K = 100;
# barrier_lower = 80; barrier_upper = 125; N = 24; R = 0.0
# value, conf_interval_lower, conf_interval_upper = Quasi_Monte_Carlo(s, K, T, r, sigma, N, R, barrier_lower, barrier_upper, is_reversed=True)
# print('the mc price and conf. interval: {:.4f}, [{:.4f}, {:.4f}]'.format(value, conf_interval_lower, conf_interval_upper))