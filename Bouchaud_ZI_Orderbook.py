#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:02:14 2018

@author: charlie
"""



import scipy.stats as st
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import collections

# Based on Toth/Bouchaud Agent-Based Models for Latent Liquidity and Concave Price Impact.pdf

# Page 4 P(f) = zeta(1 - f)**(zeta - 1)
# Zeta is the aggresiveness of the trader
class vmo_pdf(st.rv_continuous):
    agg = 2
    
    def _pdf(self, x):
        return self.agg*(1 - x)**(self.agg - 1)
    
vmo_cv = vmo_pdf(momtype=0, a=0, b=1)


class lo_pdf(st.rv_continuous):
    
    def _pdf(self, x):
        return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)
    
lo_cv = lo_pdf(momtype=0)

class Market():
    
    def __init__(self, num_traders, price):
        self.num_traders = num_traders
        self.traders = [Trader(np.random.normal(4, 2), self) for t in range(num_traders)]
        self.price = price
        self.keys = list(np.arange(start=0, stop=200, step=0.01))
        self.keys = [round(k, 2) for k in self.keys]
        self.market_depth = {k:0 for k in self.keys}
        self.price_hist = [price, price]
        self.price_hist2 = collections.deque(self.price_hist, maxlen=2)
        self.moving_average = price
        self.moving_average_hist = [price]
        self.r = 0
        self.r_hist = []
        self.r_real = 0
        self.r_real_hist = []
        self.daily_return = []
        self.uncertainty = random.randint(1, 20)
        self.volume = 0
        
    def populate_order_book(self):
        p = random.choice(self.keys)
        v = random.randint(1, 1500)
        self.market_depth[p] += v
        
    def limit_order(self):
        if self.price > self.price_hist[-1]:
            p = round(np.random.normal(self.price + (self.price - self.price_hist[-1]), 5), 2)
        if self.price < self.price_hist[-1]:
            p = round(np.random.normal(self.price - (self.price_hist[-1] - self.price), 5), 2)
        if self.price == self.price_hist[-1]:
            p = round(np.random.normal(self.price, 5), 2)
        v = random.randint(1, 1000)
        self.market_depth[p] += v
        
    def cancel_limit_order(self):
        if self.price > self.moving_average:
            p = random.choice(self.keys[0:(round(self.price*100))])
        if self.price < self.moving_average:
            p = random.choice(self.keys[(round(self.price*100)):int(self.keys[-1] * 100)])
        if self.price == self.moving_average:
            p = random.choice(self.keys)
        v = random.randint(0, self.market_depth[p])
        if self.market_depth[p] > v:
            self.market_depth[p] -= v
        else:
            self.market_depth[p] = 0
        
    def calc_moving_average(self):
        window = len(self.price_hist)
        round(window)
        if window > 10000:
            window = 10000
            window = int(window)
            self.moving_average = np.mean(self.price_hist[-window:])
            self.moving_average_hist.append(self.moving_average)
            
    def calc_return(self):
        # Calculate absolute return |rt| = |ln(pt/pt-1)| 
        self.r = np.log((self.price_hist2[-1] / self.price_hist2[-2]))
        self.r_hist.append(self.r)
        # Calculate the return relative to previous price
        self.r_real = (self.price_hist2[-1] - self.price_hist2[-2])
        self.r_real_hist.append(self.r_real)
        
            
        


class Trader():
    
    def __init__(self, zeta, market):
        # Zeta is the aggressiveness of the trader - see Toth/Bouchaud page 4
        self.zeta = zeta
        self.market = market
        self.Q = random.randint(1, 100000)
        self.Q_id = 0
        self.sign = random.choice([1, -1])
        self.meta_orders = {self.Q_id:{"p_init":0, "p_last":0, "Q":self.Q}}
    
    # Get the sign of the trade (i.e. buy or sell)
    def get_epsilon(self):
        return random.choice([1, -1])
    
    def set_Q(self, price):
        if self.Q > 0:
            pass
        else:
            self.meta_orders[self.Q_id]["p_last"] = price
            self.Q = random.randint(1, 100000)
            self.sign = self.get_epsilon()
            self.Q_id += 1
            self.meta_orders.update({self.Q_id:{"p_init":0, "p_last":0, "Q":self.Q}})
    
    # Determine th best price according to whether I want to buy or sell
    def get_best(self, sign):
        best = None
        p = self.market.price
        if sign == 1:
            while best == None:
                if p not in self.market.market_depth.keys():
                    return best
                if self.market.market_depth[p]:
                    best = p
                else:
                    p += 0.01
                    p = round(p, 2)
        if sign == -1:
            while best == None:
                if p not in self.market.market_depth.keys():
                    return best
                if self.market.market_depth[p]:
                    best = p
                else:
                    p -= 0.01
                    p = round(p, 2)
                    
        return best
    
    # Determine the opposite best price - this will be used to calculate the midpoint price
    def get_worst(self, sign):
        worst = None
        p = self.market.price
        if sign == 1:
            while worst == None:
                if p not in self.market.market_depth.keys():
                    return worst
                if self.market.market_depth[p]:
                    worst = p
                else:
                    p -= 0.01
                    p = round(p, 2)
        if sign == -1:
            while worst == None:
                if p not in self.market.market_depth.keys():
                    return worst
                if self.market.market_depth[p]:
                    worst = p
                else:
                    p += 0.01
                    p = round(p, 2)
                    
        return worst
    
    # Get the volume at the best price    
    def get_v_at_best(self, best):
        best = best
        return self.market.market_depth[best]
            
    
    # Determine the volume of the market order    
    def get_v_mo(self, rn_gen, best):
        # How much volume is at the best price?
        vol_at_best = self.get_v_at_best(best)
        # Tune my random number generator with my aggressiveness
        rn_gen.agg = self.zeta/self.market.uncertainty
        # Return the max of either 1 or the percent of the volume at best based on my aggressiveness
        v = max([rn_gen.rvs() * vol_at_best, 1])
        return v
    
    def trade(self):
        sign = self.sign
        best = self.get_best(sign)
        worst = self.get_worst(sign)
        v_best = self.get_v_at_best(best)
        v_mo = round(self.get_v_mo(vmo_cv, best))
        self.Q -= v_mo
        self.market.volume += v_mo
        self.market.market_depth[best] -= v_mo
        this_price = (best + worst) / 2
        if self.meta_orders[self.Q_id]["p_init"] == 0:
            self.meta_orders[self.Q_id]["p_init"] = this_price
        self.market.price_hist.append((best + worst) / 2)
        self.market.price_hist2.append((best + worst) / 2)
        self.market.price = round(this_price, 2)
        self.set_Q(this_price)
        
if __name__ == "__main__":
    m = Market(200, 100)
    for x in range(2000):
        #m.limit_order()
        m.populate_order_book()
    
    #plt.hist(m.market_depth.values())
        
    for x in tqdm(range(200)):
        for t in m.traders:
            m.limit_order()
            t.trade()
            m.calc_moving_average()
            m.cancel_limit_order()
            m.calc_return()
        m.daily_return.append(m.price)
        m.uncertainty = random.randint(1, 20)
        
            
    plt.plot(m.price_hist); plt.plot(m.moving_average_hist)
    ph = pd.Series(m.price_hist)
    ph_log = ph.apply(np.log)
    ph_log_returns = ph_log.diff()
    ph_log_returns = ph_log_returns.dropna()
    #pd.plotting.autocorrelation_plot(ph_log_returns)
    
    rh = pd.Series(m.r_hist)
    ph = pd.Series(m.price_hist)
    rrh = pd.Series(m.r_real_hist)
    
    
    rhc = rh.dropna()
    rhc = rhc.iloc[1000:]
    rha = rhc.abs()
    
    rha = list(rha)
    rha = list(filter((0.0).__ne__, rha))
    rha = pd.Series(rha)
    
    dr = pd.Series(m.daily_return)
    dr.plot()
    
    dr = dr.apply(np.log)
    
    
    #pd.plotting.autocorrelation_plot(rha)
    #pd.plotting.autocorrelation_plot(rhc)
    #pd.plotting.autocorrelation_plot(rhc.iloc[1000:1600])
    #pd.plotting.autocorrelation_plot(rha.iloc[1000:1050])
    
    #plt.hist(rhc, bins=100)
    
    #plt.hist(rhc, bins=50, log=True, range=(-0.1, 0.1))

