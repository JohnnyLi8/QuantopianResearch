#!/usr/bin/env python
# coding: utf-8

# In[123]:


import pandas as pd
import numpy as np
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.factors import CustomFactor, Returns, Latest
from quantopian.pipeline.classifiers import Classifier
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.research import run_pipeline
from quantopian.pipeline.classifiers.fundamentals import Sector  
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.research import run_pipeline
import scipy
from pykalman import KalmanFilter
from scipy import poly1d
import matplotlib.pyplot as plt


# In[124]:


equities = ['UPS', 'FSLR','GM']
market = ['SPY']


# In[146]:


prices = get_pricing(equities + market,
                     start_date='2015-01-01',end_date='2020-02-2', fields='price')
returns = get_pricing(equities, 
                      start_date='2015-01-01',end_date='2020-02-2', fields='price').pct_change()[1:]
cum_returns = (returns + 1).cumprod()
cum_returns.plot()


# In[147]:


x = get_pricing('SPY', fields='price', start_date=start, end_date=end)
rows = len(equities)
column = 1
index = 0
plot = rows*100 + 10


# In[148]:


delta = 1e-3
trans_cov = delta / (1 - delta) * np.eye(2) # How much random walk wiggles
obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # y is 1-dimensional, (alpha, beta) is 2-dimensional
                  initial_state_mean=[0,0],
                  initial_state_covariance=np.ones((2, 2)),
                  transition_matrices=np.eye(2),
                  observation_matrices=obs_mat,
                  observation_covariance=2,
                  transition_covariance=trans_cov)


# In[149]:


for i in range(len(equities)):
    y = get_pricing(equities[i], fields='price', start_date=start, end_date=end)
    state_means, state_covs = kf.filter(y.values)
    plot_id = str(plot + i)
    plt.subplot(plot_id)
    plt.plot(x.index, state_means[:,0], label='slope')
    plt.legend(['Beta of ' + equities[i]])


# In[150]:


step = 40
xi = np.linspace(x.min(), x.max(), 2)
for i in range(len(equities)):
    y = get_pricing(equities[i], fields='price', start_date=start, end_date=end)
    state_means, state_covs = kf.filter(y.values)
    plot_id = str(plot + i)
    plt.subplot(plot_id)
    cm = plt.get_cmap('jet')
    sc = plt.scatter(x, y, s=30, c=colors, cmap=cm, edgecolor='k', alpha=0.7)
    #cb = plt.colorbar(sc)
    #cb.ax.set_yticklabels([str(p.date()) for p in x[::len(x)//9].index])
    colors_l = np.linspace(0.1, 1, len(state_means[::step]))
    for j, beta in enumerate(state_means[::step]):
        plt.plot(xi, beta[0] * xi + beta[1], alpha=.2, lw=1, c=cm(colors_l[j]))
    # Plot the OLS regression line
    plt.plot(xi, poly1d(np.polyfit(x, y, 1))(xi), '0.4')
    plt.xlabel('SPY')
    plt.ylabel(equities[i]);
    plt.axis([min(x)-5, max(x)+5, min(y)-5, max(y)+5])
plt.subplots_adjust(hspace=0.5, )
cax = plt.axes([1, 0.1, 0.075, 0.8])
plt.colorbar(sc, cax=cax)


# In[151]:


step = 20
x_r = x.pct_change()[1:]
colors_r = np.linspace(0.1, 1, len(x_r))
for i in range(len(equities)):
    y_r = get_pricing(equities[i], fields='price', start_date=start, end_date=end).pct_change()[1:]
    xi = np.linspace(x_r.min(), x_r.max(), 2)
    state_means, state_covs = kf.filter(y_r.values)
    plot_id = str(plot + i)
    plt.subplot(plot_id)
    cm = plt.get_cmap('jet')
    colors = np.linspace(0.1, 1, len(x))
    sc = plt.scatter(x_r, y_r, s=30, c=colors_r, cmap=cm, edgecolor='k', alpha=0.7)
    #cb = plt.colorbar(sc)
    #cb.ax.set_yticklabels([str(p.date()) for p in x[::len(x)//9].index])
    colors_l = np.linspace(0.1, 1, len(state_means[::step]))
    for j, beta in enumerate(state_means[::step]):
        plt.plot(xi, beta[0] * xi + beta[1], alpha=.2, lw=1, c=cm(colors_l[j]))
    # Plot the OLS regression line
    plt.plot(xi, poly1d(np.polyfit(x_r, y_r, 1))(xi), '0.4')
    plt.xlabel('SPY')
    plt.ylabel(equities[i]);
    plt.axis([-0.03,0.03,-0.11, 0.11])
plt.subplots_adjust(hspace=0.5, )
cax = plt.axes([1, 0.1, 0.075, 0.8])
plt.colorbar(sc, cax=cax)


# In[152]:


returns.columns = map(lambda x: x.symbol, returns.columns)
print 'Covariance matrix:'
print returns.cov()


# In[159]:


R_P = np.mean(returns, axis=1)
print 'Portfolio Volatility'
print np.std(R_P)


# In[ ]:





# In[ ]:





# In[ ]:




