#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import datetime
import matplotlib.pyplot as plt

SECTOR_CODE_NAMES = {
    Sector.BASIC_MATERIALS: 'Basic Materials',
    Sector.CONSUMER_CYCLICAL: 'Consumer Cyclical',
    Sector.FINANCIAL_SERVICES: 'Financial Services',
    Sector.REAL_ESTATE: 'Real Estate',
    Sector.CONSUMER_DEFENSIVE: 'Consumer Defensive',
    Sector.HEALTHCARE: 'Healthcare',
    Sector.UTILITIES: 'Utilities',
    Sector.COMMUNICATION_SERVICES: 'Communication Services',
    Sector.ENERGY: 'Energy',
    Sector.INDUSTRIALS: 'Industrials',
    Sector.TECHNOLOGY: 'Technology',
    -1 : 'Misc'
}

SECTOR_CODES = {
    Sector.BASIC_MATERIALS: 101,
    Sector.CONSUMER_CYCLICAL: 102,
    Sector.FINANCIAL_SERVICES: 103,
    Sector.REAL_ESTATE: 104,
    Sector.CONSUMER_DEFENSIVE: 205,
    Sector.HEALTHCARE: 206,
    Sector.UTILITIES: 207,
    Sector.COMMUNICATION_SERVICES: 308,
    Sector.ENERGY: 309,
    Sector.INDUSTRIALS: 310,
    Sector.TECHNOLOGY: 311,
    -1 : 'Misc'
}
# In[91]:


sector = 311


# In[92]:


def make_pipeline():
    pipe = Pipeline()
    '''standards'''
    book_to_price = 1/Latest([Fundamentals.pb_ratio])
    earning_to_price = 1/Latest([Fundamentals.pe_ratio])
    #research_and_development = Latest([Fundamentals.research_and_development])
    gross_dividend_payment = Latest([Fundamentals.gross_dividend_payment])
    value_score = Latest([Fundamentals.value_score])
    returns = Returns(inputs=[USEquityPricing.close], window_length=2)
    '''filter'''
    mask = QTradableStocksUS()
    mask &= Sector().eq(sector)
    #mask &= value_score > 70
    mask &= USEquityPricing.volume.latest > 10000
    '''pipeline'''
    pipe = Pipeline(  
        columns={
            'Returns': returns,
            'B/P': book_to_price,
            'E/P': earning_to_price,
            'Dividends': gross_dividend_payment,
            'Value_score': value_score
        },  
        screen=mask
    )  
    return pipe 

def extract_consistent_stocks(results):
    daily_stocks = [ results.loc[results.index.levels[0][i]].index.tolist()
                for i in range(len(results.index.levels[0])) ]
    sector_stocks = set(daily_stocks[0])
    for s in daily_stocks[1:]:
        sector_stocks.intersection_update(s)
    sector_stocks = list(sector_stocks)
    return sector_stocks


# In[93]:


pipe = make_pipeline()
start_date = '2018-2-1'
end_date = '2020-2-20'
results = run_pipeline(pipe, start_date, end_date)


# In[94]:


results.head()


# In[95]:


sector_stocks = extract_consistent_stocks(results)


# In[96]:


num_of_stocks = len(sector_stocks)
print "there are %d assets in sector." % num_of_stocks


# In[97]:


tickers = [sector_stock.symbol for sector_stock in sector_stocks]
historical_prices = get_pricing(tickers,start_date='2015-01-01',end_date='2020-02-14')
SPY = get_pricing('SPY', fields='price',start_date='2015-01-01',end_date='2020-02-14')
historical_returns = historical_prices['close_price'].pct_change()[1:]
SPY_returns = SPY.pct_change()[1:]
daily_return_mean = historical_returns.mean(axis=1, skipna=True)
overall_cumreturn = (daily_return_mean + 1).cumprod()
SPY_cumreturn = (SPY_returns + 1).cumprod()


# In[98]:


overall_return.plot()
SPY_cumreturn.plot()
plt.legend(['Sector', 'SPY'])


# In[99]:


bp_ratios = []
ep_ratios = []
for sector_stock in sector_stocks:
    stock_daily_bp_ratios = [ results.loc[ (results.index.levels[0][i], 
                                          sector_stock)] ['B/P'] 
                            for i in range(len(results.index.levels[0]))]
    stock_daily_ep_ratios = [ results.loc[ (results.index.levels[0][i], 
                                          sector_stock)] ['E/P'] 
                            for i in range(len(results.index.levels[0]))]
    stock_daily_bp_ratios = list(
        scipy.stats.mstats.winsorize(stock_daily_bp_ratios, inplace=True, limits=0.03))
    stock_daily_ep_ratios = list(
        scipy.stats.mstats.winsorize(stock_daily_ep_ratios, inplace=True, limits=0.03))
    bp_ratios.append(stock_daily_bp_ratios)
    ep_ratios.append(stock_daily_ep_ratios)
BP_Ratios = pd.DataFrame( np.transpose(bp_ratios),
                columns=sector_stocks, index=results.index.levels[0])
EP_Ratios = pd.DataFrame( np.transpose(ep_ratios),
                columns=sector_stocks, index=results.index.levels[0])


# In[100]:


bp_ratio_daily_avg = BP_Ratios.mean(axis=1)
pb_ratio_daily_avg = 1/bp_ratio_daily_avg
ep_ratio_daily_avg = EP_Ratios.mean(axis=1)
pe_ratio_daily_avg = 1/ep_ratio_daily_avg


# In[101]:


plt.subplot(2, 1, 1)
plt.plot(results.index.levels[0], pb_ratio_daily_avg.values, '-')
title = 'avg performance of %d stocks in Financial Services sector' % num_of_stocks
plt.title(title)
plt.ylabel('PB Ratio')
plt.subplot(2, 1, 2)
plt.plot(results.index.levels[0], pe_ratio_daily_avg.values, '.-')
plt.ylabel('PE Ratio')
plt.show()


# In[ ]:





# In[116]:


def make_pipeline2():
    pipe = Pipeline()
    '''standards'''
    book_to_price = 1/Latest([Fundamentals.pb_ratio])
    PE = Latest([Fundamentals.pe_ratio])
    #research_and_development = Latest([Fundamentals.research_and_development])
    gross_dividend_payment = Latest([Fundamentals.gross_dividend_payment])
    value_score = Latest([Fundamentals.value_score])
    returns = Returns(inputs=[USEquityPricing.close], window_length=2)
    growth_score = Latest([Fundamentals.growth_score])
    profit_grade = Fundamentals.profitability_grade.latest
    financial_health_grade = Fundamentals.financial_health_grade.latest
    '''filter'''
    #profit_requirement = (profit_grade=='A') | (profit_grade=='B')
    mask = QTradableStocksUS()
    mask &= Sector().eq(sector)
    mask &= value_score > 30
    mask &= PE < 25
    mask &= USEquityPricing.volume.latest > 10000
    '''pipeline'''
    pipe = Pipeline(  
        columns={
            'Returns': returns,
            'B/P': book_to_price,
            'P/E': PE,
            'Dividends': gross_dividend_payment,
            'Value Score': value_score,
            'Growth_Score': growth_score,
            'Profit_Grade': profit_grade,
            'Financial_Health': financial_health_grade
        },  
        screen=mask
    )  
    return pipe 


# In[117]:


pipe = make_pipeline2()
today = '2020-2-19'
results = run_pipeline(pipe, today, today)


# In[118]:


filtered_results = results.query("Profit_Grade in ['A', 'B']").query("Financial_Health in ['A', 'B']")


# In[119]:


filtered_results


# In[ ]:





# In[ ]:





# In[ ]:




