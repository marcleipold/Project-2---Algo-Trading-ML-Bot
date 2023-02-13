#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
from pathlib import Path
import numpy as np
import alpaca_trade_api as tradeapi
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from finta import TA
from finta.utils import resample_calendar
import json
import hvplot.pandas
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Alpaca for data
import alpaca_trade_api as api
from alpaca_trade_api.rest import TimeFrame

# pandas for analysis
import pandas as pd

# Plotly for charting
#import plotly.graph_objects as go
#import plotly.express as px

# Set default charting for pandas to plotly
#pd.options.plotting.backend = "plotly"

import datetime as dt
import pytz

import os
import requests


# In[50]:


# Initial imports
import os
import requests
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import warnings
warnings.filterwarnings("ignore")




# In[51]:


load_dotenv()


# In[52]:


# Our API keys for Alpaca
API_KEY = os.getenv('ALPACA_API_KEY')

API_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')


# In[53]:


def alpaca_data_import(coin):

    # Load .env environment variables

    # Our API keys for Alpaca
    API_KEY = os.getenv('ALPACA_API_KEY')

    API_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

    #today = dt.date.today()
    # symbols we will be looking at
    #btc = "BTCUSD"
    #spy = "SPY"

    # start dates and end dates for backtest
    #start_date = "2020-01-01"
    #end_date = today 
    #end_date = '2023-02-06'

    end_date = (dt.datetime.now(pytz.timezone('US/Eastern'))-dt.timedelta(hours=1)).isoformat()

    start_date = (dt.datetime.now(pytz.timezone('US/Eastern'))-dt.timedelta(days=1080)).isoformat()

    # time frame for backtests
    timeframe = TimeFrame.Day

    # Setup instance of alpaca api
    alpaca = api.REST(API_KEY, API_SECRET_KEY)

    # # # Request historical bar data for SPY and BTC using Alpaca Data API
    # for equities, use .get_bars
    #spy_data = alpaca.get_bars(spy, timeframe, start_date, end_date).df

    # for crypto, use .get_crypto_bars, from multiple exchanges
    #btc_data = alpaca.get_crypto_bars(btc, timeframe, start_date, end_date).df
    df = alpaca.get_crypto_bars(coin, timeframe, start_date, end_date).df

    df = df[df['exchange'] == 'CBSE']

    # display crypto bar data
    #display(df)
    #display(spy_data)

    return df


# In[54]:


btc_data = alpaca_data_import('BTCUSD')


# In[ ]:





# In[55]:



# #today = dt.date.today()
# # symbols we will be looking at
# btc = "BTCUSD"
# #spy = "SPY"

# # start dates and end dates for backtest
# #start_date = "2020-01-01"
# #end_date = today 
# #end_date = '2023-02-06'

# end_date = (dt.datetime.now(pytz.timezone('US/Eastern'))-dt.timedelta(hours=1)).isoformat()

# start_date = (dt.datetime.now(pytz.timezone('US/Eastern'))-dt.timedelta(days=1080)).isoformat()

# # time frame for backtests
# timeframe = TimeFrame.Day


# In[56]:


# # Our API keys for Alpaca
# API_KEY = os.getenv('ALPACA_API_KEY')

# API_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# # Setup instance of alpaca api
# alpaca = api.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# # # # Request historical bar data for SPY and BTC using Alpaca Data API
# # for equities, use .get_bars
# #spy_data = alpaca.get_bars(spy, timeframe, start_date, end_date).df

# # for crypto, use .get_crypto_bars, from multiple exchanges
# btc_data = alpaca.get_crypto_bars(btc, timeframe, start_date, end_date).df

# btc_data = btc_data[btc_data['exchange'] == 'CBSE']

# # display crypto bar data
# display(btc_data)
# #display(spy_data)


# In[57]:


def process_data_ohlcv(df):

    ohlcv_df = df.drop(columns=['exchange','trade_count','vwap'])
    #ohlc_df = df.drop(columns=['volume'])
    ichimoku_df = TA.ICHIMOKU(ohlcv_df, tenkan_period= 20, kijun_period= 60, senkou_period= 120, chikou_period= 30)
    ichimoku_df = pd.concat([ichimoku_df, ohlcv_df['close']], axis=1)

    return ichimoku_df
#ohlcv_df

#ohlcv_df = resample_calendar(ohlcv_df, '4h')
#ohlcv_df = ohlcv_df.dropna()


# In[58]:


ichimoku_df = process_data_ohlcv(btc_data)
#ichimoku_df


# In[59]:


def get_signal(df):
    
    df['signal'] = 0
    df["signal"] = np.where((df["TENKAN"] > df["KIJUN"]) &
                                (df["close"] > df["senkou_span_a"]), 1, 0)

    df['entry/exit'] = df['signal'].diff()
    
    df['actual_returns'] = df['close'].pct_change()
    df["strategy_returns"] = df["actual_returns"] * df["signal"].shift() 

    actual_returns = df['actual_returns'].cumsum()
    
    
    return df


# In[ ]:





# In[60]:


ichimoku_signal_df =  get_signal(ichimoku_df)
#print(ichimoku_signal_df.tail())

# Making the testing and training data sets
X = ichimoku_signal_df[['TENKAN', 'KIJUN', 'senkou_span_a', 'SENKOU', 'CHIKOU', 'close',
       'signal']].shift().dropna().copy()
print(X.tail())
y = ichimoku_signal_df['entry/exit'].copy()
print(y.tail())
y.value_counts()

def train_test_data(X,y):
    # Settign training and testing parameters
    training_begin = X.index.min()
    training_end = X.index.min() + DateOffset(months=24)
    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]
    X_test = X.loc[training_end:]
    y_test = y.loc[training_end:]
    # Scaling the training and testing data
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    # Initiating Deep Neural Network
    number_input_features = len(X_train.iloc[0])
    number_output_neurons = 1
    # Defining number of hidden nodes for first layer
    hidden_nodes_layer1 = np.ceil(np.sqrt(number_input_features * number_output_neurons))
    # Defining the number of hidden nodes in layer 2
    hidden_nodes_layer2 = np.ceil(np.sqrt(hidden_nodes_layer1 * number_output_neurons))
    # Creating the Sequential model instance
    nn=Sequential()
    # Adding the first layer
    nn.add(
        Dense(
            units=hidden_nodes_layer1,
            activation='relu',
            input_dim=number_input_features
        )
    )
    # Adding second layer
    nn.add(
        Dense(
            units=hidden_nodes_layer2,
            activation='relu'
        )
    )
    # Adding the output layer
    nn.add(
        Dense(
            units=1,
            activation='sigmoid'
        )
    )
    # Reviewing the Sequential model
    print(nn.summary())
    # Compiling the Sequential model
    nn.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    # Fitting the model with the epochs nad training data
    nn.model=nn.fit(X_train_scaled, y_train, epochs=100, verbose=2)

    return nn, X_test_scaled, y_test

trained_model, X_test_scaled, y_test = train_test_data(X,y)

predictions = trained_model.predict(X_test_scaled)

print(predictions)


# In[61]:


def create_order(symbol, qty, side, order_type, time_in_force, api_key_id, api_secret_key):
    headers = {
        "Apca-Api-Key-Id": api_key_id,
        "Apca-Api-Secret-Key": api_secret_key,
        "Content-Type": "application/json"
    }

    data = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force
    }

    url = "https://paper-api.alpaca.markets/v2/orders"
    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise ValueError("Failed to create order: " + response.text)

    return response.json()


# In[62]:



api_key_id = API_KEY
api_secret_key = API_SECRET_KEY

#result = create_order("BTC/USD", "1.0", "buy", "market", "gtc", api_key_id, api_secret_key)
#print(result)


# In[69]:


ichimoku_signal_df['entry/exit'].iloc[-1] 


# In[70]:


if ichimoku_signal_df['entry/exit'].iloc[-1] == 1.0:
    create_order("BTC/USD", "1.0", "buy", "market", "gtc", api_key_id, api_secret_key)
elif ichimoku_signal_df['entry/exit'].iloc[-1] == -1.0:
    create_order("BTC/USD", "1.0", "sell", "market", "gtc", api_key_id, api_secret_key)
else:
    print('No Trades Today!')


# In[ ]:


# ohlc_df = ohlcv_df.drop(columns=['volume'])
# #ohlc_df


# In[ ]:


# ichimoku_df = TA.ICHIMOKU(ohlcv_df, tenkan_period= 20, kijun_period= 60, senkou_period= 120, chikou_period= 30)
# #ichimoku_df


# In[ ]:


# ichimoku_signal_df = pd.concat([ichimoku_df, ohlcv_df['close']], axis=1)
# #ichimoku_signal_df


# In[ ]:



# ichimoku_signal_df['signal'] = 0
# ichimoku_signal_df["signal"] = np.where((ichimoku_signal_df["TENKAN"] > ichimoku_signal_df["KIJUN"]) &
#                                 (ichimoku_signal_df["close"] > ichimoku_signal_df["senkou_span_a"]), 1, 0)

# ichimoku_signal_df['entry/exit'] = ichimoku_signal_df['signal'].diff()

# #ichimoku_signal_df


# In[ ]:


# ichimoku_signal_df['actual_returns'] = ichimoku_signal_df['close'].pct_change()
# ichimoku_signal_df["strategy_returns"] = ichimoku_signal_df["actual_returns"] * ichimoku_signal_df["signal"].shift() 


# In[ ]:


#ichimoku_signal_df


# In[ ]:


#actual_returns = ichimoku_signal_df['actual_returns'].cumsum()

# actual_returns_plot =  (1 + ichimoku_signal_df[['actual_returns']]).cumprod().hvplot(
#     color='lightblue'
# )


# In[ ]:


# strategy_returns_plot = (1 + ichimoku_signal_df[['strategy_returns']]).cumprod().hvplot(
#     color='lightgreen'
# )


# In[ ]:


#ichimoku_signal_df


# In[ ]:


#ichimoku_signal_df['entry/exit'].value_counts()


# In[ ]:


# entry = ichimoku_signal_df[ichimoku_signal_df['entry/exit']==1]['close'].hvplot.scatter(
#     color='green',
#     marker= '^'
# )

# exit = ichimoku_signal_df[ichimoku_signal_df['entry/exit']==-1]['close'].hvplot.scatter(
#     color='red',
#     marker= 'v'
# )

# close = ichimoku_signal_df['close'].hvplot(
#     color='lightgray',
    
# )

#actual_returns = actual_returns.hvplot(
 #   color='lightblue'
#)

#strategy_returns = ichimoku_signal_df['strategy_returns'].hvplot(
 #   color='lightgreen',
#)


# In[ ]:


# close * entry * exit


# In[ ]:


# actual_returns_plot * strategy_returns_plot


# In[ ]:


strategy_returns_annual_volitility = ichimoku_signal_df[['strategy_returns']].std()*np.sqrt(365)
# strategy_returns_annual_volitility


# In[ ]:


annualized_return = ichimoku_signal_df["strategy_returns"].mean() * 365
annualized_std = ichimoku_signal_df["strategy_returns"].std() * np.sqrt(365)
sharpe_ratio = round(annualized_return/annualized_std, 3)


print(f'annualized_return:{annualized_return}')
print(f'annualized_std:{annualized_std}')
print(f'sharpe_ratio:{sharpe_ratio}')


# # In[ ]:


# # ichimoku_df.hvplot()


# # In[ ]:


# from pytrends.request import TrendReq

# pytrends = TrendReq(hl='en-US', tz=360)

# def get_trends(keywords):
#     pytrends.build_payload(kw_list=keywords, timeframe='today 12-m')  
#     interest_over_time_data = pytrends.interest_over_time()
#     return interest_over_time_data

# keywords = ['bitcoin']
# trends_data = get_trends(keywords)
# print(trends_data)


# # In[ ]:


# trends_data_plot =  trends_data['bitcoin'].hvplot(
#     color='red',
    
# )


# # 
