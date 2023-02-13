#!/usr/bin/env python
# coding: utf-8

# Install modules
#!pip install alpaca_trade_api
#!pip install finta
#!pip install python-dotenv

# Initial imports

# connecting google drive to colab to use .env for alpaca api
from google.colab import drive
drive.mount('/content/drive')
import os
from dotenv import load_dotenv
dotenv_path = "/content/drive/My Drive/.env"

# import dataframe tools
import pandas as pd
import numpy as np
from finta import TA
from finta.utils import resample_calendar

# Alpaca for data
import alpaca_trade_api as api
from alpaca_trade_api.rest import TimeFrame

# datetime and py trends for google trends data
import datetime as dt
import pytz

# Initial imports
import warnings
warnings.filterwarnings("ignore")

# function to get bar data from alpaca
def alpaca_data_import(coin):

    # Load .env environment variables
    # Our API keys for Alpaca
    API_KEY = os.environ.get("ALPACA_API_KEY")
    API_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

    # Start and end date, fluid last 1080 days
    end_date = (dt.datetime.now(pytz.timezone('US/Eastern'))-dt.timedelta(hours=1)).isoformat()
    start_date = (dt.datetime.now(pytz.timezone('US/Eastern'))-dt.timedelta(days=1080)).isoformat()

    # time frame for backtests
    timeframe = TimeFrame.Day

    # Setup instance of alpaca api
    alpaca = api.REST(API_KEY, API_SECRET_KEY)

    # # # Request historical bar data for SPY and BTC using Alpaca Data API
    # for equities, use .get_bars
    # for crypto, use .get_crypto_bars, from multiple exchanges
    df = alpaca.get_crypto_bars(coin, timeframe, start_date, end_date).df
    #build dataframe only with coinbase bars
    df = df[df['exchange'] == 'CBSE']

    return df

# Calling Alpaca function to get btc bars data
btc_data = alpaca_data_import('BTCUSD')

# Processing ohlcv dataframe to use ichimoku indicator
def process_data_ohlcv(df):

    ohlcv_df = df.drop(columns=['exchange','trade_count','vwap'])
    #ohlc_df = df.drop(columns=['volume'])
    ichimoku_df = TA.ICHIMOKU(ohlcv_df, tenkan_period= 20, kijun_period= 60, senkou_period= 120, chikou_period= 30)
    ichimoku_df = pd.concat([ichimoku_df, ohlcv_df['close']], axis=1)

    return ichimoku_df

# commented out code for resampling bars for TA
#ohlcv_df
#ohlcv_df = resample_calendar(ohlcv_df, '4h')
#ohlcv_df = ohlcv_df.dropna()

# Creating ichimoku indicator dataframe
ichimoku_df = process_data_ohlcv(btc_data)

# Trade signal logic on indicator
def get_signal(df):
    # Creating signal feature
    df['signal'] = 0

    # Creating entry signal when Tenkan is greater than Kijun and price is above the cloud
    # Creates exit signal when Tenkan is less than Kijun and price falls into cloud
    df["signal"] = np.where((df["TENKAN"] > df["KIJUN"]) &
                                (df["close"] > df["senkou_span_a"]), 1, 0)
    # Prints entry/exit signal
    df['entry/exit'] = df['signal'].diff()
    
    # Calculates returns of bitcoin and our strategy
    df['actual_returns'] = df['close'].pct_change()
    df["strategy_returns"] = df["actual_returns"] * df["signal"].shift() 

    actual_returns = df['actual_returns'].cumsum()
    
    
    return df

# Creates signal dataframe
ichimoku_signal_df =  get_signal(ichimoku_df)

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

# calls model to train
trained_model, X_test_scaled, y_test = train_test_data(X,y)

#creates predictions
predictions = trained_model.predict(X_test_scaled)

print(predictions[-1])

# function to buy/sell btc through alpaca
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

# Returns latest row of daily candle post signal processing
ichimoku_signal_df['entry/exit'].iloc[-1] 

# Calls the create order function to buy or sell bitcoin from trading signal or holds until market is ready
if ichimoku_signal_df['entry/exit'].iloc[-1] == 1.0:
    create_order("BTC/USD", "1.0", "buy", "market", "gtc", api_key_id, api_secret_key)
elif ichimoku_signal_df['entry/exit'].iloc[-1] == -1.0:
    create_order("BTC/USD", "1.0", "sell", "market", "gtc", api_key_id, api_secret_key)
else:
    print('No Trades Today!')


# Strategy_returns_annual_volitility
strategy_returns_annual_volitility = ichimoku_signal_df[['strategy_returns']].std()*np.sqrt(365)

# Calculates key performance metrics of the fund
annualized_return = ichimoku_signal_df["strategy_returns"].mean() * 365
annualized_std = ichimoku_signal_df["strategy_returns"].std() * np.sqrt(365)
sharpe_ratio = round(annualized_return/annualized_std, 3)

# Displays metrics
print(f"ichimoku strategy annualized returns: {round(annualized_return,2)}")
print(f"ichimoku strategy annualized volitility: {round(annualized_std,2)}")
print(f"ichimoku strategy sharpe ratio: {sharpe_ratio}")

# Google trend data commented out for bitcoin keyword
# from pytrends.request import TrendReq

# pytrends = TrendReq(hl='en-US', tz=360)

# def get_trends(keywords):
#     pytrends.build_payload(kw_list=keywords, timeframe='today 12-m')  
#     interest_over_time_data = pytrends.interest_over_time()
#     return interest_over_time_data

# keywords = ['bitcoin']
# trends_data = get_trends(keywords)
# print(trends_data)
