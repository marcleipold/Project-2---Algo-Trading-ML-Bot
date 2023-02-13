{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "history_visible": true,
      "authorship_tag": "ABX9TyP4T/hhV5VdICMYb5lgdVSi",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/marcleipold/Project-2---Algo-Trading-ML-Bot/blob/main/ichimoku_strategy.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BCZIZKON1lp3",
        "outputId": "42dd0b5d-61ab-4edc-8f34-4b0da95f773d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n",
            "                              TENKAN      KIJUN  senkou_span_a     SENKOU  \\\n",
            "timestamp                                                                   \n",
            "2023-01-11 06:00:00+00:00  16918.225  16922.680        18825.7  20363.285   \n",
            "2023-01-12 06:00:00+00:00  17350.920  16922.680        18495.4  20363.285   \n",
            "2023-01-13 06:00:00+00:00  17721.005  17287.925        18495.4  20363.285   \n",
            "2023-01-14 06:00:00+00:00  18824.070  18390.990        18495.4  20363.285   \n",
            "2023-01-15 06:00:00+00:00  18824.070  18390.990        18495.4  20363.285   \n",
            "\n",
            "                             CHIKOU     close  signal  \n",
            "timestamp                                              \n",
            "2023-01-11 06:00:00+00:00  21802.71  17409.60     0.0  \n",
            "2023-01-12 06:00:00+00:00  21678.33  18085.57     0.0  \n",
            "2023-01-13 06:00:00+00:00  21803.77  18790.58     1.0  \n",
            "2023-01-14 06:00:00+00:00  21839.49  20843.55     1.0  \n",
            "2023-01-15 06:00:00+00:00  21806.74  20752.70     1.0  \n",
            "timestamp\n",
            "2023-02-09 06:00:00+00:00    0.0\n",
            "2023-02-10 06:00:00+00:00    0.0\n",
            "2023-02-11 06:00:00+00:00    0.0\n",
            "2023-02-12 06:00:00+00:00    0.0\n",
            "2023-02-13 06:00:00+00:00    0.0\n",
            "Name: entry/exit, dtype: float64\n",
            "Model: \"sequential_9\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " dense_27 (Dense)            (None, 3)                 24        \n",
            "                                                                 \n",
            " dense_28 (Dense)            (None, 2)                 8         \n",
            "                                                                 \n",
            " dense_29 (Dense)            (None, 1)                 3         \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 35\n",
            "Trainable params: 35\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n",
            "None\n",
            "Epoch 1/100\n",
            "23/23 - 1s - loss: 1.3013 - accuracy: 0.2011 - 894ms/epoch - 39ms/step\n",
            "Epoch 2/100\n",
            "23/23 - 0s - loss: 1.0921 - accuracy: 0.2599 - 48ms/epoch - 2ms/step\n",
            "Epoch 3/100\n",
            "23/23 - 0s - loss: 0.9255 - accuracy: 0.3611 - 46ms/epoch - 2ms/step\n",
            "Epoch 4/100\n",
            "23/23 - 0s - loss: 0.7919 - accuracy: 0.4514 - 49ms/epoch - 2ms/step\n",
            "Epoch 5/100\n",
            "23/23 - 0s - loss: 0.6936 - accuracy: 0.5718 - 47ms/epoch - 2ms/step\n",
            "Epoch 6/100\n",
            "23/23 - 0s - loss: 0.6157 - accuracy: 0.7264 - 51ms/epoch - 2ms/step\n",
            "Epoch 7/100\n",
            "23/23 - 0s - loss: 0.5551 - accuracy: 0.8235 - 48ms/epoch - 2ms/step\n",
            "Epoch 8/100\n",
            "23/23 - 0s - loss: 0.5071 - accuracy: 0.9166 - 50ms/epoch - 2ms/step\n",
            "Epoch 9/100\n",
            "23/23 - 0s - loss: 0.4683 - accuracy: 0.9357 - 47ms/epoch - 2ms/step\n",
            "Epoch 10/100\n",
            "23/23 - 0s - loss: 0.4364 - accuracy: 0.9439 - 50ms/epoch - 2ms/step\n",
            "Epoch 11/100\n",
            "23/23 - 0s - loss: 0.4091 - accuracy: 0.9713 - 47ms/epoch - 2ms/step\n",
            "Epoch 12/100\n",
            "23/23 - 0s - loss: 0.3848 - accuracy: 0.9808 - 61ms/epoch - 3ms/step\n",
            "Epoch 13/100\n",
            "23/23 - 0s - loss: 0.3622 - accuracy: 0.9850 - 47ms/epoch - 2ms/step\n",
            "Epoch 14/100\n",
            "23/23 - 0s - loss: 0.3413 - accuracy: 0.9850 - 50ms/epoch - 2ms/step\n",
            "Epoch 15/100\n",
            "23/23 - 0s - loss: 0.3221 - accuracy: 0.9850 - 57ms/epoch - 2ms/step\n",
            "Epoch 16/100\n",
            "23/23 - 0s - loss: 0.3035 - accuracy: 0.9850 - 48ms/epoch - 2ms/step\n",
            "Epoch 17/100\n",
            "23/23 - 0s - loss: 0.2867 - accuracy: 0.9850 - 56ms/epoch - 2ms/step\n",
            "Epoch 18/100\n",
            "23/23 - 0s - loss: 0.2708 - accuracy: 0.9850 - 53ms/epoch - 2ms/step\n",
            "Epoch 19/100\n",
            "23/23 - 0s - loss: 0.2559 - accuracy: 0.9850 - 54ms/epoch - 2ms/step\n",
            "Epoch 20/100\n",
            "23/23 - 0s - loss: 0.2421 - accuracy: 0.9850 - 49ms/epoch - 2ms/step\n",
            "Epoch 21/100\n",
            "23/23 - 0s - loss: 0.2291 - accuracy: 0.9850 - 54ms/epoch - 2ms/step\n",
            "Epoch 22/100\n",
            "23/23 - 0s - loss: 0.2170 - accuracy: 0.9850 - 53ms/epoch - 2ms/step\n",
            "Epoch 23/100\n",
            "23/23 - 0s - loss: 0.2059 - accuracy: 0.9850 - 47ms/epoch - 2ms/step\n",
            "Epoch 24/100\n",
            "23/23 - 0s - loss: 0.1956 - accuracy: 0.9850 - 49ms/epoch - 2ms/step\n",
            "Epoch 25/100\n",
            "23/23 - 0s - loss: 0.1858 - accuracy: 0.9850 - 73ms/epoch - 3ms/step\n",
            "Epoch 26/100\n",
            "23/23 - 0s - loss: 0.1769 - accuracy: 0.9850 - 80ms/epoch - 3ms/step\n",
            "Epoch 27/100\n",
            "23/23 - 0s - loss: 0.1685 - accuracy: 0.9850 - 78ms/epoch - 3ms/step\n",
            "Epoch 28/100\n",
            "23/23 - 0s - loss: 0.1607 - accuracy: 0.9850 - 72ms/epoch - 3ms/step\n",
            "Epoch 29/100\n",
            "23/23 - 0s - loss: 0.1531 - accuracy: 0.9850 - 72ms/epoch - 3ms/step\n",
            "Epoch 30/100\n",
            "23/23 - 0s - loss: 0.1461 - accuracy: 0.9850 - 81ms/epoch - 4ms/step\n",
            "Epoch 31/100\n",
            "23/23 - 0s - loss: 0.1398 - accuracy: 0.9850 - 75ms/epoch - 3ms/step\n",
            "Epoch 32/100\n",
            "23/23 - 0s - loss: 0.1334 - accuracy: 0.9850 - 75ms/epoch - 3ms/step\n",
            "Epoch 33/100\n",
            "23/23 - 0s - loss: 0.1275 - accuracy: 0.9850 - 74ms/epoch - 3ms/step\n",
            "Epoch 34/100\n",
            "23/23 - 0s - loss: 0.1221 - accuracy: 0.9850 - 72ms/epoch - 3ms/step\n",
            "Epoch 35/100\n",
            "23/23 - 0s - loss: 0.1167 - accuracy: 0.9850 - 73ms/epoch - 3ms/step\n",
            "Epoch 36/100\n",
            "23/23 - 0s - loss: 0.1118 - accuracy: 0.9850 - 72ms/epoch - 3ms/step\n",
            "Epoch 37/100\n",
            "23/23 - 0s - loss: 0.1069 - accuracy: 0.9850 - 74ms/epoch - 3ms/step\n",
            "Epoch 38/100\n",
            "23/23 - 0s - loss: 0.1024 - accuracy: 0.9850 - 78ms/epoch - 3ms/step\n",
            "Epoch 39/100\n",
            "23/23 - 0s - loss: 0.0979 - accuracy: 0.9850 - 75ms/epoch - 3ms/step\n",
            "Epoch 40/100\n",
            "23/23 - 0s - loss: 0.0938 - accuracy: 0.9850 - 91ms/epoch - 4ms/step\n",
            "Epoch 41/100\n",
            "23/23 - 0s - loss: 0.0899 - accuracy: 0.9850 - 89ms/epoch - 4ms/step\n",
            "Epoch 42/100\n",
            "23/23 - 0s - loss: 0.0858 - accuracy: 0.9850 - 84ms/epoch - 4ms/step\n",
            "Epoch 43/100\n",
            "23/23 - 0s - loss: 0.0822 - accuracy: 0.9850 - 87ms/epoch - 4ms/step\n",
            "Epoch 44/100\n",
            "23/23 - 0s - loss: 0.0785 - accuracy: 0.9850 - 87ms/epoch - 4ms/step\n",
            "Epoch 45/100\n",
            "23/23 - 0s - loss: 0.0751 - accuracy: 0.9850 - 80ms/epoch - 3ms/step\n",
            "Epoch 46/100\n",
            "23/23 - 0s - loss: 0.0717 - accuracy: 0.9850 - 79ms/epoch - 3ms/step\n",
            "Epoch 47/100\n",
            "23/23 - 0s - loss: 0.0684 - accuracy: 0.9850 - 79ms/epoch - 3ms/step\n",
            "Epoch 48/100\n",
            "23/23 - 0s - loss: 0.0653 - accuracy: 0.9850 - 73ms/epoch - 3ms/step\n",
            "Epoch 49/100\n",
            "23/23 - 0s - loss: 0.0623 - accuracy: 0.9850 - 76ms/epoch - 3ms/step\n",
            "Epoch 50/100\n",
            "23/23 - 0s - loss: 0.0593 - accuracy: 0.9850 - 77ms/epoch - 3ms/step\n",
            "Epoch 51/100\n",
            "23/23 - 0s - loss: 0.0564 - accuracy: 0.9850 - 81ms/epoch - 4ms/step\n",
            "Epoch 52/100\n",
            "23/23 - 0s - loss: 0.0537 - accuracy: 0.9850 - 80ms/epoch - 3ms/step\n",
            "Epoch 53/100\n",
            "23/23 - 0s - loss: 0.0509 - accuracy: 0.9850 - 78ms/epoch - 3ms/step\n",
            "Epoch 54/100\n",
            "23/23 - 0s - loss: 0.0483 - accuracy: 0.9850 - 79ms/epoch - 3ms/step\n",
            "Epoch 55/100\n",
            "23/23 - 0s - loss: 0.0460 - accuracy: 0.9850 - 84ms/epoch - 4ms/step\n",
            "Epoch 56/100\n",
            "23/23 - 0s - loss: 0.0431 - accuracy: 0.9850 - 79ms/epoch - 3ms/step\n",
            "Epoch 57/100\n",
            "23/23 - 0s - loss: 0.0408 - accuracy: 0.9850 - 69ms/epoch - 3ms/step\n",
            "Epoch 58/100\n",
            "23/23 - 0s - loss: 0.0384 - accuracy: 0.9850 - 76ms/epoch - 3ms/step\n",
            "Epoch 59/100\n",
            "23/23 - 0s - loss: 0.0363 - accuracy: 0.9850 - 79ms/epoch - 3ms/step\n",
            "Epoch 60/100\n",
            "23/23 - 0s - loss: 0.0338 - accuracy: 0.9850 - 78ms/epoch - 3ms/step\n",
            "Epoch 61/100\n",
            "23/23 - 0s - loss: 0.0315 - accuracy: 0.9850 - 76ms/epoch - 3ms/step\n",
            "Epoch 62/100\n",
            "23/23 - 0s - loss: 0.0294 - accuracy: 0.9850 - 73ms/epoch - 3ms/step\n",
            "Epoch 63/100\n",
            "23/23 - 0s - loss: 0.0274 - accuracy: 0.9850 - 69ms/epoch - 3ms/step\n",
            "Epoch 64/100\n",
            "23/23 - 0s - loss: 0.0250 - accuracy: 0.9850 - 68ms/epoch - 3ms/step\n",
            "Epoch 65/100\n",
            "23/23 - 0s - loss: 0.0230 - accuracy: 0.9850 - 72ms/epoch - 3ms/step\n",
            "Epoch 66/100\n",
            "23/23 - 0s - loss: 0.0210 - accuracy: 0.9850 - 81ms/epoch - 4ms/step\n",
            "Epoch 67/100\n",
            "23/23 - 0s - loss: 0.0188 - accuracy: 0.9850 - 79ms/epoch - 3ms/step\n",
            "Epoch 68/100\n",
            "23/23 - 0s - loss: 0.0170 - accuracy: 0.9850 - 65ms/epoch - 3ms/step\n",
            "Epoch 69/100\n",
            "23/23 - 0s - loss: 0.0148 - accuracy: 0.9850 - 68ms/epoch - 3ms/step\n",
            "Epoch 70/100\n",
            "23/23 - 0s - loss: 0.0130 - accuracy: 0.9850 - 68ms/epoch - 3ms/step\n",
            "Epoch 71/100\n",
            "23/23 - 0s - loss: 0.0108 - accuracy: 0.9850 - 80ms/epoch - 3ms/step\n",
            "Epoch 72/100\n",
            "23/23 - 0s - loss: 0.0092 - accuracy: 0.9850 - 74ms/epoch - 3ms/step\n",
            "Epoch 73/100\n",
            "23/23 - 0s - loss: 0.0071 - accuracy: 0.9850 - 72ms/epoch - 3ms/step\n",
            "Epoch 74/100\n",
            "23/23 - 0s - loss: 0.0053 - accuracy: 0.9850 - 78ms/epoch - 3ms/step\n",
            "Epoch 75/100\n",
            "23/23 - 0s - loss: 0.0033 - accuracy: 0.9850 - 77ms/epoch - 3ms/step\n",
            "Epoch 76/100\n",
            "23/23 - 0s - loss: 0.0013 - accuracy: 0.9850 - 80ms/epoch - 3ms/step\n",
            "Epoch 77/100\n",
            "23/23 - 0s - loss: -4.4745e-04 - accuracy: 0.9850 - 74ms/epoch - 3ms/step\n",
            "Epoch 78/100\n",
            "23/23 - 0s - loss: -2.3675e-03 - accuracy: 0.9850 - 85ms/epoch - 4ms/step\n",
            "Epoch 79/100\n",
            "23/23 - 0s - loss: -4.2869e-03 - accuracy: 0.9850 - 94ms/epoch - 4ms/step\n",
            "Epoch 80/100\n",
            "23/23 - 0s - loss: -6.1200e-03 - accuracy: 0.9850 - 63ms/epoch - 3ms/step\n",
            "Epoch 81/100\n",
            "23/23 - 0s - loss: -8.1615e-03 - accuracy: 0.9850 - 59ms/epoch - 3ms/step\n",
            "Epoch 82/100\n",
            "23/23 - 0s - loss: -9.9379e-03 - accuracy: 0.9850 - 51ms/epoch - 2ms/step\n",
            "Epoch 83/100\n",
            "23/23 - 0s - loss: -1.1796e-02 - accuracy: 0.9850 - 60ms/epoch - 3ms/step\n",
            "Epoch 84/100\n",
            "23/23 - 0s - loss: -1.3904e-02 - accuracy: 0.9850 - 47ms/epoch - 2ms/step\n",
            "Epoch 85/100\n",
            "23/23 - 0s - loss: -1.5847e-02 - accuracy: 0.9850 - 48ms/epoch - 2ms/step\n",
            "Epoch 86/100\n",
            "23/23 - 0s - loss: -1.7770e-02 - accuracy: 0.9850 - 48ms/epoch - 2ms/step\n",
            "Epoch 87/100\n",
            "23/23 - 0s - loss: -1.9748e-02 - accuracy: 0.9850 - 49ms/epoch - 2ms/step\n",
            "Epoch 88/100\n",
            "23/23 - 0s - loss: -2.1627e-02 - accuracy: 0.9850 - 50ms/epoch - 2ms/step\n",
            "Epoch 89/100\n",
            "23/23 - 0s - loss: -2.3466e-02 - accuracy: 0.9850 - 57ms/epoch - 2ms/step\n",
            "Epoch 90/100\n",
            "23/23 - 0s - loss: -2.5580e-02 - accuracy: 0.9850 - 49ms/epoch - 2ms/step\n",
            "Epoch 91/100\n",
            "23/23 - 0s - loss: -2.7667e-02 - accuracy: 0.9850 - 42ms/epoch - 2ms/step\n",
            "Epoch 92/100\n",
            "23/23 - 0s - loss: -2.9563e-02 - accuracy: 0.9850 - 46ms/epoch - 2ms/step\n",
            "Epoch 93/100\n",
            "23/23 - 0s - loss: -3.1819e-02 - accuracy: 0.9850 - 50ms/epoch - 2ms/step\n",
            "Epoch 94/100\n",
            "23/23 - 0s - loss: -3.3756e-02 - accuracy: 0.9850 - 49ms/epoch - 2ms/step\n",
            "Epoch 95/100\n",
            "23/23 - 0s - loss: -3.5875e-02 - accuracy: 0.9850 - 54ms/epoch - 2ms/step\n",
            "Epoch 96/100\n",
            "23/23 - 0s - loss: -3.7973e-02 - accuracy: 0.9850 - 65ms/epoch - 3ms/step\n",
            "Epoch 97/100\n",
            "23/23 - 0s - loss: -3.9976e-02 - accuracy: 0.9850 - 52ms/epoch - 2ms/step\n",
            "Epoch 98/100\n",
            "23/23 - 0s - loss: -4.2387e-02 - accuracy: 0.9850 - 48ms/epoch - 2ms/step\n",
            "Epoch 99/100\n",
            "23/23 - 0s - loss: -4.4614e-02 - accuracy: 0.9850 - 52ms/epoch - 2ms/step\n",
            "Epoch 100/100\n",
            "23/23 - 0s - loss: -4.6861e-02 - accuracy: 0.9850 - 53ms/epoch - 2ms/step\n",
            "5/5 [==============================] - 0s 3ms/step\n",
            "[0.0439519]\n",
            "No Trades Today!\n",
            "ichimoku strategy annualized returns: 0.61\n",
            "ichimoku strategy annualized volitility: 0.46\n",
            "ichimoku strategy sharpe ratio: 1.346\n"
          ]
        }
      ],
      "source": [
        "#!/usr/bin/env python\n",
        "# coding: utf-8\n",
        "\n",
        "# Install modules\n",
        "#!pip install alpaca_trade_api\n",
        "#!pip install finta\n",
        "#!pip install python-dotenv\n",
        "\n",
        "# Initial imports\n",
        "\n",
        "# connecting google drive to colab to use .env for alpaca api\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "import os\n",
        "from dotenv import load_dotenv\n",
        "dotenv_path = \"/content/drive/My Drive/.env\"\n",
        "\n",
        "# import dataframe tools\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from finta import TA\n",
        "from finta.utils import resample_calendar\n",
        "\n",
        "# Alpaca for data\n",
        "import alpaca_trade_api as api\n",
        "from alpaca_trade_api.rest import TimeFrame\n",
        "\n",
        "# datetime and py trends for google trends data\n",
        "import datetime as dt\n",
        "import pytz\n",
        "\n",
        "# Initial imports\n",
        "import warnings\n",
        "warnings.filterwarnings(\"ignore\")\n",
        "\n",
        "# function to get bar data from alpaca\n",
        "def alpaca_data_import(coin):\n",
        "\n",
        "    # Load .env environment variables\n",
        "    # Our API keys for Alpaca\n",
        "    API_KEY = os.environ.get(\"ALPACA_API_KEY\")\n",
        "    API_SECRET_KEY = os.environ.get(\"ALPACA_SECRET_KEY\")\n",
        "\n",
        "    # Start and end date, fluid last 1080 days\n",
        "    end_date = (dt.datetime.now(pytz.timezone('US/Eastern'))-dt.timedelta(hours=1)).isoformat()\n",
        "    start_date = (dt.datetime.now(pytz.timezone('US/Eastern'))-dt.timedelta(days=1080)).isoformat()\n",
        "\n",
        "    # time frame for backtests\n",
        "    timeframe = TimeFrame.Day\n",
        "\n",
        "    # Setup instance of alpaca api\n",
        "    alpaca = api.REST(API_KEY, API_SECRET_KEY)\n",
        "\n",
        "    # # # Request historical bar data for SPY and BTC using Alpaca Data API\n",
        "    # for equities, use .get_bars\n",
        "    # for crypto, use .get_crypto_bars, from multiple exchanges\n",
        "    df = alpaca.get_crypto_bars(coin, timeframe, start_date, end_date).df\n",
        "    #build dataframe only with coinbase bars\n",
        "    df = df[df['exchange'] == 'CBSE']\n",
        "\n",
        "    return df\n",
        "\n",
        "# Calling Alpaca function to get btc bars data\n",
        "btc_data = alpaca_data_import('BTCUSD')\n",
        "\n",
        "# Processing ohlcv dataframe to use ichimoku indicator\n",
        "def process_data_ohlcv(df):\n",
        "\n",
        "    ohlcv_df = df.drop(columns=['exchange','trade_count','vwap'])\n",
        "    #ohlc_df = df.drop(columns=['volume'])\n",
        "    ichimoku_df = TA.ICHIMOKU(ohlcv_df, tenkan_period= 20, kijun_period= 60, senkou_period= 120, chikou_period= 30)\n",
        "    ichimoku_df = pd.concat([ichimoku_df, ohlcv_df['close']], axis=1)\n",
        "\n",
        "    return ichimoku_df\n",
        "\n",
        "# commented out code for resampling bars for TA\n",
        "#ohlcv_df\n",
        "#ohlcv_df = resample_calendar(ohlcv_df, '4h')\n",
        "#ohlcv_df = ohlcv_df.dropna()\n",
        "\n",
        "# Creating ichimoku indicator dataframe\n",
        "ichimoku_df = process_data_ohlcv(btc_data)\n",
        "\n",
        "# Trade signal logic on indicator\n",
        "def get_signal(df):\n",
        "    # Creating signal feature\n",
        "    df['signal'] = 0\n",
        "\n",
        "    # Creating entry signal when Tenkan is greater than Kijun and price is above the cloud\n",
        "    # Creates exit signal when Tenkan is less than Kijun and price falls into cloud\n",
        "    df[\"signal\"] = np.where((df[\"TENKAN\"] > df[\"KIJUN\"]) &\n",
        "                                (df[\"close\"] > df[\"senkou_span_a\"]), 1, 0)\n",
        "    # Prints entry/exit signal\n",
        "    df['entry/exit'] = df['signal'].diff()\n",
        "    \n",
        "    # Calculates returns of bitcoin and our strategy\n",
        "    df['actual_returns'] = df['close'].pct_change()\n",
        "    df[\"strategy_returns\"] = df[\"actual_returns\"] * df[\"signal\"].shift() \n",
        "\n",
        "    actual_returns = df['actual_returns'].cumsum()\n",
        "    \n",
        "    \n",
        "    return df\n",
        "\n",
        "# Creates signal dataframe\n",
        "ichimoku_signal_df =  get_signal(ichimoku_df)\n",
        "\n",
        "# Making the testing and training data sets\n",
        "X = ichimoku_signal_df[['TENKAN', 'KIJUN', 'senkou_span_a', 'SENKOU', 'CHIKOU', 'close',\n",
        "       'signal']].shift().dropna().copy()\n",
        "print(X.tail())\n",
        "y = ichimoku_signal_df['entry/exit'].copy()\n",
        "print(y.tail())\n",
        "y.value_counts()\n",
        "\n",
        "def train_test_data(X,y):\n",
        "    # Settign training and testing parameters\n",
        "    training_begin = X.index.min()\n",
        "    training_end = X.index.min() + DateOffset(months=24)\n",
        "    X_train = X.loc[training_begin:training_end]\n",
        "    y_train = y.loc[training_begin:training_end]\n",
        "    X_test = X.loc[training_end:]\n",
        "    y_test = y.loc[training_end:]\n",
        "    # Scaling the training and testing data\n",
        "    scaler = StandardScaler()\n",
        "    X_scaler = scaler.fit(X_train)\n",
        "    X_train_scaled = X_scaler.transform(X_train)\n",
        "    X_test_scaled = X_scaler.transform(X_test)\n",
        "    # Initiating Deep Neural Network\n",
        "    number_input_features = len(X_train.iloc[0])\n",
        "    number_output_neurons = 1\n",
        "    # Defining number of hidden nodes for first layer\n",
        "    hidden_nodes_layer1 = np.ceil(np.sqrt(number_input_features * number_output_neurons))\n",
        "    # Defining the number of hidden nodes in layer 2\n",
        "    hidden_nodes_layer2 = np.ceil(np.sqrt(hidden_nodes_layer1 * number_output_neurons))\n",
        "    # Creating the Sequential model instance\n",
        "    nn=Sequential()\n",
        "    # Adding the first layer\n",
        "    nn.add(\n",
        "        Dense(\n",
        "            units=hidden_nodes_layer1,\n",
        "            activation='relu',\n",
        "            input_dim=number_input_features\n",
        "        )\n",
        "    )\n",
        "    # Adding second layer\n",
        "    nn.add(\n",
        "        Dense(\n",
        "            units=hidden_nodes_layer2,\n",
        "            activation='relu'\n",
        "        )\n",
        "    )\n",
        "    # Adding the output layer\n",
        "    nn.add(\n",
        "        Dense(\n",
        "            units=1,\n",
        "            activation='sigmoid'\n",
        "        )\n",
        "    )\n",
        "    # Reviewing the Sequential model\n",
        "    print(nn.summary())\n",
        "    # Compiling the Sequential model\n",
        "    nn.compile(\n",
        "        loss='binary_crossentropy',\n",
        "        optimizer='adam',\n",
        "        metrics=['accuracy']\n",
        "    )\n",
        "    # Fitting the model with the epochs nad training data\n",
        "    nn.model=nn.fit(X_train_scaled, y_train, epochs=100, verbose=2)\n",
        "\n",
        "    return nn, X_test_scaled, y_test\n",
        "\n",
        "# calls model to train\n",
        "trained_model, X_test_scaled, y_test = train_test_data(X,y)\n",
        "\n",
        "#creates predictions\n",
        "predictions = trained_model.predict(X_test_scaled)\n",
        "\n",
        "print(predictions[-1])\n",
        "\n",
        "# function to buy/sell btc through alpaca\n",
        "def create_order(symbol, qty, side, order_type, time_in_force, api_key_id, api_secret_key):\n",
        "    headers = {\n",
        "        \"Apca-Api-Key-Id\": api_key_id,\n",
        "        \"Apca-Api-Secret-Key\": api_secret_key,\n",
        "        \"Content-Type\": \"application/json\"\n",
        "    }\n",
        "\n",
        "    data = {\n",
        "        \"symbol\": symbol,\n",
        "        \"qty\": qty,\n",
        "        \"side\": side,\n",
        "        \"type\": order_type,\n",
        "        \"time_in_force\": time_in_force\n",
        "    }\n",
        "\n",
        "    url = \"https://paper-api.alpaca.markets/v2/orders\"\n",
        "    response = requests.post(url, headers=headers, json=data)\n",
        "\n",
        "    if response.status_code != 200:\n",
        "        raise ValueError(\"Failed to create order: \" + response.text)\n",
        "\n",
        "    return response.json()\n",
        "\n",
        "# Returns latest row of daily candle post signal processing\n",
        "ichimoku_signal_df['entry/exit'].iloc[-1] \n",
        "\n",
        "# Calls the create order function to buy or sell bitcoin from trading signal or holds until market is ready\n",
        "if ichimoku_signal_df['entry/exit'].iloc[-1] == 1.0:\n",
        "    create_order(\"BTC/USD\", \"1.0\", \"buy\", \"market\", \"gtc\", api_key_id, api_secret_key)\n",
        "elif ichimoku_signal_df['entry/exit'].iloc[-1] == -1.0:\n",
        "    create_order(\"BTC/USD\", \"1.0\", \"sell\", \"market\", \"gtc\", api_key_id, api_secret_key)\n",
        "else:\n",
        "    print('No Trades Today!')\n",
        "\n",
        "\n",
        "# Strategy_returns_annual_volitility\n",
        "strategy_returns_annual_volitility = ichimoku_signal_df[['strategy_returns']].std()*np.sqrt(365)\n",
        "\n",
        "# Calculates key performance metrics of the fund\n",
        "annualized_return = ichimoku_signal_df[\"strategy_returns\"].mean() * 365\n",
        "annualized_std = ichimoku_signal_df[\"strategy_returns\"].std() * np.sqrt(365)\n",
        "sharpe_ratio = round(annualized_return/annualized_std, 3)\n",
        "\n",
        "# Displays metrics\n",
        "print(f\"ichimoku strategy annualized returns: {round(annualized_return,2)}\")\n",
        "print(f\"ichimoku strategy annualized volitility: {round(annualized_std,2)}\")\n",
        "print(f\"ichimoku strategy sharpe ratio: {sharpe_ratio}\")\n",
        "\n",
        "# Google trend data commented out for bitcoin keyword\n",
        "# from pytrends.request import TrendReq\n",
        "\n",
        "# pytrends = TrendReq(hl='en-US', tz=360)\n",
        "\n",
        "# def get_trends(keywords):\n",
        "#     pytrends.build_payload(kw_list=keywords, timeframe='today 12-m')  \n",
        "#     interest_over_time_data = pytrends.interest_over_time()\n",
        "#     return interest_over_time_data\n",
        "\n",
        "# keywords = ['bitcoin']\n",
        "# trends_data = get_trends(keywords)\n",
        "# print(trends_data)\n",
        "\n"
      ]
    }
  ]
}