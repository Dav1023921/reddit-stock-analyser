import yfinance as yf
import numpy as np
import pandas as pd 
from datetime import datetime, timedelta
from process import get_sentiment_data
import keras
from keras import backend as k 
from tenserflow.keras.models import Sequential
from tenserflow.keras.layers import Activation
from tenserflow.keras.layers.core import Dense
from tenserflow.keras.layers import LSTM
from tensorflow.keras.losses import BinaryCrossentropy

today = datetime.today()
today_str = today.strftime("%Y-%m-%d")

pl = yf.download("NVDA", start = "2022-01-01", end = today_str)

print(pl.tail())

sentiment_values = get_sentiment_data()
train_labels = []

model = Sequential([
    Dense(32, input_shape=(2,), activation ='relu')
    LSTM(64, return_sequences=False)
    Dense(50, activation = "relu")
    LSTM(70, return_sequences=True)
    LSTM(64, return_sequences=True)
    Dense(1, activation='sigmoid')
])

model.compile(optimizer="adam", loss=BinaryCrossentropy(), metrics=['accuracy'])
model.fit(sentiment_values, train_labels, batch_size = 10, epochs = 20, shuffle = True, verbose = 2)