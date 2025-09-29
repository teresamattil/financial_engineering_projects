import yfinance as yf
import pandas as pd
import numpy as np
import datetime as date
import matplotlib.pyplot as plt

startdate = date.datetime(2000, 1, 1)
enddate = date.datetime.today()

# Use yfinance instead of pandas_datareader
ipc = yf.download('^MXX', start=startdate, end=enddate)['Close']

ipc.plot()