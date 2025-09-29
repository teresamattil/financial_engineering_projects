import yfinance as yf
import pandas as pd
import numpy as np
import datetime as date
import matplotlib.pyplot as plt

startdate = date.datetime(2020, 1, 1)
enddate = date.datetime.today()

# Use yfinance instead of pandas_datareader
ipc = yf.download('^MXX', start=startdate, end=enddate)['Close']

#ipc.plot(title="IPC (^MXX) Closing Prices")
#plt.show()


# St+1 = St* e^( (𝜇 -1/2*𝜎^2)∂t + 𝜎∂Bₜ )
last_year = np.log(ipc[-252:])
sigma = last_year.pct_change().std()*(252**0.5)
first_price = last_year.iloc[0]   # first element (scalar)
last_price_year = last_year.iloc[-1]  # last element (scalar)
mu = float(last_price_year / first_price - 1)



#Explicacion parametros
"""
1. last_year = np.log(ipc[-252:])
    ipc[-252:] → takes the last 252 daily prices, roughly one trading year.
    np.log(...) → transforms prices into log prices.
    In GBM, we work with log returns, because they are normally distributed under the model.
    Los log returns son útiles porque:
        Se suman fácilmente en el tiempo (en vez de multiplicar precios día a día).
        Son aproximadamente normales para periodos cortos, lo que hace más fácil modelar riesgos y hacer simulaciones.

2. sigma = last_year.pct_change().std()*(252**0.5)
    El daily log return es la forma logarítmica de medir el cambio porcentual de un activo de un día al siguiente.
    last_year.pct_change() → computes daily log returns.
    .std() → computes the standard deviation of daily log returns → daily volatility.
    *(252**0.5) → annualizes volatility (≈252 trading days in a year).

    Por qué raiz de 252?
        * Si los retornos diarios son independientes y normalmente distribuidos, entonces la varianza anual se escala así:
        Var(sum\_{t=1}^{N} r\_t) = N \* Var(r\_daily)
        * Tomando la desviación estándar (la raíz cuadrada de la varianza):
        sigma\_annual = sqrt(N) \* sigma\_daily
        * Para mercados de acciones, N ≈ 252 días de trading por año, por eso:
        sigma\_annual = sigma\_daily \* sqrt(252)


3. mu = (last_year[-1:].values / last_year[:1].values)-1
    Aquí estás calculando cuánto creció el índice en un año.
    En finanzas, esto se llama drift, y representa el retorno promedio esperado del índice en el tiempo, ignorando la volatilidad diaria.
    last_year[-1:] → the last log price.
    last_year[:1] → the first log price (a year ago).   
    Ratio → measures growth in log-price over the year.
    Subtract 1 → approximates the annual return.
"""

window = 100
T = 1.0
last_price = float(ipc.iloc[-1])
print("\n\n\nlast price", last_price)

# ----- Creando trayectorias

import scipy as sp

np.random.seed(10)
paths = 7
dt = T / window
S = np.zeros([window], dtype=float)
x = range(0, int(window), 1)

df = pd.DataFrame()

# figure setup
fig = plt.figure()
axis = fig.add_subplot(111)

for j in range(paths):
    S[0] = last_price
    for i in x[:-1]:
        e = np.random.normal()
        S[i+1] = S[i] + S[i]*(mu - 0.5 * sigma**2)*dt + sigma*S[i]*np.sqrt(dt)*e
        df[j] = S  # store the simulated path in DataFrame

    plt.plot(x, S, lw=2)

plt.title("Caminos con movimiento Browniano")
axis.set_xlabel("tiempo")
axis.set_ylabel("precio")
axis.grid(True)
plt.show()
