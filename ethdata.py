import ccxt
import pandas as pd
import streamlit as st
import time
import streamlit as st

# Parameters
symbol = 'ETH/USDT'
timeframe = '15m'
now = int(time.time() * 1000)
since = now - 7 * 24 * 60 * 60 * 1000  # 7 days ago in ms

# Fetch data using ccxt
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=672)

# Convert to DataFrame
df = pd.DataFrame(ohlcv, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

# Streamlit app
st.title('ETH/USDT 15m Prices - Last 7 Days (Binance)')
st.write(df[['open_time', 'open', 'high', 'low', 'close', 'volume']])
st.line_chart(df.set_index('open_time')['close']) 