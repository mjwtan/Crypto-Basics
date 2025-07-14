
import ccxt
import pandas as pd
import streamlit as st
import time

def fetch_ohlcv(symbol: str, timeframe: str, since: int, limit: int = 672):
    """Fetch OHLCV data from Binance using ccxt."""
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    return ohlcv

def process_ohlcv(ohlcv):
    """Convert raw OHLCV data to a pandas DataFrame and process timestamps."""
    df = pd.DataFrame(ohlcv, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    return df

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout='wide', page_title='Crypto Prices')
    st.title('Crypto Prices - Last 7 Days (Binance)')

    # Dropdown menus for symbol and timeframe
    symbols = ['ETH/USDT', 'BTC/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    symbol = st.selectbox('Select Trading Pair', symbols, index=0)
    timeframe = st.selectbox('Select Timeframe', timeframes, index=2)

    now = int(time.time() * 1000)
    since = now - 7 * 24 * 60 * 60 * 1000  # 7 days ago in ms

    # Fetch and process data
    ohlcv = fetch_ohlcv(symbol, timeframe, since)
    df = process_ohlcv(ohlcv)

    # Display data table
    st.write(df[['open_time', 'open', 'high', 'low', 'close', 'volume']])

    # Candlestick chart using Plotly
    import plotly.graph_objects as go
    fig = go.Figure(data=[
        go.Candlestick(
            x=df['open_time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        )
    ])
    fig.update_layout(title=f'{symbol} {timeframe} Candlestick Chart', xaxis_title='Time', yaxis_title='Price (USDT)', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Line chart for close price
    st.line_chart(df.set_index('open_time')['close'])

if __name__ == "__main__":
    main()