import ccxt
import pandas as pd
import streamlit as st
import time
import os
from dotenv import load_dotenv
import numpy as np
import requests
from bs4 import BeautifulSoup

load_dotenv()

def fetch_ohlcv(symbol: str, timeframe: str, since: int, limit: int = 1000, max_candles: int = 2500):
    """Fetch OHLCV data from Binance using ccxt, paginated, capped at max_candles (default 2500)."""
    exchange = ccxt.binance()
    all_ohlcv = []
    fetch_since = since
    while len(all_ohlcv) < max_candles:
        fetch_limit = min(limit, max_candles - len(all_ohlcv))
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=fetch_limit)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        if len(ohlcv) < fetch_limit:
            break
        fetch_since = ohlcv[-1][0] + 1  # +1ms to avoid overlap
    return all_ohlcv[:max_candles]

def process_ohlcv(ohlcv):
    """Convert raw OHLCV data to a pandas DataFrame and process timestamps."""
    df = pd.DataFrame(ohlcv, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    return df

def get_top_symbols():
    try:
        url = 'https://coinmarketcap.com/'
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        symbols = []
        for row in soup.select('table tbody tr')[:25]:
            symbol_tag = row.select_one('td:nth-child(3) .coin-item-symbol')
            if symbol_tag:
                symbol = symbol_tag.text.strip()
                symbols.append(f'{symbol}/USDT')
        if symbols:
            return symbols
    except Exception as e:
        print(f'Error scraping CoinMarketCap: {e}')
    # Fallback
    return ['ETH/USDT', 'BTC/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']

def get_timeframe_ms(timeframe: str) -> int:
    """Return the number of milliseconds in one candle for a given timeframe string."""
    tf_map = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
    }
    return tf_map.get(timeframe, 60 * 1000)


def main():
    st.set_page_config(layout='wide', page_title='Crypto App')
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Overview', 'Backtesting', 'Strategy Optimizer'])

    # Shared controls
    symbols = get_top_symbols()
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    symbol = st.sidebar.selectbox('Select Trading Pair', symbols, index=0)
    timeframe = st.sidebar.selectbox('Select Timeframe', timeframes, index=2)
    # Fetch most recent data
    now_ms = int(time.time() * 1000)
    tf_ms = get_timeframe_ms(timeframe)
    since = now_ms - (tf_ms * 2500)
    ohlcv = fetch_ohlcv(symbol, timeframe, since, max_candles=2500)
    df = process_ohlcv(ohlcv)

    if page == 'Overview':
        st.title('Crypto Prices - Last 7 Days (Binance)')
        show_sma = st.checkbox('Show SMA (20)', value=False)
        show_ema = st.checkbox('Show EMA (20)', value=False)
        show_rsi = st.checkbox('Show RSI (14)', value=False)

        # Calculate indicators
        if show_sma:
            df['SMA20'] = df['close'].rolling(window=20).mean()
        if show_ema:
            df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
        if show_rsi:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI14'] = 100 - (100 / (1 + rs))

        st.write(df[['open_time', 'open', 'high', 'low', 'close', 'volume']])

        import plotly.graph_objects as go
        fig = go.Figure()
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['open_time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        ))
        # Volume as bar chart
        fig.add_trace(go.Bar(
            x=df['open_time'],
            y=df['volume'],
            name='Volume',
            marker_color='rgba(128,128,128,0.3)',
            yaxis='y2',
            opacity=0.3
        ))
        # Overlays for indicators
        if show_sma and 'SMA20' in df:
            fig.add_trace(go.Scatter(x=df['open_time'], y=df['SMA20'], mode='lines', name='SMA 20', line=dict(color='blue')))
        if show_ema and 'EMA20' in df:
            fig.add_trace(go.Scatter(x=df['open_time'], y=df['EMA20'], mode='lines', name='EMA 20', line=dict(color='orange')))
        if show_rsi and 'RSI14' in df:
            fig.add_trace(go.Scatter(x=df['open_time'], y=df['RSI14'], mode='lines', name='RSI 14', line=dict(color='purple', dash='dot')))
        fig.update_layout(
            title=f'{symbol} {timeframe} Candlestick Chart',
            xaxis_title='Time',
            yaxis_title='Price (USDT)',
            xaxis_rangeslider_visible=False,
            yaxis=dict(title='Price (USDT)', side='left'),
            yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False, rangemode='tozero'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # News feed
        NEWSAPI_KEY = os.environ.get('NEWSAPI_KEY')
        base_coin = symbol.split('/')[0] if '/' in symbol else symbol
        if not NEWSAPI_KEY:
            st.info('Set the NEWSAPI_KEY environment variable to see news headlines.\n\nIn PowerShell, run:  $env:NEWSAPI_KEY = "your_actual_key"\nThen restart Streamlit.')
        else:
            news_url = f'https://newsapi.org/v2/everything?q={base_coin}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWSAPI_KEY}'
            try:
                import requests
                response = requests.get(news_url)
                data = response.json()
                if data.get('status') == 'ok' and data.get('articles'):
                    for article in data['articles']:
                        st.markdown(f"**[{article['title']}]({article['url']})**  ")
                        st.caption(article['source']['name'] + ' | ' + article['publishedAt'][:10])
                        st.write(article['description'] or '')
                        st.write('---')
                else:
                    st.info('No news found or API limit reached.')
            except Exception as e:
                st.error(f'Error fetching news: {e}')

    elif page == 'Backtesting':
        st.title('Backtesting Strategies')
        # Date range selection at the top
        min_date = df['open_time'].min()
        max_date = df['open_time'].max()
        col1, col2 = st.columns(2)
        with col1:
            backtest_start = st.date_input('Backtest Start Date', min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())
        with col2:
            backtest_end = st.date_input('Backtest End Date', min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())
        df = df[(df['open_time'].dt.date >= backtest_start) & (df['open_time'].dt.date <= backtest_end)].copy()
        df = pd.DataFrame(df)  # Ensure columns are pandas Series, not numpy arrays
        if df.empty:
            st.warning('No data available for the selected date range.')
            return

        strategies = ['None', 'SMA Crossover', 'EMA Crossover', 'RSI Strategy', 'Mean Reversion', 'Momentum']
        strategy = st.selectbox('Backtest Strategy', strategies, index=0)
        if strategy == 'SMA Crossover':
            sma_fast = st.number_input('SMA Fast Window', min_value=2, max_value=100, value=10)
            sma_slow = st.number_input('SMA Slow Window', min_value=2, max_value=200, value=30)
        elif strategy == 'EMA Crossover':
            ema_fast = st.number_input('EMA Fast Window', min_value=2, max_value=100, value=10)
            ema_slow = st.number_input('EMA Slow Window', min_value=2, max_value=200, value=30)
        elif strategy == 'RSI Strategy':
            rsi_period = st.number_input('RSI Period', min_value=2, max_value=50, value=14)
            rsi_buy = st.number_input('RSI Buy Threshold', min_value=1, max_value=100, value=30)
            rsi_sell = st.number_input('RSI Sell Threshold', min_value=1, max_value=100, value=70)
        elif strategy == 'Mean Reversion':
            mr_window = st.number_input('Lookback Window', min_value=2, max_value=200, value=20)
            mr_entry = st.number_input('Entry Threshold (%)', min_value=1, max_value=20, value=5, help='Enter long if price is this % below mean')
            mr_exit = st.number_input('Exit Threshold (%)', min_value=0, max_value=20, value=0, help='Exit long if price returns to mean or above this %')
        elif strategy == 'Momentum':
            mom_window = st.number_input('Momentum Lookback (candles)', min_value=1, max_value=200, value=10)
            mom_entry = st.number_input('Entry Threshold (%)', min_value=1, max_value=50, value=3, help='Enter long if price rises this % over lookback')
            mom_exit = st.number_input('Exit Threshold (%)', min_value=1, max_value=50, value=2, help='Exit if price falls this % from entry')
            mom_max_hold = st.number_input('Max Hold (candles)', min_value=1, max_value=200, value=20, help='Exit after this many candles if not stopped')
        # Calculate indicators for backtest
        if strategy == 'SMA Crossover':
            df['sma_fast'] = df['close'].rolling(window=sma_fast).mean()
            df['sma_slow'] = df['close'].rolling(window=sma_slow).mean()
            df['signal'] = (df['sma_fast'] > df['sma_slow']).astype(int)
            df['signal'] = df['signal'].diff().fillna(0)
        elif strategy == 'EMA Crossover':
            df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
            df['signal'] = (df['ema_fast'] > df['ema_slow']).astype(int)
            df['signal'] = df['signal'].diff().fillna(0)
        elif strategy == 'RSI Strategy':
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            # Generate signals: 1 for buy, -1 for sell, 0 otherwise
            df['signal'] = 0
            df.loc[(df['RSI'] < rsi_buy) & (df['RSI'].shift(1) >= rsi_buy), 'signal'] = 1  # Cross below buy threshold
            df.loc[(df['RSI'] > rsi_sell) & (df['RSI'].shift(1) <= rsi_sell), 'signal'] = -1  # Cross above sell threshold
        elif strategy == 'Mean Reversion':
            df['mr_mean'] = df['close'].rolling(window=mr_window).mean()
            df['mr_pct'] = (df['close'] - df['mr_mean']) / df['mr_mean'] * 100
            df['signal'] = 0
            # Entry: price below mean by entry threshold
            df.loc[(df['mr_pct'] < -mr_entry) & (df['mr_pct'].shift(1) >= -mr_entry), 'signal'] = 1
            # Exit: price returns to mean or above exit threshold
            df.loc[(df['mr_pct'] > -mr_exit) & (df['mr_pct'].shift(1) <= -mr_exit), 'signal'] = -1
        elif strategy == 'Momentum':
            df['mom_return'] = (df['close'] / df['close'].shift(mom_window) - 1) * 100
            df['signal'] = 0
            # Entry: price up by entry threshold over lookback
            df.loc[(df['mom_return'] > mom_entry) & (df['mom_return'].shift(1) <= mom_entry), 'signal'] = 1
            # Exit: price falls by exit threshold from entry or max hold reached (handled in trade logic)
        else:
            df['signal'] = 0

        # Portfolio rules controls
        if strategy != 'None':
            st.subheader('Portfolio Rules')
            colp1, colp2, colp3 = st.columns(3)
            with colp1:
                risk_pct = st.slider('Risk Per Trade (%)', min_value=1, max_value=100, value=100, step=1, help='Percent of portfolio risked per trade')
            with colp2:
                compounding = st.radio('Compounding', ['Yes', 'No'], index=0, help='Reinvest profits (Yes) or use fixed capital (No)')
            with colp3:
                reinvest = st.radio('Reinvest Profits', ['Yes', 'No'], index=0, help='Alias for compounding')

            # --- Trades Table and Portfolio Simulation (shared logic) ---
            trades = []
            equity_curve = []
            position = None
            entry_idx = None
            capital = 1000.0  # Default for stats, will be replaced in simulation section
            current_equity = capital
            hold_count = 0
            for idx, row in df.iterrows():
                if strategy == 'Momentum':
                    if row['signal'] == 1 and position is None:
                        position = 'long'
                        entry_idx = idx
                        entry_equity = current_equity
                        entry_price = row['close']
                        hold_count = 0
                    elif position == 'long':
                        hold_count += 1
                        price_drop = (row['close'] - entry_price) / entry_price * 100
                        if price_drop <= -mom_exit or hold_count >= mom_max_hold or row['signal'] == -1:
                            entry_time = df.at[entry_idx, 'open_time']
                            exit_time = row['open_time']
                            exit_price = row['close']
                            trade_return = (exit_price - entry_price) / entry_price
                            duration = (exit_time - entry_time).total_seconds() / 3600
                            risked = entry_equity * (risk_pct / 100) if compounding == 'Yes' else capital * (risk_pct / 100)
                            not_risked = entry_equity - risked if compounding == 'Yes' else capital - risked
                            profit = risked * trade_return
                            current_equity = not_risked + risked + profit if reinvest == 'Yes' else current_equity + profit
                            equity_curve.append(current_equity)
                            trades.append({
                                'Entry Time': entry_time,
                                'Entry Price': entry_price,
                                'Exit Time': exit_time,
                                'Exit Price': exit_price,
                                'Return (%)': trade_return * 100,
                                'Duration (h)': duration,
                                'Profit': profit,
                                'Equity After': current_equity
                            })
                            position = None
                            entry_idx = None
                else:
                    if row['signal'] == 1 and position is None:
                        position = 'long'
                        entry_idx = idx
                        entry_equity = current_equity
                    elif row['signal'] == -1 and position == 'long':
                        entry_time = df.at[entry_idx, 'open_time']
                        entry_price = df.at[entry_idx, 'close']
                        exit_time = row['open_time']
                        exit_price = row['close']
                        trade_return = (exit_price - entry_price) / entry_price
                        duration = (exit_time - entry_time).total_seconds() / 3600
                        risked = entry_equity * (risk_pct / 100) if compounding == 'Yes' else capital * (risk_pct / 100)
                        not_risked = entry_equity - risked if compounding == 'Yes' else capital - risked
                        profit = risked * trade_return
                        current_equity = not_risked + risked + profit if reinvest == 'Yes' else current_equity + profit
                        equity_curve.append(current_equity)
                        trades.append({
                            'Entry Time': entry_time,
                            'Entry Price': entry_price,
                            'Exit Time': exit_time,
                            'Exit Price': exit_price,
                            'Return (%)': trade_return * 100,
                            'Duration (h)': duration,
                            'Profit': profit,
                            'Equity After': current_equity
                        })
                        position = None
                        entry_idx = None
            # If still in a position at the end, close at last available price/time
            if position == 'long' and entry_idx is not None:
                entry_time = df.at[entry_idx, 'open_time']
                entry_price = df.at[entry_idx, 'close']
                exit_time = df.iloc[-1]['open_time']
                exit_price = df.iloc[-1]['close']
                trade_return = (exit_price - entry_price) / entry_price
                duration = (exit_time - entry_time).total_seconds() / 3600
                risked = current_equity * (risk_pct / 100) if compounding == 'Yes' else capital * (risk_pct / 100)
                not_risked = current_equity - risked if compounding == 'Yes' else capital - risked
                profit = risked * trade_return
                current_equity = not_risked + risked + profit if reinvest == 'Yes' else current_equity + profit
                equity_curve.append(current_equity)
                if df.iloc[-1]['signal'] == -1:
                    trades.append({
                        'Entry Time': entry_time,
                        'Entry Price': entry_price,
                        'Exit Time': exit_time,
                        'Exit Price': exit_price,
                        'Return (%)': trade_return * 100,
                        'Duration (h)': duration,
                        'Profit': profit,
                        'Equity After': current_equity
                    })
            trades_df = pd.DataFrame(trades)
            # Calculate stats from equity_curve
            if equity_curve:
                total_return = (equity_curve[-1] / capital) - 1
                equity_curve_pd = pd.Series([capital] + equity_curve)
                running_max = equity_curve_pd.cummax()
                drawdown = (equity_curve_pd - running_max) / running_max
                max_drawdown = drawdown.min()
            else:
                total_return = 0
                max_drawdown = 0
            win_trades = (trades_df['Profit'] > 0).sum() if not trades_df.empty else 0
            loss_trades = (trades_df['Profit'] < 0).sum() if not trades_df.empty else 0
            win_rate = (trades_df['Profit'] > 0).mean() * 100 if not trades_df.empty else 0
            avg_trade_return = trades_df['Return (%)'].mean() if not trades_df.empty else 0
            num_trades = len(trades_df)
            sharpe = (trades_df['Profit'].mean() / trades_df['Profit'].std(ddof=0)) * (252 ** 0.5) if trades_df['Profit'].std(ddof=0) != 0 else float('nan')

            st.subheader('Backtest Results')
            st.write(f"Total Return: {total_return:.2%}")
            st.write(f"Winning Trades: {win_trades}")
            st.write(f"Losing Trades: {loss_trades}")
            st.write(f"Max Drawdown: {max_drawdown:.2%}")
            st.write(f"Sharpe Ratio: {sharpe:.2f}")
            st.write(f"Win Rate: {win_rate:.2f}%")
            st.write(f"Average Trade Return: {avg_trade_return:.2f}%")
            st.write(f"Number of Trades: {num_trades}")

            if not trades_df.empty:
                st.subheader('Trades Table')
                st.dataframe(trades_df, use_container_width=True)

        # Plot chart with signals
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
        if strategy == 'SMA Crossover':
            fig.add_trace(go.Scatter(x=df['open_time'], y=df['sma_fast'], mode='lines', name=f'SMA {sma_fast}', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['open_time'], y=df['sma_slow'], mode='lines', name=f'SMA {sma_slow}', line=dict(color='orange')))
        elif strategy == 'EMA Crossover':
            fig.add_trace(go.Scatter(x=df['open_time'], y=df['ema_fast'], mode='lines', name=f'EMA {ema_fast}', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['open_time'], y=df['ema_slow'], mode='lines', name=f'EMA {ema_slow}', line=dict(color='orange')))
        elif strategy == 'RSI Strategy':
            fig.add_trace(go.Scatter(x=df['open_time'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
            # Plot buy/sell markers for RSI signals
            buy_signals = df[df['signal'] == 1]
            sell_signals = df[df['signal'] == -1]
            fig.add_trace(go.Scatter(x=buy_signals['open_time'], y=buy_signals['close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy'))
            fig.add_trace(go.Scatter(x=sell_signals['open_time'], y=sell_signals['close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell'))
        elif strategy == 'Mean Reversion':
            fig.add_trace(go.Scatter(x=df['open_time'], y=df['mr_mean'], mode='lines', name=f'Mean ({mr_window})', line=dict(color='purple', dash='dot')))
            # Plot buy/sell markers for mean reversion signals
            buy_signals = df[df['signal'] == 1]
            sell_signals = df[df['signal'] == -1]
            fig.add_trace(go.Scatter(x=buy_signals['open_time'], y=buy_signals['close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy'))
            fig.add_trace(go.Scatter(x=sell_signals['open_time'], y=sell_signals['close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell'))
        # Plot buy/sell signals for all strategies except RSI and Mean Reversion (already handled above)
        if strategy not in ['None', 'RSI Strategy', 'Mean Reversion']:
            buy_signals = df[df['signal'] == 1]
            sell_signals = df[df['signal'] == -1]
            fig.add_trace(go.Scatter(x=buy_signals['open_time'], y=buy_signals['close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy'))
            fig.add_trace(go.Scatter(x=sell_signals['open_time'], y=sell_signals['close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell'))
        fig.update_layout(title=f'{symbol} {timeframe} Backtest Chart', xaxis_title='Time', yaxis_title='Price (USDT)', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- Simulation Section ---
        if strategy != 'None':
            st.subheader('Simulate Strategy Over Custom Period')
            start_capital = st.number_input('Starting Capital (USDT)', min_value=1.0, value=1000.0)
            # Use the same portfolio rules as above
            sim_df = df.copy()
            sim_equity = [float(start_capital)]
            position = None
            entry_price = None
            entry_equity = float(start_capital)
            for i, row in sim_df.iterrows():
                if row['signal'] == 1 and position is None:
                    position = 'long'
                    entry_price = float(row['close'])
                    entry_equity = sim_equity[-1]
                elif row['signal'] == -1 and position == 'long' and entry_price is not None:
                    trade_return = (float(row['close']) / entry_price) - 1
                    risked = entry_equity * (risk_pct / 100) if compounding == 'Yes' else float(start_capital) * (risk_pct / 100)
                    not_risked = entry_equity - risked if compounding == 'Yes' else float(start_capital) - risked
                    profit = risked * trade_return
                    new_equity = not_risked + risked + profit if reinvest == 'Yes' else sim_equity[-1] + profit
                    sim_equity.append(new_equity)
                    position = None
                    entry_price = None
                else:
                    sim_equity.append(sim_equity[-1])
            # Always close any open position at the last price
            if position == 'long' and entry_price is not None:
                trade_return = (float(sim_df.iloc[-1]['close']) / entry_price) - 1
                risked = sim_equity[-1] * (risk_pct / 100) if compounding == 'Yes' else float(start_capital) * (risk_pct / 100)
                not_risked = sim_equity[-1] - risked if compounding == 'Yes' else float(start_capital) - risked
                profit = risked * trade_return
                new_equity = not_risked + risked + profit if reinvest == 'Yes' else sim_equity[-1] + profit
                sim_equity.append(new_equity)
            sim_equity = [x for x in sim_equity if x is not None]
            # --- Buy & Hold and Cash Comparisons ---
            if not sim_df.empty:
                buy_hold_curve = [float(start_capital) * (sim_df.iloc[i]['close'] / sim_df.iloc[0]['close']) for i in range(len(sim_df))]
                buy_hold_curve.append(buy_hold_curve[-1])  # match length
                cash_curve = [float(start_capital)] * len(sim_equity)
            else:
                buy_hold_curve = [float(start_capital)] * len(sim_equity)
                cash_curve = [float(start_capital)] * len(sim_equity)
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=sim_equity, mode='lines', name='Strategy'))
            fig.add_trace(go.Scatter(y=buy_hold_curve, mode='lines', name='Buy & Hold'))
            fig.add_trace(go.Scatter(y=cash_curve, mode='lines', name='Cash'))
            fig.update_layout(title='Equity Curve Comparison', xaxis_title='Step', yaxis_title='Equity (USDT)')
            st.plotly_chart(fig, use_container_width=True)
            sim_total_return = (sim_equity[-1] / start_capital) - 1 if sim_equity else 0
            buy_hold_return = (buy_hold_curve[-1] / start_capital) - 1 if buy_hold_curve else 0
            st.write(f"Simulated Total Return (Strategy): {sim_total_return:.2%}")
            st.write(f"Buy & Hold Return: {buy_hold_return:.2%}")
            st.write(f"Final Capital (Strategy): {sim_equity[-1]:.2f} USDT")
            st.write(f"Final Capital (Buy & Hold): {buy_hold_curve[-1]:.2f} USDT")
            st.write(f"Final Capital (Cash): {cash_curve[-1]:.2f} USDT")

    elif page == 'Strategy Optimizer':
        st.title('Strategy Optimizer')
        st.write('Run a grid search over strategy parameters to find the best-performing set.')
        # Always load most recent data for selected symbol/timeframe
        now_ms = int(time.time() * 1000)
        tf_ms = get_timeframe_ms(timeframe)
        opt_since = now_ms - (tf_ms * 2500)
        opt_ohlcv = fetch_ohlcv(symbol, timeframe, since=opt_since, max_candles=2500)
        df_opt_full = process_ohlcv(opt_ohlcv)
        # Date range selection for optimizer
        all_dates = df_opt_full['open_time'].dt.date
        min_date = all_dates.min()
        max_date = all_dates.max()
        # --- Fix: Use session state to reset date pickers if symbol/timeframe changes ---
        if 'opt_start' not in st.session_state or st.session_state.get('last_symbol') != symbol or st.session_state.get('last_timeframe') != timeframe:
            st.session_state['opt_start'] = min_date
            st.session_state['opt_end'] = max_date
            st.session_state['last_symbol'] = symbol
            st.session_state['last_timeframe'] = timeframe
        col1, col2 = st.columns(2)
        with col1:
            opt_start = st.date_input('Optimizer Start Date', min_value=min_date, max_value=max_date, value=st.session_state['opt_start'], key='opt_start')
        with col2:
            opt_end = st.date_input('Optimizer End Date', min_value=min_date, max_value=max_date, value=st.session_state['opt_end'], key='opt_end')
        df_opt_range = df_opt_full[(df_opt_full['open_time'].dt.date >= opt_start) & (df_opt_full['open_time'].dt.date <= opt_end)].copy()
        if df_opt_range.empty:
            st.warning('No data available for the selected date range.')
            return
        # Select strategy
        opt_strategies = ['SMA Crossover', 'EMA Crossover', 'RSI Strategy']
        opt_strategy = st.selectbox('Optimize Strategy', opt_strategies)
        # Parameter grid selection
        if opt_strategy == 'SMA Crossover':
            sma_fast_range = st.slider('SMA Fast Window Range', 2, 50, (5, 15))
            sma_slow_range = st.slider('SMA Slow Window Range', 10, 100, (20, 50))
            fast_vals = list(range(sma_fast_range[0], sma_fast_range[1]+1))
            slow_vals = list(range(sma_slow_range[0], sma_slow_range[1]+1))
            param_grid = [(f, s) for f in fast_vals for s in slow_vals if f < s]
        elif opt_strategy == 'EMA Crossover':
            ema_fast_range = st.slider('EMA Fast Window Range', 2, 50, (5, 15))
            ema_slow_range = st.slider('EMA Slow Window Range', 10, 100, (20, 50))
            fast_vals = list(range(ema_fast_range[0], ema_fast_range[1]+1))
            slow_vals = list(range(ema_slow_range[0], ema_slow_range[1]+1))
            param_grid = [(f, s) for f in fast_vals for s in slow_vals if f < s]
        elif opt_strategy == 'RSI Strategy':
            rsi_period_range = st.slider('RSI Period Range', 2, 30, (7, 21))
            rsi_buy_range = st.slider('RSI Buy Threshold Range', 5, 50, (20, 35))
            rsi_sell_range = st.slider('RSI Sell Threshold Range', 50, 90, (65, 80))
            period_vals = list(range(rsi_period_range[0], rsi_period_range[1]+1))
            buy_vals = list(range(rsi_buy_range[0], rsi_buy_range[1]+1))
            sell_vals = list(range(rsi_sell_range[0], rsi_sell_range[1]+1))
            param_grid = [(p, b, s) for p in period_vals for b in buy_vals for s in sell_vals if b < s]
        # Run grid search
        st.write('Running grid search...')
        results = []
        for params in param_grid:
            df_opt = df_opt_range.copy()
            if opt_strategy == 'SMA Crossover':
                f, s = params
                df_opt['sma_fast'] = df_opt['close'].rolling(window=f).mean()
                df_opt['sma_slow'] = df_opt['close'].rolling(window=s).mean()
                df_opt['signal'] = (df_opt['sma_fast'] > df_opt['sma_slow']).astype(int)
                df_opt['signal'] = df_opt['signal'].diff().fillna(0)
            elif opt_strategy == 'EMA Crossover':
                f, s = params
                df_opt['ema_fast'] = df_opt['close'].ewm(span=f, adjust=False).mean()
                df_opt['ema_slow'] = df_opt['close'].ewm(span=s, adjust=False).mean()
                df_opt['signal'] = (df_opt['ema_fast'] > df_opt['ema_slow']).astype(int)
                df_opt['signal'] = df_opt['signal'].diff().fillna(0)
            elif opt_strategy == 'RSI Strategy':
                p, b, s = params
                delta = df_opt['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=p).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
                rs = gain / loss
                df_opt['RSI'] = 100 - (100 / (1 + rs))
                df_opt['signal'] = 0
                df_opt.loc[(df_opt['RSI'] < b) & (df_opt['RSI'].shift(1) >= b), 'signal'] = 1
                df_opt.loc[(df_opt['RSI'] > s) & (df_opt['RSI'].shift(1) <= s), 'signal'] = -1
            # Simple backtest: buy on 1, sell on -1, full compounding
            capital = 1000.0
            equity = capital
            position = None
            entry_price = None
            for i, row in df_opt.iterrows():
                if row['signal'] == 1 and position is None:
                    position = 'long'
                    entry_price = row['close']
                elif row['signal'] == -1 and position == 'long' and entry_price is not None:
                    trade_return = (row['close'] / entry_price) - 1
                    equity = equity * (1 + trade_return)
                    position = None
                    entry_price = None
            # If open at end, close at last price
            if position == 'long' and entry_price is not None:
                trade_return = (df_opt.iloc[-1]['close'] / entry_price) - 1
                equity = equity * (1 + trade_return)
            total_return = (equity / capital) - 1
            results.append((params, total_return))
        # Show results as table or heatmap
        st.write('Results Table (Top 20):')
        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
        st.dataframe(pd.DataFrame([
            {'Params': str(r[0]), 'Total Return (%)': f'{r[1]*100:.2f}'} for r in results_sorted[:20]
        ]))
        # Show best equity curve
        best_params = results_sorted[0][0]
        st.write(f'Best Parameters: {best_params}')
        # Re-run best equity curve
        df_best = df_opt_range.copy()
        if opt_strategy == 'SMA Crossover':
            f, s = best_params
            df_best['sma_fast'] = df_best['close'].rolling(window=f).mean()
            df_best['sma_slow'] = df_best['close'].rolling(window=s).mean()
            df_best['signal'] = (df_best['sma_fast'] > df_best['sma_slow']).astype(int)
            df_best['signal'] = df_best['signal'].diff().fillna(0)
        elif opt_strategy == 'EMA Crossover':
            f, s = best_params
            df_best['ema_fast'] = df_best['close'].ewm(span=f, adjust=False).mean()
            df_best['ema_slow'] = df_best['close'].ewm(span=s, adjust=False).mean()
            df_best['signal'] = (df_best['ema_fast'] > df_best['ema_slow']).astype(int)
            df_best['signal'] = df_best['signal'].diff().fillna(0)
        elif opt_strategy == 'RSI Strategy':
            p, b, s = best_params
            delta = df_best['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
            rs = gain / loss
            df_best['RSI'] = 100 - (100 / (1 + rs))
            df_best['signal'] = 0
            df_best.loc[(df_best['RSI'] < b) & (df_best['RSI'].shift(1) >= b), 'signal'] = 1
            df_best.loc[(df_best['RSI'] > s) & (df_best['RSI'].shift(1) <= s), 'signal'] = -1
        # Equity curve for best
        capital = 1000.0
        equity_curve = [capital]
        position = None
        entry_price = None
        for i, row in df_best.iterrows():
            if row['signal'] == 1 and position is None:
                position = 'long'
                entry_price = row['close']
            elif row['signal'] == -1 and position == 'long' and entry_price is not None:
                trade_return = (row['close'] / entry_price) - 1
                equity_curve.append(equity_curve[-1] * (1 + trade_return))
                position = None
                entry_price = None
            else:
                equity_curve.append(equity_curve[-1])
        if position == 'long' and entry_price is not None:
            trade_return = (df_best.iloc[-1]['close'] / entry_price) - 1
            equity_curve.append(equity_curve[-1] * (1 + trade_return))
        import plotly.graph_objects as go
        fig = go.Figure()
        # Overlay equity and price (secondary y-axis)
        fig.add_trace(go.Scatter(y=equity_curve, mode='lines', name='Best Equity Curve', yaxis='y1'))
        fig.add_trace(go.Scatter(y=df_best['close'].values, mode='lines', name='Price', yaxis='y2', line=dict(color='gray', dash='dot')))
        fig.update_layout(
            title='Best Parameter Equity Curve (with Price)',
            xaxis_title='Step',
            yaxis=dict(title='Equity (USDT)', side='left'),
            yaxis2=dict(title='Price', overlaying='y', side='right', showgrid=False),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()