import streamlit as st
import ccxt
import pandas as pd
import ta
import requests
import plotly.graph_objects as go
import joblib

st.set_page_config(layout="wide")
st.title("🤖 Scalping Signal + Machine Learning (30min)")

symbol = st.text_input("Trading-Paar (z.B. BTC/USDT)", value="BTC/USDT")
exchange = ccxt.bitget()

# Telegram-Funktion
def send_telegram_message(message):
    token = st.secrets["telegram"]["token"]
    chat_id = st.secrets["telegram"]["chat_id"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram Fehler:", e)

# Daten abrufen
def get_data(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='30m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['time','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

try:
    df = get_data(symbol)

    # Indikatoren
    df['ema5'] = ta.trend.EMAIndicator(df['close'], 5).ema_indicator()
    df['ema10'] = ta.trend.EMAIndicator(df['close'], 10).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    stoch = ta.momentum.StochRSIIndicator(df['close'])
    df['stoch_k'] = stoch.stochrsi_k()
    df['stoch_d'] = stoch.stochrsi_d()
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    macd = ta.trend.MACD(df['close'])
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['volume_avg'] = df['volume'].rolling(20).mean()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Candle Patterns
    bullish_engulfing = (
        prev['close'] < prev['open'] and
        latest['close'] > latest['open'] and
        latest['close'] > prev['open'] and
        latest['open'] < prev['close']
    )
    bearish_engulfing = (
        prev['close'] > prev['open'] and
        latest['close'] < latest['open'] and
        latest['close'] < prev['open'] and
        latest['open'] > prev['close']
    )

    # Bedingungen LONG / SHORT
    long_conditions = {
        'EMA5 > EMA10': latest['ema5'] > latest['ema10'],
        'RSI 45-60': 45 < latest['rsi'] < 60,
        'Stoch %K > %D': latest['stoch_k'] > latest['stoch_d'],
        'Close > BB Lower': latest['close'] > latest['bb_lower'],
        'MACD > Signal': latest['macd_line'] > latest['macd_signal'],
        'Volume > Avg': latest['volume'] > latest['volume_avg'],
        'ADX > 20': latest['adx'] > 20,
        'Bullish Engulfing': bullish_engulfing
    }

    short_conditions = {
        'EMA5 < EMA10': latest['ema5'] < latest['ema10'],
        'RSI 40-55': 40 < latest['rsi'] < 55,
        'Stoch %K < %D': latest['stoch_k'] < latest['stoch_d'],
        'Close < BB Upper': latest['close'] < latest['bb_upper'],
        'MACD < Signal': latest['macd_line'] < latest['macd_signal'],
        'Volume > Avg': latest['volume'] > latest['volume_avg'],
        'ADX > 20': latest['adx'] > 20,
        'Bearish Engulfing': bearish_engulfing
    }

    long_score = sum(long_conditions.values())
    short_score = sum(short_conditions.values())

    # ML-Modell laden
    model = joblib.load("trade_model.pkl")

    features = pd.DataFrame([{
        'ema5': latest['ema5'],
        'ema10': latest['ema10'],
        'rsi': latest['rsi'],
        'stoch_k': latest['stoch_k'],
        'stoch_d': latest['stoch_d'],
        'macd_line': latest['macd_line'],
        'macd_signal': latest['macd_signal'],
        'adx': latest['adx'],
        'volume': latest['volume'],
        'volume_avg': latest['volume_avg'],
        'atr': latest['atr']
    }])

    prediction = model.predict(features)[0]

    st.subheader("🔍 Klassischer Score")
    st.write(f"✅ LONG Score: {long_score}/8")
    st.write(f"🔻 SHORT Score: {short_score}/8")

    st.subheader("🤖 ML-Modell-Vorhersage")
    if prediction == 1:
        st.success("📈 ML sagt: LONG")
        ml_signal = "LONG"
    elif prediction == -1:
        st.error("📉 ML sagt: SHORT")
        ml_signal = "SHORT"
    else:
        st.warning("⏸ ML sagt: NEUTRAL")
        ml_signal = "NONE"

    # Trade-Ziele
    entry = latest['close']
    tp = entry + 2 * latest['atr']
    sl = entry - 1.5 * latest['atr']
    rr = (tp - entry) / (entry - sl)

    st.subheader("🎯 Trade-Ziele")
    st.write(f"Entry: {entry:.2f}, TP: {tp:.2f}, SL: {sl:.2f}, RRR: {rr:.2f}")

    # Telegram
    if 'last_signal' not in st.session_state:
        st.session_state.last_signal = None

    signal_msg = None
    if ml_signal != "NONE" and ml_signal != st.session_state.last_signal:
        signal_msg = f"🤖 {ml_signal} von ML für {symbol}\nEntry: {entry:.2f} | TP: {tp:.2f} | SL: {sl:.2f}"
        st.session_state.last_signal = ml_signal

    if signal_msg:
        send_telegram_message(signal_msg)

    # Anzeige Indikator-Tabelle
    st.subheader("📋 Indikator-Werte")
    st.dataframe(pd.DataFrame({
        'Wert': {
            'EMA5': latest['ema5'],
            'EMA10': latest['ema10'],
            'RSI': latest['rsi'],
            'Stoch K': latest['stoch_k'],
            'Stoch D': latest['stoch_d'],
            'MACD': latest['macd_line'],
            'MACD Signal': latest['macd_signal'],
            'ADX': latest['adx'],
            'Volume': latest['volume'],
            'Vol Avg': latest['volume_avg'],
            'ATR': latest['atr'],
            'Close': latest['close']
        }
    }).T.round(2))

    st.subheader("🧠 Score Details (LONG)")
    st.dataframe(pd.DataFrame(long_conditions, index=["Erfüllt"]).T)

    st.subheader("🧠 Score Details (SHORT)")
    st.dataframe(pd.DataFrame(short_conditions, index=["Erfüllt"]).T)

    # Chart
    st.subheader("📊 30-Minuten Chart")
    chart_df = df.tail(100)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_df['time'],
        open=chart_df['open'],
        high=chart_df['high'],
        low=chart_df['low'],
        close=chart_df['close'],
        name='Candles'))
    fig.add_trace(go.Scatter(x=chart_df['time'], y=chart_df['ema5'], name='EMA5', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=chart_df['time'], y=chart_df['ema10'], name='EMA10', line=dict(color='orange')))
    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("❌ Fehler aufgetreten:")
    st.text(str(e))
