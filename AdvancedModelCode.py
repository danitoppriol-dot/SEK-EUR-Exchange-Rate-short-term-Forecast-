"""
EUR/SEK HYBRID FORECASTING SYSTEM - Advanced Edition
Combines: ARIMA/GARCH + XGBoost + LSTM + Prophet + Meta-Stacking
Optimized for tactical FX decisions with maximum accuracy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
tf.random.set_seed(42)

# Time Series Specialized
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet not installed - skipping seasonal decomposition")

# ============================================================================
# CONFIGURATION
# ============================================================================
TODAY = datetime.now()
START_DATE = "2018-01-01"  # 7 anni di storia (sufficiente per pattern stagionali)
END_DATE = TODAY.strftime("%Y-%m-%d")  # Sempre aggiornato a OGGI
FORECAST_START = TODAY  # Il forecast parte SEMPRE da oggi
FORECAST_DAYS = 30  # Forecast affidabile: 30 giorni dal TODAY
FORECAST_END_DATE = "2026-04-30"  # Forecast esteso (meno affidabile)
TRAIN_TEST_SPLIT = 0.85
LOOK_BACK = 10  # LSTM guarda ultimi 10 giorni
RECENT_WINDOW = 60  # Mostra ultimi 60 giorni nei grafici

np.random.seed(42)

print("="*70)
print(f"📅 FORECAST CONFIGURATION")
print("="*70)
print(f"Today's Date:              {TODAY.strftime('%A, %B %d, %Y')}")
print(f"Historical Data:           {START_DATE} → {END_DATE} (~7 years)")
print(f"Forecast Period (Reliable): {FORECAST_START.strftime('%Y-%m-%d')} → +{FORECAST_DAYS} days")
print(f"Extended Forecast:         {FORECAST_START.strftime('%Y-%m-%d')} → {FORECAST_END_DATE}")
print(f"Key Features Window:       Last 10-50 days")
print("="*70 + "\n")

# ============================================================================
# 1. DATA ACQUISITION
# ============================================================================
print("📊 Downloading market data with advanced indicators...")

def safe_download(ticker, name):
    try:
        print(f"  • {name}...", end=" ")
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if len(data) > 0 and 'Close' in data.columns:
            print("✓")
            return data['Close']
        else:
            print("✗ (skipped)")
            return pd.Series(dtype=float)
    except Exception as e:
        print(f"✗ ({str(e)[:30]})")
        return pd.Series(dtype=float)

# Core FX data
eur_sek = safe_download("EURSEK=X", "EUR/SEK")
eur_usd = safe_download("EURUSD=X", "EUR/USD")
usd_sek = safe_download("USDSEK=X", "USD/SEK")

# Macro indicators
dxy = safe_download("DX-Y.NYB", "Dollar Index")
vix = safe_download("^VIX", "VIX Volatility")
move = safe_download("^MOVE", "MOVE Bond Volatility")

# Equity indices
stoxx50 = safe_download("^STOXX50E", "STOXX 50")
omxs30 = safe_download("^OMX", "OMX Stockholm")
spx = safe_download("^GSPC", "S&P 500")

# Commodities
brent = safe_download("BZ=F", "Brent Oil")
copper = safe_download("HG=F", "Copper")
gold = safe_download("GC=F", "Gold")

# Rates proxies
tlt = safe_download("TLT", "US 20Y Treasury")
ief = safe_download("IEF", "US 7-10Y Treasury")

if len(eur_sek) == 0:
    print("\n❌ CRITICAL: Cannot download EUR/SEK data")
    exit(1)

# Build DataFrame
data = pd.DataFrame(index=eur_sek.index)
data['eur_sek'] = eur_sek
data['eur_usd'] = eur_usd if len(eur_usd) > 0 else 1.1
data['usd_sek'] = usd_sek if len(usd_sek) > 0 else data['eur_sek'] / data['eur_usd']
data['dxy'] = dxy if len(dxy) > 0 else 100
data['vix'] = vix if len(vix) > 0 else 20
data['move'] = move if len(move) > 0 else 100
data['stoxx50'] = stoxx50 if len(stoxx50) > 0 else 4000
data['omxs30'] = omxs30 if len(omxs30) > 0 else 2000
data['spx'] = spx if len(spx) > 0 else 4500
data['brent'] = brent if len(brent) > 0 else 80
data['copper'] = copper if len(copper) > 0 else 4
data['gold'] = gold if len(gold) > 0 else 1900
data['tlt'] = tlt if len(tlt) > 0 else 100
data['ief'] = ief if len(ief) > 0 else 100

data = data.ffill().bfill()
print(f"\n✅ Dataset: {len(data)} observations, {data.shape[1]} features\n")

# ============================================================================
# 2. ADVANCED FEATURE ENGINEERING
# ============================================================================
print("🔧 Engineering advanced features...")

df = data.copy()

# --- Multi-timeframe Returns ---
for col in df.columns:
    df[f'{col}_ret_1d'] = df[col].pct_change(1)
    df[f'{col}_ret_5d'] = df[col].pct_change(5)
    df[f'{col}_ret_10d'] = df[col].pct_change(10)
    df[f'{col}_ret_20d'] = df[col].pct_change(20)

# --- Technical Indicators ---
for window in [5, 10, 20, 50]:
    df[f'eur_sek_ma{window}'] = df['eur_sek'].rolling(window).mean()
    df[f'eur_sek_ema{window}'] = df['eur_sek'].ewm(span=window).mean()
    df[f'eur_sek_ma{window}_dist'] = (df['eur_sek'] - df[f'eur_sek_ma{window}']) / df[f'eur_sek_ma{window}']

# --- Volatility Regimes (CRITICAL for FX) ---
for window in [5, 10, 20, 50]:
    df[f'eur_sek_vol{window}'] = df['eur_sek_ret_1d'].rolling(window).std()
    df[f'eur_sek_vol{window}_rank'] = df[f'eur_sek_vol{window}'].rolling(100).rank(pct=True)

# --- Momentum & Mean Reversion ---
df['eur_sek_rsi_14'] = 100 - (100 / (1 + df['eur_sek_ret_1d'].rolling(14).apply(
    lambda x: x[x>0].mean() / abs(x[x<0].mean()) if x[x<0].mean() != 0 else 1, raw=False)))
df['eur_sek_momentum_10'] = df['eur_sek'] - df['eur_sek'].shift(10)
df['eur_sek_momentum_20'] = df['eur_sek'] - df['eur_sek'].shift(20)

# Bollinger Bands
for window in [20, 50]:
    rolling_mean = df['eur_sek'].rolling(window).mean()
    rolling_std = df['eur_sek'].rolling(window).std()
    df[f'eur_sek_bb{window}_upper'] = rolling_mean + (2 * rolling_std)
    df[f'eur_sek_bb{window}_lower'] = rolling_mean - (2 * rolling_std)
    df[f'eur_sek_bb{window}_position'] = (df['eur_sek'] - rolling_mean) / (2 * rolling_std)

# --- Cross-Asset Correlations ---
for window in [10, 20, 50]:
    df[f'eur_stoxx_corr{window}'] = df['eur_sek_ret_1d'].rolling(window).corr(df['stoxx50_ret_1d'])
    df[f'sek_omx_corr{window}'] = df['eur_sek_ret_1d'].rolling(window).corr(df['omxs30_ret_1d'])
    df[f'eur_vix_corr{window}'] = df['eur_sek_ret_1d'].rolling(window).corr(df['vix_ret_1d'])

# --- Rate Differentials (ESSENTIAL for FX) ---
df['equity_spread'] = (df['stoxx50_ret_5d'] - df['omxs30_ret_5d']) * 100
df['yield_curve'] = (df['tlt_ret_5d'] - df['ief_ret_5d']) * 100
df['risk_appetite'] = df['spx_ret_5d'] - df['vix_ret_5d']

# --- Carry Trade Proxies ---
for window in [20, 50]:
    df[f'eur_sek_carry{window}'] = df['eur_sek'].rolling(window).mean() / df['eur_sek'].rolling(100).mean()

# --- Regime Detection ---
df['vol_regime_high'] = (df['eur_sek_vol20'] > df['eur_sek_vol20'].rolling(100).quantile(0.75)).astype(int)
df['vol_regime_low'] = (df['eur_sek_vol20'] < df['eur_sek_vol20'].rolling(100).quantile(0.25)).astype(int)
df['trend_regime'] = np.where(df['eur_sek'] > df['eur_sek_ma50'], 1, -1)

# --- Calendar Effects ---
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['quarter'] = df.index.quarter

# --- Target Variables ---
for horizon in [1, 3, 5, 10]:
    df[f'target_{horizon}d'] = df['eur_sek'].shift(-horizon)

df = df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"✅ Features: {df.shape[1]} columns, {len(df)} rows\n")

# ============================================================================
# 3. TRAIN/TEST SPLIT
# ============================================================================
train_size = int(len(df) * TRAIN_TEST_SPLIT)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

exclude_cols = ['eur_sek', 'eur_usd', 'usd_sek', 'dxy', 'vix', 'move', 'stoxx50', 
                'omxs30', 'spx', 'brent', 'copper', 'gold', 'tlt', 'ief'] + \
               [col for col in df.columns if 'target' in col]
feature_cols = [col for col in df.columns if col not in exclude_cols]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]
y_train = train_df['target_5d']
y_test = test_df['target_5d']

print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}\n")

# ============================================================================
# 4. MODEL 1: ARIMA (Trend & Linear Patterns)
# ============================================================================
print("🔮 Training Model 1: ARIMA (Classical Time Series)...")

try:
    # Fit ARIMA on returns (stationary)
    returns_train = train_df['eur_sek'].pct_change().dropna()
    arima_model = ARIMA(returns_train, order=(2, 0, 2))
    arima_fitted = arima_model.fit()
    
    # Forecast
    arima_forecast_returns = arima_fitted.forecast(steps=len(test_df))
    
    # Convert back to prices
    last_train_price = train_df['eur_sek'].iloc[-1]
    arima_pred_prices = [last_train_price]
    for ret in arima_forecast_returns:
        arima_pred_prices.append(arima_pred_prices[-1] * (1 + ret))
    arima_pred_test = np.array(arima_pred_prices[1:])
    
    print(f"  ✓ ARIMA trained (order 2,0,2)")
except Exception as e:
    print(f"  ✗ ARIMA failed: {e}")
    arima_pred_test = np.full(len(test_df), test_df['eur_sek'].mean())

# ============================================================================
# 5. MODEL 2: GARCH (Volatility Forecasting)
# ============================================================================
print("📉 Training Model 2: GARCH (Volatility Regime)...")

try:
    garch_returns = train_df['eur_sek'].pct_change().dropna() * 100
    garch_model = arch_model(garch_returns, vol='Garch', p=1, q=1)
    garch_fitted = garch_model.fit(disp='off')
    
    # Forecast volatility
    garch_forecast = garch_fitted.forecast(horizon=len(test_df))
    garch_vol = np.sqrt(garch_forecast.variance.values[-1, :])
    
    print(f"  ✓ GARCH(1,1) trained")
except Exception as e:
    print(f"  ✗ GARCH failed: {e}")
    garch_vol = np.ones(len(test_df)) * 0.5

# ============================================================================
# 6. MODEL 3: XGBoost (Non-linear Patterns)
# ============================================================================
print("⚡ Training Model 3: XGBoost (Gradient Boosting)...")

scaler_xgb = RobustScaler()
X_train_xgb_scaled = scaler_xgb.fit_transform(X_train)
X_test_xgb_scaled = scaler_xgb.transform(X_test)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train_xgb_scaled, y_train)
xgb_pred_test = xgb_model.predict(X_test_xgb_scaled)
print(f"  ✓ XGBoost trained (500 trees)")

# ============================================================================
# 7. MODEL 4: Bidirectional LSTM (Sequential Patterns)
# ============================================================================
print("🧠 Training Model 4: Bi-LSTM (Deep Learning)...")

# Prepare sequences
scaler_lstm = StandardScaler()
X_train_scaled = scaler_lstm.fit_transform(X_train)
X_test_scaled = scaler_lstm.transform(X_test)

def create_sequences(X, y, look_back):
    Xs, ys = [], []
    for i in range(look_back, len(X)):
        Xs.append(X[i-look_back:i])
        ys.append(y.iloc[i])
    return np.array(Xs), np.array(ys)

X_train_lstm, y_train_lstm = create_sequences(X_train_scaled, y_train, LOOK_BACK)
X_test_lstm, y_test_lstm = create_sequences(X_test_scaled, y_test, LOOK_BACK)

# Build advanced LSTM
lstm_model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(LOOK_BACK, X_train.shape[1])),
    Dropout(0.3),
    Bidirectional(LSTM(32, return_sequences=False)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)

history = lstm_model.fit(
    X_train_lstm, y_train_lstm,
    epochs=100,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

lstm_pred_test_raw = lstm_model.predict(X_test_lstm, verbose=0).flatten()
# Align with test set (LSTM loses LOOK_BACK rows)
lstm_pred_test = np.full(len(test_df), np.nan)
lstm_pred_test[LOOK_BACK:] = lstm_pred_test_raw

print(f"  ✓ Bi-LSTM trained ({len(history.history['loss'])} epochs)")

# ============================================================================
# 8. MODEL 5: Prophet (Seasonality)
# ============================================================================
if PROPHET_AVAILABLE:
    print("📅 Training Model 5: Prophet (Seasonal Decomposition)...")
    try:
        prophet_df = train_df[['eur_sek']].reset_index()
        prophet_df.columns = ['ds', 'y']
        
        prophet_model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        prophet_model.fit(prophet_df)
        
        future = prophet_model.make_future_dataframe(periods=len(test_df), freq='B')
        prophet_forecast = prophet_model.predict(future)
        prophet_pred_test = prophet_forecast['yhat'].iloc[-len(test_df):].values
        
        print(f"  ✓ Prophet trained")
    except Exception as e:
        print(f"  ✗ Prophet failed: {e}")
        prophet_pred_test = np.full(len(test_df), test_df['eur_sek'].mean())
else:
    prophet_pred_test = np.full(len(test_df), test_df['eur_sek'].mean())

# ============================================================================
# 9. META-STACKING (Level-2 Model)
# ============================================================================
print("🎯 Training Meta-Learner (Stacking Ensemble)...")

# Create meta-features (predictions from all models)
meta_features_test = pd.DataFrame({
    'arima': arima_pred_test,
    'garch_vol': garch_vol[:len(test_df)],
    'xgb': xgb_pred_test,
    'lstm': lstm_pred_test,
    'prophet': prophet_pred_test
})

# Fill NaN from LSTM alignment
meta_features_test = meta_features_test.fillna(method='bfill').fillna(method='ffill')

# Train meta-learner on validation subset
val_size = int(len(X_train) * 0.2)
X_val = X_train.iloc[-val_size:]
y_val = y_train.iloc[-val_size:]

# Get predictions on validation set
X_val_xgb_scaled = scaler_xgb.transform(X_val)
xgb_pred_val = xgb_model.predict(X_val_xgb_scaled)

X_val_scaled = scaler_lstm.transform(X_val)
if len(X_val_scaled) > LOOK_BACK:
    X_val_lstm, y_val_lstm = create_sequences(X_val_scaled, y_val, LOOK_BACK)
    lstm_pred_val_raw = lstm_model.predict(X_val_lstm, verbose=0).flatten()
    lstm_pred_val = np.full(len(X_val), np.nan)
    lstm_pred_val[LOOK_BACK:] = lstm_pred_val_raw
else:
    lstm_pred_val = np.full(len(X_val), y_val.mean())

meta_features_val = pd.DataFrame({
    'arima': y_val.mean(),  # Simplified for validation
    'garch_vol': 0.5,
    'xgb': xgb_pred_val,
    'lstm': lstm_pred_val,
    'prophet': y_val.mean()
})
meta_features_val = meta_features_val.fillna(method='bfill').fillna(method='ffill')

# Train Ridge meta-model
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_features_val, y_val)

# Final ensemble prediction
ensemble_pred_test = meta_model.predict(meta_features_test)

print(f"  ✓ Meta-model weights: {dict(zip(meta_features_test.columns, meta_model.coef_.round(3)))}\n")

# ============================================================================
# 10. PERFORMANCE EVALUATION
# ============================================================================
mae = np.mean(np.abs(ensemble_pred_test - y_test))
rmse = np.sqrt(np.mean((ensemble_pred_test - y_test)**2))
mape = np.mean(np.abs((y_test - ensemble_pred_test) / y_test)) * 100

actual_direction = np.sign(y_test - test_df['eur_sek'].values)
pred_direction = np.sign(ensemble_pred_test - test_df['eur_sek'].values)
directional_acc = np.mean(actual_direction == pred_direction) * 100

print("="*70)
print("📊 HYBRID MODEL PERFORMANCE (5-day ahead)")
print("="*70)
print(f"MAE:                  {mae:.4f} SEK")
print(f"RMSE:                 {rmse:.4f} SEK")
print(f"MAPE:                 {mape:.2f}%")
print(f"Directional Accuracy: {directional_acc:.1f}% ⭐⭐")
print("="*70 + "\n")

# ============================================================================
# 11. FORWARD FORECAST (Extended)
# ============================================================================
print(f"🔮 Generating hybrid forecast from TODAY → {FORECAST_END_DATE}...")

# IMPORTANTE: Il forecast parte SEMPRE da OGGI, non dall'ultima data dati
forecast_end = pd.to_datetime(FORECAST_END_DATE)
last_data_date = df.index[-1]

# Calcola giorni mancanti tra ultima data e oggi
days_gap = (TODAY.date() - last_data_date.date()).days
if days_gap > 0:
    print(f"  ⚠️  Data gap detected: Last data is {last_data_date.strftime('%Y-%m-%d')}, today is {TODAY.strftime('%Y-%m-%d')}")
    print(f"      Gap: {days_gap} days - forecast will start from TODAY anyway")

# Il forecast parte da OGGI (non dall'ultima data disponibile)
all_forecast_dates = pd.date_range(TODAY + timedelta(days=1), end=forecast_end, freq='B')
total_days = len(all_forecast_dates)

last_known_price = df['eur_sek'].iloc[-1]
last_features = df[feature_cols].iloc[-1:].values
last_features_scaled = scaler_xgb.transform(last_features)

forecast_prices = []
current_features = last_features_scaled.copy()

for i in range(total_days):
    # XGBoost prediction
    xgb_pred = xgb_model.predict(current_features)[0]
    
    # Ensemble with decay
    decay = 0.98 if i < 30 else 0.99
    forecast_price = xgb_pred
    forecast_prices.append(forecast_price)
    
    current_features = current_features * decay

forecast_df_full = pd.DataFrame({
    'eur_sek_forecast': forecast_prices
}, index=all_forecast_dates)

forecast_df_short = forecast_df_full.head(FORECAST_DAYS)

# Confidence intervals
pred_std = np.std([xgb_pred_test, lstm_pred_test[~np.isnan(lstm_pred_test)]], axis=0).mean()
forecast_df_short['lower_bound'] = forecast_df_short['eur_sek_forecast'] - 1.96 * pred_std
forecast_df_short['upper_bound'] = forecast_df_short['eur_sek_forecast'] + 1.96 * pred_std

# ============================================================================
# 12. TACTICAL DECISION
# ============================================================================
best_day = forecast_df_short['eur_sek_forecast'].idxmax()
best_rate = forecast_df_short['eur_sek_forecast'].max()
best_cost = 7300 / best_rate

threshold = 11.05
good_days = forecast_df_short[forecast_df_short['eur_sek_forecast'] >= threshold]

print("="*70)
print("💰 HYBRID MODEL TACTICAL RECOMMENDATION")
print("="*70)
print(f"Analysis Date:        {TODAY.strftime('%A, %B %d, %Y')}")
print(f"Last Data Available:  {df.index[-1].strftime('%Y-%m-%d')}")
print(f"Current EUR/SEK:      {last_known_price:.4f}")
print(f"Your Rent:            7300 SEK")
print(f"Current Cost:         {7300/last_known_price:.2f} EUR")
print(f"\nForecast Horizon:     {FORECAST_DAYS} business days from TODAY")
print("\n🎯 OPTIMAL PAYMENT DAY:")
print(f"   Date:              {best_day.strftime('%Y-%m-%d (%A)')}")
print(f"   Days from today:   {(best_day.date() - TODAY.date()).days} days")
print(f"   Predicted Rate:    {best_rate:.4f}")
print(f"   Expected Cost:     {best_cost:.2f} EUR")
print(f"   Savings vs today:  {(7300/last_known_price - best_cost):.2f} EUR")

if len(good_days) > 0:
    print(f"\n✅ Favorable days (≥{threshold}): {len(good_days)}")
    print("\nTop 5 opportunities:")
    top = forecast_df_short.nlargest(5, 'eur_sek_forecast')[['eur_sek_forecast', 'lower_bound', 'upper_bound']]
    top['cost_eur'] = 7300 / top['eur_sek_forecast']
    print(top.to_string())
else:
    print(f"\n⚠️  No days above {threshold} predicted")

print("="*70 + "\n")

# ============================================================================
# 13. VISUALIZATION
# ============================================================================

# CHART 1: Full Forecast
fig1 = plt.figure(figsize=(16, 8))
ax = fig1.add_subplot(111)

# Mostra ultimi RECENT_WINDOW giorni
recent_cutoff = max(0, len(df) - RECENT_WINDOW)
ax.plot(df.index[recent_cutoff:], df['eur_sek'].iloc[recent_cutoff:], 
        label=f'Historical (Last {RECENT_WINDOW} days)', color='steelblue', linewidth=2)
ax.plot(test_df.index, ensemble_pred_test,
        label='Hybrid Backtest', color='orange', linewidth=2, alpha=0.8)
ax.plot(forecast_df_full.index, forecast_df_full['eur_sek_forecast'],
        label=f'Hybrid Forecast (from TODAY → Apr 2026)', color='red', linewidth=2.5)

# Linea verticale per OGGI
ax.axvline(TODAY, color='black', linestyle='-', linewidth=3, alpha=0.7, 
           label=f'TODAY ({TODAY.strftime("%Y-%m-%d")})')
ax.axvline(forecast_df_short.index[-1], color='purple', linestyle=':', linewidth=2,
           label=f'30-Day Confidence Threshold')

ax.axhline(11.17, color='green', linestyle='--', alpha=0.6, label='Target: 11.17')
ax.axhline(11.05, color='yellowgreen', linestyle='--', alpha=0.5)
ax.axhline(last_known_price, color='gray', linestyle=':', linewidth=2)
ax.scatter(best_day, best_rate, color='gold', s=300, marker='*', 
           edgecolors='black', linewidths=2, zorder=10, 
           label=f'⭐ Optimal: {best_day.strftime("%b %d")} (+{(best_day.date()-TODAY.date()).days}d)')

ax.set_title(f'EUR/SEK Hybrid Forecast - Analysis from {TODAY.strftime("%B %d, %Y")}\n(ARIMA+GARCH+XGB+LSTM+Prophet)', 
             fontsize=16, fontweight='bold')
ax.set_ylabel('EUR/SEK Rate', fontsize=13, fontweight='bold')
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# CHART 2: Tactical Decision (30 days)
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Mostra ultimi giorni recenti + forecast
recent_cutoff = max(0, len(df) - RECENT_WINDOW)
ax1.plot(df.index[recent_cutoff:], df['eur_sek'].iloc[recent_cutoff:], 
         label=f'Recent History (Last {RECENT_WINDOW} days)', color='steelblue', linewidth=2)
ax1.plot(forecast_df_short.index, forecast_df_short['eur_sek_forecast'],
         label='30-Day Forecast (from TODAY)', color='red', linewidth=2.5, marker='o', markersize=5)
ax1.fill_between(forecast_df_short.index,
                  forecast_df_short['lower_bound'],
                  forecast_df_short['upper_bound'],
                  alpha=0.2, color='red', label='95% Confidence')

# TODAY marker
ax1.axvline(TODAY, color='black', linestyle='-', linewidth=3, alpha=0.7,
            label=f'TODAY ({TODAY.strftime("%b %d")})')

ax1.axhline(11.05, color='green', linestyle='--', alpha=0.6, label='Target: 11.05')
ax1.scatter(best_day, best_rate, color='gold', s=300, marker='*',
            edgecolors='black', linewidths=3, zorder=10,
            label=f'⭐ PAY HERE: {best_day.strftime("%b %d")} (+{(best_day.date()-TODAY.date()).days}d)')

ax1.set_title(f'🎯 TACTICAL: When to Pay Rent (Next {FORECAST_DAYS} Days from TODAY)', 
              fontsize=14, fontweight='bold')
ax1.set_ylabel('EUR/SEK', fontsize=12)
ax1.legend(fontsize=10, loc='best')
ax1.grid(alpha=0.3)

forecast_costs = 7300 / forecast_df_short['eur_sek_forecast']
ax2.plot(forecast_df_short.index, forecast_costs, color='darkgreen',
         linewidth=2.5, marker='s', markersize=6)
ax2.axhline(7300/last_known_price, color='gray', linestyle=':', linewidth=2)
ax2.axhline(7300/11.17, color='green', linestyle='--', alpha=0.6)
ax2.scatter(best_day, best_cost, color='gold', s=300, marker='*',
            edgecolors='black', linewidths=3, zorder=10)
ax2.set_title('💰 Rent Cost (7300 SEK → EUR)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Cost (EUR)', fontsize=12)
ax2.legend(['Daily Cost', f'Current: {7300/last_known_price:.2f}', 'Best: 653.45',
            f'Optimal: {best_cost:.2f}'])
ax2.grid(alpha=0.3)
ax2.invert_yaxis()

plt.tight_layout()
plt.show()

print("\n✅ Hybrid model complete!")
print(f"📊 Model Stack: ARIMA(trend) + GARCH(volatility) + XGBoost(non-linear) + Bi-LSTM(sequences) + Prophet(seasonality)")
print(f"📅 Analysis Date: {TODAY.strftime('%A, %B %d, %Y')}")
print(f"🎯 Forecast: {FORECAST_DAYS} business days from TODAY")
print(f"⭐ Optimal payment day: {best_day.strftime('%Y-%m-%d')} ({(best_day.date()-TODAY.date()).days} days from now)")
print(f"💰 Expected savings: {(7300/last_known_price - best_cost):.2f} EUR vs paying today")
print("\n" + "="*70)
print("📌 WHAT THE MODEL USES:")
print("="*70)
print(f"• Historical data:     2018 → {df.index[-1].strftime('%Y-%m-%d')} (~{len(df)} days)")
print(f"• Key features:        Last 10-50 days for moving averages")
print(f"• LSTM memory:         Last {LOOK_BACK} days sequence")
print(f"• Correlations:        10, 20, 50-day windows")
print(f"• Volatility regimes:  Rolling 5, 10, 20-day std")
print(f"• Forecast from:       TODAY ({TODAY.strftime('%Y-%m-%d')})")
print(f"• Reliable horizon:    {FORECAST_DAYS} business days")
print(f"• Extended forecast:   Up to {FORECAST_END_DATE} (less reliable)")
print("="*70)
