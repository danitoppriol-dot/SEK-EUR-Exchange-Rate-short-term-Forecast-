"""
EUR/SEK Professional Forecasting System
Optimized for short-term tactical FX decisions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

START_DATE = "2018-01-01"  # 7 anni di dati (più bilanciato)
END_DATE = datetime.now().strftime("%Y-%m-%d")
FORECAST_DAYS = 30  # Previsione massima 30 giorni
TRAIN_TEST_SPLIT = 0.85
ENSEMBLE_MODELS = 3  # Numero di modelli da combinare

print(" Downloading market data...")

def safe_download(ticker, name):
    """Download con gestione errori"""
    try:
        print(f"  Downloading {name}...")
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if len(data) > 0 and 'Close' in data.columns:
            return data['Close']
        else:
            print(f"    No data for {name}, skipping...")
            return pd.Series(dtype=float)
    except Exception as e:
        print(f"    Error downloading {name}: {e}")
        return pd.Series(dtype=float)

# Download dati
eur_sek = safe_download("EURSEK=X", "EUR/SEK")
eur_usd = safe_download("EURUSD=X", "EUR/USD")
usd_sek = safe_download("USDSEK=X", "USD/SEK")
dxy = safe_download("DX-Y.NYB", "Dollar Index")
vix = safe_download("^VIX", "VIX")
stoxx50 = safe_download("^STOXX50E", "STOXX 50")
omxs30 = safe_download("^OMX", "OMX Stockholm")
brent = safe_download("BZ=F", "Brent Oil")
copper = safe_download("HG=F", "Copper")

if len(eur_sek) == 0:
    print("\n ERRORE CRITICO: Impossibile scaricare EUR/SEK")
    print("Possibili cause:")
    print("  1. Problema di connessione internet")
    print("  2. Yahoo Finance temporaneamente non disponibile")
    print("  3. Ticker cambiato (prova 'SEK=X' invece)")
    print("\nRiprova tra qualche minuto o controlla la tua connessione.")
    exit(1)

print(f"\n EUR/SEK data loaded: {len(eur_sek)} observations")

data = pd.DataFrame(index=eur_sek.index)
data['eur_sek'] = eur_sek

if len(eur_usd) > 0:
    data['eur_usd'] = eur_usd
else:
    data['eur_usd'] = 1.1  # Valore fisso come fallback
    
if len(usd_sek) > 0:
    data['usd_sek'] = usd_sek
else:
    data['usd_sek'] = data['eur_sek'] / data['eur_usd']  

if len(dxy) > 0:
    data['dxy'] = dxy
else:
    data['dxy'] = 100 

if len(vix) > 0:
    data['vix'] = vix
else:
    data['vix'] = 20  

if len(stoxx50) > 0:
    data['stoxx50'] = stoxx50
else:
    data['stoxx50'] = 4000 

if len(omxs30) > 0:
    data['omxs30'] = omxs30
else:
    data['omxs30'] = 2000  

if len(brent) > 0:
    data['brent'] = brent
else:
    data['brent'] = 80  

if len(copper) > 0:
    data['copper'] = copper
else:
    data['copper'] = 4

# Riempi valori mancanti
data = data.ffill().bfill()

print(f"Total features loaded: {data.shape[1]} columns, {len(data)} rows")


print(" Engineering features...")

df = data.copy()

for col in df.columns:
    df[f'{col}_ret_1d'] = df[col].pct_change(1)
    df[f'{col}_ret_5d'] = df[col].pct_change(5)
    df[f'{col}_ret_10d'] = df[col].pct_change(10)

for window in [5, 10, 20, 50]:
    df[f'eur_sek_ma{window}'] = df['eur_sek'].rolling(window).mean()
    df[f'eur_sek_ma{window}_dist'] = (df['eur_sek'] - df[f'eur_sek_ma{window}']) / df[f'eur_sek_ma{window}']

for window in [5, 10, 20]:
    df[f'eur_sek_vol{window}'] = df['eur_sek_ret_1d'].rolling(window).std()
    df[f'vix_ma{window}'] = df['vix'].rolling(window).mean()

df['eur_stoxx_corr'] = df['eur_sek_ret_1d'].rolling(20).corr(df['stoxx50_ret_1d'])
df['sek_omx_corr'] = df['eur_sek_ret_1d'].rolling(20).corr(df['omxs30_ret_1d'])

df['eur_sek_rsi'] = 100 - (100 / (1 + df['eur_sek_ret_1d'].rolling(14).apply(lambda x: x[x>0].mean() / abs(x[x<0].mean()) if x[x<0].mean() != 0 else 1, raw=False)))
df['eur_sek_momentum'] = df['eur_sek'] - df['eur_sek'].shift(10)

df['equity_spread'] = (df['stoxx50_ret_5d'] - df['omxs30_ret_5d']) * 100

df['eur_sek_carry_proxy'] = df['eur_sek'].rolling(20).mean() / df['eur_sek'].rolling(50).mean()

df['vol_regime'] = (df['eur_sek_vol20'] > df['eur_sek_vol20'].rolling(50).mean()).astype(int)

for horizon in [1, 5, 10]:
    df[f'target_{horizon}d'] = df['eur_sek'].shift(-horizon)

df = df.replace([np.inf, -np.inf], np.nan).dropna()

print(f" Features created: {df.shape[1]} columns, {len(df)} rows")

train_size = int(len(df) * TRAIN_TEST_SPLIT)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

exclude_cols = ['eur_sek', 'eur_usd', 'usd_sek', 'dxy', 'vix', 'stoxx50', 'omxs30', 'brent', 'copper'] + \
               [col for col in df.columns if 'target' in col]
feature_cols = [col for col in df.columns if col not in exclude_cols]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

y_train = train_df['target_5d']
y_test = test_df['target_5d']

print(f"📈 Training set: {len(X_train)} | Test set: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(" Training ensemble models...")

models = {
    'XGBoost': xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        random_state=42,
        verbose=0
    )
}

predictions = {}
for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train_scaled, y_train)
    predictions[name] = model.predict(X_test_scaled)

weights = {'XGBoost': 0.5, 'GradientBoosting': 0.3, 'RandomForest': 0.2}
ensemble_pred = sum(predictions[name] * weights[name] for name in models.keys())

mae = np.mean(np.abs(ensemble_pred - y_test))
rmse = np.sqrt(np.mean((ensemble_pred - y_test)**2))
mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100

actual_direction = np.sign(y_test - test_df['eur_sek'].values)
pred_direction = np.sign(ensemble_pred - test_df['eur_sek'].values)
directional_acc = np.mean(actual_direction == pred_direction) * 100

print("\n" + "="*70)
print(" BACKTEST PERFORMANCE (5-day ahead predictions)")
print("="*70)
print(f"MAE (Mean Absolute Error):    {mae:.4f} SEK")
print(f"RMSE:                          {rmse:.4f} SEK")
print(f"MAPE:                          {mape:.2f}%")
print(f"Directional Accuracy:          {directional_acc:.1f}% ")
print("="*70)

print("\n Generating extended forecast until April 2026...")

FORECAST_END_DATE = "2026-04-30"
forecast_end = pd.to_datetime(FORECAST_END_DATE)
last_date = df.index[-1]

all_forecast_dates = pd.date_range(last_date + timedelta(days=1), end=forecast_end, freq='B')
total_forecast_days = len(all_forecast_dates)

print(f"  Forecasting {total_forecast_days} business days ({FORECAST_DAYS} short-term + {total_forecast_days-FORECAST_DAYS} long-term)")

last_known_price = df['eur_sek'].iloc[-1]
last_features = df[feature_cols].iloc[-1:].values
last_features_scaled = scaler.transform(last_features)

forecast_prices_short = []  # Primi 30 giorni (più affidabile)
forecast_prices_long = []   # Oltre 30 giorni (meno affidabile)

current_features = last_features_scaled.copy()

for i in range(total_forecast_days):
    preds = {name: model.predict(current_features)[0] for name, model in models.items()}
    
    forecast_price = sum(preds[name] * weights[name] for name in models.keys())
    
    if i < FORECAST_DAYS:
        forecast_prices_short.append(forecast_price)
    else:
        forecast_prices_long.append(forecast_price)
    
    decay = 0.95 if i < FORECAST_DAYS else 0.99  # Meno aggiornamenti nel lungo periodo
    current_features = current_features * decay

forecast_df_short = pd.DataFrame({
    'date': all_forecast_dates[:FORECAST_DAYS],
    'eur_sek_forecast': forecast_prices_short
}).set_index('date')

forecast_df_long = pd.DataFrame({
    'date': all_forecast_dates[FORECAST_DAYS:],
    'eur_sek_forecast': forecast_prices_long
}).set_index('date')

# DataFrame completo
forecast_df_full = pd.DataFrame({
    'date': all_forecast_dates,
    'eur_sek_forecast': forecast_prices_short + forecast_prices_long
}).set_index('date')

forecast_df = forecast_df_short

pred_std = np.std([predictions[name] for name in models.keys()], axis=0).mean()
forecast_df['lower_bound'] = forecast_df['eur_sek_forecast'] - 1.96 * pred_std
forecast_df['upper_bound'] = forecast_df['eur_sek_forecast'] + 1.96 * pred_std

print("\n" + "="*70)
print(" TACTICAL ANALYSIS - RENT PAYMENT OPTIMIZATION")
print("="*70)
print(f"Current EUR/SEK Rate:          {last_known_price:.4f}")
print(f"Your Rent:                     7300 SEK")
print(f"Current Cost in EUR:           {7300/last_known_price:.2f} EUR")
print(f"\nTarget Rate (Best case):       11.1700 (Cost: {7300/11.17:.2f} EUR)")
print(f"Acceptable Rate:               11.0500 (Cost: {7300/11.05:.2f} EUR)")
print(f"Poor Rate (avoid):             10.9000 (Cost: {7300/10.90:.2f} EUR)")
print("="*70)

# Trova il giorno ottimale
best_day = forecast_df['eur_sek_forecast'].idxmax()
best_rate = forecast_df['eur_sek_forecast'].max()
best_cost = 7300 / best_rate

# Giorni sopra threshold
threshold = 11.05
good_days = forecast_df[forecast_df['eur_sek_forecast'] >= threshold]

print(f"\n OPTIMAL PAYMENT DAY (According to forecast):")
print(f"   Date:                       {best_day.strftime('%Y-%m-%d (%A)')}")
print(f"   Forecasted Rate:            {best_rate:.4f}")
print(f"   Expected Cost:              {best_cost:.2f} EUR")
print(f"   Savings vs Current:         {(7300/last_known_price - best_cost):.2f} EUR")

if len(good_days) > 0:
    print(f"\n Days above {threshold} threshold: {len(good_days)}")
    print("\nTop 5 favorable days:")
    top_days = forecast_df.nlargest(5, 'eur_sek_forecast')[['eur_sek_forecast', 'lower_bound', 'upper_bound']]
    top_days['cost_eur'] = 7300 / top_days['eur_sek_forecast']
    print(top_days.to_string())
else:
    print(f"\n  WARNING: No days forecasted above {threshold} in next 30 days")
    print("   Recommendation: Pay early or wait for better market conditions")


fig1 = plt.figure(figsize=(16, 8))
ax1 = fig1.add_subplot(111)

ax1.plot(df.index, df['eur_sek'], 
         label='Historical Data', color='steelblue', linewidth=1.5, alpha=0.8)

ax1.plot(test_df.index, ensemble_pred, 
         label='Backtest (Validation)', color='orange', linewidth=2, alpha=0.7)

ax1.plot(forecast_df_full.index, forecast_df_full['eur_sek_forecast'], 
         label=f'Full Forecast → {FORECAST_END_DATE}', 
         color='red', linewidth=2.5, linestyle='-', alpha=0.9)

ax1.axvline(forecast_df_short.index[-1], color='purple', linestyle=':', 
            linewidth=2, label='30-Day Reliability Threshold', alpha=0.6)

ax1.axhline(11.17, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Target: 11.17')
ax1.axhline(11.05, color='yellowgreen', linestyle='--', alpha=0.5, linewidth=1.5, label='Acceptable: 11.05')
ax1.axhline(10.90, color='salmon', linestyle='--', alpha=0.5, linewidth=1.5, label='Poor: 10.90')
ax1.axhline(last_known_price, color='black', linestyle=':', linewidth=2, 
            alpha=0.7, label=f'Current: {last_known_price:.4f}')

ax1.set_title(f'EUR/SEK Complete Forecast: Now → April 2026\n⚠️ Uncertainty increases significantly beyond 30 days', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Date', fontsize=13, fontweight='bold')
ax1.set_ylabel('EUR/SEK Exchange Rate', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')

final_forecast = forecast_df_full['eur_sek_forecast'].iloc[-1]
ax1.annotate(f'Apr 2026\nForecast: {final_forecast:.4f}', 
             xy=(forecast_df_full.index[-1], final_forecast),
             xytext=(-80, 30), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2),
             fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

fig2 = plt.figure(figsize=(16, 9))

ax2a = fig2.add_subplot(211)
ax2a.plot(forecast_df_full.index, forecast_df_full['eur_sek_forecast'], 
          color='crimson', linewidth=3, marker='o', markersize=3, 
          label='Forecasted EUR/SEK')

ax2a.fill_between(forecast_df_short.index, 
                   forecast_df_short['eur_sek_forecast'].min() - 0.1,
                   forecast_df_short['eur_sek_forecast'].max() + 0.1,
                   color='green', alpha=0.1, label='High Confidence (30 days)')

ax2a.axhline(11.17, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax2a.axhline(11.05, color='yellowgreen', linestyle='--', linewidth=1.5, alpha=0.6)
ax2a.axhline(10.90, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
ax2a.axhline(last_known_price, color='black', linestyle=':', linewidth=2, alpha=0.8)

ax2a.scatter(best_day, best_rate, color='gold', s=300, zorder=10, 
             edgecolors='black', linewidths=3, marker='*',
             label=f'⭐ Optimal Day: {best_day.strftime("%Y-%m-%d")}')

ax2a.set_title(f'Detailed Forecast Period: {forecast_df_full.index[0].strftime("%Y-%m-%d")} → {FORECAST_END_DATE}', 
               fontsize=14, fontweight='bold')
ax2a.set_ylabel('EUR/SEK Rate', fontsize=12, fontweight='bold')
ax2a.legend(loc='best', fontsize=10)
ax2a.grid(True, alpha=0.3)

ax2b = fig2.add_subplot(212)
forecast_costs_full = 7300 / forecast_df_full['eur_sek_forecast']

ax2b.plot(forecast_df_full.index, forecast_costs_full, 
          color='darkgreen', linewidth=3, marker='s', markersize=4,
          label='Rent Cost (7300 SEK → EUR)')

favorable_mask = forecast_df_full['eur_sek_forecast'] >= 11.05
ax2b.fill_between(forecast_df_full.index, 
                   forecast_costs_full.min() - 5,
                   forecast_costs_full.max() + 5,
                   where=favorable_mask,
                   color='lightgreen', alpha=0.3, label='Favorable Rates (≥11.05)')

ax2b.axhline(7300/last_known_price, color='black', linestyle=':', 
             linewidth=2, label=f'Current Cost: {7300/last_known_price:.2f} EUR')
ax2b.axhline(7300/11.17, color='green', linestyle='--', 
             linewidth=2, alpha=0.7, label='Best Case: 653.45 EUR')
ax2b.axhline(7300/10.90, color='red', linestyle='--', 
             linewidth=2, alpha=0.7, label='Worst Case: 669.72 EUR')

ax2b.scatter(best_day, best_cost, color='gold', s=300, zorder=10,
             edgecolors='black', linewidths=3, marker='*',
             label=f'⭐ Minimum Cost: {best_cost:.2f} EUR')

ax2b.set_title('Rent Payment Cost Evolution (Lower is Better)', fontsize=12, fontweight='bold')
ax2b.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2b.set_ylabel('Cost in EUR', fontsize=12, fontweight='bold')
ax2b.legend(loc='best', fontsize=10)
ax2b.grid(True, alpha=0.3)
ax2b.invert_yaxis()  # Inverti per mostrare "meno = meglio"

plt.tight_layout()
plt.show()

fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(14, 10))

ax3a.plot(df.index[-100:], df['eur_sek'].iloc[-100:], 
         label='Historical (Last 100 days)', color='steelblue', linewidth=2)
ax3a.plot(test_df.index, ensemble_pred, 
         label='Backtest Prediction', color='orange', linewidth=1.5, alpha=0.7)
ax3a.plot(forecast_df.index, forecast_df['eur_sek_forecast'], 
         label='30-Day Forecast (Tactical)', color='red', linewidth=2.5, marker='o', markersize=5)
ax3a.fill_between(forecast_df.index, 
                  forecast_df['lower_bound'], 
                  forecast_df['upper_bound'],
                  alpha=0.2, color='red', label='Confidence Interval (95%)')
ax3a.axhline(11.05, color='green', linestyle='--', alpha=0.6, label='Target Rate (11.05)')
ax3a.axhline(last_known_price, color='gray', linestyle=':', alpha=0.6, label=f'Current ({last_known_price:.4f})')
ax3a.scatter(best_day, best_rate, color='gold', s=250, zorder=5, 
            edgecolors='black', linewidths=2.5, marker='*', label=f'⭐ PAY HERE: {best_day.strftime("%b %d")}')
ax3a.set_title(' TACTICAL DECISION: When to Pay Your Rent (Next 30 Days)', 
               fontsize=14, fontweight='bold')
ax3a.set_ylabel('EUR/SEK Rate', fontsize=12)
ax3a.legend(loc='best', fontsize=10)
ax3a.grid(True, alpha=0.3)

# Panel 2: Cost Analysis
forecast_costs = 7300 / forecast_df['eur_sek_forecast']
ax3b.plot(forecast_df.index, forecast_costs, color='darkgreen', 
          linewidth=2.5, marker='s', markersize=6, label='Daily Rent Cost')
ax3b.axhline(7300/last_known_price, color='gray', linestyle=':', linewidth=2, 
             label=f'Current Cost ({7300/last_known_price:.2f} EUR)')
ax3b.axhline(7300/11.17, color='green', linestyle='--', alpha=0.6, 
             label='Best Case (653.45 EUR)')
ax3b.scatter(best_day, best_cost, color='gold', s=250, zorder=5, 
            edgecolors='black', linewidths=2.5, marker='*',
            label=f' Optimal: {best_cost:.2f} EUR (Save {(7300/last_known_price - best_cost):.2f} EUR)')
ax3b.set_title(' Rent Cost in EUR (7300 SEK) - Lower is Better!', 
               fontsize=12, fontweight='bold')
ax3b.set_xlabel('Date', fontsize=12)
ax3b.set_ylabel('Cost (EUR)', fontsize=12)
ax3b.legend(loc='best', fontsize=10)
ax3b.grid(True, alpha=0.3)
ax3b.invert_yaxis()

plt.tight_layout()
plt.show()

print("\n Generated 3 comprehensive charts:")
print("     Full Historical + Extended Forecast (Now → April 2026)")
print("     Detailed Forecast Period Only (Focus on future movements)")
print("     Tactical Decision Chart (Next 30 days - MOST RELIABLE)")
print("\n  Remember: Charts 1-2 show extended forecast for visualization only.")
print("   The 30-day tactical chart (#3) is the most actionable for your decision!\n")

print("\n" + "="*70)
print("  IMPORTANT DISCLAIMERS")
print("="*70)
print(f"• FX forecasting is inherently uncertain - model accuracy is ~{directional_acc:.1f}%")
print("• Actual rates depend on: ECB/Riksbank decisions, macro data, geopolitics")
print("• Use this as ONE input for decisions, not the only factor")
print("• Consider using LIMIT ORDERS on your payment platform")
print("• Monitor real-time rates daily near optimal window")
print("• Factor in platform fees (Wise typically better than banks)")
print("="*70)

print("\n✅ Analysis complete! Review charts and table above for decision support.")
