# Focused Optuna RF tuning script
import sys, subprocess, json, os
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import joblib

OUT_DIR = Path(r"a:/Software Projects/Delhi-AQI-Model/results")
OUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(r"a:/Software Projects/Delhi-AQI-Model/data")

try:
    import optuna
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"], stdout=subprocess.DEVNULL)
    import optuna

# Load merged dataset
merged_path = DATA_DIR / 'merged_aqi_fire_wind.csv'
if not merged_path.exists():
    raise FileNotFoundError(f"Missing merged dataset at {merged_path}")
merged = pd.read_csv(merged_path, parse_dates=['date']).sort_values('date').reset_index(drop=True)
merged = merged[(merged['date']>=pd.to_datetime('2021-01-01')) & (merged['date']<=pd.to_datetime('2024-12-31'))].copy()
merged = merged.fillna(0)

features = [c for c in ['PM2.5','PM10','NO2','SO2','CO','Ozone','weighted_frp_fw','weighted_frp_aligned','mean_dist_km','min_dist_km','mean_wind_kmh','std_wind_kmh','mean_wind_dir_cos','mean_wind_dir_sin','total_frp_lag1','fire_count_lag1','mean_wind_lag1','weighted_frp_lag1','fire_wind_alignment01'] if c in merged.columns]
train = merged[merged['date'] < pd.to_datetime('2024-01-01')]
test = merged[merged['date'] >= pd.to_datetime('2024-01-01')]
X_train = train[features].values; X_test = test[features].values
y_train = train['AQI'].values; y_test = test['AQI'].values

tscv = TimeSeriesSplit(n_splits=5)

def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 40),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 12),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6),
        'max_features': trial.suggest_categorical('max_features', ['sqrt','log2', 0.5, 0.7]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }
    cv_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        Xtr, Xval = X_train[train_idx], X_train[val_idx]
        ytr, yval = y_train[train_idx], y_train[val_idx]
        model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        model.fit(Xtr, ytr)
        pred = model.predict(Xval)
        cv_scores.append(np.sqrt(mean_squared_error(yval, pred)))
    return np.mean(cv_scores)

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    N_TRIALS = int(os.environ.get('OPTUNA_RF_TRIALS', 60))
    study.optimize(rf_objective, n_trials=N_TRIALS)

    print('RF Optuna best params:', study.best_params, 'best_value (RMSE):', study.best_value)

    best_rf = RandomForestRegressor(random_state=42, n_jobs=-1, **study.best_params)
    best_rf.fit(X_train, y_train)
    pred_test = best_rf.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    print('Test RMSE for best RF:', test_rmse)

    trials_df = study.trials_dataframe(attrs=('number','value','params','state'))
    trials_df.to_csv(OUT_DIR / 'optuna_rf_trials.csv', index=False)
    with open(OUT_DIR / 'rf_best_params.json','w') as fh:
        json.dump(study.best_params, fh, indent=2)
    joblib.dump(best_rf, OUT_DIR / 'best_rf_optuna.pkl')

    print('Saved trials to results/optuna_rf_trials.csv and model to results/best_rf_optuna.pkl')
