import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Генерація синтетичних даних для демонстрації
np.random.seed(0)
df = pd.DataFrame({
    'Дата': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Ціна': np.random.normal(0, 1, 100).cumsum() + np.sin(np.linspace(0, 10, 100)) * 10
})
df.set_index('Дата', inplace=True)

# Аналіз тренду та сезонності
decomposition = seasonal_decompose(df['Ціна'], model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(df['Ціна'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Налаштування та прогнозування моделі ARIMA
model_arima = ARIMA(df['Ціна'], order=(5, 1, 0))
model_fit_arima = model_arima.fit()
forecast_arima = model_fit_arima.forecast(steps=6)
print('Прогноз ARIMA:', forecast_arima)

# Налаштування та прогнозування моделі SARIMA
model_sarima = SARIMAX(df['Ціна'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
results_sarima = model_sarima.fit()
df['forecast_sarima'] = results_sarima.predict(start=len(df)-10, end=len(df)+5, dynamic=True)
df[['Ціна', 'forecast_sarima']].plot()
plt.show()

# Перевірка стаціонарності
result = adfuller(df['Ціна'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Векторні авторегресійні моделі (VAR)
df['Інший_часовий_ряд'] = np.random.normal(0, 1, 100).cumsum() + np.cos(np.linspace(0, 10, 100)) * 5
model_var = VAR(df[['Ціна', 'Інший_часовий_ряд']])
results_var = model_var.fit()
print(results_var.summary())

# Машинне навчання
# Додавання фіч: лаги попередніх значень ціни та тренду
df['trend'] = np.arange(len(df))
for lag in range(1, 4):
    df[f'lag_{lag}'] = df['Ціна'].shift(lag)

# Видалення рядків з NaN (через лаги)
df.dropna(inplace=True)

# Підготовка даних
X = df[['trend'] + [f'lag_{lag}' for lag in range(1, 4)]].values  # Використання тренду і лагів як фіч
y = df['Ціна'].values

# Нормалізація даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Розділення на тренувальні та тестові набори
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Створення та тренування моделі
model_rf = RandomForestRegressor(n_estimators=200)
model_rf.fit(X_train, y_train)

# Прогнозування
predictions_rf = model_rf.predict(X_test)

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(df.index[len(df) - len(y_test):], y_test, label='Actual')
plt.plot(df.index[len(df) - len(y_test):], predictions_rf, label='Predicted')
plt.legend()
plt.show()
