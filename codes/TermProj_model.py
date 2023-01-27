import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
import sys
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import sklearn.metrics as matrics

sys.path.append('../TStoolbox.py')
from TStoolbox import AutoCorr_plot, Cal_AutoCorr, ACF_PACF_Plot, AutoCorr_subplot, GPAC_table, forecast_average_method, \
    forecast_naive_method, forecast_drift_method, forecast_simple_exponential_method, differencing, convert_differencing

lags = 50
interval=24
# interval = 8760
# ====================== data loading ===================== #
df_train_raw = pd.read_csv('train.csv', index_col=[0], parse_dates=True)
df_train = df_train_raw.iloc[interval+1:]
df_test = pd.read_csv('test.csv', index_col=[0], parse_dates=True)

df_train.index.freq = 'H'
df_test.index.freq = 'H'
# ====================== feature selection ===================== #
# ============ standardization
from sklearn.preprocessing import StandardScaler

col = ['humidity', 'pressure', 'weather', 'wind_direction', 'wind_speed']
x_train = df_train.filter(col)
x_test = df_test.filter(col)
y_train = df_train['original']
y_test = df_test['f_temp']
y_train_sta = df_train['differenced']
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train, y_train)
X_test = scaler.transform(x_test)

X_train = pd.DataFrame(X_train, columns=x_train.columns, index=x_train.index)
X_test = pd.DataFrame(X_test, columns=x_test.columns, index=x_test.index)

# ============ PCA
from TStoolbox import draw_PCA

draw_PCA(X_train)

# ============ SVD
x_t_x = (X_train.T @ X_train)
v, s, d = LA.svd(x_t_x)
print('singular values=', s)
print('*' * 50)
# ============ Condition Number
print('condition number=', LA.cond(X_train))
print('*' * 50)
# ============ backward selection
from TStoolbox import backwards_selection, vif_selection

back_table, back_model = backwards_selection(X_train, y_train, threshold=0.05)
print(back_table)
print('*' * 50)
# ============ vif selection
vif_table, vif_model = vif_selection(X_train, y_train, threshold=3)
print(vif_table)
print('*' * 50)
# ====================== Simple Modeling ===================== #
# ============ Holt-Winters Method
from statsmodels.tsa.holtwinters import ExponentialSmoothing

hwmodel = ExponentialSmoothing(y_train,
                               trend='add',
                               seasonal='add',
                               seasonal_periods=24).fit()

df_train['holt-winter_add'] = hwmodel.fittedvalues
plt.figure()
y_train.plot(label='original')
df_train['holt-winter_add'].plot(label='holt-winter additive')
plt.title('Holt-Winter Method')
plt.legend()
plt.ylabel('Temperature(\N{DEGREE SIGN}F)')
plt.xlabel('')
plt.show()

# Holt-Winters Forecast
hwforecast = hwmodel.forecast(steps=len(y_test))
# hwforecast = hwmodel.forecast(steps=2000)
# hwforecast = hwmodel.predict(start=df_test.index[0], end=df_test.index[-1])
plt.figure(figsize=(12, 6))
y_test.plot(label='test set')
hwforecast.plot(label='forecast')
plt.title('Holt-Winter Forecast')
plt.ylabel('Temperature(\N{DEGREE SIGN}F)')
plt.legend()
plt.tight_layout()
plt.show()

# first 500 hours forecasting
plt.figure(figsize=(10, 6))
y_test[:501].plot(label='test set first 1000')
hwforecast[:501].plot(label='forecast')
plt.title('Holt-Winter Forecast for 1000 Hours')
plt.ylabel('Temperature(\N{DEGREE SIGN}F)')
plt.legend()
plt.tight_layout()
plt.show()

# evaluation
print(f"Mean square error of Holt-Winter method: {mean_squared_error(y_test, hwforecast):.4f}")
print('*' * 50)
# ============ Average Method
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
result = forecast_average_method(y_train, y_test)
# result[1] = convert_differencing(result[1], y_test, 24)
# plt.figure(figsize=(10, 6))
y_train.plot(label='train set', ax=ax[0,0])
y_test.plot(label='test set', ax=ax[0,0])
result[0].plot(label='one-step prediction', ax=ax[0,0])
result[1].plot(label='h-step forecast', ax=ax[0,0])
ax[0,0].set_title('Average Method Forecast')
ax[0,0].legend()
# plt.tight_layout()
# plt.show()
# # evaluation
# res_e = y_train - result[0]
# fore_e = y_test - result[1]
# fig, ax = plt.subplots(2, 1, figsize=(10, 10))
# AutoCorr_subplot(res_e[1:], lags, 'Residual Error', ax[0])
# AutoCorr_subplot(fore_e, lags, 'Forecast Error', ax[1])
# plt.tight_layout()
# plt.show()

# ============ Naive Method
result = forecast_naive_method(y_train, y_test)
# plt.figure(figsize=(10, 6))
y_train.plot(label='train set', ax=ax[0,1])
y_test.plot(label='test set', ax=ax[0,1])
result[0].plot(label='one-step prediction', ax=ax[0,1])
result[1].plot(label='h-step forecast', ax=ax[0,1])
ax[0,1].set_title('Naive Method Forecast')
ax[0,1].set_ylabel('Temperature(\N{DEGREE SIGN}F)')
ax[0,1].legend()
# plt.tight_layout()
# plt.show()

# ============ Drift Method
result = forecast_drift_method(y_train, y_test)
# plt.figure(figsize=(10, 6))
y_train.plot(label='train set', ax=ax[1,0])
y_test.plot(label='test set', ax=ax[1,0])
result[0].plot(label='one-step prediction', ax=ax[1,0])
result[1].plot(label='h-step forecast', ax=ax[1,0])
ax[1,0].set_title('Drift Method Forecast')
ax[1,0].set_ylabel('Temperature(\N{DEGREE SIGN}F)')
ax[1,0].legend()
# plt.tight_layout()
# plt.show()

# ============ SES Method
result = forecast_simple_exponential_method(y_train, y_test, 0.5)
# plt.figure(figsize=(10, 6))
y_train.plot(label='train set', ax=ax[1,1])
y_test.plot(label='test set', ax=ax[1,1])
result[0].plot(label='one-step prediction', ax=ax[1,1])
result[1].plot(label='h-step forecast', ax=ax[1,1])
ax[1,1].set_title('Simple Exponential Smoothing Method Forecast')
ax[1,1].set_ylabel('Temperature(\N{DEGREE SIGN}F)')
ax[1, 1].legend()
plt.tight_layout()
plt.show()

# ====================== Multiple Linear Regression ===================== #
print(back_model.summary())
print('*' * 50)
back_predicted = back_model.predict(sm.add_constant(X_test))
plt.figure(figsize=(12, 6))
df_test['f_temp'].plot(label='test set')
back_predicted.plot(label='forecast')
plt.title('Multiple Linear Regression on Temperature')
plt.ylabel('Temperature(\N{DEGREE SIGN}F)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
df_test['f_temp'].iloc[:1000].plot(label='test set')
back_predicted.iloc[:1000].plot(label='forecast')
plt.title('Multiple Linear Regression on Temperature Forecast 1000 Hours')
plt.ylabel('Temperature(\N{DEGREE SIGN}F)')
plt.legend()
plt.tight_layout()
plt.show()

# ============ evaluation
# print("Mean absolute error =", round(sm.mean_absolute_error(y_test, back_predicted), 2))
# print("Median absolute error =", round(sm.median_absolute_error(y_test, back_predicted), 2))
# print("Explain variance score =", round(sm.explained_variance_score(y_test, back_predicted), 2))
# print("R2 score =", round(sm.r2_score(y_test, back_predicted), 2))

print('t-test result for each coefficients:\n')
print(back_model.pvalues.round(4))
print('*' * 50)
print('f-test result for model:\n')
print(f'(F-statistics: {back_model.fvalue:.4f}, p-value:{back_model.f_pvalue:.4f})')
print('*' * 50)
# MSE
e = y_test - back_predicted
print("Mean squared error for linear regression =", round(matrics.mean_squared_error(y_test, back_predicted), 2))
print('*' * 50)
# ACF
# AutoCorr_plot(e, lags, 'residual error')
ACF_PACF_Plot(e, lags)
# Q-value
from TStoolbox import Q_value, chi_square_test
Q = Q_value(e, lags)
print(f'Q value of residual error with linear regression is {Q:.2f}')
print('*' * 50)
DOF = lags - len(x_train.columns)
alpha = 0.01
# chi-square test
print("Chi-square test on residual error from linear regression:")
chi_square_test(Q, alpha, DOF)
print('*' * 50)
# residual variance and mean
print(f'mean of linear regression residual error is {np.mean(e):.2f},\n'
      f'variance of linear regression residual error is {np.var(e):.2f}')
print('*' * 50)

# ====================== ARMA ===================== #
arma_y = df_train['differenced']
# ============ order determination
# GPAC table
from TStoolbox import GPAC_table, SARIMA_parameter_estimation, zero_pole
ry_sta = Cal_AutoCorr(arma_y, lags)
table = GPAC_table(ry_sta, 7, 7)
fig = plt.figure()
sns.heatmap(table, annot=True, fmt='.2f', cmap='BrBG')
plt.title('Generalized Partial Autocorrection(GPAC) Table ARMA model')
plt.show()
# ACF/PACF
ACF_PACF_Plot(arma_y, lags)
# order pick 1: na = 1, nb = 0
# order pick 2: na = 3, nb = 1

# ============ coefficients estimation
arma_model1 = SARIMA_parameter_estimation(arma_y, 2, 1)
print(arma_model1.summary())
print('*' * 50)
zero_pole(arma_model1.arparams, arma_model1.maparams)
print('*' * 50)
arma_model2 = SARIMA_parameter_estimation(arma_y, 0, 1)
print(arma_model2.summary())
print('*' * 50)
# zero_pole(arma_model.arparams, arma_model.maparams)
na = 1
nb = 0
arma_model = SARIMA_parameter_estimation(arma_y, na, nb)
print(arma_model.summary())
print('*' * 50)

# prediction
arma_pred = arma_model.predict(start=1, end=len(arma_y)-1) # one step prediction
res_e = arma_y[1:] - arma_pred
AutoCorr_plot(res_e, lags, 'Residual Error of ARMA')
arma_fore = arma_model.forecast(steps=len(y_test)) # h step prediction

plt.figure()
arma_pred.plot(label='one-step forecast')
arma_y.plot(label='train set')
plt.title('ARMA Model One Step Prediction')
plt.xlabel("")
plt.ylabel('Temperature(\N{DEGREE SIGN}F)')
plt.legend()
plt.show()

arma_fore = convert_differencing(arma_fore, y_train[-(interval+len(y_test)):]).iloc[interval:]
arma_fore.index = y_test.index
forecast_e = y_test - arma_fore
AutoCorr_plot(forecast_e, lags, 'Forecast Error of ARMA')
# chi-sqaure test
q_arma_pre = Q_value(res_e, lags)
print(f'Q value for ARMA residual error: {q_arma_pre}')
print('*' * 50)
print('Chi-square test on residual error from ARMA:')
chi_square_test(q_arma_pre, 0.05, lags-na-nb)
print('*' * 50)
q_arma_fore = Q_value(forecast_e, lags)
print(f'Q value for ARMA forecast error: {q_arma_fore}')
print('*' * 50)
print('Chi-square test on forcast error from ARMA:')
chi_square_test(q_arma_fore, 0.05, lags-na-nb)
print('*' * 50)
# plot forecast
plt.figure()
arma_fore.plot(label='h-step forecast')
y_test.plot(label='test set')
plt.title('ARMA Model Forecast')
plt.ylabel('Temperature(\N{DEGREE SIGN}F)')
plt.xlabel("")
plt.legend()
plt.show()

print("Mean squared error =", round(matrics.mean_squared_error(y_test, arma_fore), 2))
print('*' * 50)
# ====================== SARIMA ===================== #
# ============ order determination
# sarima_y = differencing(arma_y, 1)
# sarima_y.index = arma_y.index[1:]
# ACF_PACF_Plot(sarima_y, 80)

# ======================
# now we have SARIMA(x, 1, x)x(0, 1, 1, 24)
# ======================
# ============ coefficients estimation
# gridSearch
# from sktime.forecasting.arima import AutoARIMA
# forecaster = AutoARIMA(start_p = 0,
#                        start_q = 0,
#                        max_p = 3, # order of ar
#                        max_q = 3, # order of ma
#                        start_P = 0,
#                        max_P = 0,
#                        start_Q = 1,
#                        max_Q = 1,
#                        d = 1,
#                        D = 1,
#                        sp = 24,
#                        seasonal=True,
#                        stationary=True)
# forecaster.fit(y_train)
# print(forecaster.summary())
# # now we have SARIMA(0, 1, 2)x(0, 1, 0)
#
# sarima = AutoARIMA(start_p = 0,
#                        start_q = 0,
#                        max_p = 3, # order of ar
#                        max_q = 2, # order of ma
#                        start_P = 0,
#                        max_P = 3,
#                        start_Q = 0,
#                        max_Q = 3,
#                        d = 1,
#                        D = 1,
#                         sp= 24,
#                        seasonal=True,
#                        stationary=True,
#                        stepwise=False)
# sarima.fit(arma_y)
# print(sarima.summary())
# # now we have SARIMA(2, 1, 2)x(0, 1, 0)
na = 0
d = 0
nb = 1
Na = 0
Nb = 1
D = 0
S = 24
dof = lags - na - nb - Na - Nb
sarima_model = SARIMA_parameter_estimation(arma_y, na, nb, d, Na, Nb, D, S)
print(sarima_model.summary())
print('*' * 50)
sarima_pred = sarima_model.predict(start=1, end=len(y_train)-1) # one step prediction
sarima_forecast = sarima_model.forecast(steps=len(y_test)) # h-step prediction
# invert diff predict
# sarima_pred = convert_differencing(sarima_pred, y_train)
# sarima_pred.index = y_train.index
# index = sarima_pred.index
# sarima_pred = convert_differencing(sarima_pred, y_train[-interval:], interval)
# # sarima_pred.index = y_train.index
# sarima_pred.index = index

# invert differencing forecast
sarima_forecast = convert_differencing(sarima_forecast, y_train[-(interval+len(y_test)):], interval=interval).iloc[interval:]
sarima_forecast.index = y_test.index

# index = sarima_forecast.index
# sarima_forecast = convert_differencing(sarima_forecast, y_train[-(1+len(y_test)):]).iloc[1:]
# sarima_forecast.index = index

# prediction error
res_sarima = y_train[1:] - sarima_pred
AutoCorr_plot(res_sarima, lags, 'Residual Error from SARIMA')
ACF_PACF_Plot(res_sarima, lags)
q_sarima_pre = Q_value(res_sarima, lags)
print(f'Q value for SARIMA residual error: {q_sarima_pre}')
print('*' * 50)
print('Chi-square test on residual error from SARIMA:')
chi_square_test(q_sarima_pre, 0.05, dof)
print('*' * 50)

# forecast error
forecast_sarima = y_test - sarima_forecast
ACF_PACF_Plot(forecast_sarima, lags)

plt.figure()
sarima_pred.plot(label='1-step prediction')
y_train.plot(label='train set')
plt.title('SARIMA Model Prediction')
plt.ylabel('Temperature(\N{DEGREE SIGN}F)')
plt.legend()
plt.show()

plt.figure()
sarima_forecast.plot(label='h-step forecast')
y_test.plot(label='test set')
plt.title('SARIMA Model Forecast')
plt.ylabel('Temperature(\N{DEGREE SIGN}F)')
plt.xlabel("")
plt.legend()
plt.show()

plt.figure()
sarima_forecast.iloc[:1000].plot(label='h-step forecast')
y_test.iloc[:1000].plot(label='test set')
plt.title('SARIMA Model Forecast for First 1000 Hour')
plt.ylabel('Temperature(\N{DEGREE SIGN}F)')
plt.xlabel("")
plt.legend()
plt.show()
q_sarima_fore = Q_value(forecast_sarima, lags)
print(f'Q value for SARIMA forecast error: {q_sarima_pre}')
print('*' * 50)
print('Chi-square test on forecast error from SARIMA:')
chi_square_test(q_sarima_fore, 0.05, dof)
print('*' * 50)
print("Mean squared error =", round(matrics.mean_squared_error(y_test, sarima_forecast), 2))
print('*' * 50)
