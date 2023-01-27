import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
import sys
sys.path.append('../TStoolbox.py')
from TStoolbox import check_nan, plot_missing, AutoCorr_plot, \
    correlation_matrix_plot, kpss_test, ADF_Cal, Cal_rolling_mean_var

# ====================== data loading ===================== #
temp = pd.read_csv('datasets/temperature.csv')
city = pd.read_csv('datasets/city_attributes.csv')
humidity = pd.read_csv('datasets/humidity.csv')
pressure = pd.read_csv('datasets/pressure.csv')
weather = pd.read_csv('datasets/weather_description.csv')
wind_direction = pd.read_csv('datasets/wind_direction.csv')
wind_speed = pd.read_csv('datasets/wind_speed.csv')
dfs = [temp, humidity, pressure, weather, wind_direction, wind_speed, city]
df_names = ['temp', 'humidity', 'pressure', 'weather', 'wind_direction', 'wind_speed', 'city']
print([f'shape of {df_names[x]}: {dfs[x].shape} \n' for x in range(len(dfs))])
print('*' * 50)


def get_city(city):
    """
    generate temp, humidity, wind information, pressure, and weather description of given city.
    :param city: str
    :return: pd.DataFrame
    """
    dfs = [temp, humidity, pressure, weather, wind_direction, wind_speed]
    col = [f'{city}_temp', 'humidity', 'pressure', 'weather', 'wind_direction', 'wind_speed']
    df = pd.DataFrame(index=pd.to_datetime(temp.datetime))
    for z in range(len(col)):
        df[col[z]] = dfs[z][city].values
    return df


cities = city.City
print(cities)
print('*' * 50)
# ==========define our city here
c = cities[0]
df = get_city(c)
print(f'Head of {c} temperature dataset')
print(df.head())
print('*' * 50)
col = df.columns


def kelvin_to_celsius(kelvin):
    """
    convert kelvin temp to celsius temp
    :param kelvin: float
    :return: float
    """
    return kelvin - 273.15

def kelvin_to_f(kelvin):
    """
    convert kelvin temp to fahrenheit temp
    :param kelvin: float
    :return: float
    """
    return (kelvin - 273.15)*9/5 + 32


# ====================== clean data ===================== #
# ============ check nan values
temp_nan = pd.DataFrame(check_nan(df).data)
print('Missing values:\n', temp_nan)
plot_missing(df)
print('*' * 50)
# ============ remove nan values
# truncate tail
lst_i = df.last_valid_index()
df = df[:lst_i]
# impute numeric values
df = df.interpolate(method='linear', limit_direction='backward')
print('Missing values after impute numerical values:\n',pd.DataFrame(check_nan(df).data))
print('*' * 50)
# impute categorical values
df.weather = df.weather.replace(df.weather.iloc[0], df.weather.values[1])
print('Missing values after impute categorical values\n',pd.DataFrame(check_nan(df).data))
print('*' * 50)

# category to numeric
print('weathers categories in dataset')
print(df.weather.value_counts())
print('*' * 50)
labels = enumerate(set(df['weather']))
label2int = {}
for i, l in labels:
    label2int[l] = i
print("numeric labels for weather", label2int)
print('*' * 50)
df['weather'] = df['weather'].apply(lambda x: label2int[x])


# ====================== visualization ===================== #
dep = col[0]
cityname = dep[:dep.index('_')]
plt.figure(figsize=(10, 6))
df['f_temp'] = df[dep].apply(kelvin_to_f)
df['cel_temp'] = df[dep].apply(kelvin_to_celsius)
tempf = df['f_temp'].copy()
df['f_temp'].plot()
plt.title(f"Hourly Temperature of {cityname}")
plt.ylabel(u'Temperature(\N{DEGREE SIGN}F)')
plt.xlabel('Time')
plt.grid()
plt.show()

# ============ ACF
from TStoolbox import ACF_PACF_Plot
AutoCorr_plot(df[dep], 20, label=f'Temperature in {cityname}')
ACF_PACF_Plot(df[dep], 40)

# ============ CorrelationMatrix
correlation_matrix_plot(df[col], 'pearson')

# ============ Train Test Split
df_train = df[:round(len(df)*0.80)]
df_test = df[round(len(df)*0.80):]
y_train = df_train['f_temp']
print(f'train set length: {df_train.shape[0]}')
print(f'test set length: {df_test.shape[0]}')
print('*' * 50)

# # ====================== stationarity ===================== #
# ============ check stationary
kpss_test(y_train)
ADF_Cal(y_train)
Cal_rolling_mean_var(df_train, 'f_temp') # not stationary
# ============ STL Decomposition
from TStoolbox import STL_trend_strength
from statsmodels.tsa.seasonal import STL
res = STL(y_train).fit()
# res.plot()
# plt.show()
T, S, R = res.trend, res.seasonal, res.resid
print('strength of trend and seasonality original data')
STL_trend_strength(T, S, R)
print('*' * 50)
# show decomposition
fig = plt.figure()
T.plot(label='trend')
S.plot(label='seasonality')
R.plot(label='reminder')
plt.legend()
plt.title(f'Temperature of {cityname} STL Decomposition')
plt.xlabel('time')
plt.ylabel('temp(\N{DEGREE SIGN}F)')
plt.show()
# ============ seasonal differencing
from TStoolbox import differencing
INTERVAL = 24
# INTERVAL = 8760
y_train_sta = differencing(y_train, INTERVAL)
y_train_sta.index = y_train.index[INTERVAL:]
# ============ re-check
kpss_test(y_train_sta)
ADF_Cal(y_train_sta)
Cal_rolling_mean_var(pd.DataFrame(y_train_sta, columns=['f_temp']), 'f_temp')

# show decomposition
res1 = STL(y_train_sta).fit()
T, S, R = res1.trend, res1.seasonal, res1.resid
print('strength of trend and seasonality after sesonal differencing')
STL_trend_strength(T, S, R)
print('*' * 50)

fig = plt.figure()
T.plot(label='trend')
S.plot(label='seasonality')
R.plot(label='reminder')
plt.legend()
plt.title(f'Seasonal Differenced Temperature of {cityname} STL Decomposition')
plt.xlabel('time')
plt.ylabel('temp(\N{DEGREE SIGN}F)')
plt.show()

# ============ non-seasonal differencing
index = y_train_sta.index
y_train_sta = differencing(y_train_sta, 1)
y_train_sta.index = index[1:]
# show decomposition
res1 = STL(y_train_sta).fit()
T, S, R = res1.trend, res1.seasonal, res1.resid
print('strength of trend and seasonality after detrended seasonal differencing')
STL_trend_strength(T, S, R)
print('*' * 50)

fig = plt.figure(figsize=(10, 6))
T.plot(label='trend')
S.plot(label='seasonality')
R.plot(label='reminder')
plt.legend()
plt.title(f'Detrended Seasonal Differenced Temperature of {cityname} STL Decomposition')
plt.xlabel('time')
plt.ylabel('temp(\N{DEGREE SIGN}F)')
plt.show()

# show data
plt.figure()
y_train_sta.plot()
plt.title(f"Seasonal Differenced Hourly Temperature of {cityname}")
plt.ylabel(u'Temperature(\N{DEGREE SIGN}F)')
plt.xlabel('Time')
plt.grid()
plt.show()
# ============ MA Decomposition
# from TStoolbox import rolling_avg, detrended
# fig, ax = plt.subplots(2, 1, figsize=(16, 10))
# order = [3, 7]
# for i in range(0, 2):
#     ax[i].plot(y_train, label='original')
#     Tt = rolling_avg(y_train, order[i])
#     ax[i].plot(Tt, label=f'{order[i]}-MA')
#     ax[i].plot(detrended(y_train, Tt, 'add'), label='detrended')
#     ax[i].legend()
#     ax[i].set_title(f'Moving Average')
#     ax[i].set_ylabel('temperature(\N{DEGREE SIGN}F)')
#     ax[i].set_xlabel('time')
#
# plt.tight_layout()
# plt.show()

# ============ detrend & seasonally adjusted
# Temp_detrend = y_train - T
# Temp_ssadj = Temp_detrend - S
# y_train_ssadj = differencing(y_train_sta, interval=INTERVAL)
# y_train_ssadj.index = df_train.index[len(df_train)-len(y_train_ssadj):]
# res2 = STL(y_train_ssadj).fit()
# res2.plot()
# plt.title('After Adjusting Seasonality')
# plt.show()
#
# print('strength of trend and seasonality after detrend and adjusting seasonality')
# STL_trend_strength(res2.trend, res2.seasonal, res2.resid)
#
# fig = plt.figure()
# y_train.plot(label='raw data')
# y_train_sta.plot(label='differenced data')
# y_train_ssadj.plot(label='seasonal differenced data')
# # Temp_detrend.plot(label='detrended data')
# # Temp_ssadj.plot(label='seasonal adjusted data')
# plt.title(f'temperature of {cityname}')
# plt.legend()
# plt.xlabel('time')
# plt.ylabel('temp(\N{DEGREE SIGN}F)')
# plt.show()
# # check stationary after detrend and seasonal adjust
# Cal_rolling_mean_var(pd.DataFrame(y_train_ssadj, columns=['f_temp']), 'f_temp')
# Cal_rolling_mean_var(pd.DataFrame(Temp_ssadj, columns=['f_temp']), 'f_temp')


# ====================================
# for now we have:
# df ---> raw dataset
# tempf ---> raw dependent variable
# df_train, df_test ---> original train, test set
# y_train ---> train set dependent variable
# y_train_sta ---> stationary but seasonal y_train (first order differencing)
# y_train_ssadj ---> 1st order non-seasonal differenced and 1st order seasonal differenced(k=8760)
# Temp_detrend ---> detrended but seasonal y_train (STL decomposed)
# Temp_ssadj ---> detrended and non-seasonal y_train (STL decomposed)
# ====================================
output = pd.DataFrame({'original': y_train, 'differenced': y_train_sta,
                       # 'detrended':Temp_detrend,
                       # 'non-seasonal': y_train_ssadj,
                       'humidity': df_train.humidity, 'pressure': df_train.pressure,
                       'weather': df_train.weather, 'wind_direction': df_train.wind_direction,
                       'wind_speed': df_train.wind_speed})
output_test = df_test.copy()
output_test.columns = ['original'] + list(output_test.columns[1:])
output.to_csv('train.csv')
output_test.to_csv('test.csv')

