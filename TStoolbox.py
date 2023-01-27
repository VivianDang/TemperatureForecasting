import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================= data clean =============================
from dash import dash_table
def check_nan(df):
    """
    The missing value checker

    :return: DataTable
    table of variable and number of missing values in variable
    """
    # missing values check
    df_nan = pd.DataFrame([[var, df[var].isna().sum()] for var in df.columns],
                          columns=['var', 'number of missing values'])
    dt_nan = dash_table.DataTable(df_nan.to_dict('records'), [{"name": i, "id": i} for i in df_nan.columns])
    return dt_nan

# def avg_n_surrounding(yt, cur_i, n):
#     """
#
#     :param yt:
#     :param cur_i: int index of missing value
#     :param n:
#     :return:
#     """
#     con = pd.concat([yt.iloc[cur_i-n:cur_i], yt.iloc[cur_i+1:cur_i+n+1]])
#     return np.mean(con)
#
# def impute_missing_by_surrounding_avg(yt, n):
#     """
#
#     :param yt:
#     :param n: int number of surrounding values one side
#     :return:
#     """
#
#     yt_ = yt.copy()
#     missing_i = yt_[yt_.isna()].index
#     for i in range(len(missing_i)-n):
#         id = missing_i[i]
#         print(id)
#         yt_.iloc[id] = avg_n_surrounding(yt_, id, n)
#     return yt_


import seaborn as sns
def plot_missing(df):
    """
    plot missing values by index of missing values
    reference https://www.kaggle.com/code/selahattinsanli/visualizing-missing-data
    :param df:
    :return:
    """
    df = df.reset_index(drop=True)

    fig = plt.figure(figsize=(9, 6))
    sns.heatmap(df.isna(), cmap=sns.color_palette(['lightgrey', 'tan']))
    plt.title('Missing Values')
    plt.grid(axis='x')
    plt.ylabel('index')
    plt.show()


# ================================= stationary =============================
def Cal_rolling_mean_var(df, feature):
    """
    Plot rolling mean and rolling variance to check stationary
    :param df: pd.DataFrame
    :param feature: str
    :return: plot
    """
    output = pd.DataFrame(index=np.arange(0, len(df)), columns=['rol_mean', 'rol_var'])
    for i in range(len(df)):
        rol_mean = np.mean(df[feature].head(i + 1))
        rol_var = np.var(df[feature].head(i + 1))
        output.iloc[i] = [rol_mean, rol_var]

    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(2, 1, 1)
    output.rol_mean.plot()
    plt.title(f'Rolling Mean - {feature}')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.legend(['Varying mean'])

    fig.add_subplot(2, 1, 2)
    output.rol_var.plot()
    plt.title(f'Rolling Variance - {feature}')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.legend(['Varying variance'])

    plt.tight_layout()
    plt.show()


from statsmodels.tsa.stattools import adfuller

def ADF_Cal(x):
    """
    Augmented Dicky Fuller(ADF) test.
    H0: The process is trend non-stationary.
    :param x: np.Series
    :return:
    """
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))



from statsmodels.tsa.stattools import kpss


def kpss_test(timeseries):
    """
    Kwiatkowski-Phillips-Schmidt-Shin(KPSS) test to test stationary.
    H0: the process id trend stationary.
    :param timeseries: np.Series
    :return:
    """
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'LagsUsed'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)


def differencing(y, interval=1):
    """
    Perform Differencing on y
    :param y: pd.Series time series data
    :param interval: int for seasonal data
    :return: pd.Series differenced data
    """
    diff = list()
    for i in range(interval, len(y)):
        value = float(y.iloc[i] - y.iloc[i-interval])
        diff.append(value)
    return pd.Series(diff)

def convert_differencing(differenced, original, interval=1):
    """
    Convert differencing on differenced data
    :param differenced: pd.Series differenced data
    :param original: pd.Series original data
    :param interval: int the interval performed
    :return: pd.Series original data
    """
    result = original.copy()
    for i in range(len(differenced)):
        result.iloc[i+interval] = float(differenced.iloc[i] + original.iloc[i])
    return result

# ================================= correlation coefficient =============================
def pearson_coef(x, y):
    """
    calculate the pearson's correlation coefficient between random variables x and y

    :param x: variable x
    :param y: variable y
    :return: pearson correlation coefficient between x and y
    """
    mu_x = np.sum(x)/(len(x)-1)
    mu_y = np.sum(y)/(len(y)-1)
    diff_x = x - mu_x
    diff_y = y - mu_y
    r_x_y = np.sum(diff_x*diff_y) / (np.sqrt(np.sum(diff_x**2)) * np.sqrt(np.sum(diff_y**2)))
    return r_x_y


def correlation_matrix_plot(df, method):
    """
    Plot correlation matrix heatmap with method
    :param df: pd.DataFrame
    :param method: str
    :return:
    """
    corr = df.corr(method=method)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='BrBG')
    plt.title(f'{method} correlation matrix')
    plt.show()


# ================================= Autocorrection(ACF) =============================
def Cal_AutoCorr(y, lag):
    """
    Calculate the Autocorrelation
    :param y: Series
    :param lag: int number of lags
    :return:
    """
    y_mean = np.mean(y)
    result = []
    denom = np.sum((y-y_mean)**2)
    for i in range(lag+1):
        sum = 0
        for j in range(i+1, len(y)+1):
            sum += (y[j-1] - y_mean) * (y[j-i-1] - y_mean)
        result.append(sum/denom)
    return result


def AutoCorr_plot(y, lag, label):
    """
    Plot the Autocorrelation function
    :param y: data
    :param lag: int number of lags
    :param label: label of y as title
    :return:
    """
    # y_mean = np.mean(y)
    # result = []
    # denom = np.sum((y-y_mean)**2)
    # for i in range(lag+1):
    #     sum = 0
    #     for j in range(i+1, len(y)+1):
    #         sum += (y[j-1] - y_mean) * (y[j-i-1] - y_mean)
    #     result.append(sum/denom)
    result = Cal_AutoCorr(y, lag)

    x = np.arange(-lag, lag+1, 1)
    result = result[::-1] + result[1:]
    plt.figure()
    (markers, stemlines, baseline) = plt.stem(x,result,markerfmt='o')
    plt.setp(markers, color = 'red', marker = 'o')
    plt.setp(baseline, color = 'gray', linewidth = 2,linestyle = '-')
    m = 1.96/np.sqrt(len(y))
    plt.axhspan(-m,m, alpha = 0.2, color = 'blue')
    plt.ylabel('Magnitude')
    plt.xlabel('Lags')
    plt.title(f'Autocorrelation Function of {label}')
    plt.show()


def AutoCorr_subplot(y, lag, label, ax):
    """
    Generate plot of the Autocorrelation function on ax
    :param y: data
    :param lag: number of lags
    :param label: label of y
    :param ax: ax
    :return:
    """
    # y_mean = np.mean(y)
    # result = []
    # denom = np.sum((y-y_mean)**2)
    # for i in range(lag+1):
    #     sum = 0
    #     for j in range(i+1, len(y)+1):
    #         sum += (y.iloc[j-1] - y_mean) * (y.iloc[j-i-1] - y_mean)
    #     result.append(sum/denom)
    result = Cal_AutoCorr(y, lag)

    x = np.arange(-lag, lag+1, 1)
    result = result[::-1] + result[1:]
    # result = pd.Series(result)
    plt.figure()
    (markers, stemlines, baseline) = ax.stem(x,result,markerfmt='o')
    plt.setp(markers, color = 'red', marker = 'o')
    plt.setp(baseline, color = 'gray', linewidth = 2,linestyle = '-')
    m = 1.96/np.sqrt(len(y))
    ax.axhspan(-m,m, alpha = 0.2, color = 'blue')
    ax.set_ylabel('Magnitude')
    ax.set_xlabel('Lags')
    ax.set_title(f'Autocorrelation Function of {label}')
    # plt.show()


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def ACF_PACF_Plot(y, lags):
    """
    Plot ACF and partial ACF
    :param y: pd.Series raw data
    :param lags: int number of lags to show
    :return:
    """
    # acf = sm.tsa.stattools.acf(y, nlags=lags)
    # pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()


# ================================= simple forecast method =============================
def forecast_average_method(train, test):
    """
    Average method forecast
    :param train: pd.Series time series
    :param test: pd.Series time series
    :return: one-step prediction, h-step prediction
    """
    one_step = [np.nan]
    for i in range(1, len(train)):
        # one_step.append(np.mean(train[:i], axis=0).values[0])
        one_step.append(np.mean(train[:i], axis=0))
    # h_step = [np.mean(train, axis=0).values[0]] * len(test)
    h_step = [np.mean(train, axis=0)] * len(test)
    return pd.Series(one_step, index=train.index), pd.Series(h_step, index=test.index)


def forecast_naive_method(train, test):
    """

    :param train:
    :param test:
    :return:
    """
    one_step = [np.nan]
    for i in range(1, len(train)):
        # one_step.append(train.iloc[i-1].values[0])
        one_step.append(train.iloc[i - 1])
    # h_step = [train.iloc[-1].values[0]] * len(test)
    h_step = [train.iloc[-1]] * len(test)
    return pd.Series(one_step, index=train.index), pd.Series(h_step, index=test.index)


def forecast_drift_method(train, test):
    """

    :param train:
    :param test:
    :return:
    """
    one_step = [np.nan, np.nan]
    h_step = []
    # y1 = train.iloc[0].values[0]
    y1 = train.iloc[0]
    for i in range(2, len(train)):
        # yt = train.iloc[i-1].values[0]
        yt = train.iloc[i - 1]
        one_step.append(yt + (yt-y1)/(i-1))
    # yt = train.iloc[-1].values[0]
    yt = train.iloc[-1]
    for i in range(1, len(test)+1):
        h_step.append(yt + i * (yt-y1) / (len(train)-1))
    return pd.Series(one_step, index=train.index), pd.Series(h_step, index=test.index)


def forecast_simple_exponential_method(train, test, alpha):
    """

    :param train:
    :param test:
    :param alpha: float damping factor from [0, 1]
    :return:
    """
    # one_step = [train.iloc[0].values[0]]
    one_step = [train.iloc[0]]
    for i in range(1, len(train)):
        # one_step.append(alpha * train.iloc[i-1].values[0] + (1-alpha) * one_step[i-1])
        one_step.append(alpha * train.iloc[i - 1] + (1 - alpha) * one_step[i - 1])
    # h_step = [alpha * train.iloc[-1].values[0] + (1-alpha) * one_step[-1]] * len(test)
    h_step = [alpha * train.iloc[-1] + (1 - alpha) * one_step[-1]] * len(test)
    return pd.Series(one_step, index=train.index), pd.Series(h_step, index=test.index)


def plot_forecasting_method(train, test, forecast, label):
    """
    Plot h-step prediction with forecast method.

    :param train: time series df
    :param test: time series df
    :param forecast: predicted values
    :return:
    """
    fig, ax = plt.subplots()
    ax.plot(train, label='Train')
    ax.plot(test, label='Test')
    ax.plot(forecast, label=f'{label} Method h-step prediction')
    plt.legend()
    plt.title(f'{label} Method & Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()


# ================================= residual checking =============================
def Cal_MSE(train, test, pred, forecast):
    """
    return min square error (e^2) for prediction and forecasting
    :param train:
    :param test:
    :param pred: 1-step prediction
    :param forecast: h-step prediction
    :return:
    """
    train_col = train.columns[0]
    test_col = test.columns[0]
    pre_error = train[train_col] - pred
    mse_pre = np.mean(pre_error**2)
    forecast_error = test[test_col] - forecast
    mse_fore = np.mean(forecast_error**2)
    result_pre = pd.DataFrame({'e': pre_error, 'e^2': pre_error**2, 'MSE':mse_pre}, index=train.index)
    result_fore = pd.DataFrame({'e': forecast_error, 'e^2': forecast_error**2, 'MSE':mse_fore}, index=test.index)
    return result_pre, result_fore


def Q_value(x, lags):
    """

    :param x: pd.Series
    :param lags: int number of lags
    :return: float q value
    """
    re = Cal_AutoCorr(x, lags)
    Q = len(x) * np.sum(np.square(re[lags:]))
    return Q


from scipy.stats import chi2
def chi_square_test(stat, alpha, dof):
    """
    Perform chi square test and display result
    :param stat: float statistics value
    :param alpha: float p-value threshold
    :param dof: int degree of freedom
    :return:
    """
    critical_chi = chi2.ppf(1 - alpha, dof)
    if stat < critical_chi:
        print('The error is white')
    else:
        print('The error is not white')

# ================================= feature reduction =============================
from sklearn.decomposition import PCA
def draw_PCA(x):
    """
    Plot PCA to check minimum number of necessary features
    :param x: DataFrame normalized data
    :return:
    """
    pca_ = PCA(n_components=len(x.columns), svd_solver='full')
    pca_.fit(x)

    plt.figure()
    x = np.arange(1, len(np.cumsum(pca_.explained_variance_ratio_)) + 1, 1)
    plt.plot(x, np.cumsum(pca_.explained_variance_ratio_))
    plt.xticks(x)
    plt.ylabel('cumulative explained values')
    plt.xlabel('number of component')
    plt.title('PCA Analysis')
    plt.grid()
    plt.show()


import statsmodels.api as sm
def backwards_selection(x, y, threshold=None):
    """
    Perform backwards selection on p-values
    :param threshold: threshold to determine p value significance, feature with p-value > threshold will be eliminated.
    :param x: pd.DataFrame multiple features
    :param y: pd.Series target y
    :return: Analysis table, result model
    """
    predictors = list(x.columns)
    formula = []
    aic = []
    bic = []
    r_adj = []
    while len(predictors) != 0:
        model = sm.OLS(y, sm.add_constant(x[predictors])).fit()
        formula.append(dict.fromkeys(predictors, int(1)))
        aic.append(model.aic)
        bic.append(model.bic)
        r_adj.append(model.rsquared_adj)
        # all p-values except intercept
        pvalue = model.pvalues
        worst_p = pvalue.max()
        if threshold is not None:
            if worst_p > threshold:
                predictors.remove(pvalue.idxmax())
            else:
                break
        else:
            predictors.remove(pvalue.idxmax())
    result = pd.DataFrame(formula).astype(float)
    result['AIC'] = aic
    result['BIC'] = bic
    result['Adj_R^2'] = r_adj
    return result, model


from statsmodels.stats.outliers_influence import variance_inflation_factor
def vif_selection(x, y, threshold=3):
    """
    Perform selection by VIF value
    :param x: DataFrame
    :param y: DataFrame
    :param threshold: indicate remove, default=3
    :return: Analysis table, result model
    """
    predictors = list(x.columns)
    formula = []
    aic = []
    bic = []
    r_adj = []
    while len(predictors) != 0:
        formula.append(dict.fromkeys(predictors, int(1)))
        model = sm.OLS(y, sm.add_constant(x[predictors])).fit()
        aic.append(model.aic)
        bic.append(model.bic)
        r_adj.append(model.rsquared_adj)
        # remove the max vif
        vif = {predictors[i]: variance_inflation_factor(x[predictors].values, i) for i in range(len(predictors))}
        max_vif = max(vif, key=vif.get)
        if vif[max_vif] > threshold:
            predictors.remove(max_vif)
        else:
            break
    result = pd.DataFrame(formula).astype(float)
    result['AIC'] = aic
    result['BIC'] = bic
    result['Adj_R^2'] = r_adj
    return result, model


# ================================= model coefficient estimation =============================
from numpy import linalg as LA
def linear_regression_model(x, y):
    """
    Estimate coefficients of linear model
    :param x: DataFrame
    :param y: DataFrame
    :return: beta
    """
    X = np.insert(x.values, 0, [1]*len(x), axis=1)
    Y = y.values
    beta = LA.inv(X.T @ X) @ X.T @ Y
    return beta


# ================================= time series decomposition =============================
def rolling_avg(df, m):
    """
    helper function to calculate rolling mean base on odd and even order.
    :param df: pd.DataFrame
    :param m: int order
    :return: pd.Series
    """
    output = df.copy()
    # m is odd
    if m%2 != 0:
        k = (m-1) // 2
        output.iloc[:k] = np.nan
        output.iloc[-k:] = np.nan
        for i in range(k, len(df)-k):
            output.iloc[i] = np.sum(df.iloc[i-k:i+k+1]) / m
    # m is even
    else:
        k = m // 2
        output.iloc[:k] = np.nan
        output.iloc[-k + 1:] = np.nan
        for i in range(k, len(df) - k+1):
            output.iloc[i] = np.sum(df.iloc[i - k:i + k]) / m
    return output


def rolling_avg_folding(MA, m1, m2):
    """
    second time rolling average when m1 is even
    :param MA: pd.Series moving average output from first time
    :param m1: int order of first time
    :param m2: int order of second time
    :return: pd.Series
    """
    k = m1 // 2
    MA.iloc[k - 1:-k] = rolling_avg(MA.iloc[k:-k + 1], m2)
    MA.iloc[-k] = np.nan
    return MA


def rolling_avg_input(df):
    """
    calculate rolling average with input from keyboard
    :param df: pd.DataFrame
    :return: pd.Series
    """
    # catch invalid input
    while True:
        try:
            m = int(input('enter order of moving average: '))
            if not ((m == 1) or (m == 2)):
                break
            else:
                print('order 1 and 2 are not accepted')
        except:
            print('order must be an integer: ')
    # get rolling mean
    output = rolling_avg(df, m)
    # if order is even, calc rolling mean of rolling mean
    if m%2 == 0:
        while True:
            try:
                m2 = int(input('enter second order: '))
                if not (m2%2==1):
                    break
                else:
                    print('second order must be even')
            except:
                print('order must be an integer: ')

        output = rolling_avg_folding(output, m, m2)

    return output


def detrended(yt, Tt, decomp):
    """
    Detrend rolling average from data
    :param yt: original
    :param Tt: MA
    :param decomp: str Additive or Multiplicative
    :return:
    """
    if decomp == 'add':
        return yt - Tt
    elif decomp == 'mult':
        return yt / Tt


def STL_trend_strength(T, S, R):
    """
    Print strength of trend and seasonality
    :param T: pd.Series trend
    :param S: pd.Series seasonality
    :param R: pd.Series residual
    :return:
    """
    F = np.maximum(0, 1-(np.var(np.array(R))/np.var(np.array(T+R))))
    print(f'The strength of trend for this dataset is {100*F:.2f}%')

    Fs = np.maximum(0, 1 - (np.var(np.array(R)) / np.var(np.array(S + R))))
    print(f'The strength of seasonality for this dataset is {100 * Fs:.2f}%')


# ================================= ARMA model =============================
def balance_parameter(coefar, coefma):
    ar_len = len(coefar)
    ma_len = len(coefma)
    if ar_len > ma_len:
        coefma += [0] * (ar_len-ma_len)
    elif ma_len > ar_len:
        coefar += [0] * (ma_len-ar_len)
    else:
        return coefar, coefma
    return coefar, coefma


def ARMA_simulate(mean, std, n, denom, num):
    """
    Simulator of AR or MA
    :param mean: int mean of e
    :param std: int std of e
    :param n: int sample size
    :param denom: list coef of AR
    :param num: list coef of MA
    :return:
    """
    # np.random.seed(123)
    denom = np.r_[1, np.array(denom)]
    num = np.r_[1, np.array(num)]
    e = np.random.normal(mean, std, size=n)
    system = (num, denom, 1)
    _, y = signal.dlsim(system, e)

    return y


import statsmodels.api as sm
def ARMA_process(mean, std, n, arparams, maparams):
    """
    Simulator of AR or MA
    :param mean: int mean of e
    :param std: int std of e
    :param n: int sample size
    :param arparams: list coef of AR
    :param maparams: list coef of MA
    :return:
    """
    np.random.seed(123)
    ar = np.r_[1, arparams]
    ma = np.r_[1, maparams]
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    y_mean = mean * sum(ma) / sum(ar)
    y = arma_process.generate_sample(n, scale=std) + float(y_mean)

    return arma_process, y


def GPAC_table(ry, J, K):
    """
    Return GPAC table
    :param ry: pd.Series ACF
    :param J: int number of row in table
    :param K: int number of column in table
    :return: pd.DataFrame GPAC table
    """
    sol = np.zeros((J, K))
    for j in range(J):
        for k in range(1, K + 1):
            den = np.zeros((k, k))
            i = j
            for c in range(k):
                col = [ry[np.abs(s)] for s in range(i, i + k)]
                den[:, c] = col
                i -= 1

            num = den.copy()
            num[:, -1] = [ry[np.abs(s)] for s in range(j + 1, j + k + 1)]
            sol[j, k - 1] = np.linalg.det(num) / np.linalg.det(den)
    return pd.DataFrame(sol, columns=np.arange(1, K+1))


from scipy import signal
def ARMA_simulate_input():
    """
    Simulator of AR or MA with keyboard input
    :param mean: int mean of e
    :param std: int std of e
    :param n: int sample size
    :param denom: list coef of AR
    :param num: list coef of MA
    :return:
    """
    equation = "y(t)"
    n = int(input("Enter the number of data samples:"))
    m = float(input("Enter the mean of white noise:"))
    v = float(input("Enter the variance of the white noise:"))
    na = int(input("Enter AR order:"))
    nb = int(input("Enter MA order:"))
    coefar = np.zeros(max(na, nb))
    coefma = np.zeros(max(na, nb))
    for i in range(1, na + 1):
        coef = input(f'Enter a{i}:')
        coefar[i - 1] = float(coef)
        try:
            m = int(coef)
            equation += m * f"y(t-{i})"
        except:
            if float(coef) < 0:
                equation += f"{float(coef)}y(t-{i})"
            else:
                equation += f" + {float(coef)}y(t-{i})"

    equation += " = e(t)"
    for i in range(1, nb + 1):
        coef = input(f'Enter b{i}:')
        coefma[i - 1] = float(coef)
        try:
            m = int(coef)
            equation += m * f"y(t-{i})"
        except:
            if float(coef) < 0:
                equation += f"{float(coef)}y(t-{i})"
            else:
                equation += f"+{float(coef)}y(t-{i})"

    print(f"time difference equation: {equation}")

    coefar = np.r_[1, coefar]
    coefma = np.r_[1, coefma]
    y = ARMA_simulate(m, np.sqrt(v), n, coefar, coefma)
    return y


def ARMA_GPAC_table_input():
    """
    Generate and plot GPAC table from keyboard input
    :return:
    """
    n = int(input("Enter the number of data samples:"))
    m = float(input("Enter the mean of white noise:"))
    v = float(input("Enter the variance of the white noise:"))
    na = int(input("Enter AR order:"))
    nb = int(input("Enter MA order:"))
    coefar = np.zeros(max(na, nb))
    coefma = np.zeros(max(na, nb))
    for i in range(1, na + 1):
        coef = input(f'Enter a{i}:')
        coefar[i - 1] = float(coef)

    for i in range(1, nb + 1):
        coef = input(f'Enter b{i}:')
        coefma[i - 1] = float(coef)
    process, y = ARMA_process(m, np.sqrt(v), n, coefar, coefma)
    ry = process.acf(15)
    result = GPAC_table(ry, 7, 7)
    return result


import statsmodels.tsa.api
# import statsmodels
def SARIMA_parameter_estimation(y, na, nb, d=0, Na=0, Nb=0, D=0, S=0):
    """
    Return SARIMA model. Note the coefficients of model in summary is opposite.
    :param y: pd.Series time series data
    :param na: int non-seasonal AR order
    :param nb: int non-seasonal MA order
    :param d: int non-seasonal differencing order
    :param Na: int seasonal AR order
    :param Nb: int seasonal MA order
    :param D: int seasonal differencing order
    :param S: int seasonality
    :return:
    """
    model = statsmodels.tsa.api.SARIMAX(y, order=(na, d, nb),
                                        seasonal_order=(Na, D, Nb, S)).fit()
    # for i in range(na):
    #     print(f'The AR coefficient a{i+1} is {-model.arparams[i]}')
    # for i in range(nb):
    #     print(f'The MA coefficient a{i+1} is {model.maparams[i]}')
    # for i in range(Na):
    #     print(f'The AR coefficient a{i+S} is {-model.arparams[i]}')
    # for i in range(Nb):
    #     print(f'The MA coefficient a{i+S} is {model.maparams[i]}')
    return model


def zero_pole(ar_coef, ma_coef):
    """

    :param ar_coef: np.array coefficients of AR
    :param ma_coef: np.array coefficients of MA
    :return:
    """
    print("roots of AR: ",np.roots(ar_coef))
    print("roots of MA: ",np.roots(ma_coef))


# ================================= other =============================
# Cramer algorithm
# reference: https://github.com/guiriosoficial/CramersRule
# Laplace algorithm
def laplace(matrix, val=1):
    # Set the matrix length
    n = len(matrix)

    # Return Det if Matrix length is 1
    if n == 1:
        return val * matrix[0][0]
    else:
        sign = -1
        det = 0
        for i in range(n):
            mtx = []
            for j in range(1, n):
                buff = []
                for k in range(n):
                    if k != i:
                        buff.append(matrix[j][k])
                mtx.append(buff)
            sign *= -1
            det += val * laplace(mtx, sign * matrix[0][i])
        return det


def cramer(matrix, results, order):
    # Calc and Show Main Matrix Determinant
    main_det = laplace(matrix)
    print(f'\nMain matrix determinant: {main_det}')

    # Build New Matrix with Substitutions
    if main_det != 0:
        resolution = []
        for r in range(order):
            matrix_sub = []
            for i in range(order):
                matrix_sub.append([])
                for j in range(order):
                    if j == r:
                        matrix_sub[i].append(results[i])
                    else:
                        matrix_sub[i].append(matrix[i][j])

            # Show Actual Matrix with Substitution
            print(f'\nMatrix with replacement at COLUMN {r + 1}:')
            for line in matrix_sub:
                for val in line:
                    print(f'{val:^8}', end=' ')
                print()

            # Calc Determinant with Substitution
            sub_det = laplace(matrix_sub)
            print(f'Determinant of this matrix: {sub_det}')

            # Calc and Save Final Resolution
            resolution.append(sub_det / main_det)

    # Return Resolution to Display in Main
        return resolution

    # Display Resolution if Main Det is 0
    else:
        return 0