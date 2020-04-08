from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

from GeneralRegression.GeneralRegression import GenericRegressor


# Fourier base generator
def fourier(X, n=1, l=1.):
    points = []
    for x in X:
        point = []
        for deg in range(n + 1):
            if deg == 0:
                point.append(1.)
            else:
                point.append(np.sin(deg * x[0] / l))
                point.append(np.cos(deg * x[0] / l))
        point.append(x[0])
        point.append(x[0] ** 2)
        points.append(np.array(point))
    return np.array(points)


# Fourier base generator
def mixed(X, p_d=3, f_d=1, l=1., e_d=2):
    points = []
    for x in X:
        point = []
        point.append(1.)
        for deg in range(1, f_d + 1):
            point.append(np.sin(deg * x[0] / l))
            point.append(np.cos(deg * x[0] / l))
        for deg in range(1, p_d + 1):
            point.append(x[0] ** deg)
        for deg in range(e_d + 1):
            point.append((x[0] ** deg) * np.exp(x[0]))
            point.append((x[0] ** deg) * np.exp(-x[0]))
        points.append(np.array(point))
    return np.array(points)


def aggregate(df, method='monthly'):
    from pandas import DataFrame
    new_df = DataFrame()
    min_date = min(df['date'])
    max_date = max(df['date'])
    cur_date = min_date
    clmns = list(df.columns)
    clmns.pop(clmns.index('date'))
    idx = 0.
    stp = .1
    while cur_date < max_date:
        if method == 'monthly':
            nxt_date = cur_date + timedelta(days=31)
            nxt_date = nxt_date - timedelta(days=(nxt_date.day - 1))
        t_df = df[(df['date'] >= cur_date) & (df['date'] < nxt_date)].mean()
        dct = {clmn: [t_df[clmn]] for clmn in clmns}
        dct['date'] = [cur_date]
        dct['t'] = idx
        row = DataFrame(dct)
        new_df = new_df.append(row, ignore_index=True)
        cur_date = nxt_date
        idx += stp
    return new_df


df = pd.read_csv("../data/RemandCnt.csv", parse_dates=['date'])

def plot_population(df, center='Regina', yrs=2):
    c_df = aggregate(df)
    x = c_df['t'].values
    y_s = c_df['SEN_%s'%center].values
    y_r = c_df['REM_%s'%center].values
    x_t = x.reshape(-1, 1)
    y_ts = y_s.reshape(-1, 1)
    y_tr = y_r.reshape(-1, 1)
    x_f = np.linspace(10., max(x)+yrs*1.2, 200)
    model1 = GenericRegressor(mixed, regressor=BayesianRidge, **dict(p_d=2, f_d=2, l=3., e_d=0))
    model1.fit(x_t, y_ts)
    plt.scatter(x, y_ts, color='navy', s=5, marker='o', label="Sentenced actual counts")
    y_ps = model1.predict(x_f.reshape(-1, 1))
    plt.plot(x_f, y_ps, color='salmon', linewidth=1, label="Sentenced")
    plt.fill_between(x_f,
                         y_ps - model1.ci_band,
                         y_ps + model1.ci_band,
                         color='salmon',
                         alpha=0.1)
    model2 = GenericRegressor(mixed, regressor=BayesianRidge, **dict(p_d=2, f_d=2, l=3., e_d=0))
    model2.fit(x_t, y_tr)
    plt.scatter(x, y_tr, color='navy', s=5, marker='o', label="Remand  actual counts")
    y_pr = model2.predict(x_f.reshape(-1, 1))
    plt.plot(x_f, y_pr, color='Green', linewidth=1, label="Remand")
    plt.fill_between(x_f,
                         y_pr - model2.ci_band,
                         y_pr + model2.ci_band,
                         color='green',
                         alpha=0.1)
    model3 = GenericRegressor(mixed, regressor=BayesianRidge, **dict(p_d=2, f_d=2, l=3., e_d=0))
    model3.fit(x_t, (y_tr + y_ts))
    plt.scatter(x, (y_tr + y_ts), color='navy', s=5, marker='o', label="Total custody counts")
    y_prs = model3.predict(x_f.reshape(-1, 1))
    plt.plot(x_f, y_prs, color='Green', linewidth=1, label="Total")
    plt.fill_between(x_f,
                         y_prs - model3.ci_band,
                         y_prs + model3.ci_band,
                         color='orange',
                         alpha=0.1)

    plt.legend(loc=2)
    plt.show()

plot_population(df)