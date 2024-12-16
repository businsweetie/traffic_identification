import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaincc
from scipy.stats import expon, lognorm, uniform, weibull_min, ttest_ind, mannwhitneyu, kstest

def get_intevals_from_df(df):
    df_intervals = []
    for i in range(1, df.shape[1]):
        df_intervals.append(df[i] - df[i-1])
    return pd.DataFrame(np.array(df_intervals).T.tolist())

def get_cdf_from_intervals(df_intervals):
    emperical_intervals = df_intervals.iloc[0].tolist()
    emperical_intervals = sorted(emperical_intervals)

    df_result = pd.DataFrame()
    df_result['emperical'] = pd.Series(emperical_intervals).unique()
    df_result['value_counts'] = pd.Series(emperical_intervals).value_counts(sort=False).to_list()
    df_result['prob'] = df_result['value_counts'] / 2999
    distrib = [0., df_result['prob'].loc[0]]

    for i in range(2, len(df_result)):
        distrib.append(df_result['prob'].loc[i] + distrib[i-1])
    df_result['cdf_emp'] = distrib

    return df_result

def distribution_function_exp(x, lmbd):
    return expon.cdf(x, scale = 1 / lmbd)

def distribution_function_gamma(x, alpha, beta):
    return gammaincc(alpha, x / beta)

def distribution_function_hexp(x, p, lmbd1, lmbd2):
    f = p * expon.cdf(x, scale = 1 / lmbd1) + (1 - p) * expon.cdf(x, scale = 1 / lmbd2)
    return f

def distribution_function_lognorm(x, mu, sigma):
    return lognorm.cdf(x, sigma, scale=np.exp(mu))

def distribution_function_uniform(x, a, b):
    return uniform.cdf(x, a, b-a)

def distribution_function_weibull(x, theta, k):
    return weibull_min.cdf(x, k, loc=0, scale=theta)

def kolmogorov(emp_function, distribution_function):
    abs_list = abs(emp_function - distribution_function)
    return abs_list.max()

def sum_diag_matrix(matrix, matrix_size):
    for i in range(matrix_size):
        for j in range(matrix_size):
            if i==j:
                matrix[i][j] = -(sum(matrix[i]) - matrix[i][j])
    return matrix

def calculation_pi(q_matrix_for_solve, matrix_size):
    v = np.zeros(matrix_size)
    v[-1] = 1
    pi = np.linalg.solve(q_matrix_for_solve, v)
    return pi

def choose_time_work(q_matrix, pi, T, K, k):
    p = 0
    rnd = random.random()

    for i in range (len(pi)):
        if rnd >= (p + pi[i]):
            p += pi[i]
        else:
            k = i
    rnd_list = []
    for v in range(K):
        if v != k:
            rnd = random.expovariate(q_matrix[k][v])
            rnd_list.append(rnd)
        if v == k:
            rnd = 10e100
            rnd_list.append(rnd)
    temp = min(rnd_list)
    k = rnd_list.index(min(rnd_list))
    T += temp
    return k, T

def cycle(lmbd_matrix, T, k, s, S, t0, time_list):
    while s < S:
        thau = random.expovariate(lmbd_matrix[k])
        if t0 + thau < T:
            t0 += thau
            s += 1
            time_list.append(t0)
        else:
            return s, T, time_list
    return s, T, time_list

def get_x_pred(test_df, q_matrix, lmbd_matrix, pi):
    X_pred = []
    p = 0
    K = test_df['k_size'][0]
    v = 0
    S = 3000
    T = 0
    s = 0
    t0 = 0
    time_list=[]
    k = 0

    while s < S:
        k, T = choose_time_work(q_matrix, pi, T, K, k)
        s, t0, time_list = cycle(lmbd_matrix, T, k, s, S, t0, time_list)

    X_pred.append(time_list)
    return pd.DataFrame(X_pred)

def get_mmpp_param(y_pred, test_for_mmmp):
    k_size = test_for_mmmp['k_size'][0]
    test_list = y_pred[0, : k_size ** 2]
    l = 0
    q_matrix = []
    for _ in range(0, k_size):
        sub = []
        for _ in range(0, k_size):
            sub.append(test_list[l])
            l += 1
        q_matrix.append(sub)

    q_matrix = sum_diag_matrix(q_matrix, k_size)

    q_matrix_for_solve = q_matrix.copy()
    q_matrix_for_solve[k_size-1] = np.ones(k_size)

    pi = calculation_pi(q_matrix_for_solve, k_size)

    lmbd_matrix = y_pred[0, 16:16 + k_size]
    return q_matrix, lmbd_matrix, pi

def mmpp_test_param(X_pred, test_values, dict_out):

    #pred_values = X_pred.values[0]
    #test_values = test_values.values[0]

    dict_out[' '] = ' '
    stat, p_value = kstest(test_values, X_pred)
    dict_out['Расстояние Колмогорова:'] = "{:.3f}".format(stat)
    dict_out['  '] = ' '

    stat, p_value = ttest_ind(test_values, X_pred)
    dict_out[f"t-критерий Стьюдента: statistic={stat:.3f}, p-value={p_value:.3f}"] = ' '
    if p_value < 0.05:
        dict_out['Вывод t-критерий Стьюдента: Отклоняем нулевую гипотезу (существенная разница между двумя выборками)'] = ' '
    else:
        dict_out['Вывод t-критерий Стьюдента: Не отвергаем нулевую гипотезу (нет существенной разницы между двумя выборками,\
            \n об отсутствии различий в средних значениях между выборками)'] = ' '
    dict_out['   '] = ' '
        
    stat, p_value = mannwhitneyu(test_values, X_pred)
    dict_out[f"U-критерий Манна — Уитни: statistic={stat:.3f}, p-value={p_value:.3f}"] = ' '
    if p_value < 0.05:
        dict_out['Вывод U-критерий Манна — Уитни: Отклоняем нулевую гипотезу (существенная разница между двумя выборками)'] = ' '
    else:
        dict_out['Вывод U-критерий Манна — Уитни: Не отвергаем нулевую гипотезу (нет существенной разницы между двумя выборками,\
            \n об отсутствии различий в медианах между выборками)'] = ' '
    dict_out['     '] = ' '
        
    return dict_out

def kolmogorov_plot(df, theor_cdf, file_name, process_type, x_lim_min=0, xticks_flag=None, xticks_arr=None, xticks_arr_change=None):
    df['cdf_theor'] = theor_cdf
    k = np.argmax(np.abs(df['cdf_emp'] - df['cdf_theor']))
    ks_stat = np.abs(df['cdf_emp'][k] - df['cdf_theor'][k])
    y = (df['cdf_emp'][k] + df['cdf_theor'][k]) / 2
    
    fig = plt.figure(figsize=(5, 3))
    plt.plot('emperical', 'cdf_emp', data=df, label='Эмпирическая ФРВ')
    plt.plot('emperical', 'cdf_theor', data=df, label='Теоретическая ФРВ', linestyle=(0, (5, 5)))
    plt.errorbar(x=df['emperical'][k], y=y, yerr=ks_stat/2, color='k',
        capsize=4, mew=2, label=r"$\Delta$ =" + f"{ks_stat:.3f}")
    if xticks_flag != None:
        plt.xticks(xticks_arr, xticks_arr_change)
    #plt.xticks([0.0, 50000, 100000, 150000, 200000], ['0', r'$1*10^{-3}$', r'$1,75*10^{-3}$'])
    #plt.xticks([0.0, 50000, 100000, 150000, 200000], [0, 50000, 100000, 150000, 200000])
    #plt.xticks([0.0, 0.001, 0.00175], [0.0, 0.001, 0.00175])
    #plt.xticks([0.0, 0.0002, 0.0004, 0.0006, 0.0008], [0.0, 0.0002, 0.0004, 0.0006, 0.0008])
    #plt.xticks([0.0, 0.0005, 0.001, 0.0015, 0.002], [0.0, 0.0005, 0.001, 0.0015, 0.002])
    plt.legend(loc=4);
    plt.grid(alpha=0.4, color='k')
    plt.xlim(x_lim_min, max(df['emperical']))
    plt.ylim(-0.01, 1.008)
    plt.xlabel('Интервалы между моментами времени \nнаступления событий', fontsize=10)
    plt.ylabel('Вероятность', fontsize=10)
    if os.path.exists('C:/Users/Dashulya/YandexDisk/Dasha_dis/pic/cdf/color/'+file_name) is not True:
        os.mkdir('C:/Users/Dashulya/YandexDisk/Dasha_dis/pic/cdf/color/'+file_name)
    fig.savefig('C:/Users/Dashulya/YandexDisk/Dasha_dis/pic/cdf/color/'+file_name+'/'+process_type+'.pdf', bbox_inches="tight");
    
    fig = plt.figure(figsize=(5, 3))
    plt.plot('emperical', 'cdf_emp', data=df, label='Эмпирическая ФРВ', color='grey')
    plt.plot('emperical', 'cdf_theor', data=df, label='Теоретическая ФРВ', color='k', linestyle=(0, (5, 5)))
    plt.errorbar(x=df['emperical'][k], y=y, yerr=ks_stat/2, color='k',
        capsize=4, mew=2, label=r"$\Delta$ =" + f"{ks_stat:.3f}")
    if xticks_flag != None:
        plt.xticks(xticks_arr, xticks_arr_change)
    #plt.xticks([0.0, 50000, 100000, 150000, 200000], ['0', r'$1*10^{-3}$', r'$1,75*10^{-3}$'])
    #plt.xticks([0.0, 50000, 100000, 150000, 200000], [0, 50000, 100000, 150000, 200000])
    #plt.xticks([0.0, 0.001, 0.00175], [0.0, 0.001, 0.00175])
    #plt.xticks([0.0, 0.0002, 0.0004, 0.0006, 0.0008], [0.0, 0.0002, 0.0004, 0.0006, 0.0008])
    #plt.xticks([0.0, 0.0005, 0.001, 0.0015, 0.002], [0.0, 0.0005, 0.001, 0.0015, 0.002])
    plt.legend(loc=4);
    plt.grid(alpha=0.4, color='k')
    plt.xlim(x_lim_min, max(df['emperical']))
    plt.ylim(-0.01, 1.008)
    plt.xlabel('Интервалы между моментами времени \nнаступления событий', fontsize=10)
    plt.ylabel('Вероятность', fontsize=10)
    if os.path.exists('C:/Users/Dashulya/YandexDisk/Dasha_dis/pic/cdf/grey/'+file_name) is not True:
        os.mkdir('C:/Users/Dashulya/YandexDisk/Dasha_dis/pic/cdf/grey/'+file_name)
    fig.savefig('C:/Users/Dashulya/YandexDisk/Dasha_dis/pic/cdf/grey/'+file_name+'/'+process_type+'.pdf', bbox_inches="tight");

def hist_plot_test(data, file_name, bins_size=100, figsize=(10,3)):
    fig = plt.figure(figsize=figsize)
    plt.hist(data, bins=bins_size, edgecolor='black')
    plt.grid(alpha=0.1, color='k')
    plt.xlim(min(data), max(data))
    plt.xlabel('Время', fontsize=10)
    plt.ylabel('Вероятность', fontsize=10)
    fig.savefig('pic/hist/color/' + file_name + "_bins_size_" + str(bins_size) + '.pdf', bbox_inches="tight");

    fig = plt.figure(figsize=figsize)
    plt.hist(data, bins=bins_size, color="grey", edgecolor='black')
    plt.grid(alpha=0.1, color='k')
    plt.xlim(min(data), max(data))
    plt.xlabel('Время', fontsize=10)
    plt.ylabel('Вероятность', fontsize=10)
    fig.savefig('pic/hist/grey/'+ file_name + "_bins_size_" + str(bins_size) + '.pdf', bbox_inches="tight");