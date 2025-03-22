import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.special import gammaincc, gamma, erfc
from scipy.stats import expon, lognorm, uniform, weibull_min, ttest_ind, mannwhitneyu, kstest, f, pareto, invgamma

def get_intervals_from_df(df):
    """
    Вычисляет интервалы между соседними столбцами DataFrame.
    """
    df_intervals = df.diff(axis=1).iloc[:, 1:]
    df_intervals.columns = range(df_intervals.shape[1])
    return df_intervals
    

def get_cdf_from_intervals(df_intervals):
    """
    Создает DataFrame с эмпирическими интервалами, их частотами и эмпирической функцией распределения (CDF).
    """
    emperical_intervals = np.sort(df_intervals.iloc[0].values) 
    unique_vals, counts = np.unique(emperical_intervals, return_counts=True) 

    total_intervals = np.sum(counts)

    df_result = pd.DataFrame({
        'emperical': unique_vals,
        'value_counts': counts,
        'prob': counts / total_intervals 
    })

    df_result['cdf_emp'] = np.cumsum(df_result['prob'])

    return df_result


def calculate_statistics(df_intervals):
    n_observations = df_intervals.shape[0]
    stat_M = np.zeros((n_observations, 10))

    # Векторизованные вычисления для основных статистик
    stat_M[:, 0] = np.mean(df_intervals, axis=1)
    stat_M[:, 1] = np.var(df_intervals, axis=1)
    stat_M[:, 2] = np.std(df_intervals, axis=1)
    stat_M[:, 3] = stat_M[:, 2] / stat_M[:, 0]  # Коэффициент вариации

    # Квантили. Избегаем повторного вычисления квантилей
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    quantiles_values = np.quantile(df_intervals, quantiles, axis=1)
    stat_M[:, 4:10] = quantiles_values.T # Транспонируем, чтобы правильно разместить

    return stat_M


def ecdf_with_all_x(data1, data2):
    all_data = np.concatenate((data1, data2))
    all_x = np.sort(np.unique(all_data))
    y1 = np.array([np.mean(data1 <= x) for x in all_x])
    y2 = np.array([np.mean(data2 <= x) for x in all_x])
    y = np.vstack((y1, y2))
    return all_x, y

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

def distribution_function_levi(x, mu, c):
    mask = x > mu
    result = np.zeros_like(x, dtype=float)
    result[mask] = erfc(np.sqrt(c / (2 * (x[mask] - mu))))
    return result

def distribution_function_phisher(x, dfn, dfd):
    return f.cdf(x, dfn, dfd)

def distribution_function_pareto(x, b, scale):
    return pareto.cdf(x, b, scale=scale)

def distribution_function_invgamma(x, a, scale):
    return invgamma.cdf(x, a, scale=scale)

def distribution_function_lomax(x, alpha, lmbd):
    return 1 - (1 + x / lmbd) ** (-alpha)

def distribution_function_burr(x, c, k, lmbd):
    """
    Кумулятивная функция распределения Burr XII с параметрами:
    c (масштаб), k (параметр формы), lam (параметр хвоста).
    Для x < 0 считается, что F(x)=0.
    """
    cdf = np.zeros_like(x)
    valid = x >= 0
    cdf[valid] =  1 - (1 + (x[valid] / lmbd) ** c) ** (-k)
    return cdf

def distribution_function_phrechet(x, alpha, s, m):
    """
    Кумулятивная функция распределения Фреше с 3 параметрами.
    x - наблюдения,
    alpha - параметр формы,
    s - параметр масштаба (s > 0),
    m - параметр сдвига (локация).
    Для x < m считается, что F(x) = 0.
    """
    cdf = np.zeros_like(x)
    valid = x > m
    cdf[valid] = np.exp(-(((x[valid] - m) / s) ** (-alpha)))
    return cdf

def beta_func(z1, z2):
    return (gamma(z1)*gamma(z2))/(gamma(z1+z2))

# def distribution_function(x, dist_type, *params):
#     if dist_type == 'exp':
#         lmbd = params[0]
#         return expon.cdf(x, scale=1 / lmbd)
    
#     elif dist_type == 'gamma':
#         alpha, beta = params
#         return gammaincc(alpha, x / beta)
    
#     elif dist_type == 'hexp':
#         p, lmbd1, lmbd2 = params
#         return p * expon.cdf(x, scale=1 / lmbd1) + (1 - p) * expon.cdf(x, scale=1 / lmbd2)
    
#     elif dist_type == 'lognorm':
#         mu, sigma = params
#         return lognorm.cdf(x, sigma, scale=np.exp(mu))
    
#     elif dist_type == 'uniform':
#         a, b = params
#         return uniform.cdf(x, a, b - a)
    
#     elif dist_type == 'weibull':
#         theta, k = params
#         return weibull_min.cdf(x, k, loc=0, scale=theta)
#     else:
#         raise ValueError(f"Неизвестный тип распределения: {dist_type}")
    
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


def kolmogorov_plot(data_source, theor_cdf, file_name, process_type, x_lim_min=0,
                             xticks_flag=False, xticks_arr=None, xticks_arr_change=None,
                             mmpp_mode=False):
    main_path = os.getcwd().replace(os.sep, '/')
    colors = ['color', 'grey']

    if mmpp_mode:
        x, cdf1, cdf2 = data_source, theor_cdf[0], theor_cdf[1] 
        k = np.argmax(np.abs(cdf1 - cdf2))
        ks_stat = np.abs(cdf1[k] - cdf2[k])
        y = (cdf1[k] + cdf2[k]) / 2
    else:
        df = data_source.copy()  # Avoid modifying the original DataFrame
        df['cdf_theor'] = theor_cdf
        k = np.argmax(np.abs(df['cdf_emp'] - df['cdf_theor']))
        ks_stat = np.abs(df['cdf_emp'][k] - df['cdf_theor'][k])
        y = (df['cdf_emp'][k] + df['cdf_theor'][k]) / 2
        x = df['emperical']

    for color in colors:
        fig = plt.figure(figsize=(5, 3))

        if mmpp_mode:
            if color == 'grey':
                plt.plot(x, cdf1, label='Эмпирическая ФРВ', color='grey')
                plt.plot(x, cdf2, label='Теоретическая ФРВ', color='k', linestyle=(0, (5, 5)))
            else:
                plt.plot(x, cdf1, label='Эмпирическая ФРВ')
                plt.plot(x, cdf2, label='Теоретическая ФРВ', linestyle=(0, (5, 5)))
            plt.errorbar(x=x[k], y=y, yerr=ks_stat/2, color='k', capsize=4, mew=2, label=r"$\Delta$ =" + f"{ks_stat:.3f}")
        else:
            if color == 'grey':
                plt.plot('emperical', 'cdf_emp', data=df, label='Эмпирическая ФРВ', color='grey')
                plt.plot('emperical', 'cdf_theor', data=df, label='Теоретическая ФРВ', color='k', linestyle=(0, (5, 5)))
            else:
                plt.plot('emperical', 'cdf_emp', data=df, label='Эмпирическая ФРВ')
                plt.plot('emperical', 'cdf_theor', data=df, label='Теоретическая ФРВ', linestyle=(0, (5, 5)))
            plt.errorbar(x=df['emperical'][k], y=y, yerr=ks_stat/2, color='k', capsize=4, mew=2, label=r"$\Delta$ =" + f"{ks_stat:.3f}")

        if xticks_flag:
            plt.xticks(xticks_arr, xticks_arr_change)

        plt.legend(loc=4)
        plt.grid(alpha=0.4, color='k')
        plt.xlim(x_lim_min, max(x)) 
        plt.ylim(-0.01, 1.008)
        plt.xlabel('Интервалы между моментами времени \nнаступления событий', fontsize=10)
        plt.ylabel('Вероятность', fontsize=10)

        save_path = f'{main_path}/pic/cdf/{color}/{file_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        fig.savefig(f'{save_path}/{process_type}.pdf', bbox_inches="tight")

def hist_plot_test(data, file_name, bins_size=100, figsize=(10,3)):
    main_path = os.getcwd().replace(os.sep, '/')
    colors = ['color', 'grey']

    for color in colors:
        fig = plt.figure(figsize=figsize)
        if color == 'grey':
            plt.hist(data, bins=bins_size, color="grey", edgecolor='black', density=False) 
        else:
            plt.hist(data, bins=bins_size, edgecolor='black', density=False)
        
        plt.grid(alpha=0.1, color='k')
        plt.xlim(min(data), max(data))
        plt.xlabel('Длина интервалов', fontsize=10)
        plt.ylabel('Количество интервалов', fontsize=10)
        
        save_path = f'{main_path}/pic/hist/{color}/{file_name}'
        
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            
        fig.savefig(f'{save_path}/_bins_size_{bins_size}.pdf', bbox_inches="tight")