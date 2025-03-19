import os
import math
import pandas as pd
import numpy as np
from scipy.special import gamma

from functions import *
from model_importer import *
from moment_method import *
from report_builder import write_txt, write_matrix, clean_txt

    
def start():
    main_path = os.getcwd().replace(os.sep, '/')

    file_name = 'test'
    
    df_for_mmmp = df_moments = pd.read_csv(main_path + "/" + file_name + '.csv', sep=';', header=None)
    df_intervals = get_intervals_from_df(df_moments)
    df_stat = calculate_statistics(df_intervals)

    print(df_moments.shape)
    print(df_intervals.shape)
    
    lmbd_emp = df_intervals.shape[1] / np.sum(df_intervals, axis=1)
    df_result = get_cdf_from_intervals(df_intervals)
    hist_plot_test(df_intervals.iloc[0].tolist(), file_name, 100)
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    lmbd_pois = pois_model.predict(df_intervals)[0]
    # class_process_prob = classification_model.predict_proba(df_intervals)
    # recurr_class_process_prob = recurr_classification_model.predict_proba(df_moments)
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    out_dict_pois = {}
    clean_txt(file_name, 'pois')
    out_dict_pois['ПУАССОНОВСКИЙ ПОТОК:'] = ' ' #"{:.2f}%".format(class_process_prob[0][0] * 100)
    if lmbd_pois > 0:
        out_dict_pois['Интенсивность:'] = "{:.3f}".format(lmbd_pois)
        out_dict_pois['Интенсивность эмпирическая:'] = "{:.3f}".format(lmbd_emp[0])
        theor_cdf = [distribution_function_exp(df_result['emperical'][i], lmbd_pois) for i in range(len(df_result['emperical']))]
        out_dict_pois['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_pois['Значение интенсивности некорректно'] = ''
    kolmogorov_plot(df_result, theor_cdf, file_name, 'pois', x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'pois', out_dict_pois)

    # #----------------------------------------------------------------------------------------------------------------------------------
    
    # out_dict_mmpp = {}
    # clean_txt(file_name, 'mmpp')
    # out_dict_mmpp['MMPP ПОТОК:'] = ' ' #"{:.2f}%".format(class_process_prob[0][1] * 100)
    # out_dict_mmpp['------------------------------------------------------------------------------'] = ' '
    # #k_size_prob = mmpp_classification_model.predict_proba(df_moments)

    # out_dict_mmpp['Количество состояний 2:'] = ' ' #"{:.2f}%".format(k_size_prob[0][0] * 100)
    # out_dict_mmpp['  '] = ' '
    # df_for_mmmp['k_size'] = 2
    # y_pred = mmpp_regression_model.predict(df_for_mmmp)
    # q_matrix, lmbd_matrix, pi = get_mmpp_param(y_pred, df_for_mmmp)
    # q_matrix_for_print = np.asmatrix(q_matrix)
    # lmbd_matrix_for_print = np.asmatrix(lmbd_matrix)
    # out_dict_mmpp["Матрица инфинитезимальных характеристик, Q"] = ' '
    # write_txt(file_name, 'mmpp', out_dict_mmpp)
    # write_matrix(file_name, 'mmpp', q_matrix_for_print)
    # out_dict_mmpp = {}
    # out_dict_mmpp['  '] = ' '
    # out_dict_mmpp['Матрица условных интенсивностей, Lambda'] = ' '
    # write_txt(file_name, 'mmpp', out_dict_mmpp)
    # write_matrix(file_name, 'mmpp', lmbd_matrix_for_print)
    # out_dict_mmpp = {}
    # X_pred = get_x_pred(df_for_mmmp, q_matrix, lmbd_matrix, pi)
    # X_pred = get_intervals_from_df(X_pred)
    # df_mmpp = get_cdf_from_intervals(X_pred)
    # q_matrix[-1] = [1, 1]
    # b = np.array([0, 1])
    # x = np.linalg.solve(q_matrix, b)
    # e = np.ones(2)
    # lmbd_theor = sum(x * lmbd_matrix * e)
    # out_dict_mmpp['  '] = ' '
    # out_dict_mmpp['Интенсивность эмпирическая:'] = "{:.3f}".format(lmbd_emp[0])
    # out_dict_mmpp['Интенсивность теоретическая:'] = "{:.3f}".format(lmbd_theor)
    # x, cdf = ecdf_with_all_x(df_mmpp['emperical'], df_result['emperical'])
    # out_dict_mmpp = mmpp_test_param(df_mmpp['emperical'], df_result['emperical'], out_dict_mmpp)
    # out_dict_mmpp['------------------------------------------------------------------------------'] = ' '
    # kolmogorov_plot(x, cdf, file_name, "mmpp_2", mmpp_mode=True)
    # write_txt(file_name, 'mmpp', out_dict_mmpp)
    
    # out_dict_mmpp = {}
    # out_dict_mmpp['Количество состояний 3:'] = ' ' #"{:.2f}%".format(k_size_prob[0][1] * 100)
    # out_dict_mmpp['  '] = ' '
    # df_for_mmmp['k_size'] = 3
    # y_pred = mmpp_regression_model.predict(df_for_mmmp)
    # q_matrix, lmbd_matrix, pi = get_mmpp_param(y_pred, df_for_mmmp)
    # q_matrix_for_print = np.asmatrix(q_matrix)
    # lmbd_matrix_for_print = np.asmatrix(lmbd_matrix)
    # out_dict_mmpp["Матрица инфинитезимальных характеристик, Q"] = ' '
    # write_txt(file_name, 'mmpp', out_dict_mmpp)
    # write_matrix(file_name, 'mmpp', q_matrix_for_print)
    # out_dict_mmpp = {}
    # out_dict_mmpp['  '] = ' '
    # out_dict_mmpp['Матрица условных интенсивностей, Lambda'] = ' '
    # write_txt(file_name, 'mmpp', out_dict_mmpp)
    # write_matrix(file_name, 'mmpp', lmbd_matrix_for_print)
    # out_dict_mmpp = {}
    # X_pred = get_x_pred(df_for_mmmp, q_matrix, lmbd_matrix, pi)
    # X_pred = get_intervals_from_df(X_pred)
    # df_mmpp = get_cdf_from_intervals(X_pred)
    # q_matrix[-1] = [1, 1, 1]
    # b = np.array([0, 0, 1])
    # x = np.linalg.solve(q_matrix, b)
    # e = np.ones(3)
    # lmbd_theor = sum(x * lmbd_matrix * e)
    # out_dict_mmpp['  '] = ' '
    # out_dict_mmpp['Интенсивность эмпирическая:'] = "{:.3f}".format(lmbd_emp[0])
    # out_dict_mmpp['Интенсивность теоретическая:'] = "{:.3f}".format(lmbd_theor)
    # x, cdf = ecdf_with_all_x(df_mmpp['emperical'], df_result['emperical'])
    # out_dict_mmpp = mmpp_test_param(df_mmpp['emperical'], df_result['emperical'], out_dict_mmpp)
    # out_dict_mmpp['------------------------------------------------------------------------------'] = ' '
    # kolmogorov_plot(x, cdf, file_name, "mmpp_3", mmpp_mode=True)
    # write_txt(file_name, 'mmpp', out_dict_mmpp)
    
    # out_dict_mmpp = {}
    # out_dict_mmpp['Количество состояний 4:'] = ' ' #"{:.2f}%".format(k_size_prob[0][2] * 100)
    # out_dict_mmpp['  '] = ' '
    # df_for_mmmp['k_size'] = 4
    # y_pred = mmpp_regression_model.predict(df_for_mmmp)
    # q_matrix, lmbd_matrix, pi = get_mmpp_param(y_pred, df_for_mmmp)
    # q_matrix_for_print = np.asmatrix(q_matrix)
    # lmbd_matrix_for_print = np.asmatrix(lmbd_matrix)
    # out_dict_mmpp["Матрица инфинитезимальных характеристик, Q"] = ' '
    # write_txt(file_name, 'mmpp', out_dict_mmpp)
    # write_matrix(file_name, 'mmpp', q_matrix_for_print)
    # out_dict_mmpp = {}
    # out_dict_mmpp['  '] = ' '
    # out_dict_mmpp['Матрица условных интенсивностей, Lambda'] = ' '
    # write_txt(file_name, 'mmpp', out_dict_mmpp)
    # write_matrix(file_name, 'mmpp', lmbd_matrix_for_print)
    # out_dict_mmpp = {}
    # X_pred = get_x_pred(df_for_mmmp, q_matrix, lmbd_matrix, pi)
    # X_pred = get_intervals_from_df(X_pred)
    # df_mmpp = get_cdf_from_intervals(X_pred)
    # q_matrix[-1] = [1, 1, 1, 1]
    # b = np.array([0, 0, 0, 1])
    # x = np.linalg.solve(q_matrix, b)
    # e = np.ones(4)
    # lmbd_theor = sum(x * lmbd_matrix * e)
    # out_dict_mmpp['  '] = ' '
    # out_dict_mmpp['Интенсивность эмпирическая:'] = "{:.3f}".format(lmbd_emp[0])
    # out_dict_mmpp['Интенсивность теоретическая:'] = "{:.3f}".format(lmbd_theor)
    # x, cdf = ecdf_with_all_x(df_mmpp['emperical'], df_result['emperical'])
    # out_dict_mmpp = mmpp_test_param(df_mmpp['emperical'], df_result['emperical'], out_dict_mmpp)
    # kolmogorov_plot(x, cdf, file_name, "mmpp_4", mmpp_mode=True)
    # write_txt(file_name, 'mmpp', out_dict_mmpp)

    # #----------------------------------------------------------------------------------------------------------------------------------

    out_dict_recurr = {}
    clean_txt(file_name, 'recurr')
    out_dict_recurr["РЕКУРРЕНТНЫЙ ПОТОК:"] = ' ' #"{:.2f}%".format(class_process_prob[0][2] * 100)
    out_dict_recurr['  '] = ' '
    out_dict_recurr['РАСПРЕДЕЛЕНИЕ ДЛИН ИНТЕРВАЛОВ'] = ' '
    out_dict_recurr['------------------------------------------------------------------------------'] = ' '

    out_dict_recurr['1. ГAММА-РАСПРЕДЕЛЕНИЕ:'] = ' ' #"{:.2f}%".format(recurr_class_process_prob[0][0] * 100)
    out_dict_recurr['   '] = ' '
    # alpha = recurr_gamma_alpha_model.predict(df_intervals)[0]
    # beta = recurr_gamma_beta_model.predict(df_intervals)[0]
    model_param = recurr_gamma_model.predict(df_intervals)
    print('model_param', model_param)
    alpha_mm, beta_mm = gamma_method_moments(df_intervals)
    if model_param[0][0] > 0 and model_param[0][1] > 0:
        out_dict_recurr['МЕТОД МОМЕНТОВ'] = ' '
        out_dict_recurr['ММ Параметр формы:'] = "{:.3f}".format(alpha_mm[0])
        out_dict_recurr['ММ Параметр масштаба:'] = "{:.3f}".format(beta_mm[0])
        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр формы:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр масштаба:'] = "{:.3f}".format(model_param[0][1])
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / (model_param[0][0] * model_param[0][1]))
        theor_cdf = [distribution_function_gamma(df_result['emperical'][i], model_param[0][0], model_param[0][1]) for i in range(len(df_result['emperical']))]
        theor_cdf = list(reversed(theor_cdf))
        out_dict_recurr['    '] = ' '
        out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_recurr['Значение интенсивности некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_gamma", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)
    out_dict_recurr = {}
    
    
    


    # out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    # out_dict_recurr['2. ГИПЕРЭКСПОНЕНЦИАЛЬНОЕ РАСПРЕДЕЛЕНИЕ:'] = ' ' #"{:.2f}%".format(recurr_class_process_prob[0][1] * 100)
    # out_dict_recurr['   '] = ' '
    # model_param[0][0] = recurr_hyper_lmbd1_model.predict(df_intervals)[0]
    # model_param[0][1] = recurr_hyper_lmbd2_model.predict(df_intervals)[0]
    # p = recurr_hyper_p_model.predict(df_intervals)[0]
    # lmbd1_mm, lmbd2_mm, p_mm = hexp_method_moments(df_intervals)
    # if model_param[0][0] > 0 and model_param[0][1] > 0 and p > 0:
    #     out_dict_recurr['МЕТОД МОМЕНТОВ'] = ' '
    #     out_dict_recurr['ММ Параметр 1:'] = "{:.3f}".format(lmbd1_mm[0])
    #     out_dict_recurr['ММ Параметр 2:'] = "{:.3f}".format(lmbd2_mm[0])
    #     out_dict_recurr['ММ Вероятность:'] = "{:.1f}".format(p_mm[0])
    #     out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
    #     out_dict_recurr['Параметр 1:'] = "{:.3f}".format(model_param[0][0])
    #     out_dict_recurr['Параметр 2:'] = "{:.3f}".format(model_param[0][1])
    #     out_dict_recurr['Вероятность:'] = "{:.1f}".format(p)
    #     out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
    #     out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / ((p / model_param[0][0]) + ((1 - p) / model_param[0][1])))
    #     theor_cdf = [distribution_function_hexp(df_result['emperical'][i], p, model_param[0][0], model_param[0][1]) for i in range(len(df_result['emperical']))]
    #     #theor_cdf_mm = [distribution_function_hexp(df_result['emperical'][i], p_mm[0], lmbd1_mm[0], lmbd2_mm[0]) for i in range(len(df_result['emperical']))]
    #     out_dict_recurr['    '] = ' '
    #     out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    #     #out_dict_recurr['ММ Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf_mm))
    # else:
    #     out_dict_recurr['Значение интенсивности некорректно'] = ' '
    # out_dict_recurr['     '] = ' '
    # kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_hyp", x_lim_min=0, xticks_flag=False)
    # write_txt(file_name, 'recurr', out_dict_recurr)
    # out_dict_recurr = {}

    out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    out_dict_recurr['2. ГИПЕРЭКСПОНЕНЦИАЛЬНОЕ РАСПРЕДЕЛЕНИЕ:'] = ' ' #"{:.2f}%".format(recurr_class_process_prob[0][1] * 100)
    out_dict_recurr['   '] = ' '
    model_param = recurr_hexp_model.predict(df_intervals)
    lmbd1_mm, lmbd2_mm, p_mm = hexp_method_moments(df_intervals)
    out_dict_recurr['МЕТОД МОМЕНТОВ'] = ' '
    out_dict_recurr['ММ Параметр 1:'] = "{:.3f}".format(lmbd1_mm[0])
    out_dict_recurr['ММ Параметр 2:'] = "{:.3f}".format(lmbd2_mm[0])
    out_dict_recurr['ММ Вероятность:'] = "{:.1f}".format(p_mm[0])
    if model_param[0][0] > 0 and model_param[0][1] > 0 and model_param[0][2] > 0:
        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр 1:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр 2:'] = "{:.3f}".format(model_param[0][1])
        out_dict_recurr['Вероятность:'] = "{:.1f}".format(model_param[0][2])
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / ((model_param[0][2] / model_param[0][0]) + ((1 - model_param[0][2]) / model_param[0][1])))
        theor_cdf = [distribution_function_hexp(df_result['emperical'][i], model_param[0][2], model_param[0][0], model_param[0][1]) for i in range(len(df_result['emperical']))]
        #theor_cdf_mm = [distribution_function_hexp(df_result['emperical'][i], p_mm[0], lmbd1_mm[0], lmbd2_mm[0]) for i in range(len(df_result['emperical']))]
        out_dict_recurr['    '] = ' '
        out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
        #out_dict_recurr['ММ Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf_mm))
    else:
        out_dict_recurr['Значение интенсивности некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_hyp", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)
    out_dict_recurr = {}
    
    


    # out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    # out_dict_recurr['3. ЛОГНОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ:'] = ' ' #"{:.2f}%".format(recurr_class_process_prob[0][2] * 100)
    # out_dict_recurr['   '] = ' '
    # mu = recurr_lognorm_mu_model.predict(df_intervals)[0]
    # sigma = recurr_lognorm_sigma_model.predict(df_intervals)[0]
    # mm_param = lognorm_method_moments(df_intervals)
    # if sigma > 0:
    #     out_dict_recurr['МЕТОД МОМЕНТОВ'] = ' '
    #     out_dict_recurr['ММ Параметр mu:'] = "{:.3f}".format(mm_param[0][0])
    #     out_dict_recurr['ММ Параметр sigma:'] = "{:.3f}".format(mm_param[0][1])
    #     out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
    #     out_dict_recurr['Параметр mu:'] = "{:.3f}".format(mu)
    #     out_dict_recurr['Параметр sigma:'] = "{:.3f}".format(sigma)
    #     out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
    #     out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / (math.exp(mu + pow(sigma, 2) / 2)))
    #     theor_cdf = [distribution_function_lognorm(df_result['emperical'][i], mu, sigma) for i in range(len(df_result['emperical']))]
    #     out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    # else:
    #     out_dict_recurr['Значение интенсивности некорректно'] = ' '
    # out_dict_recurr['     '] = ' '
    # kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_lognorm", x_lim_min=0, xticks_flag=False)
    # write_txt(file_name, 'recurr', out_dict_recurr)
    # out_dict_recurr = {}



    # out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    # out_dict_recurr['4. РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ:'] = ' ' #"{:.2f}%".format(recurr_class_process_prob[0][3] * 100)
    # out_dict_recurr['   '] = ' '
    # a = recurr_uniform_a_model.predict(df_intervals)[0]
    # b = recurr_uniform_b_model.predict(df_intervals)[0]
    # mm_param = uni_method_moments(df_intervals)
    # print(mm_param)
    # if b > a:
    #     out_dict_recurr['МЕТОД МОМЕНТОВ'] = ' '
    #     out_dict_recurr['ММ Параметр a:'] = "{:.3f}".format(mm_param[0][0])
    #     out_dict_recurr['ММ Параметр b:'] = "{:.3f}".format(mm_param[0][1])
    #     out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
    #     out_dict_recurr['Параметр a:'] = "{:.3f}".format(a)
    #     out_dict_recurr['Параметр b:'] = "{:.3f}".format(b)
    #     out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
    #     out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / ((a + b) / 2))
    #     theor_cdf = [distribution_function_uniform(df_result['emperical'][i], a, b) for i in range(len(df_result['emperical']))]
    #     out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    # else:
    #     out_dict_recurr['Значение интенсивности некорректно'] = ' '
    # out_dict_recurr['     '] = ' '
    # kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_uni", x_lim_min=0, xticks_flag=False)
    # write_txt(file_name, 'recurr', out_dict_recurr)
    # out_dict_recurr = {}
    


    # out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    # out_dict_recurr['5. РАСПРЕДЕЛЕНИЕ ВЕЙБУЛЛА:'] = ' ' #"{:.2f}%".format(recurr_class_process_prob[0][4] * 100)
    # out_dict_recurr['   '] = ' '
    # theta = recurr_weibull_theta_model.predict(df_intervals)[0]
    # k = recurr_weibull_k_model.predict(df_intervals)[0]
    # mm_param = weibull_method_moments(df_intervals)
    # print(mm_param)
    # if theta > 0 and k > 0:
    #     out_dict_recurr['МЕТОД МОМЕНТОВ'] = ' '
    #     out_dict_recurr['ММ Параметр theta:'] = "{:.3f}".format(mm_param[0][0])
    #     out_dict_recurr['ММ Параметр k:'] = "{:.3f}".format(mm_param[0][1])
    #     out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
    #     out_dict_recurr['Параметр theta:'] = "{:.3f}".format(theta)
    #     out_dict_recurr['Параметр k:'] = "{:.3f}".format(k)
    #     out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
    #     out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / (theta * gamma(1 + 1 / k)))
    #     theor_cdf = [distribution_function_weibull(df_result['emperical'][i], theta, k) for i in range(len(df_result['emperical']))]
    #     out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    # else:
    #     out_dict_recurr['Значение интенсивности некорректно'] = ' '
    # out_dict_recurr['     '] = ' '
    # kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_weibull", x_lim_min=0, xticks_flag=False)
    # write_txt(file_name, 'recurr', out_dict_recurr)



    out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    out_dict_recurr['6. РАСПРЕДЕЛЕНИЕ ЛЕВИ:'] = ' '
    out_dict_recurr['   '] = ' '
    model_param = recurr_levi_model.predict(df_intervals)
    mm_param = levi_method_moments(df_intervals)
    print(model_param)
    print(mm_param)
    if model_param[0][0] > 0 and model_param[0][1] > 0:
        out_dict_recurr['МЕТОД НАИМЕНЬШИХ КВАДРАТОВ'] = ' '
        out_dict_recurr['МНК Параметр mu местоположение:'] = "{:.3f}".format(mm_param[0])
        out_dict_recurr['МНК Параметр c масштаб:'] = "{:.3f}".format(mm_param[1])
        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр mu местоположение:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр c масштаб:'] = "{:.3f}".format(model_param[0][1])
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        # out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / (theta * gamma(1 + 1 / k)))
        # theor_cdf = [distribution_function_weibull(df_result['emperical'][i], theta, k) for i in range(len(df_result['emperical']))]
        # out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_recurr['Значение интенсивности некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_levi", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)

    
start()