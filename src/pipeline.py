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
    catboost_models = load_catboost_models()
    xgb_models = load_xgb_models()

    file_name = 'test'
    
    df_moments = pd.read_csv(main_path + "/" + file_name + '.csv', sep=';', header=None)
    df_for_mmmp = df_moments.iloc[:, :10000]
    df_intervals = get_intervals_from_df(df_moments)
    df_stat = calculate_statistics(df_intervals)

    lmbd_emp = df_intervals.shape[1] / np.sum(df_intervals, axis=1)
    df_result = get_cdf_from_intervals(df_intervals)
    hist_plot_test(df_intervals.iloc[0].tolist(), file_name, 100)
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    pois_model = catboost_models["pois"]
    lmbd_pois = pois_model.predict(df_intervals)[0]
    # # class_process_prob = classification_model.predict_proba(df_intervals)
    # # recurr_class_process_prob = recurr_classification_model.predict_proba(df_moments)
    
    # #----------------------------------------------------------------------------------------------------------------------------------
    
    out_dict_pois = {}
    clean_txt(file_name, 'pois')
    out_dict_pois['ПУАССОНОВСКИЙ ПОТОК:'] = ' ' #"{:.2f}%".format(class_process_prob[0][0] * 100)
    if lmbd_pois > 0:
        out_dict_pois['Интенсивность:'] = "{:.3f}".format(lmbd_pois)
        out_dict_pois['Интенсивность эмпирическая:'] = "{:.3f}".format(lmbd_emp[0])
        theor_cdf = [distribution_function_exp(df_result['emperical'][i], lmbd_pois) for i in range(len(df_result['emperical']))]
        out_dict_pois['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_pois['Значение параметров некорректно'] = ''
    kolmogorov_plot(df_result, theor_cdf, file_name, 'pois', x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'pois', out_dict_pois)

    # # #----------------------------------------------------------------------------------------------------------------------------------
    
    out_dict_mmpp = {}
    clean_txt(file_name, 'mmpp')
    # # out_dict_mmpp['MMPP ПОТОК:'] = ' ' #"{:.2f}%".format(class_process_prob[0][1] * 100)
    # # out_dict_mmpp['------------------------------------------------------------------------------'] = ' '
    # # #k_size_prob = mmpp_classification_model.predict_proba(df_moments)

    out_dict_mmpp['Количество состояний 2:'] = ' '
    out_dict_mmpp['  '] = ' '
    df_for_mmmp['k_size'] = 2
    mmpp_regression_model = xgb_models["mmpp_k2"]
    y_pred = mmpp_regression_model.predict(df_for_mmmp)
    q_matrix, lmbd_matrix, pi = get_mmpp_param(y_pred, df_for_mmmp)
    q_matrix_for_print = np.asmatrix(q_matrix)
    lmbd_matrix_for_print = np.asmatrix(lmbd_matrix)
    out_dict_mmpp["Матрица инфинитезимальных характеристик, Q"] = ' '
    write_txt(file_name, 'mmpp', out_dict_mmpp)
    write_matrix(file_name, 'mmpp', q_matrix_for_print)
    out_dict_mmpp = {}
    out_dict_mmpp['  '] = ' '
    out_dict_mmpp['Матрица условных интенсивностей, Lambda'] = ' '
    write_txt(file_name, 'mmpp', out_dict_mmpp)
    write_matrix(file_name, 'mmpp', lmbd_matrix_for_print)
    out_dict_mmpp = {}
    X_pred = get_x_pred(df_for_mmmp, q_matrix, lmbd_matrix, pi)
    X_pred = get_intervals_from_df(X_pred)
    df_mmpp = get_cdf_from_intervals(X_pred)
    q_matrix[-1] = [1, 1]
    b = np.array([0, 1])
    x = np.linalg.solve(q_matrix, b)
    e = np.ones(2)
    lmbd_theor = sum(x * lmbd_matrix * e)
    out_dict_mmpp['  '] = ' '
    out_dict_mmpp['Интенсивность эмпирическая:'] = "{:.3f}".format(lmbd_emp[0])
    out_dict_mmpp['Интенсивность теоретическая:'] = "{:.3f}".format(lmbd_theor)
    x, cdf = ecdf_with_all_x(df_mmpp['emperical'], df_result['emperical'])
    out_dict_mmpp = mmpp_test_param(df_mmpp['emperical'], df_result['emperical'], out_dict_mmpp)
    out_dict_mmpp['------------------------------------------------------------------------------'] = ' '
    kolmogorov_plot(x, cdf, file_name, "mmpp_2", mmpp_mode=True)
    write_txt(file_name, 'mmpp', out_dict_mmpp)
    
    out_dict_mmpp = {}
    out_dict_mmpp['Количество состояний 3:'] = ' ' #"{:.2f}%".format(k_size_prob[0][1] * 100)
    out_dict_mmpp['  '] = ' '
    df_for_mmmp['k_size'] = 3
    mmpp_regression_model = xgb_models["mmpp_k3"]
    y_pred = mmpp_regression_model.predict(df_for_mmmp)
    q_matrix, lmbd_matrix, pi = get_mmpp_param(y_pred, df_for_mmmp)
    q_matrix_for_print = np.asmatrix(q_matrix)
    lmbd_matrix_for_print = np.asmatrix(lmbd_matrix)
    out_dict_mmpp["Матрица инфинитезимальных характеристик, Q"] = ' '
    write_txt(file_name, 'mmpp', out_dict_mmpp)
    write_matrix(file_name, 'mmpp', q_matrix_for_print)
    out_dict_mmpp = {}
    out_dict_mmpp['  '] = ' '
    out_dict_mmpp['Матрица условных интенсивностей, Lambda'] = ' '
    write_txt(file_name, 'mmpp', out_dict_mmpp)
    write_matrix(file_name, 'mmpp', lmbd_matrix_for_print)
    out_dict_mmpp = {}
    X_pred = get_x_pred(df_for_mmmp, q_matrix, lmbd_matrix, pi)
    X_pred = get_intervals_from_df(X_pred)
    df_mmpp = get_cdf_from_intervals(X_pred)
    q_matrix[-1] = [1, 1, 1]
    b = np.array([0, 0, 1])
    x = np.linalg.solve(q_matrix, b)
    e = np.ones(3)
    lmbd_theor = sum(x * lmbd_matrix * e)
    out_dict_mmpp['  '] = ' '
    out_dict_mmpp['Интенсивность эмпирическая:'] = "{:.3f}".format(lmbd_emp[0])
    out_dict_mmpp['Интенсивность теоретическая:'] = "{:.3f}".format(lmbd_theor)
    x, cdf = ecdf_with_all_x(df_mmpp['emperical'], df_result['emperical'])
    out_dict_mmpp = mmpp_test_param(df_mmpp['emperical'], df_result['emperical'], out_dict_mmpp)
    out_dict_mmpp['------------------------------------------------------------------------------'] = ' '
    kolmogorov_plot(x, cdf, file_name, "mmpp_3", mmpp_mode=True)
    write_txt(file_name, 'mmpp', out_dict_mmpp)
    
    out_dict_mmpp = {}
    out_dict_mmpp['Количество состояний 4:'] = ' ' #"{:.2f}%".format(k_size_prob[0][2] * 100)
    out_dict_mmpp['  '] = ' '
    df_for_mmmp['k_size'] = 4
    mmpp_regression_model = xgb_models["mmpp_k4"]
    y_pred = mmpp_regression_model.predict(df_for_mmmp)
    q_matrix, lmbd_matrix, pi = get_mmpp_param(y_pred, df_for_mmmp)
    q_matrix_for_print = np.asmatrix(q_matrix)
    lmbd_matrix_for_print = np.asmatrix(lmbd_matrix)
    out_dict_mmpp["Матрица инфинитезимальных характеристик, Q"] = ' '
    write_txt(file_name, 'mmpp', out_dict_mmpp)
    write_matrix(file_name, 'mmpp', q_matrix_for_print)
    out_dict_mmpp = {}
    out_dict_mmpp['  '] = ' '
    out_dict_mmpp['Матрица условных интенсивностей, Lambda'] = ' '
    write_txt(file_name, 'mmpp', out_dict_mmpp)
    write_matrix(file_name, 'mmpp', lmbd_matrix_for_print)
    out_dict_mmpp = {}
    X_pred = get_x_pred(df_for_mmmp, q_matrix, lmbd_matrix, pi)
    X_pred = get_intervals_from_df(X_pred)
    df_mmpp = get_cdf_from_intervals(X_pred)
    q_matrix[-1] = [1, 1, 1, 1]
    b = np.array([0, 0, 0, 1])
    x = np.linalg.solve(q_matrix, b)
    e = np.ones(4)
    lmbd_theor = sum(x * lmbd_matrix * e)
    out_dict_mmpp['  '] = ' '
    out_dict_mmpp['Интенсивность эмпирическая:'] = "{:.3f}".format(lmbd_emp[0])
    out_dict_mmpp['Интенсивность теоретическая:'] = "{:.3f}".format(lmbd_theor)
    x, cdf = ecdf_with_all_x(df_mmpp['emperical'], df_result['emperical'])
    out_dict_mmpp = mmpp_test_param(df_mmpp['emperical'], df_result['emperical'], out_dict_mmpp)
    kolmogorov_plot(x, cdf, file_name, "mmpp_4", mmpp_mode=True)
    write_txt(file_name, 'mmpp', out_dict_mmpp)

    # Вызов функции для каждого количества состояний
    # for k in [2, 3, 4]:
    #     if k == 2:
    #         mmpp_regression_model = xgb_models["mmpp_k2"]
    #     if k == 3:
    #         mmpp_regression_model = xgb_models["mmpp_k3"]
    #     if k == 4:
    #         mmpp_regression_model = xgb_models["mmpp_k4"]
    #     process_mmpp(file_name, df_for_mmmp, mmpp_regression_model, lmbd_emp, df_result, k)
    # # #----------------------------------------------------------------------------------------------------------------------------------

    out_dict_recurr = {}
    clean_txt(file_name, 'recurr')
    out_dict_recurr["РЕКУРРЕНТНЫЙ ПОТОК:"] = ' ' 
    out_dict_recurr['  '] = ' '
    out_dict_recurr['РАСПРЕДЕЛЕНИЕ ДЛИН ИНТЕРВАЛОВ'] = ' '
    out_dict_recurr['------------------------------------------------------------------------------'] = ' '

    out_dict_recurr['1. ГAММА-РАСПРЕДЕЛЕНИЕ:'] = ' ' 
    out_dict_recurr['   '] = ' '
    recurr_gamma_model = catboost_models["gamma"]
    model_param = recurr_gamma_model.predict(df_intervals)
    alpha_mm, beta_mm = gamma_method_moments(df_intervals)
    if model_param[0][0] > 0 and model_param[0][1] > 0:
        out_dict_recurr['МЕТОД МОМЕНТОВ'] = ' '
        out_dict_recurr['ММ Параметр формы:'] = "{:.3f}".format(alpha_mm[0])
        out_dict_recurr['ММ Параметр масштаба:'] = "{:.3f}".format(beta_mm[0])

        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр формы:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр масштаба:'] = "{:.3f}".format(model_param[0][1])

        out_dict_recurr['РАСЧЕТ ИНТЕНСИВНОСТИ'] = ' '
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / (model_param[0][0] * model_param[0][1]))
        theor_cdf = [distribution_function_gamma(df_result['emperical'][i], model_param[0][0], model_param[0][1]) for i in range(len(df_result['emperical']))]
        theor_cdf = list(reversed(theor_cdf))
        out_dict_recurr['    '] = ' '
        out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_recurr['Значение параметров некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_gamma", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)
    out_dict_recurr = {}
    
    

    out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    out_dict_recurr['2. ГИПЕРЭКСПОНЕНЦИАЛЬНОЕ РАСПРЕДЕЛЕНИЕ:'] = ' '
    out_dict_recurr['   '] = ' '
    recurr_hexp_model = catboost_models["hexp"]
    model_param = recurr_hexp_model.predict(df_intervals)
    lmbd1_mm, lmbd2_mm, p_mm = hexp_method_moments(df_intervals)
    if model_param[0][0] > 0 and model_param[0][1] > 0 and model_param[0][2] > 0:
        out_dict_recurr['МЕТОД МОМЕНТОВ'] = ' '
        out_dict_recurr['ММ Параметр 1:'] = "{:.3f}".format(lmbd1_mm[0])
        out_dict_recurr['ММ Параметр 2:'] = "{:.3f}".format(lmbd2_mm[0])
        out_dict_recurr['ММ Вероятность:'] = "{:.1f}".format(p_mm[0])

        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр 1:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр 2:'] = "{:.3f}".format(model_param[0][1])
        out_dict_recurr['Вероятность:'] = "{:.1f}".format(model_param[0][2])

        out_dict_recurr['РАСЧЕТ ИНТЕНСИВНОСТИ'] = ' '
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / ((model_param[0][2] / model_param[0][0]) + ((1 - model_param[0][2]) / model_param[0][1])))
        theor_cdf = [distribution_function_hexp(df_result['emperical'][i], model_param[0][2], model_param[0][0], model_param[0][1]) for i in range(len(df_result['emperical']))]
        out_dict_recurr['    '] = ' '
        out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_recurr['Значение параметров некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_hyp", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)
    out_dict_recurr = {}
    
    

    out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    out_dict_recurr['3. ЛОГНОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ:'] = ' ' 
    out_dict_recurr['   '] = ' '
    recurr_lognorm_model = catboost_models["lognorm"]
    model_param = recurr_lognorm_model.predict(df_intervals)
    mm_param = lognorm_method_moments(df_intervals)
    if model_param[0][1] > 0:
        out_dict_recurr['МЕТОД МОМЕНТОВ'] = ' '
        out_dict_recurr['ММ Параметр mu:'] = "{:.3f}".format(mm_param[0][0])
        out_dict_recurr['ММ Параметр sigma:'] = "{:.3f}".format(mm_param[0][1])

        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр mu:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр sigma:'] = "{:.3f}".format(model_param[0][1])

        out_dict_recurr['РАСЧЕТ ИНТЕНСИВНОСТИ'] = ' '
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / (math.exp(model_param[0][0] + pow(model_param[0][1], 2) / 2)))
        theor_cdf = [distribution_function_lognorm(df_result['emperical'][i], model_param[0][0], model_param[0][1]) for i in range(len(df_result['emperical']))]
        out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_recurr['Значение параметров некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_lognorm", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)
    out_dict_recurr = {}



    out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    out_dict_recurr['4. РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ:'] = ' '
    out_dict_recurr['   '] = ' '
    recurr_uniform_model = catboost_models["uniform"]
    model_param = recurr_uniform_model.predict(df_intervals)
    mm_param = uni_method_moments(df_intervals)
    if model_param[0][1] > model_param[0][0]:
        out_dict_recurr['МЕТОД МОМЕНТОВ'] = ' '
        out_dict_recurr['ММ Параметр a:'] = "{:.3f}".format(mm_param[0][0])
        out_dict_recurr['ММ Параметр b:'] = "{:.3f}".format(mm_param[0][1])

        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр a:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр b:'] = "{:.3f}".format(model_param[0][1])

        out_dict_recurr['РАСЧЕТ ИНТЕНСИВНОСТИ'] = ' '
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / ((model_param[0][0] + model_param[0][1]) / 2))
        theor_cdf = [distribution_function_uniform(df_result['emperical'][i], model_param[0][0], model_param[0][1]) for i in range(len(df_result['emperical']))]
        out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_recurr['Значение параметров некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_uni", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)
    out_dict_recurr = {}
    


    out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    out_dict_recurr['5. РАСПРЕДЕЛЕНИЕ ВЕЙБУЛЛА:'] = ' ' 
    out_dict_recurr['   '] = ' '
    recurr_weibull_model = catboost_models["weibull"]
    model_param = recurr_weibull_model.predict(df_intervals)
    mm_param = weibull_method_moments(df_intervals)
    if model_param[0][0] > 0 and model_param[0][1] > 0:
        out_dict_recurr['МЕТОД МОМЕНТОВ'] = ' '
        out_dict_recurr['ММ Параметр theta:'] = "{:.3f}".format(mm_param[0][0])
        out_dict_recurr['ММ Параметр k:'] = "{:.3f}".format(mm_param[0][1])

        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр theta:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр k:'] = "{:.3f}".format(model_param[0][1])

        out_dict_recurr['РАСЧЕТ ИНТЕНСИВНОСТИ'] = ' '
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / (model_param[0][0] * gamma(1 + 1 / model_param[0][1])))
        theor_cdf = [distribution_function_weibull(df_result['emperical'][i], model_param[0][0], model_param[0][1]) for i in range(len(df_result['emperical']))]
        out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_recurr['Значение параметров некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_weibull", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)
    out_dict_recurr = {}



    out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    out_dict_recurr['6. РАСПРЕДЕЛЕНИЕ ЛЕВИ:'] = ' '
    out_dict_recurr['   '] = ' '
    recurr_levi_model = catboost_models["levi"]
    model_param = recurr_levi_model.predict(df_intervals)
    mm_param = levi_method_moments(df_intervals)
    if model_param[0][0] > 0 and model_param[0][1] > 0:
        out_dict_recurr['МЕТОД НАИМЕНЬШИХ КВАДРАТОВ'] = ' '
        out_dict_recurr['МНК Параметр mu:'] = "{:.3f}".format(mm_param[0])
        out_dict_recurr['МНК Параметр c:'] = "{:.3f}".format(mm_param[1])

        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр mu:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр c:'] = "{:.3f}".format(model_param[0][1])

        out_dict_recurr['РАСЧЕТ ИНТЕНСИВНОСТИ'] = ' '
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        out_dict_recurr['Интенсивность теоретическая:'] = '-'
        theor_cdf = [distribution_function_levi(df_result['emperical'][i], model_param[0][0], model_param[0][1]) for i in range(len(df_result['emperical']))]
        out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_recurr['Значение параметров некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_levi", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)
    out_dict_recurr = {}



    out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    out_dict_recurr['7. РАСПРЕДЕЛЕНИЕ ФИШЕРА:'] = ' '
    out_dict_recurr['   '] = ' '
    recurr_phisher_model = catboost_models["phisher"]
    model_param = recurr_phisher_model.predict(df_intervals)
    mm_param = phisher_method_moments(df_intervals)
    if model_param[0][0] > 0 and model_param[0][1] > 0:
        out_dict_recurr['МЕТОД НАИМЕНЬШИХ КВАДРАТОВ'] = ' '
        out_dict_recurr['МНК Параметр d1:'] = "{:.3f}".format(mm_param[0])
        out_dict_recurr['МНК Параметр d2:'] = "{:.3f}".format(mm_param[1])

        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр d1:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр d2:'] = "{:.3f}".format(model_param[0][1])

        out_dict_recurr['РАСЧЕТ ИНТЕНСИВНОСТИ'] = ' '
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        if model_param[0][1] > 2:
            out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / (model_param[0][1]/(model_param[0][1]-2)))
        else:
            out_dict_recurr['Интенсивность теоретическая:'] = '-'
        theor_cdf = [distribution_function_phisher(df_result['emperical'][i], model_param[0][0], model_param[0][1]) for i in range(len(df_result['emperical']))]
        out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_recurr['Значение параметров некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_phisher", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)
    out_dict_recurr = {}


    out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    out_dict_recurr['8. РАСПРЕДЕЛЕНИЕ ПАРЕТО:'] = ' '
    out_dict_recurr['   '] = ' '
    recurr_pareto_model = catboost_models["pareto"]
    model_param = recurr_pareto_model.predict(df_intervals)
    mm_param = pareto_method_moments(df_intervals)
    if model_param[0][0] > 0 and model_param[0][1] > 0:
        out_dict_recurr['МЕТОД НАИМЕНЬШИХ КВАДРАТОВ'] = ' '
        out_dict_recurr['МНК Параметр x_m:'] = "{:.3f}".format(mm_param[0])
        out_dict_recurr['МНК Параметр alpha:'] = "{:.3f}".format(mm_param[1])

        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр x_m:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр alpha:'] = "{:.3f}".format(model_param[0][1])

        out_dict_recurr['РАСЧЕТ ИНТЕНСИВНОСТИ'] = ' '
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        if model_param[0][1] > 1:
            out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / ((model_param[0][0] * model_param[0][1])/(model_param[0][1]-1)))
        else:
            out_dict_recurr['Интенсивность теоретическая:'] = '-'
        theor_cdf = [distribution_function_pareto(df_result['emperical'][i], model_param[0][0], model_param[0][1]) for i in range(len(df_result['emperical']))]
        out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_recurr['Значение параметров некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_pareto", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)
    out_dict_recurr = {}



    out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    out_dict_recurr['9. ОБРАТНОЕ ГАММА РАСПРЕДЕЛЕНИЕ:'] = ' '
    out_dict_recurr['   '] = ' '
    recurr_invgamma_model = catboost_models["invgamma"]
    model_param = recurr_invgamma_model.predict(df_intervals)
    mm_param = inverse_gamma_method_moments(df_intervals)
    if model_param[0][0] > 0 and model_param[0][1] > 0:
        out_dict_recurr['МЕТОД НАИМЕНЬШИХ КВАДРАТОВ'] = ' '
        out_dict_recurr['МНК Параметр alpha:'] = "{:.3f}".format(mm_param[0])
        out_dict_recurr['МНК Параметр beta:'] = "{:.3f}".format(mm_param[1])

        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр alpha:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр beta:'] = "{:.3f}".format(model_param[0][1])

        out_dict_recurr['РАСЧЕТ ИНТЕНСИВНОСТИ'] = ' '
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        if model_param[0][0] > 1:
            out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / (model_param[0][1]/(model_param[0][0]-1)))
        else:
            out_dict_recurr['Интенсивность теоретическая:'] = '-'
        theor_cdf = [distribution_function_invgamma(df_result['emperical'][i], model_param[0][0], model_param[0][1]) for i in range(len(df_result['emperical']))]
        out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_recurr['Значение параметров некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_invgamma", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)
    out_dict_recurr = {}



    out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    out_dict_recurr['10. РАСПРЕДЕЛЕНИЕ ЛОМАКСА:'] = ' '
    out_dict_recurr['   '] = ' '
    recurr_lomax_model = catboost_models["lomax"]
    model_param = recurr_lomax_model.predict(df_intervals)
    mm_param = lomax_method_moments(df_intervals)
    if model_param[0][0] > 0 and model_param[0][1] > 0:
        out_dict_recurr['МЕТОД НАИМЕНЬШИХ КВАДРАТОВ'] = ' '
        out_dict_recurr['МНК Параметр alpha:'] = "{:.3f}".format(mm_param[0])
        out_dict_recurr['МНК Параметр lmbd:'] = "{:.3f}".format(mm_param[1])

        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр alpha:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр lmbd:'] = "{:.3f}".format(model_param[0][1])

        out_dict_recurr['РАСЧЕТ ИНТЕНСИВНОСТИ'] = ' '
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        if model_param[0][0] > 1:
            out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / (model_param[0][1]/(model_param[0][0]-1)))
        else:
            out_dict_recurr['Интенсивность теоретическая:'] = '-'
        theor_cdf = [distribution_function_lomax(df_result['emperical'][i], model_param[0][1], model_param[0][0]) for i in range(len(df_result['emperical']))]
        out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_recurr['Значение параметров некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_lomax", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)
    out_dict_recurr = {}



    out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    out_dict_recurr['11. РАСПРЕДЕЛЕНИЕ БУРА XII:'] = ' '
    out_dict_recurr['   '] = ' '
    recurr_burr_model = catboost_models["burr"]
    model_param = recurr_burr_model.predict(df_intervals)
    mm_param = burr_method_moments(df_intervals)
    if model_param[0][0] > 0 and model_param[0][1] > 0:
        out_dict_recurr['МЕТОД НАИМЕНЬШИХ КВАДРАТОВ'] = ' '
        out_dict_recurr['МНК Параметр c:'] = "{:.3f}".format(mm_param[0])
        out_dict_recurr['МНК Параметр k:'] = "{:.3f}".format(mm_param[1])
        out_dict_recurr['МНК Параметр lmbd:'] = "{:.3f}".format(mm_param[2])

        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр c:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр k:'] = "{:.3f}".format(model_param[0][1])
        out_dict_recurr['Параметр lmbd:'] = "{:.3f}".format(model_param[0][2])

        out_dict_recurr['РАСЧЕТ ИНТЕНСИВНОСТИ'] = ' '
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / (model_param[0][1] * beta_func((model_param[0][1]-1)/model_param[0][0], 1+(1/model_param[0][0]))))
        theor_cdf = [distribution_function_burr(df_result['emperical'][i], model_param[0][0], model_param[0][1], model_param[0][2]) for i in range(len(df_result['emperical']))]
        out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_recurr['Значение параметров некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_burr", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)
    out_dict_recurr = {}
    


    out_dict_recurr['------------------------------------------------------------------------------'] = ' '
    out_dict_recurr['12. РАСПРЕДЕЛЕНИЕ ФРЕШЕ:'] = ' '
    out_dict_recurr['   '] = ' '
    recurr_frechet_model = catboost_models["frechet"]
    model_param = recurr_frechet_model.predict(df_intervals)
    mm_param = phreshet_method_moments(df_intervals.values[0])
    if model_param[0][0] > 0 and model_param[0][1] > 0:
        out_dict_recurr['МЕТОД НАИМЕНЬШИХ КВАДРАТОВ'] = ' '
        out_dict_recurr['МНК Параметр alpha:'] = "{:.3f}".format(mm_param[0])
        out_dict_recurr['МНК Параметр s:'] = "{:.3f}".format(mm_param[1])
        out_dict_recurr['МНК Параметр m:'] = "{:.3f}".format(mm_param[2])

        out_dict_recurr['ОЦЕНКА МОДЕЛЬЮ'] = ' '
        out_dict_recurr['Параметр alpha:'] = "{:.3f}".format(model_param[0][0])
        out_dict_recurr['Параметр s:'] = "{:.3f}".format(model_param[0][1])
        out_dict_recurr['Параметр m:'] = "{:.3f}".format(model_param[0][2])

        out_dict_recurr['РАСЧЕТ ИНТЕНСИВНОСТИ'] = ' '
        out_dict_recurr["Интенсивность эмпирическая:"] = "{:.3f}".format(lmbd_emp[0])
        if model_param[0][0]>1:
            out_dict_recurr['Интенсивность теоретическая:'] = "{:.3f}".format(1 / (model_param[0][2] + model_param[0][1]*gamma(1-1/model_param[0][0])))
        else:
            out_dict_recurr['Интенсивность теоретическая:'] = '-'
        theor_cdf = [distribution_function_phrechet(df_result['emperical'][i], model_param[0][0], model_param[0][1], model_param[0][2]) for i in range(len(df_result['emperical']))]
        out_dict_recurr['Расстояние Колмогорова:'] = "{:.3f}".format(kolmogorov(df_result['cdf_emp'], theor_cdf))
    else:
        out_dict_recurr['Значение параметров некорректно'] = ' '
    out_dict_recurr['     '] = ' '
    kolmogorov_plot(df_result, theor_cdf, file_name, "recurr_phrechet", x_lim_min=0, xticks_flag=False)
    write_txt(file_name, 'recurr', out_dict_recurr)
    out_dict_recurr = {}
start()