import numpy as np
from scipy.special import gamma
from scipy.optimize import fsolve
from sympy import symbols, Eq, solve
from scipy import optimize
from scipy.special import erfc
from scipy import stats

def gamma_method_moments(df_intervals):
    mean_emp = df_intervals.mean(axis=1)
    var_emp = df_intervals.var(axis=1)

    alpha = mean_emp ** 2 / var_emp
    beta = var_emp / mean_emp
    return alpha, beta

def lognorm_method_moments(df_intervals):
    mean_empirical = df_intervals.mean(axis=1)
    var_emp = df_intervals.var(axis=1)
    results = []

    for mean, var in zip(mean_empirical, var_emp):
        if var <= 0 or mean <= 0:
            results.append(None)  # Недопустимые значения дисперсии или среднего
            continue

        def equations(x):
            mu, sigma = x
            eq1 = np.exp(mu + sigma**2 / 2) - mean
            eq2 = (np.exp(sigma**2)-1)*np.exp(2*mu+sigma**2) - var
            return [eq1, eq2]
        try:
            # Используем fsolve с начальной точкой
            initial_guess = (np.log(mean), np.sqrt(np.log(var/mean**2 + 1)))
            mu, sigma = fsolve(equations, initial_guess)

            if sigma > 0:  # Проверяем, что sigma положительное
                results.append((mu, sigma))
            else:
                results.append(None) # Недопустимое значение sigma
        except Exception as e:
            print(f"Ошибка при решении для mean={mean}, var={var}: {e}")
            results.append(None)  # Если решение не найдено

    return np.array(results, dtype=object)  # Возвращаем NumPy массив

def uni_method_moments(df_intervals):
    mean_empirical = df_intervals.mean(axis=1)
    var_emp = df_intervals.var(axis=1)

    a, b = symbols('a b')  # Определяем символьные переменные

    results = []  # Список для хранения результатов
    for mean_emp, var_emp in zip(mean_empirical, var_emp):

        # Проверка на допустимость var_emp (дисперсия должна быть неотрицательной)
        if var_emp <= 0:
            results.append(None)  # Если var_emp <= 0, возвращаем None
            continue

        # Создание системы уравнений (внутри цикла для каждой пары mean и var)
        system_of_equations = [Eq((a + b) / 2, mean_emp),
                               Eq((b - a)**2 / 12, var_emp)]

        try:
            # Решение системы уравнений (внутри цикла)
            solutions = solve(system_of_equations, (a, b))

            # Проверка наличия решений
            if solutions:
                # Вывод результатов
                # Выберем одно из решений (обычно первое) и преобразуем его в числовые значения.
                a_sol, b_sol = solutions[0] # Берем первое решение

                # Преобразуем sympy объекты в numpy float
                a_val = float(a_sol)
                b_val = float(b_sol)

                results.append((a_val, b_val))
            else:
                results.append(None)  # Если решений нет, возвращаем None

        except NotImplementedError as e:
            print(f"Не удалось решить символьно для mean={mean_emp}, var={var_emp}: {e}")
            results.append(None)
        except Exception as e:
            print(f"Другая ошибка при решении для mean={mean_emp}, var={var_emp}: {e}")
            results.append(None)

    return np.array(results, dtype=object)

def weibull_method_moments(df_intervals):
    mean_empirical = df_intervals.mean(axis=1)
    moment2_empirical = (df_intervals**2).mean(axis=1)

    results = []

    for m1, m2 in zip(mean_empirical, moment2_empirical):
        # Проверка на допустимость моментов (m2 > m1**2, m1 > 0)
        if m2 <= m1**2 or m1 <= 0:
            results.append(None)
            continue

        def weibull_moment_equation(k, m1, m2):
            """Уравнение момента для оценки параметра k."""
            return (m2 / m1**2) - (gamma(1 + 2/k) / (gamma(1 + 1/k)**2))

        # Начальное приближение для k
        initial_guess_k = 1.0

        try:
            # Решение уравнения для k
            estimated_k = fsolve(weibull_moment_equation, initial_guess_k, args=(m1, m2))[0]

            # Проверка валидности estimated_k
            if estimated_k <= 0:
                results.append(None)
                continue

            # Решение для лямбды
            estimated_lambda = m1 / gamma(1 + 1/estimated_k)

            results.append((estimated_k, estimated_lambda))

        except Exception as e:
            print(f"Ошибка при оценке параметров для m1={m1}, m2={m2}: {e}")
            results.append(None)

    return np.array(results, dtype=object)

def hexp_method_moments(df_intervals):
    mean_empirical = df_intervals.mean(axis=1)
    moment2_empirical = (df_intervals**2).mean(axis=1)

    cv = np.sqrt(moment2_empirical-mean_empirical**2)/mean_empirical
    p = 0.5*(1-np.sqrt((cv-1)/(cv+1)))
    l1 = 2*(p/mean_empirical)
    l2 = 2*((1-p)/mean_empirical)

    return l2, l1, p

def levi_method_moments(df_intervals):
    sample_sorted = np.sort(df_intervals)
    ecdf = np.arange(1, len(df_intervals) + 1) / len(df_intervals)
    
    def levy_cdf(x, mu, c):
        mask = x > mu
        result = np.zeros_like(x, dtype=float)
        result[mask] = erfc(np.sqrt(c / (2 * (x[mask] - mu))))
        return result

    def error_function(params, x, ecdf_values):
        mu, c = params
        # Ограничения на параметры
        if c <= 0 or mu >= np.min(x):
            return 1e10  # Большое значение как штраф
        theoretical_cdf = levy_cdf(x, mu, c)
        return np.sum((theoretical_cdf - ecdf_values)**2)
    
    mu_init = np.median(df_intervals) - 1
    c_init = 1 / (2 * np.mean(1 / (df_intervals - mu_init)))

    result = optimize.minimize(
        error_function,
        [mu_init, c_init],
        args=(sample_sorted, ecdf),
        bounds=[(None, np.min(df_intervals)), (1e-10, None)]  # μ < min(sample), c > 0
    )
    return result.x

def phisher_method_moments(df_intervals):
    sample_sorted = np.sort(df_intervals)
    ecdf = np.arange(1, len(df_intervals) + 1) / len(df_intervals)
    
    def fisher_cdf(x, dfn, dfd):
        """
        Кумулятивная функция распределения Фишера (Fisher-Snedecor).
        dfn - степени свободы числителя,
        dfd - степени свободы знаменателя.
        """
        return stats.f.cdf(x, dfn, dfd)
    
    def error_function(params, x, ecdf_values):
        """
        Функция ошибки – сумма квадратов разностей между
        теоретической ФРС, полученной для параметров dfn и dfd,
        и эмпирической ФРС.
        """
        dfn, dfd = params
        # Ограничения на параметры: степени свободы должны быть положительны
        #if dfn <= 0 or dfd <= 0:
        #    return 1e10  # штрафное большое значение ошибки
        theoretical_cdf = fisher_cdf(x, dfn, dfd)
        return np.sum((theoretical_cdf - ecdf_values) ** 2)
    
    # Начальные предположения для степеней свободы
    initial_guess = [0.7, 3]
    
    # Оптимизация для минимизации функции ошибки
    result = optimize.minimize(
        error_function,
        initial_guess,
        args=(sample_sorted, ecdf),
        bounds=[(1e-10, None), (1e-10, None)]  # dfn > 0, dfd > 0
    )
    
    return result.x

def pareto_method_moments(df_intervals):
        # Сортировка выборки и вычисление эмпирической ФРС
    sample_sorted = np.sort(df_intervals)
    ecdf = np.arange(1, len(df_intervals) + 1) / len(df_intervals)
    
    def pareto_cdf(x, b, scale):
        return stats.pareto.cdf(x, b, scale=scale)
    
    def error_function(params, x, ecdf_values):
        """
        Функция ошибки – сумма квадратов разностей между
        теоретической ФРС, полученной для параметров b и scale,
        и эмпирической ФРС.
        """
        b, scale = params
        # Ограничения на параметры: параметры должны быть положительны
        if b <= 0 or scale <= 0:
            return 1e10  # штрафное большое значение ошибки
        theoretical_cdf = pareto_cdf(x, b, scale)
        return np.sum((theoretical_cdf - ecdf_values) ** 2)
    
    # Начальные предположения для параметров методом моментов
    sample_mean = df_intervals.mean(axis=1)[0]
    sample_min = df_intervals.min(axis=1)[0]
    
    b_init = 1 / (sample_mean / sample_min - 1)
    scale_init = sample_min
    
    initial_guess = [b_init, scale_init]
    
    # Оптимизация для минимизации функции ошибки
    result = optimize.minimize(
        error_function,
        initial_guess,
        args=(sample_sorted, ecdf),
        bounds=[(1e-10, None), (1e-10, None)]  # b > 0, scale > 0
    )
    
    return result.x

def inverse_gamma_method_moments(df_intervals):
    sample_sorted = np.sort(df_intervals)
    ecdf = np.arange(1, len(sample_sorted) + 1) / len(sample_sorted)

    def invgamma_cdf(x, a, scale):
        # Вычисление теоретической ФРС для обратного гамма распределения с использованием scipy.stats
        return stats.invgamma.cdf(x, a, scale=scale)

    def error_function(params, x, ecdf_values):
        a, scale = params
        if a <= 0 or scale <= 0:
            return 1e10  # штрафное большое значение ошибки
        theoretical_cdf = invgamma_cdf(x, a, scale)
        return np.sum((theoretical_cdf - ecdf_values) ** 2)

    # Начальные предположения для параметров методом моментов:
    # Для обратного гамма распределения, математическое ожидание равно scale/(a-1) при a > 1.
    # Выберем a_init = 3.0, тогда scale_init = sample_mean * (a_init - 1)
    sample_mean = df_intervals.mean(axis=1)[0]
    a_init = 3.0
    scale_init = sample_mean * (a_init - 1)

    initial_guess = [a_init, scale_init]

    # Оптимизация функции ошибки с ограничениями: a > 0 и scale > 0
    result = optimize.minimize(
        error_function,
        initial_guess,
        args=(sample_sorted, ecdf),
        bounds=[(1e-10, None), (1e-10, None)]
    )

    return result.x

def lomax_method_moments(df_intervals):
    sample_sorted = np.sort(df_intervals)
    ecdf = np.arange(1, len(sample_sorted) + 1) / len(sample_sorted)
    
    def lomax_cdf(x, alpha, lambda_):
        """
        Кумулятивная функция распределения Ломакса.
        F(x; alpha, lambda) = 1 - (1 + x/lambda)^(-alpha)
        """
        return 1 - (1 + x / lambda_) ** (-alpha)
    
    def error_function(params, x, ecdf_values):
        alpha, lambda_ = params
        theoretical_cdf = lomax_cdf(x, alpha, lambda_)
        return np.sum((theoretical_cdf - ecdf_values) ** 2)
    
    # Начальные предположения для параметров: shape и scale
    m_sample = df_intervals.mean(axis=1)[0]
    v_sample = df_intervals.var(axis=1)[0]
    # Если дисперсия меньше или равна квадрату среднего, используем запасное значение
    if v_sample <= m_sample**2:
        initial_guess = [0.7, 3]
    else:
        alpha_est = 2 * v_sample / (v_sample - m_sample**2)
        lambda_est = m_sample * (alpha_est - 1)
        initial_guess = [alpha_est, lambda_est]
    
    result = optimize.minimize(
        error_function,
        initial_guess,
        args=(sample_sorted, ecdf),
        bounds=[(1e-10, None), (1e-10, None)]  # alpha > 0, lambda > 0
    )
    return result.x

def burr_method_moments(df_intervals):
    # Сортировка выборки и вычисление эмпирической ФРС
    sample_sorted = np.sort(df_intervals)
    ecdf = np.arange(1, len(sample_sorted) + 1) / len(sample_sorted)
    
    def burr_cdf(x, c, k, lmbd):
        """
        Кумулятивная функция распределения Burr XII с параметрами:
        c (масштаб), k (параметр формы), lmbd (параметр хвоста).
        Для x < 0 считается, что F(x)=0.
        """
        cdf = np.zeros_like(x)
        valid = x >= 0
        cdf[valid] = 1 - (1 + (x[valid] / lmbd) ** c) ** (-k)
        return cdf
    
    def error_function(params, x, ecdf_values):
        c, k, lmbd = params
        theoretical_cdf = burr_cdf(x, c, k, lmbd)
        return np.sum((theoretical_cdf - ecdf_values) ** 2)
    
    # Начальные предположения: если доступны true_c, true_k, true_lam, использовать их, иначе задать 1.0
    # Estimate initial parameters using method of moments
    mean = df_intervals.mean(axis=1)[0]
    var = df_intervals.var(axis=1)[0]
    skew = stats.skew(df_intervals, axis=1)[0]

    
    # Rough approximations for Burr XII parameters
    initial_k = max(1.0, abs(skew))  # k affects the shape/skewness
    initial_c = max(1.0, mean)       # c affects the scale
    initial_lam = max(1.0, var/mean) # lambda affects the tail behavior
    
    initial_guess = [initial_c, initial_k, initial_lam]
    # Параметры должны быть положительными
    bounds = [(1e-10, None), (1e-10, None), (1e-10, None)]
    
    result = optimize.minimize(
        error_function,
        initial_guess,
        args=(sample_sorted, ecdf),
        bounds=bounds
    )
    
    return result.x

def phreshet_method_moments(df_intervals):
    sample_sorted = np.sort(df_intervals)
    ecdf = np.arange(1, len(sample_sorted) + 1) / len(sample_sorted)
    
    def frechet_cdf(x, alpha, s, m):
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
    
    def error_function(params, x, ecdf_values):
        alpha, s, m = params
        theoretical_cdf = frechet_cdf(x, alpha, s, m)
        return np.sum((theoretical_cdf - ecdf_values) ** 2)
    
    # Начальные предположения: alpha > 0, s > 0, m должно быть меньше минимального значения выборки
    initial_guess = [1.0, 1.0, sample_sorted[0] - 0.1]
    # Ограничения: alpha > 0, s > 0, m <= min(sample)
    bounds = [(1e-10, None), (1e-10, None), (None, sample_sorted[0])]
    
    result = optimize.minimize(
        error_function,
        initial_guess,
        args=(sample_sorted, ecdf),
        bounds=bounds
    )
    
    return result.x