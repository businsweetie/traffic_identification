import os
import logging
# import tensorflow as tf
from xgboost import XGBRegressor
from log_control import LogController
from catboost import CatBoostRegressor#, CatBoostClassifier


# # # -------------------------------------------------------------------------------

# # mmpp_classification_model = CatBoostClassifier()
# # mmpp_classification_model.load_model(main_path + "/src/trained_models/mmpp/mmpp_size_4_classification_not_interval")

# # mmpp_regression_model = XGBRegressor()
# # mmpp_regression_model.load_model(main_path + "/src/trained_models/mmpp/model_xgb_k4_not_intervals.json")
# # log.info('mmpp model added')

# # # -------------------------------------------------------------------------------

def load_catboost_models():
    main_path = os.getcwd().replace(os.sep, '/')
    
    logger_controller = LogController()
    logger_controller.initialize(main_path, 'output.log', logging.INFO)
    log = logging.getLogger()

    models = {
        "pois": "new_model/pois/pois_stat",
        "gamma": "new_model/gamma/models/gamma_stat",
        "hexp": "new_model/hexp/models/hexp_stat",
        "lognorm": "new_model/lognorm/lognorm_stat",
        "uniform": "new_model/uniform/uniform_stat",
        "weibull": "new_model/weibull/weibull_stat",
        "levi": "new_model/levi/levi_stat_model.cbm",
        "phisher": "new_model/phisher/phisher_stat_model.cbm",
        "pareto": "new_model/phisher/phisher_stat_model.cbm",
        "invgamma": "new_model/pareto/pareto_stat_model.cbm",
        "lomax": "new_model/lomax/lomax_stat_model.cbm",
        "burr": "new_model/burr/BurrXII_stat_model.cbm",
        "frechet": "new_model/frechet/Frechet_inter_model.cbm",
    }

    loaded_models = {}
    for name, path in models.items():
        model = CatBoostRegressor()
        model.load_model(f"{main_path}/src/trained_models/{path}")
        loaded_models[name] = model
        log.info(f'{name} model added')

    return loaded_models

def load_xgb_models():
    main_path = os.getcwd().replace(os.sep, '/')
    
    logger_controller = LogController()
    logger_controller.initialize(main_path, 'output.log', logging.INFO)
    log = logging.getLogger()

    models = {
        "mmpp_k2": "new_model/mmpp/model_xgb_k2_moments.json",
        "mmpp_k3": "new_model/mmpp/model_xgb_k3_moments.json",
        "mmpp_k4": "new_model/mmpp/model_xgb_k4_moments.json",
    }

    loaded_models = {}
    for name, path in models.items():
        model = XGBRegressor()
        model.load_model(f"{main_path}/src/trained_models/{path}")
        loaded_models[name] = model
        log.info(f'{name} model added')

    return loaded_models