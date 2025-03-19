import os
import logging
import tensorflow as tf
from xgboost import XGBRegressor
from log_control import LogController
from catboost import CatBoostClassifier, CatBoostRegressor

main_path = os.getcwd().replace(os.sep, '/')

logger_controller = LogController()
logger_controller.initialize(main_path, 'output.log', logging.INFO)
log = logging.getLogger()

# classification_model = CatBoostClassifier()
# classification_model.load_model(main_path + "/src/trained_models/classification/all_class_model")
# recurr_classification_model = CatBoostClassifier()
# recurr_classification_model.load_model(main_path + "/src/trained_models/classification/recurr_classification")
# log.info('Classification model added')

# # --------------------------------------------------------------------------

pois_model = CatBoostRegressor()
pois_model.load_model(main_path + "/src/trained_models/new_model/pois/pois_stat")
# pois_model = tf.keras.models.load_model(main_path + "/src/trained_models/new_model/pois/pois_stat")
log.info('pois model added')

# # -------------------------------------------------------------------------------

# mmpp_classification_model = CatBoostClassifier()
# mmpp_classification_model.load_model(main_path + "/src/trained_models/mmpp/mmpp_size_4_classification_not_interval")

# mmpp_regression_model = XGBRegressor()
# mmpp_regression_model.load_model(main_path + "/src/trained_models/mmpp/model_xgb_k4_not_intervals.json")
# log.info('mmpp model added')

# # -------------------------------------------------------------------------------

# recurr_gamma_alpha_model = CatBoostRegressor()
# recurr_gamma_beta_model = CatBoostRegressor()
# recurr_gamma_alpha_model.load_model(main_path + "/src/trained_models/gamma/recurr_gamma_alpha")
# recurr_gamma_beta_model.load_model(main_path + "/src/trained_models/gamma/recurr_gamma_beta")
# log.info('gamma model added')

recurr_gamma_model = CatBoostRegressor()
recurr_gamma_model.load_model(main_path + "/src/trained_models/new_model/gamma/models/gamma_stat")
log.info('gamma model added')

# recurr_hyper_lmbd1_model = CatBoostRegressor()
# recurr_hyper_lmbd2_model = CatBoostRegressor()
# recurr_hyper_p_model = CatBoostRegressor()
# recurr_hyper_lmbd1_model.load_model(main_path + "/src/trained_models/hyper/recurr_hyper_lmbd1")
# recurr_hyper_lmbd2_model.load_model(main_path + "/src/trained_models/hyper/recurr_hyper_lmbd2")
# recurr_hyper_p_model.load_model(main_path + "/src/trained_models/hyper/recurr_hyper_p")
# log.info('hyper model added')

recurr_hexp_model = CatBoostRegressor()
recurr_hexp_model.load_model(main_path + "/src/trained_models/new_model/hexp/models/hexp_stat")
log.info('hyper model added')

# recurr_lognorm_mu_model = CatBoostRegressor()
# recurr_lognorm_sigma_model = CatBoostRegressor()
# recurr_lognorm_mu_model.load_model(main_path + "/src/trained_models/lognorm/recurr_lognorm_mu")
# recurr_lognorm_sigma_model.load_model(main_path + "/src/trained_models/lognorm/recurr_lognorm_sigma")
# log.info('lognorm model added')

recurr_lognorm_model = CatBoostRegressor()
recurr_lognorm_model.load_model(main_path + "\src\trained_models\new_model\lognorm\lognorm_stat")
log.info('lognorm model added')

# recurr_uniform_a_model = CatBoostRegressor()
# recurr_uniform_b_model = CatBoostRegressor()
# recurr_uniform_a_model.load_model(main_path + "/src/trained_models/uniform/recurr_uniform_a")
# recurr_uniform_b_model.load_model(main_path + "/src/trained_models/uniform/recurr_uniform_b")
# log.info('uniform model added')

recurr_uni_model = CatBoostRegressor()
recurr_uni_model.load_model(main_path + "\src\trained_models\new_model\uniform\uni_stat")
log.info('uniform model added')

# recurr_weibull_theta_model = CatBoostRegressor()
# recurr_weibull_k_model = CatBoostRegressor()
# recurr_weibull_theta_model.load_model(main_path + "/src/trained_models/weibull/recurr_weibull_theta")
# recurr_weibull_k_model.load_model(main_path + "/src/trained_models/weibull/recurr_weibull_k")
# log.info('weibull model added')

recurr_weibull_model = CatBoostRegressor()
recurr_weibull_model.load_model(main_path + "/src/trained_models/weibull/recurr_weibull_theta")
log.info('weibull model added')

recurr_levi_model = CatBoostRegressor()
recurr_levi_model.load_model(main_path + "/src/trained_models/new_model/levi/levi_stat_model.cbm")
log.info('levi model added')

recurr_phisher_model = CatBoostRegressor()
recurr_phisher_model.load_model(main_path + "\src\trained_models\new_model\phisher\phisher_stat_model.cbm")
log.info('phisher model added')

recurr_pareto_model = CatBoostRegressor()
recurr_pareto_model.load_model(main_path + "\src\trained_models\new_model\phisher\phisher_stat_model.cbm")
log.info('pareto model added')

recurr_invgamma_model = CatBoostRegressor()
recurr_invgamma_model.load_model(main_path + "\src\trained_models\new_model\pareto\pareto_stat_model.cbm")
log.info('invgamma model added')

recurr_lomax_model = CatBoostRegressor()
recurr_lomax_model.load_model(main_path + "\src\trained_models\new_model\lomax\lomax_stat_model.cbm")
log.info('lomax model added')

recurr_burr_model = CatBoostRegressor()
recurr_burr_model.load_model(main_path + "\src\trained_models\new_model\burr\BurrXII_stat_model.cbm")
log.info('burr model added')

recurr_frechet_model = CatBoostRegressor()
recurr_frechet_model.load_model(main_path + "\src\trained_models\new_model\frechet\Frechet_stat_model.cbm")
log.info('frechet model added')

recurr_frechet_model = CatBoostRegressor()
recurr_frechet_model.load_model(main_path + "\src\trained_models\new_model\frechet\Frechet_stat_model.cbm")
log.info('frechet model added')