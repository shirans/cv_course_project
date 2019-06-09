from sacred import Experiment
from argparse import Namespace

import logging

from models.training_utils import choose_model
from utils.const import Models_types
from utils.data_processing import load_model
from utils.evaluation import evaluate_results

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)

ex = Experiment('EyeSegnmentation')
ex.logger = logger


@ex.config
def cfg():
    data_path = 'data/drive/training'
    data_path_training = 'data/drive/training'
    data_path_validation = 'data/drive/validation'
    data_path_test = 'data/drive/test'
    num_epochs = 3
    batch_size = 1
    plot_loss = True
    models_output_path = 'model_outputs/v1'
    is_save_model = True
    model_load_path = None
    # model_load_path = 'model_outputs/v1/20190601-170049_10kepoch_FC'
    display_images = True
    model_type = Models_types.FC


@ex.automain
def main(_run):
    args = Namespace(**_run.config)
    logger.info(args)

    training_data, validatoin_data = load_model(args)
    # ------------------------------------------------------------------------- #

    model = choose_model(args, training_data)
    # TEST
    print("evaluate on training data")
    evaluate_results(args, model, training_data)
    print("evaluate on validation data")
    evaluate_results(args, model, validatoin_data)
    # print("evaluate on test data")
    # evaluate_results(args, model, test_data)
