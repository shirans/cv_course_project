import torch
from sacred import Experiment
from argparse import Namespace

import logging

from models.training_utils import choose_model
from utils.const import Models_types
from utils.data_processing import load_input
from utils.evaluation import evaluate

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
    num_epochs = 10000
    batch_size = 1
    plot_loss = True
    models_output_path = 'model_outputs/v1'
    is_save_model = False
    model_load_path = None
    # model_load_path = 'model_outputs/v1/20190601-170049_10kepoch_FC'
    # model_load_path = 'model_outputs/v1/20190609-194734_10000'
    display_images = True
    model_type = Models_types.UNET_V1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cuda"
    print("device:", device)


@ex.automain
def main(_run):
    args = Namespace(**_run.config)

    logger.info(args)
    training_data, validatoin_data = load_input(args)
    model = choose_model(args, training_data)
    # TEST
    evaluate(args, model, training_data, validatoin_data)
