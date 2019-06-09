from sacred import Experiment
from argparse import Namespace

import logging

from models.training_utils import choose_model
from utils.data_processing import load_input
from utils.evaluation import evaluate

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)

ex = Experiment('EyeSegnmentation')
ex.logger = logger


@ex.automain
def main(_run):
    args = Namespace(**_run.config)
    logger.info(args)
    training_data, validatoin_data = load_input(args)
    model = choose_model(args, training_data)
    # TEST
    evaluate(args, model, training_data, validatoin_data)
