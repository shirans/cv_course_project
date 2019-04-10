import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sacred import Experiment
from argparse import Namespace
from loader import EyeLoader
import logging
from load_data import load_data

from models.fc import FC

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)

ex = Experiment('EyeSegnmentation')
ex.logger = logger


@ex.config
def cfg():
    data_path = 'data/drive/training'
    num_epochs = 30


def train(epoch, model, dataset, optimizer, args):
    total_loss = 0
    loss_func = nn.BCEWithLogitsLoss(reduction='none')
    model.train()
    for i, (image, mask, segmentation) in enumerate(dataset):
        model.zero_grad()
        output = model(image)
        loss = (loss_func(output, segmentation) * mask).mean()
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss


@ex.automain
def main(_run):
    args = Namespace(**_run.config)
    logger.info(args)

    training_data = DataLoader(EyeLoader(args.data_path, augment=True), shuffle=True, batch_size=1)
    model = FC()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(1, args.num_epochs + 1):
        loss = train(i, model, training_data, optimizer, args)
        print("loss in epoch %d is:" % i, loss)


# def main():
#     dirname = os.path.dirname(__file__)
#     _path_to_data_dir = os.path.join(dirname, 'data')
#     load_data.load(_path_to_data_dir)
#
#
#     x = torch.rand(5, 3)
#     print(x)
#
#
# main()
