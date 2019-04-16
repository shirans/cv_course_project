import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms.functional as TF
from sacred import Experiment
from argparse import Namespace

from analysis import plot_loss
from load_data.eye_dataset import EyeDataset
import logging
import numpy as np
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


def loss_func(output, segmentation, mask):
    loss_func = nn.BCEWithLogitsLoss(reduction='none')
    loss_results = loss_func(output, segmentation)
    return (loss_results * mask).mean()


def train(epoch, model, dataset, optimizer, args):
    total_loss = 0
    model.train()  # sets the model in training mode
    for i, (image, mask, segmentation) in enumerate(dataset):
        model.zero_grad()
        output = model(image)
        loss = loss_func(output, segmentation, mask)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss


@ex.automain
def main(_run):
    args = Namespace(**_run.config)
    logger.info(args)

    loader = EyeDataset(args.data_path, augment=True)
    validation_split = .2
    dataset_size = len(loader)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    shuffle_dataset = True
    random_seed = 42

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # training_data = DataLoader(loader, shuffle=True, batch_size=1, sampler=train_sampler)
    training_data = DataLoader(loader, batch_size=1, sampler=train_sampler)
    test_data = DataLoader(loader, batch_size=1, sampler=valid_sampler)
    model = FC()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_history = []

    for i in range(1, args.num_epochs + 1):
        loss = train(i, model, training_data, optimizer, args)
        loss_history.append(loss)
        print("loss in epoch %d is:" % i, loss)

    stats = {'loss_history': loss_history}

    plot_loss(stats)
    total_loss = 0


    # TEST
    for i, (image, mask, segmentation) in enumerate(test_data):
        net_out = model(image)
        print(net_out.data.numpy().shape())
        TF.to_pil_image(net_out.data.numpy()).show()
        net_out = net_out.data.numpy()

        tag_results = nn.BCEWithLogitsLoss(reduction='none')(net_out, segmentation) * mask
        # print(net_out)
