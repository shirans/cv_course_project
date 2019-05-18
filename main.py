import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sacred import Experiment
from argparse import Namespace
from torchvision import transforms

from analysis import plot_loss
from load_data.eye_dataset import EyeDataset
import logging
import numpy as np

from models.FC_UNET import FC_UNET
from models.FC_UNET_2 import CleanU_Net
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
    batch_size = 4
    plot_loss = True


def loss_func(output, segmentation, mask):
    loss_func = nn.BCEWithLogitsLoss(reduction='none')
    loss_results = loss_func(output, segmentation)
    return (loss_results * mask).mean()


def train(epoch, model, dataset, optimizer, args):
    total_loss = 0
    model.train()  # sets the model in training mode
    for i, (image_batch, mask, segmentation) in enumerate(dataset):
        # if you want to see the images and the segmentation
        # transforms.ToPILImage(mode='RGB')(image_batch[0, :, :, :]).show()
        # transforms.ToPILImage(mode='L')(segmentation[0, :, :, :]).show()
        model.zero_grad()
        output = model(image_batch)
        loss = loss_func(output, segmentation, mask)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss


def split_dataset_to_train_and_test(loader, batch_size):
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
    test_sampler = SubsetRandomSampler(val_indices)

    training_data = DataLoader(loader, batch_size=batch_size, sampler=train_sampler)
    test_data = DataLoader(loader, batch_size=batch_size, sampler=test_sampler)
    return training_data, test_data


@ex.automain
def main(_run):
    args = Namespace(**_run.config)
    logger.info(args)

    loader = EyeDataset(args.data_path, augment=True)
    # training_data = DataLoader(loader, shuffle=True, batch_size=1, sampler=train_sampler)
    training_data, test_data = split_dataset_to_train_and_test(loader, args.batch_size)

    model = FC_UNET()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_history = []

    for i in range(1, args.num_epochs + 1):
        loss = train(i, model, training_data, optimizer, args)
        loss_history.append(loss)
        print("loss in epoch %d is:" % i, loss)

    if args.plot_loss:
        stats = {'loss_history': loss_history}
        plot_loss(stats)


    # TEST
    for i, (image_batch, mask, segmentation) in enumerate(test_data):
        net_out = F.sigmoid(model(image_batch))
        for i in range(0, image_batch.shape[0]):
            image = net_out[i, :, :, :]
            # I want to get 0 for <0.5 and 1 for >=0.5
            network_predict = torch.round(torch.add(net_out, 0.5))
            if network_predict.min() == network_predict.max():
                print("all image values are is the same:", network_predict.min())
            else:
                transforms.ToPILImage(mode='L')(image).show()
                transforms.ToPILImage(mode='L')(segmentation[i, :, :, :]).show()