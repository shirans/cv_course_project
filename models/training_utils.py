import os
import time

import torch
from torch import nn as nn

from analysis import plot_loss
from models.fc import FC
from pytorchtools import EarlyStopping
import logging

from utils.const import choose_type

logger = logging.getLogger(__name__)


def loss_func(output, segmentation, mask):
    loss_func = nn.BCEWithLogitsLoss(reduction='none')
    loss_results = loss_func(output, segmentation)
    return (loss_results * mask).mean()


def train_model(args, model, training_data):
    logger.info("training model")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_history = []

    # initialize the early_stopping object

    early_stopping = EarlyStopping(verbose=False)

    for i in range(1, args.num_epochs + 1):
        loss = train(i, model, training_data, optimizer, args)
        loss_history.append(loss)
        print("loss in epoch %d is:" % i, loss)
        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    if args.plot_loss:
        stats = {'loss_history': loss_history}
        plot_loss(stats)
    if args.is_save_model:
        save_model(args, model)


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


def save_model(args, model):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_output_path = os.path.join(args.models_output_path,
                                     '{}_{}'.format(timestr, args.num_epochs))
    state = model.state_dict()
    logger.info("saving model to path: {}".format(model_output_path))
    torch.save(state, model_output_path)


def choose_model(args, training_data):
    path = args.model_load_path
    if path is None:
        logger.info("creating a new model")
        choose_type(args.model_type)
        model = choose_type(args.model_type)
        train_model(args, model, training_data)
        return model
    logger.info("loading model from path: {}".format(path))
    model = choose_type(args.model_type)
    load = torch.load(path)
    model.load_state_dict(load)
    return model
