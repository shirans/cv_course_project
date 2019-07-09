import os
import time

import torch
from torch import nn as nn

from analysis import plot_loss, plot_f1
from pytorchtools import EarlyStopping
import logging
import sklearn.metrics
import numpy as np
from utils.const import choose_type
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def f1_score_func(output, segmentation, mask):
    # The F1 score can be interpreted as a weighted average of the precision and recall,
    # where an F1 score reaches its best value at 1 and worst score at 0.
    # The formula for the F1 score is:
    # F1 = 2 * (precision * recall) / (precision + recall)

    probas    = F.sigmoid(output)
    outputSeg = probas.data.numpy().reshape(probas.size(2)*probas.size(3))
    outputSeg = outputSeg-outputSeg.min()
    outputSeg = outputSeg/outputSeg.max()
    outputSeg = np.where(outputSeg > 0.5, 1, 0)
    segmentationSeg = segmentation.data.numpy().reshape(outputSeg.size)
    segmentationSeg = np.where(segmentationSeg > 0.5, 1, 0)
    maskReshaped    = mask.data.numpy().reshape(outputSeg.size)==1
    f1_score  = sklearn.metrics.f1_score(segmentationSeg[maskReshaped],outputSeg[maskReshaped])
    #epsilon = 1e-7
    #segmentation = segmentation>0.5
    #probas = output>0.5
    #TP = torch.sum(probas * segmentation)
    #precision = TP / (torch.sum(probas) + epsilon)
    #recall = TP / (torch.sum(segmentation) + epsilon)
    #f1 = 2 * precision * recall / (precision + recall + epsilon)
    #f1_score = f1.clamp(min=epsilon, max=1 - epsilon)
    return f1_score


def loss_func(output, segmentation, mask):
    loss_func = nn.BCEWithLogitsLoss(reduction='none')
    loss_results = loss_func(output, segmentation)
    return (loss_results * mask).mean()


def train_model(args, model, training_data, validation_data):
    logger.info("training model")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_history = []
    loss_val_history = []
    f1_history = []
    f1_val_history = []
    # initialize the early_stopping object

    early_stopping = EarlyStopping(verbose=False, patience=args.patience, epsilon=args.epsilon)

    for i in range(1, args.num_epochs + 1):
        loss, loss_val, f1, f1_val = train_epoch(i, model, training_data, optimizer, args, validation_data)
        loss_history.append(loss)
        loss_val_history.append(loss_val)
        f1_history.append(f1)
        f1_val_history.append(f1_val)
        print("Epoch # %d" % i)
        print("Loss : Training- %.5f" % loss ,", Validation- %.5f" % loss_val)
        #print("Validation loss in epoch %d is:" % i, loss_val)
        print("F1 score : Training- %.5f" % f1 ,", Validation- %.5f" % f1_val)
        #print("Validation f1 in epoch %d is:" % i, f1_val)

        # early stopping
        early_stopping(loss_val, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    if args.plot_loss:
        stats = {'loss_history': loss_history, 'loss_val_history': loss_val_history}
        plot_loss(stats)
        stats = {'f1_history': f1_history, 'f1_val_history': f1_val_history}
        plot_f1(stats)
    if args.is_save_model:
        save_model(args, model)


def train_epoch(epoch, model, dataset, optimizer, args, validation_data):
    total_loss = 0
    total_val_loss = 0
    total_f1_score = 0
    total_f1_score_val = 0

    model.train()  # sets the model in training mode

    for i, (image_batch, mask, segmentation) in enumerate(dataset):
        # if you want to see the images and the segmentation
        # transforms.ToPILImage(mode='RGB')(image_batch[0, :, :, :]).show()
        # transforms.ToPILImage(mode='L')(segmentation[0, :, :, :]).show()
        image_batch = image_batch.to(args.device)
        model.zero_grad()
        output = model(image_batch)
        f1_score = f1_score_func(output, segmentation, mask)
        total_f1_score += f1_score.item()

        loss = loss_func(output, segmentation, mask)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    model.eval()
    for i, (image_batch, mask, segmentation) in enumerate(validation_data):
        output = model(image_batch)
        loss_val = loss_func(output, segmentation, mask)
        total_val_loss += loss_val.item()
        f1_score_val=f1_score_func(output, segmentation, mask)
        total_f1_score_val+=f1_score_val.item()

    return total_loss/len(dataset), total_val_loss/len(validation_data), total_f1_score/len(dataset), total_f1_score_val/len(validation_data)


def save_model(args, model):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_output_path = os.path.join(args.models_output_path,
                                     '{}_{}'.format(timestr, args.num_epochs))
    state = model.state_dict()
    logger.info("saving model to path: {}".format(model_output_path))
    torch.save(state, model_output_path)


def choose_model(args, training_data, validation_data):
    path = args.model_load_path
    if path is None:
        logger.info("creating a new model")
        choose_type(args.model_type)
        model = choose_type(args.model_type)
        start = time.time()
        train_model(args, model, training_data, validation_data)
        end = time.time()
        logger.info("time took to train:{}".format(end - start))

        return model
    logger.info("loading model from path: {}".format(path))
    model = choose_type(args.model_type)
    load = torch.load(path)
    model.load_state_dict(load)
    return model
