import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sacred import Experiment
from argparse import Namespace
from torchvision import transforms

from analysis import plot_loss
from load_data.eye_dataset import EyeDataset, EyeDatasetOverfitCorners, EyeDatasetOverfitCenter
import logging
import numpy as np
from models.fc import FC
from models.unet import UNET

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
    num_epochs = 1000
    batch_size = 1
    plot_loss = True
    checkpoint_path = 'checkpoints/v1'
    is_save_model = False
    # model_load_path = None
    model_load_path = 'checkpoints/v1/20190601-170049_10kepoch_FC'
    display_images = True


def loss_func(output, segmentation, mask):
    loss_func = nn.BCEWithLogitsLoss(reduction='none')
    loss_results = loss_func(output, segmentation)
    return (loss_results * mask).mean()


def train_model(args, model, training_data):
    logger.info("training model")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_history = []
    for i in range(1, args.num_epochs + 1):
        loss = train(i, model, training_data, optimizer, args)
        loss_history.append(loss)
        print("loss in epoch %d is:" % i, loss)
    if args.plot_loss:
        stats = {'loss_history': loss_history}
        plot_loss(stats)
    if args.is_save_model:
        save_model(args, args.num_epochs, model)


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


def evaluate_image(i, image, segmentation, mask):
    image_np = image.data.numpy()
    new_shape = (image_np.shape[1], image_np.shape[2])
    total_elements = new_shape[0] * new_shape[1]
    # reshape from (1, 128, 128 ) to (128,128)
    image_np = image_np.reshape(new_shape)
    # if value > 0.5 category is 1 else 0
    image_np = np.where(image_np > 0.5, 1, 0)
    seg_np = segmentation[i, :, :, :].data.numpy().reshape(new_shape)
    # all indexes in which prediction is correct
    equality = image_np == seg_np
    sum_equals = np.sum(equality)
    # correct prediction by category
    count_ones_true = ((image_np == 1) & (equality)).sum()
    count_zero_true = ((image_np == 0) & (equality)).sum()
    truth_set = np.bincount(seg_np.reshape(128 * 128).astype(int))
    expected_zeros = truth_set[0]
    expected_ones = truth_set[1]
    #  sanity check
    if ((count_zero_true + count_ones_true > total_elements)
            or (expected_zeros + expected_zeros == total_elements)
            or (count_ones_true > expected_ones)
            or (count_zero_true > expected_zeros)):
        print("error in calculation")
    success_all = sum_equals / (new_shape[0] * new_shape[1])
    sucecss_zeros = count_zero_true / expected_zeros
    success_ones = count_ones_true / expected_ones
    # print("prediction success total {}, zeros {}, ones: {}".format(
    #     success_all, sucecss_zeros, success_ones))
    return success_all, sucecss_zeros, success_ones


def save_model(args, epoch, model):
    state = model.state_dict()
    logger.info("saving model to path:", )
    timestr = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.checkpoint_path,
                        '{}'.format(timestr))
    torch.save(state, path)


def choose_model(args, training_data):
    path = args.model_load_path
    if path is None:
        logger.info("creating a new model")
        model = FC()
        train_model(args, model, training_data)
        return model
    logger.info("loading model from path: {}".format(path))
    model = FC()
    load = torch.load(path)
    model.load_state_dict(load)
    return model


@ex.automain
def main(_run):
    args = Namespace(**_run.config)
    logger.info(args)

    # ------ Michals modification: split train and validation in advance ------ #
    # train and validation images should be placed in args.data_path_training and args.data_path_validation
    # last 4 images (#37-40) are used as validation
    loader_train = EyeDatasetOverfitCenter(args.data_path_training, augment=True, normalization=True)
    loader_val = EyeDatasetOverfitCenter(args.data_path_validation, augment=True, normalization=True)

    # loader = EyeDataset(args.data_path, augment=True)
    ## training_data = DataLoader(loader, shuffle=True, batch_size=1, sampler=train_sampler)
    # training_data, test_data = split_dataset_to_train_and_test(loader, args.batch_size)

    training_data = DataLoader(loader_train, batch_size=args.batch_size)
    test_data = DataLoader(loader_val, batch_size=args.batch_size)
    # ------------------------------------------------------------------------- #

    model = choose_model(args, training_data)
    # TEST
    print("start segmentation on test")
    num_images = 0
    results = []
    results_zero = []
    results_one = []
    for i, (image_batch, mask, segmentation) in enumerate(training_data):
        net_out = model(image_batch)
        net_out = F.sigmoid(net_out)
        for i in range(0, image_batch.shape[0]):
            image = net_out[i, :, :, :]
            if image[image > 0.5].size()[0] == 0:
                print("all image values are below 0.5")
            success_all, success_zero, success_ones = evaluate_image(i, image, segmentation, mask[i,:,:,:])
            results.append(success_all)
            results_zero.append(success_zero)
            results_one.append(success_ones)
            if num_images % 10 == 0 and args.display_images:
                transforms.ToPILImage(mode='L')(image).show()
                transforms.ToPILImage(mode='L')(segmentation[i, :, :, :]).show()
            num_images = num_images + 1
    print("prediction success total {}, zeros {}, ones: {}".format(
        np.average(results), np.average(results_zero), np.average(results_one)))
