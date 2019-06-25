import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader

from load_data.eye_dataset import EyeDatasetOverfitCenter
from load_data.eye_dataset import EyeDataset

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


def load_input(args):
    # ------ Michals modification: split train and validation in advance ------ #
    # train and validation images should be placed in args.data_path_training and args.data_path_validation
    # last 4 images (#37-40) are used as validation
    # loader_train = EyeDatasetOverfitCenter(args.data_path_training, augment=True, normalization=True)
    # loader_val = EyeDatasetOverfitCenter(args.data_path_validation, augment=False, normalization=True)
    # loader_test = EyeDatasetOverfitCenter(args.data_path_test, augment=False, normalization=True)
    loader_train = EyeDataset(args.data_path_training, augment=True, normalization=True, is_crop=True)
    loader_val = EyeDataset(args.data_path_validation, augment=False, normalization=True, is_crop=False)
    loader_test = EyeDataset(args.data_path_test, augment=False, normalization=True, is_crop=False)

    # loader = EyeDataset(args.data_path, augment=True)
    ## training_data = DataLoader(loader, shuffle=True, batch_size=1, sampler=train_sampler)
    # training_data, test_data = split_dataset_to_train_and_test(loader, args.batch_size)
    training_data = DataLoader(loader_train, batch_size=args.batch_size, num_workers=args.num_workers)
    validation_data = DataLoader(loader_val, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data = DataLoader(loader_test, batch_size=args.batch_size, num_workers=args.num_workers)
    return training_data, validation_data, test_data