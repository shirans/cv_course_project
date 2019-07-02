import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torchvision import transforms


def evaluate_results(args, model, data):
    num_images = 0
    results = []
    results_zero = []
    results_one = []
    total_images = len(data)
    print("data size:{}".format(len(data)))
    for i, (image_batch, mask, segmentation) in enumerate(data):
        image_batch = image_batch.to(args.device)
        net_out = model(image_batch)
        net_out = F.sigmoid(net_out)
        for i in range(0, image_batch.shape[0]):
            prediction = net_out[i, :, :, :]
            if prediction[prediction > 0.5].size()[0] == 0:
                print("all image values are below 0.5")
            success_all, success_zero, success_ones = evaluate_image(i, prediction, segmentation, mask[i, :, :, :])
            results.append(success_all)
            results_zero.append(success_zero)
            results_one.append(success_ones)
            if args.display_images and num_images % int(total_images / 2) == 1:
                display_img(i, prediction, segmentation)
            num_images = num_images + 1
    plt.show()
    print("prediction success total {}, zeros {}, ones: {}".format(
        np.average(results), np.average(results_zero), np.average(results_one)))


def display_img(i, prediction, segmentation):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    im_pred = transforms.ToPILImage(mode='L')(prediction)
    image_plot = plt.imshow(im_pred)
    im_seg = transforms.ToPILImage(mode='L')(segmentation[i, :, :, :])
    a.set_title('prediction')
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(im_seg)
    a.set_title('Segmentation')
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    plt.show(block=True)


def evaluate_image(i, prediction, segmentation, mask):
    prediction_np = prediction.data.numpy()
    new_shape = (prediction_np.shape[1], prediction_np.shape[2])

    prediction_np = prediction_np.reshape(new_shape)
    seg_np = segmentation[i, :, :, :].data.numpy().reshape(new_shape)
    mask_np = mask.data.numpy().reshape(new_shape)
    mask_arr = mask_np == 1

    total_elements = np.sum(mask_arr).astype(int)

    # if value > 0.5 category is 1 else 0
    prediction_np = np.where(prediction_np > 0.5, 1, 0)

    # all indexes in which prediction is correct
    equality = prediction_np[mask_arr] == seg_np[mask_arr]
    sum_equals = np.sum(equality)
    # correct prediction by category
    count_ones_true = ((prediction_np[mask_arr] == 1) & (equality)).sum()
    count_zero_true = ((prediction_np[mask_arr] == 0) & (equality)).sum()

    expected_ones, expected_zeros = get_expected_bin_counts(seg_np, mask_arr, total_elements)


    #  sanity check
    if ((count_zero_true + count_ones_true > total_elements)
            or (expected_zeros + expected_ones != total_elements)
            or (count_ones_true > expected_ones)
            or (count_zero_true > expected_zeros)):
        print("error in calculation")

    success_all = sum_equals / (total_elements)
    sucecss_zeros = count_zero_true / expected_zeros
    success_ones = count_ones_true / expected_ones
    # print("prediction success total {}, zeros {}, ones: {}".format(
    #     success_all, sucecss_zeros, success_ones))
    return success_all, sucecss_zeros, success_ones


def get_expected_bin_counts(seg_np, mask_arr, total_elements):
    truth_set = np.bincount(seg_np[mask_arr].reshape(total_elements).astype(int))
    expected_zeros = truth_set[0]
    expected_ones = truth_set[1]
    return expected_ones, expected_zeros


def evaluate(args, model, training_data, validation_data, test_data):
    print("evaluate on training data")
    evaluate_results(args, model, training_data)
    print("evaluate on validation data")
    evaluate_results(args, model, validation_data)
    print("evaluate on test data")
    evaluate_results(args, model, test_data)