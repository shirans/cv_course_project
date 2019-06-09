import numpy as np
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torchvision import transforms


def evaluate_results(args, model, data):
    num_images = 0
    results = []
    results_zero = []
    results_one = []
    total_images = len(data)
    for i, (image_batch, mask, segmentation) in enumerate(data):
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
            if num_images % int(total_images / 2) == 1 and args.display_images:
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

                plt.show(block=False)
            num_images = num_images + 1
    plt.show()
    print("prediction success total {}, zeros {}, ones: {}".format(
        np.average(results), np.average(results_zero), np.average(results_one)))


def evaluate_image(i, prediction, segmentation, mask):
    prediction_np = prediction.data.numpy()
    new_shape = (prediction_np.shape[1], prediction_np.shape[2])
    total_elements = new_shape[0] * new_shape[1]
    # reshape from (1, 128, 128 ) to (128,128)
    prediction_np = prediction_np.reshape(new_shape)
    # if value > 0.5 category is 1 else 0
    prediction_np = np.where(prediction_np > 0.5, 1, 0)
    seg_np = segmentation[i, :, :, :].data.numpy().reshape(new_shape)
    # all indexes in which prediction is correct
    equality = prediction_np == seg_np
    sum_equals = np.sum(equality)
    # correct prediction by category
    count_ones_true = ((prediction_np == 1) & (equality)).sum()
    count_zero_true = ((prediction_np == 0) & (equality)).sum()
    truth_set = np.bincount(seg_np.reshape(128 * 128).astype(int))
    expected_zeros = truth_set[0]
    expected_ones = truth_set[1]

    #  sanity check
    if ((count_zero_true + count_ones_true > total_elements)
            or (expected_zeros + expected_ones != total_elements)
            or (count_ones_true > expected_ones)
            or (count_zero_true > expected_zeros)):
        print("error in calculation")

    success_all = sum_equals / (new_shape[0] * new_shape[1])
    sucecss_zeros = count_zero_true / expected_zeros
    success_ones = count_ones_true / expected_ones
    # print("prediction success total {}, zeros {}, ones: {}".format(
    #     success_all, sucecss_zeros, success_ones))
    return success_all, sucecss_zeros, success_ones
