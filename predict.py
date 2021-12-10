import tensorflow as tf
import numpy as np
import os
from utils import plot_img
from os.path import dirname, abspath
import datetime
from data_loader import point_to_images, convert, segy_read
import matplotlib.pyplot as plt
from cross_train_wrapper import rms_amplitudes


"""
Function to predict with a trained model
This implementation is for the cross-supervised workflow, thus 2 models are used
Can easily be adjusted for the baseline scenario
"""
def predict(output_dir, model_a, model_b, seis_data, labels, rms_amp):
    init_shape = seis_data.shape

    seis_data = np.pad(seis_data, ((0, 0), (0, 352 - seis_data.shape[1]), (0, 416 - seis_data.shape[2]), (0, 0)))
    labels = np.pad(labels, ((0, 0), (0, 352 - labels.shape[1]), (0, 416 - labels.shape[2]), (0, 0)))

    if rms_amp:
        rms_train = rms_amplitudes(seis_data, 3)[..., np.newaxis]
        seis_data = np.concatenate([seis_data, rms_train], axis=-1)

    prediction_a = model_a.predict(seis_data)
    prediction_b = model_b.predict(seis_data)

    prediction = (prediction_a + prediction_b) / 2

    plot_img(prediction[:, :init_shape[1], :init_shape[2], :], None, plt.figure(), output_dir, 'pred_test')

    acc_file = open(output_dir + 'acc_file.txt', 'w+')
    acc_file.write('thresh\tacc\n')
    labels = np.argmax(labels, axis=-1)

    # apply different probability thresholds to turn probabilities into labels
    # get the accuracy for each threshold
    for thresh in np.linspace(0.5, 1, 20):
        pred_label = (prediction[..., 1] > thresh).astype(np.int32) # np.argmax(prediction, axis=-1)
        plot_img(pred_label[:, :init_shape[1], :init_shape[2]], None, plt.figure(), output_dir, f'pred_label_thresh-{np.round(thresh, 2)}')
        acc = np.sum(pred_label[:, :init_shape[1], :init_shape[2]].flatten() ==
                     labels[:, :init_shape[1], :init_shape[2]].flatten()) / len(labels[:, :init_shape[1], :init_shape[2]].flatten())

        acc_file.write(f'{np.round(thresh, 2)}\t{np.round(acc, 3)}\n')

    acc_file.close()


if __name__ == '__main__':
    n_classes = 2
    # whether to further train a model on its own prediction
    train = False
    # slice number which to train on
    slice_num = 1026
    # whether to add rms amplitudes
    rms_amp = False

    homedir = dirname(dirname(abspath(__file__)))
    # where the model is stored
    model_dir = 'C:/_PROJECTS/CNN_for_seismic_classification/DSRG/output/'
    # data directory inside the root directory
    test_data_dir = homedir + '/data/interpretation_points/riped/test'
    # path to segy
    segy_filename = homedir + '/data/3d_cube_cropped_flattened.segy'
    # numeric precision with which to load and read segy
    inp_res = np.float16
    now = datetime.datetime.now()
    # directory to save all output files to
    output_dir = homedir + f'/output/prediction_slice-1029_cross-supervised_{now.strftime("%Y-%m-%d_%H-%M")}/'
    # paths to saved models
    model_a = tf.keras.models.load_model(model_dir + 'trained_a')
    model_b = tf.keras.models.load_model(model_dir + 'trained_b')

    segy_obj = segy_read(segy_filename, mode='create', read_direc='full', inp_res=inp_res)
    segy_obj.cube_num = 1
    segy_obj.data = np.expand_dims(segy_obj.data, axis=len(segy_obj.data.shape))

    # list of test files
    test_files = [test_data_dir + x for x in os.listdir(test_data_dir) if
                   os.path.isfile(test_data_dir + x) and 'z_' in x]
    # a list of facies names, used to process train/test data files, should correspond to the facies names
    # used in the train/test file names
    facies_list = ['ch', 'fault'][:n_classes - 1]

    z_points = convert(test_files, facies_list)
    seis_data, labels, _ = point_to_images(z_points, segy_obj)

    plot_img(labels, None, plt.figure(), output_dir, 'label')
    plot_img(seis_data.astype(np.float32), None, plt.figure(), output_dir, 'seis_data', 'seismic')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if train:
        for model in [model_a, model_b]:
            x = np.expand_dims(segy_obj.data[:, :, slice_num - segy_obj.t_start, :], axis=0)[:, 30:350, 50:370, :]
            plot_img(x[..., 0].astype(np.float32), None, plt.figure(), output_dir, "seismic", "seismic")

            rms_train = rms_amplitudes(x, 3)[..., np.newaxis]
            x = np.concatenate([x, rms_train], axis=-1)

            prob = model.predict(x)
            plot_img(prob, None, plt.figure(), output_dir, "prob_init")
            y = (prob > 0.6).astype(np.int32)

            plot_img(y, None, plt.figure(), output_dir, "y_init")

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                          loss=tf.keras.losses.CategoricalCrossentropy())

            model.fit(x, y, epochs=40)

            prob = model.predict(x)
            plot_img(prob, None, plt.figure(), output_dir, "prob_after_training")

    predict(output_dir, model_a, model_b, seis_data, labels, rms_amp)