from os.path import dirname, abspath
import os
import numpy as np
from cross_train_wrapper import train_wrapper
import datetime


def main():
    # number of epochs to pretrain models before using their outputs to supervise each other
    cross_burnins = [50]
    # number of epochs to pretrain models before making updates
    burnin_eps = [200]
    # how frequently to make updates
    # an update process will be run every nth epoch
    period = 1
    # width of the neighborhood for class 1
    hood_kernels1 = 30 #[3, 4, 5, 6] #
    # width of the neighborhood for class 2
    hood_kernels2 = 30 #[5, 6, 7, 8, 9, 10] #
    # width of the neighborhood for class 3
    hood_kernels3 = 20
    # type of the model to use as a backbone
    '''for the 'base' OOD samples calculation is not implemented!'''
    model_type = 'unet' # 'unet', 'modified_unet' or 'base'
    # loss function to use
    loss_type = 'focal' # options are 'CE', 'focal, 'superloss'
    # percentile threshold for the distance for selecting OOD samples
    mahal_thresh = 80
    # whether to add rms amplitudes to the input
    rms_amp = False
    # whether to weight loss for examples from different classes differently
    weight_loss = False
    # weight applied to channels
    weight_chan = 0.5
    focal_gamma = 3

    # probability thresholds to consider pixels for getting labeled
    # class 0 (background)
    threshs_1 = [0.8] #[0.8, 0.9]
    # class 1 (channels)
    threshs_2 = [0.6] #[0.6, 0.7]
    # class 2 (fault zone) if used
    thresh_3 = 0.8

    epochs = 200
    lr = 1e-4

    # parameters of the patches extraction process: size and overlap of patches in %
    window_size = 64
    overlap = 50

    # number of classes to use, for the RIPED data the max is 3
    n_classes = 2
    # set numbers of z-slices to use for training
    # None uses 1026 and 1034
    slice_n = None

    # whether to make OOD samples calculations
    show_ood_maps = False
    # whether to use CRF
    use_crf = True
    # whether to use SISIM
    use_sisim = False

    hood_kernels = [hood_kernels1, hood_kernels2, hood_kernels3]

    # root directory that contains directory with the code
    homedir = dirname(dirname(abspath(__file__)))
    # data directory inside the root directory
    data_dir = homedir + '/data/'
    # directory with training data points
    train_data_dir = data_dir + 'interpretation_points/riped/'
    # list of training files
    train_files = [train_data_dir + x for x in os.listdir(train_data_dir) if
                   os.path.isfile(train_data_dir + x) and 'z_' in x]
    # path to segy
    segy_filename = [data_dir + '3d_cube_cropped_flattened.segy']
    # numeric precision with which to load and read segy
    inp_res = np.float16
    # a list of facies names, used to process train/test data files, should correspond to the facies names
    # used in the train/test file names
    facies_list = ['ch', 'fault'][:n_classes - 1]

    now = datetime.datetime.now()
    # directory to save all output files to
    output_dir = homedir + f'/output/sss_UNET_CROSS_{n_classes}-class'
    if rms_amp:
        output_dir += '_RMS-concat'
    if weight_loss:
        output_dir += '_weighted'
    if use_sisim:
        output_dir += '_sisim'
    else:
        output_dir += '_no-sisim'
    if use_crf:
        output_dir += '_crf'#-no-bilat
    else:
        output_dir += '_no-crf'

    output_dir += f'_{loss_type}-loss'

    kernels_str = '-'.join(str(x) for x in hood_kernels[:n_classes])

    output_dir += f'_kernel-{kernels_str}_{now.strftime("%Y-%m-%d_%H-%M")}/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    runtime_f = open(output_dir + 'runtimes.txt', 'w+')
    acc_f = open(output_dir + 'accuracies.txt', 'w+')
    acc_f.write(f"acc_ch\tacc_tot\tburnin_ep\tcross_burnin\tthresh_1\tthresh_2\n")

    for burnin_ep in burnin_eps:
        for thresh_1 in threshs_1:
            for thresh_2 in threshs_2:
                for cross_burnin in cross_burnins:
                    output_dir_i = output_dir + f'burnin-{burnin_ep}_cross_burnin-{cross_burnin}_thr_1-{thresh_1}_thr_2-{thresh_2}'
                    runtime, acc_ch, acc_total = train_wrapper(lr, epochs, output_dir_i, segy_filename, inp_res, train_files,
                                                               facies_list, [thresh_1, thresh_2, thresh_3][:n_classes],
                                                               use_crf, use_sisim, model_type, burnin_ep, period, hood_kernels[:n_classes],
                                                               window_size, overlap, loss_type,  mahal_thresh, weight_loss, cross_burnin,
                                                               weight_chan, rms_amp, focal_gamma, slice_n)

                    runtime_f.write(f"{runtime} min\n")
                    acc_f.write(f"{round(acc_ch, 2)}\t{round(acc_total, 2)}\t{burnin_ep}\t{cross_burnin}\t{thresh_1}\t{thresh_2}\n")

    runtime_f.close()
    acc_f.close()

    return


if __name__ == "__main__":
    main()
