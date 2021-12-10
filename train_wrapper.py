import time
import models
import os
from data_loader import load_data
from data_loader import patches_creator
from utils import *
import copy
from tensorflow_addons.losses import SigmoidFocalCrossEntropy


"""
Main function managing training and the labels update process
"""
def train_wrapper(lr, epochs, output_dir, segy_filename, inp_res, train_files, facies_list,
                  p_thresh, use_crf, use_sisim, model_type, burnin_ep, period, hood_kernels,
                  window_size, overlap, loss_type, mahal_thresh, weight_loss, weight_chan,
                  rms_amp, show_ood_maps, slice_n=1026):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x_train, y_train, y_true, test_seis = load_data(segy_filename, inp_res, facies_list, train_files, slice_n)
    x_train = x_train[..., np.newaxis]
    test_seis = test_seis[np.newaxis, ...]
    n_classes = len(facies_list) + 1

    # add a channel with rms amplitudes to the input
    if rms_amp:
        rms_train = rms_amplitudes(x_train, 3)[..., np.newaxis]
        rms_test = rms_amplitudes(test_seis, 3)[..., np.newaxis]
        x_train = np.concatenate([x_train, rms_train], axis=-1)
        test_seis = np.concatenate([test_seis, rms_test], axis=-1)

    x_train_init = copy.deepcopy(x_train)

    fig = plt.figure()
    plot_img(x_train[..., 0].astype(np.float32), None, fig, output_dir, "seismic", "seismic")
    plot_img(test_seis.astype(np.float32), None, fig, output_dir, "test_seismic", "seismic")
    plot_img(y_train, None, fig, output_dir, "y_init")

    # initialize a model
    if model_type == 'base':
        model = models.BaseClassifier(n_classes, x_train.shape[-1])
    elif model_type == 'unet':
        model = models.build_unet(channels=x_train.shape[-1],
                                  num_classes=n_classes,
                                  layer_depth=5,
                                  filters_root=32)
    elif model_type == 'modified_unet':
        model = models.build_modified_unet(channels=x_train.shape[-1],
                                           num_classes=n_classes,
                                           layer_depth=5,
                                           filters_root=32)
    else:
        raise ValueError("model type is invalid")

    train_loss_results = []

    optimizer = tf.keras.optimizers.Adam(lr)
    superloss = SuperLoss(n_classes, 0.25)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1)

    depthwise_adv = DepthwiseAdv(hood_kernels)
    p_thresh = tf.convert_to_tensor(p_thresh, dtype=tf.float64)
    ood_map = tf.zeros_like(y_train, dtype=tf.float32)

    p_crf = None
    sim_res_exp = None

    # main training/update loop
    start = time.time()
    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()

        y_updated = []
        x_data = []
        for i, (x, y) in enumerate(train_data):
            # each epoch starts with an iteration of standard model training
            with tf.GradientTape() as tape:
                window_step = int(np.ceil(window_size * (1 - overlap / 100)))
                x_train, y_train = patches_creator(x.numpy(), y.numpy(), window_size, window_step)

                pred = model(x_train, training=True)
                # loss is calculated for labeled pixels only
                y_labeled = tf.gather_nd(y_train, tf.where(tf.reduce_sum(y_train, axis=-1) == 1))
                sm_labeled = tf.gather_nd(pred, tf.where(tf.reduce_sum(y_train, axis=-1) == 1))

                if loss_type == 'CE' or 'superloss':
                    loss = tf.keras.losses.CategoricalCrossentropy(reduction='none')(y_labeled, sm_labeled)
                    if loss_type == 'superloss':
                        loss = superloss.get_loss(loss)
                    if weight_loss:
                        loss = tf.where(tf.argmax(y_labeled, axis=-1) == 0, loss * (2 - weight_chan), loss * weight_chan)
                    loss = tf.reduce_mean(loss)

                if loss_type == 'focal':
                    loss = SigmoidFocalCrossEntropy(reduction='none', gamma=2)(y_labeled, sm_labeled)

                    if weight_loss:
                        loss = tf.where(tf.argmax(y_labeled, axis=-1) == 0, loss * (2 - weight_chan), loss * weight_chan)

                    loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            epoch_loss_avg.update_state(loss)

            # below is the label update part
            softmax = model(x, training=False)
            plot_img(softmax.numpy(), epoch, fig, output_dir, f"softmax_{i}")

            if epoch >= burnin_ep and epoch % period == 0:
                # get crf and/or sisim outputs if used
                if use_crf:
                    p_crf = get_crf(x, softmax, output_dir, epoch)
                    p_crf = tf.cast(p_crf, tf.float64)

                if use_sisim and (epoch == 0 or (epoch + 1) % 10 == 0):
                    mask_hood = tf.where(tf.reduce_max(y, axis=-1) == 0,
                                         tf.cast(tf.reduce_max(depthwise_adv.get_hood(y), axis=-1) > 0, tf.float64), 0)
                    sim_mask_idc = tf.where(mask_hood > 0).numpy()
                    sim_res = get_sim(y, sim_mask_idc)
                    plot_img(sim_res, epoch, fig, output_dir, f"sim_res_{i}")
                    sim_res_exp = tf.expand_dims(sim_res, axis=0)

                    sim_res_exp = tf.map_fn(lambda x: expand_sparse(x[0], x[1]),
                                            elems=(tf.repeat(sim_res_exp, n_classes, axis=0),
                                                   tf.range(n_classes, dtype=tf.float64)),
                                            fn_output_signature=tf.int32)

                    sim_res_exp = tf.transpose(sim_res_exp, perm=[1, 2, 0])
                    sim_res_exp = tf.expand_dims(sim_res_exp, axis=0)

                # OOD map update
                if epoch >= burnin_ep + 1 and show_ood_maps:
                    # get input pixels representations
                    feat_maps_out = tf.keras.Model(model.inputs, model.get_layer('features').output)
                    feat_maps = feat_maps_out(x)
                    y_hood = depthwise_adv.get_hood(y)
                    # make an update depending on the model type
                    if model_type == 'modified_unet':
                        ood_map = update_ood_features(ood_map, y, y_hood, feat_maps, mahal_thresh, epoch, fig, output_dir)
                    else:
                        ood_map = update_ood(ood_map, y, y_hood, feat_maps, mahal_thresh, epoch, fig, output_dir)
                    plot_img(ood_map, epoch, fig, output_dir, f"ood_map")

                y_to_upd_sum = 1
                k = 0

                # make updates while there is no more pixels meeting all the criteria
                while y_to_upd_sum > 0:
                    y_hood = depthwise_adv.get_hood(y)

                    y_to_upd = get_update_locs(softmax, y, y_hood, p_thresh, p_crf=p_crf, sim_res=sim_res_exp)

                    y_to_upd_sum = tf.reduce_sum(y_to_upd) # this variable keeps track of if there are pixels to update

                    y_pred_sparse = tf.expand_dims(tf.argmax(softmax, axis=-1), 0)
                    y_to_upd = tf.transpose(y_to_upd, perm=[3, 0, 1, 2])

                    y_updates = tf.map_fn(lambda x: get_y_updates(x[0], x[1], x[2]),
                                          elems=(tf.repeat(y_pred_sparse, n_classes, axis=0), y_to_upd,
                                                 tf.range(n_classes, dtype=tf.int64)),
                                          fn_output_signature=tf.float64)

                    y_updates = tf.transpose(y_updates, perm=[1, 2, 3, 0])
                    y = tf.where(y == 1, y, tf.cast(y_updates, tf.double))

                    k += 1

                y_updated.append(y[0])
                x_data.append(x[0])
                plot_img(y, epoch, fig, output_dir, f"y_{i}")

        # overwrite the training set with updated labels
        if epoch >= burnin_ep and epoch % period == 0 and len(y_updated) > 0:
            train_data = tf.data.Dataset.from_tensor_slices((x_data, y_updated)).batch(1)

        train_loss_results.append(epoch_loss_avg.result())

        print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

    if show_ood_maps:
        ood_map = tf.where(np.array(y_updated[0]) == 0, 0, ood_map)
        plot_img(ood_map, None, fig, output_dir, f"ood_map_fin")

    model.save(output_dir + 'trained')

    # filling parts that were not labeled with labels obtained from predicted probabilities
    train_iter = train_data.as_numpy_iterator()
    y_pred = np.array([s[1][0] for s in train_iter])
    softmax = model(np.array(x_train_init), training=False)
    sm_labels = (softmax.numpy() > 0.5).astype(int)
    idx_missing = np.where(np.sum(y_pred, axis=-1) == 0)
    y_pred[idx_missing[0], idx_missing[1], idx_missing[2]] = sm_labels[idx_missing[0], idx_missing[1], idx_missing[2]]

    plot_img(y_pred, None, fig, output_dir, f"y_final")

    test_pred_prob = model(test_seis)
    plot_img(test_pred_prob, None, fig, output_dir, f"test_prob")
    plot_img(tf.argmax(test_pred_prob, axis=-1), None, fig, output_dir, f"test_y")

    # calculation of accuracies for channels only and the total
    acc_total = np.sum(y_pred[..., 1].flatten() == y_true[..., 1].flatten()) / len(y_true[..., 1].flatten())
    acc_ch = np.sum(y_pred[..., 1].flatten()[y_true[..., 1].flatten() == 1] == 1) / np.sum(y_true[..., 1].flatten() == 1)
    runtime = (time.time() - start) / 60 / epochs

    return runtime, acc_ch, acc_total

