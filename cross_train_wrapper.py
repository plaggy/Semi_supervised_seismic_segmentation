import time
import models
import os
from data_loader import load_data
from data_loader import patches_creator
from utils import *
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
import gc
import copy


"""
Main function managing training and the labels update process with pseudo cross supervision
"""
def train_wrapper(lr, epochs, output_dir, segy_filename, inp_res, train_files, facies_list,
                  p_thresh, use_crf, use_sisim, model_type, burnin_ep, period, hood_kernels,
                  window_size, overlap, loss_type, mahal_thresh, weight_loss, cross_burnin, weight_chan,
                  rms_amp, focal_gamma, show_ood_maps, slice_n=1026):

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
    plot_img(x_train.astype(np.float32), None, fig, output_dir, "seismic", "seismic")
    plot_img(test_seis.astype(np.float32), None, fig, output_dir, "test_seismic", "seismic")
    plot_img(y_train, None, fig, output_dir, "y_init")

    # initialize a model
    if model_type == 'base':
        model_a = models.BaseClassifier(n_classes, x_train.shape[-1])
        model_b = models.BaseClassifier(n_classes, x_train.shape[-1])
    elif model_type == 'unet':
        model_a = models.build_unet(channels=x_train.shape[-1],
                                    num_classes=n_classes,
                                    layer_depth=5,
                                    filters_root=32)

        model_b = models.build_unet(channels=x_train.shape[-1],
                                    num_classes=n_classes,
                                    layer_depth=5,
                                    filters_root=32)
    elif model_type == 'modified_unet':
        model_a = models.build_modified_unet(channels=x_train.shape[-1],
                                             num_classes=n_classes,
                                             layer_depth=5,
                                             filters_root=32)
        model_b = models.build_modified_unet(channels=x_train.shape[-1],
                                             num_classes=n_classes,
                                             layer_depth=5,
                                             filters_root=32)
    else:
        raise ValueError("model type is invalid")

    optimizer_a = tf.keras.optimizers.Adam(lr)
    optimizer_b = tf.keras.optimizers.Adam(lr)
    superloss = SuperLoss(n_classes, 0.25)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(1)

    depthwise_adv = DepthwiseAdv(hood_kernels)
    p_thresh = tf.convert_to_tensor(p_thresh, dtype=tf.float64)
    ood_map_a = tf.zeros_like(y_train, dtype=tf.float32)
    ood_map_b = tf.zeros_like(y_train, dtype=tf.float32)

    p_crf = None
    sim_res_exp = None

    # main training/update loop
    start = time.time()
    for epoch in range(epochs):
        epoch_loss_avg_a = tf.keras.metrics.Mean()
        epoch_loss_avg_b = tf.keras.metrics.Mean()

        y_updated = []
        x_data = []
        for i, (x, y) in enumerate(train_data):
            # each epoch starts with an iteration of standard model training
            window_step = int(np.ceil(window_size * (1 - overlap / 100)))
            x_train, y_train = patches_creator(x.numpy(), y.numpy(), window_size, window_step)

            patches_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)

            for (x_patches, y_patches) in patches_data:
                with tf.GradientTape(persistent=True) as tape:
                    pred_a = model_a(x_patches, training=True)
                    pred_b = model_b(x_patches, training=True)
                    y_labeled = tf.gather_nd(y_patches, tf.where(tf.reduce_sum(y_patches, axis=-1) == 1))
                    sm_labeled_a = tf.gather_nd(pred_a, tf.where(tf.reduce_sum(y_patches, axis=-1) == 1))
                    sm_labeled_b = tf.gather_nd(pred_b, tf.where(tf.reduce_sum(y_patches, axis=-1) == 1))

                    # calculation of loss for labeled pixels
                    if loss_type == 'CE' or 'superloss':
                        loss_a = tf.keras.losses.CategoricalCrossentropy(reduction='none')(y_labeled, sm_labeled_a)
                        loss_b = tf.keras.losses.CategoricalCrossentropy(reduction='none')(y_labeled, sm_labeled_b)
                        if loss_type == 'superloss':
                            loss_a = superloss.get_loss(loss_a)
                            loss_b = superloss.get_loss(loss_b)

                        if weight_loss:
                            loss_a = tf.where(tf.argmax(y_labeled, axis=-1) == 0, loss_a * (2 - weight_chan),
                                              loss_a * weight_chan)
                            loss_b = tf.where(tf.argmax(y_labeled, axis=-1) == 0, loss_b * (2 - weight_chan),
                                              loss_b * weight_chan)

                        loss_a = tf.reduce_mean(loss_a)
                        loss_b = tf.reduce_mean(loss_b)

                    if loss_type == 'focal':
                        loss_a = SigmoidFocalCrossEntropy(reduction='none', gamma=focal_gamma)(y_labeled, sm_labeled_a)
                        loss_b = SigmoidFocalCrossEntropy(reduction='none', gamma=focal_gamma)(y_labeled, sm_labeled_b)

                        if weight_loss:
                            loss_a = tf.where(tf.argmax(y_labeled, axis=-1) == 0, loss_a * (2 - weight_chan),
                                              loss_a * weight_chan)
                            loss_b = tf.where(tf.argmax(y_labeled, axis=-1) == 0, loss_b * (2 - weight_chan),
                                              loss_b * weight_chan)

                        loss_a = tf.reduce_mean(loss_a)
                        loss_b = tf.reduce_mean(loss_b)

                    # calculation of loss for unlabeled pixels - pseudo cross supervision
                    if epoch >= cross_burnin:
                        sm_unlabeled_a = tf.gather_nd(pred_a, tf.where(tf.reduce_sum(y_patches, axis=-1) == 0))
                        sm_unlabeled_b = tf.gather_nd(pred_b, tf.where(tf.reduce_sum(y_patches, axis=-1) == 0))

                        pred_a = tf.where(pred_a > 0.5, 1, 0)
                        pred_b = tf.where(pred_b > 0.5, 1, 0)
                        pred_a = tf.gather_nd(pred_a, tf.where(tf.reduce_sum(y_patches, axis=-1) == 0))
                        pred_b = tf.gather_nd(pred_b, tf.where(tf.reduce_sum(y_patches, axis=-1) == 0))

                        if loss_type == 'CE' or 'superloss':
                            loss_ul_a = tf.keras.losses.CategoricalCrossentropy(reduction='none')(pred_b, sm_unlabeled_a)
                            loss_ul_b = tf.keras.losses.CategoricalCrossentropy(reduction='none')(pred_a, sm_unlabeled_b)
                            if loss_type == 'superloss':
                                loss_ul_a = superloss.get_loss(loss_ul_a)
                                loss_ul_b = superloss.get_loss(loss_ul_b)

                            loss_ul_a = tf.reduce_mean(loss_ul_a)
                            loss_ul_b = tf.reduce_mean(loss_ul_b)

                        if loss_type == 'focal':
                            loss_ul_a = SigmoidFocalCrossEntropy(reduction='none', gamma=focal_gamma)(pred_b, sm_unlabeled_a)
                            loss_ul_b = SigmoidFocalCrossEntropy(reduction='none', gamma=focal_gamma)(pred_a, sm_unlabeled_b)
                            loss_ul_a = tf.reduce_mean(loss_ul_a)
                            loss_ul_b = tf.reduce_mean(loss_ul_b)

                        loss_a += loss_ul_a
                        loss_b += loss_ul_b

                grads = tape.gradient(loss_a, model_a.trainable_weights)
                optimizer_a.apply_gradients(zip(grads, model_a.trainable_weights))
                grads = tape.gradient(loss_b, model_b.trainable_weights)
                optimizer_b.apply_gradients(zip(grads, model_b.trainable_weights))

                epoch_loss_avg_a.update_state(loss_a)
                epoch_loss_avg_b.update_state(loss_b)

            # a single softmax for label updates is just an average of models' outputs
            softmax_a = model_a(x, training=False)
            plot_img(softmax_a.numpy(), epoch, fig, output_dir, f"softmax_a_{i}")

            softmax_b = model_b(x, training=False)
            plot_img(softmax_b.numpy(), epoch, fig, output_dir, f"softmax_b_{i}")

            softmax = (softmax_a + softmax_b) / 2
            plot_img(softmax.numpy(), epoch, fig, output_dir, f"softmax_avg_{i}")

            # below is the label update part
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
                                                   tf.range(n_classes, dtype=tf.int64)),
                                            fn_output_signature=tf.int32)

                    sim_res_exp = tf.transpose(sim_res_exp, perm=[1, 2, 0])
                    sim_res_exp = tf.expand_dims(sim_res_exp, axis=0)

                # OOD maps update
                if epoch >= burnin_ep + 1 and show_ood_maps:
                    # get input pixels representations
                    feat_maps_out_a = tf.keras.Model(model_a.inputs, model_a.get_layer('features').output)
                    feat_maps_a = feat_maps_out_a(x)
                    feat_maps_out_b = tf.keras.Model(model_b.inputs, model_b.get_layer('features').output)
                    feat_maps_b = feat_maps_out_b(x)
                    # make an update depending on the model type
                    if model_type == 'unet':
                        plot_img(feat_maps_a, epoch, fig, output_dir, "feat_map_a")
                        plot_img(feat_maps_b, epoch, fig, output_dir, "feat_map_b")
                    y_hood = depthwise_adv.get_hood(y)
                    if model_type == 'unet_mahal':
                        ood_map_a = update_ood_features(ood_map_a, y, y_hood, feat_maps_a, mahal_thresh, epoch, fig, output_dir)
                    else:
                        ood_map_b = update_ood(ood_map_b, y, y_hood, feat_maps_b, mahal_thresh, epoch, fig, output_dir)
                    plot_img(ood_map_a, epoch, fig, output_dir, "ood_map_a")
                    plot_img(ood_map_b, epoch, fig, output_dir, "ood_map_b")

                y_to_upd_sum = 1
                k = 0
                # make updates while there is no more pixels meeting all the criteria
                while y_to_upd_sum > 0:
                    y_hood = depthwise_adv.get_hood(y)

                    y_to_upd = get_update_locs(softmax, y, y_hood, p_thresh, p_crf=p_crf, sim_res=sim_res_exp)

                    y_to_upd_sum = tf.reduce_sum(y_to_upd)

                    y_pred_sparse = tf.expand_dims(tf.argmax(softmax, axis=-1), 0)
                    y_to_upd = tf.transpose(y_to_upd, perm=[3, 0, 1, 2])

                    y_updates = tf.map_fn(lambda x: get_y_updates(x[0], x[1], x[2]),
                                          elems=(tf.repeat(y_pred_sparse, n_classes, axis=0), y_to_upd,
                                                 tf.range(n_classes, dtype=tf.int64)),
                                          fn_output_signature=tf.float64)

                    y_updates = tf.transpose(y_updates, perm=[1, 2, 3, 0])
                    y = tf.where(y == 1, y, tf.cast(y_updates, tf.double))

                    if k == 0 or (k + 1) % 5 == 0:
                        plot_img(y, epoch, fig, output_dir, f"iter{k}_y_{i}")

                    k += 1

                y_updated.append(y[0])
                x_data.append(x[0])
                plot_img(y, epoch, fig, output_dir, f"y_{i}")

        # overwrite the training set with updated labels
        if epoch >= burnin_ep and epoch % period == 0 and len(y_updated) > 0:
            train_data = tf.data.Dataset.from_tensor_slices((x_data, y_updated)).batch(1)

        gc.collect()

        print("Epoch {:03d}: Loss a: {:.3f}: Loss b: {:.3f}".format(epoch, epoch_loss_avg_a.result(), epoch_loss_avg_b.result()))

    if show_ood_maps:
        ood_map_a = tf.where(np.array(y_updated) == 0, 0, ood_map_a)
        plot_img(ood_map_a, None, fig, output_dir, f"ood_map_a_fin")
        ood_map_b = tf.where(y_updated == 0, 0, ood_map_b)
        plot_img(ood_map_b, None, fig, output_dir, f"ood_map_b_fin")

    model_a.save(output_dir + 'trained_a')
    model_b.save(output_dir + 'trained_b')

    # filling parts that were not labeled with labels obtained from predicted probabilities
    train_iter = train_data.as_numpy_iterator()
    y_pred = np.array([s[1][0] for s in train_iter])
    softmax = (model_a(np.array(x_train_init), training=False) + model_b(np.array(x_train_init), training=False)) / 2
    sm_labels = (softmax.numpy() > 0.5).astype(int)
    idx_missing = np.where(np.sum(y_pred, axis=-1) == 0)
    y_pred[idx_missing[0], idx_missing[1], idx_missing[2]] = sm_labels[idx_missing[0], idx_missing[1], idx_missing[2]]

    plot_img(y_pred, None, fig, output_dir, f"y_final")

    test_pred_prob = (model_a(test_seis) + model_b(test_seis)) / 2
    plot_img(test_pred_prob, None, fig, output_dir, f"test_prob")
    plot_img(tf.argmax(test_pred_prob, axis=-1), None, fig, output_dir, f"test_y")

    # calculation of accuracies for channels only and the total
    acc_total = np.sum(y_pred[..., 1].flatten() == y_true[..., 1].flatten()) / len(y_true[..., 1].flatten())
    acc_ch = np.sum(y_pred[..., 1].flatten()[y_true[..., 1].flatten() == 1] == 1) / np.sum(y_true[..., 1].flatten() == 1)
    runtime = (time.time() - start) / 60 / epochs

    return runtime, acc_ch, acc_total

