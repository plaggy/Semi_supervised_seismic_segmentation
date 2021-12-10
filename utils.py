import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from sklearn.covariance import MinCovDet
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import geostatspy.GSLIB as GSLIB
import geostats_masked_sisim as geostats
import pandas as pd


"""
The function to create an array with updates to labels of the same shape as
labels themselves. Uses model predictions and update locations in the form of a binary
array as input.
"""
@tf.function
def get_y_updates(y_pred, y_to_upd, channel):
    y_updates_ch = tf.where(y_to_upd > 0, tf.cast(y_pred == channel, tf.float64), 0)

    return y_updates_ch


"""
This class is desinged to find neighborhoods for each input channel individually.
It's implemented by applying a convolutional operation with the kernel size specified to each channel 
"""
class DepthwiseAdv:
    def __init__(self, hood_kernels):
        self.hood_kernels = tf.convert_to_tensor(hood_kernels, dtype=tf.int32)

    def get_hood(self, inp):
        inp = tf.transpose(inp, perm=[3, 1, 2, 0])
        inp = tf.map_fn(lambda x: self._channel_conv(x[0], x[1]),
                        elems=(inp, self.hood_kernels),
                        fn_output_signature=tf.float64)
        inp = tf.transpose(inp, perm=[3, 1, 2, 0])
        return inp

    def _channel_conv(self, inp, kernel):
        inp = tf.expand_dims(inp, axis=0)
        res = tf.keras.layers.Conv2D(1, (kernel.numpy(), kernel.numpy()), padding='same', use_bias=False,
                                     kernel_initializer=tf.keras.initializers.Ones, trainable=False)(inp)
        return res[0]


"""
The function obtains the SISIM results for pixels defined by a set of indices sim_mask_idc (index mask)
"""
def get_sim(y, sim_mask_idc):
    y_sparse = tf.where(tf.reduce_sum(y, axis=-1) > 0, tf.argmax(y, axis=-1), -1).numpy()[0]
    xy_cols = []
    n_classes = y.shape[-1]
    for i in range(n_classes):
        idx = np.where(y_sparse == i)
        idx = np.concatenate([idx[1][:, np.newaxis], idx[0][:, np.newaxis]], axis=1)
        idx = np.concatenate([idx, np.ones((len(idx), 1)) * i], axis=1)
        xy_cols.append(idx)

    xy_cols = np.concatenate(xy_cols)

    sisim_df = pd.DataFrame(xy_cols, columns=['X', 'Y', 'Facies'], dtype=int)
    varios = []
    varios.append(GSLIB.make_variogram(nug=0.0, nst=1, it1=1, cc1=1.0, azi1=0, hmaj1=150,
                                       hmin1=150))  # shale indicator variogram
    varios.append(GSLIB.make_variogram(nug=0.0, nst=1, it1=1, cc1=1.0, azi1=340, hmaj1=100,
                                       hmin1=70))  # sand indicator variogram
    varios.append(GSLIB.make_variogram(nug=0.0, nst=1, it1=1, cc1=1.0, azi1=0, hmaj1=50,
                                       hmin1=20))  # sand indicator variogram
    thresh = [x for x in range(n_classes)]
    dummy_trend = np.zeros((10, 10))
    gcdf = [0.78, 0.19, 0.03][:n_classes]  # the global proportions of the categories
    start = time.time()
    sim_res = geostats.sisim(sisim_df, 'X', 'Y', 'Facies', ivtype=0, koption=0, ncut=n_classes, thresh=thresh,
                             gcdf=gcdf, trend=dummy_trend, tmin=-999, tmax=999, zmin=0.0, zmax=1.0,
                             ltail=1, ltpar=1, middle=1, mpar=0, utail=1, utpar=2, nx=y.shape[2],
                             xmn=0, xsiz=1, ny=y.shape[1], ymn=0, ysiz=1, seed=73074, ndmin=0,
                             ndmax=10, nodmax=10, mults=1, nmult=3, noct=-1, radius=100, ktype=0,
                             vario=varios[:n_classes], mask_idc=sim_mask_idc) #
    print(f"sim time: {(time.time() - start) / 60} min")

    return sim_res


"""
Outputs the result of a CRF application. The part with the pairwise bilateral kernel
is commented out, it was affecting the result negatively.
"""
def get_crf(x, softmax, output_dir, ep, bilat_sxy=250, bilat_rgb=20, bilat_compat=20, gauss_sxy=1, gauss_compat=3):
    n_classes = softmax.shape[-1]
    d = dcrf.DenseCRF2D(x.shape[2], x.shape[1], n_classes)
    U = tf.transpose(softmax, perm=[3, 0, 1, 2])
    U = tf.reshape(U, (n_classes, -1)).numpy()
    U = unary_from_softmax(U)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=gauss_sxy, compat=gauss_compat, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    img = np.repeat(x[0].numpy(), 3, axis=-1).astype(np.uint8)
    # uncomment to use the pairwise bilateral kernel
    '''
    d.addPairwiseBilateral(sxy=bilat_sxy, srgb=bilat_rgb, rgbim=img,
                           compat=bilat_compat,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    '''
    Q = d.inference(4)
    p_crf = np.array(Q).reshape((n_classes, x.shape[1], x.shape[2]))
    p_crf = np.transpose(p_crf, [1, 2, 0])

    # uncomment to print out CRF outputs
    '''
    if ep == 0 or (ep + 1) % 5 == 0:
        for c in range(p_crf.shape[-1]):
            plt.imsave(output_dir + f"p_crf_ep_{ep}_c{c}.jpg", p_crf[..., c])
    '''

    return p_crf


"""
Transforms a sparse label map into a multi-channel one
"""
@tf.function
def expand_sparse(pred_sparse, channel):
    pred_channel = tf.where(pred_sparse == channel, 1, 0)

    return pred_channel


"""
Calculates the locations for updates applying all the constraints specified
"""
@tf.function
def get_update_locs(softmax, y, y_hood, p_thresh, p_crf=None, sim_res=None):
    y = tf.repeat(tf.reduce_sum(y, axis=-1, keepdims=True), tf.shape(softmax)[-1], axis=-1)
    y_to_upd = tf.zeros_like(y_hood)

    if not (p_crf is None) | (sim_res is None):
        idx = tf.where((softmax > p_thresh) & (y == 0) & (y_hood > 0)
                       & (p_crf > p_thresh) & (sim_res == 1))
    elif not p_crf is None:
        idx = tf.where((softmax > p_thresh) & (y == 0) & (y_hood > 0)
                       & (p_crf > p_thresh))
    elif not sim_res is None:
        idx = tf.where((softmax > p_thresh) & (y == 0) & (y_hood > 0)
                       & (sim_res == 1))
    else:
        y_to_upd = tf.where((softmax > p_thresh) & (y == 0) & (y_hood > 0), y_hood, 0)
        return y_to_upd

    if len(idx) > 0:
        y_to_upd = tf.tensor_scatter_nd_update(y_to_upd, idx, tf.gather_nd(y_hood, idx))

    return y_to_upd


"""
The function is to update the OOD label map using features extracted from an additional 
convolutional layer as input (with modified unet). The Mahalanobis distance is used to quantify
the difference between labeled pixels and pixels being considered for an update.
The percentile threshold is used to mark pixels as OOD. 
"""
def update_ood_features(ood_map, y, y_hood, feat_maps, p_thresh, epoch, fig, output_dir):
    means = []
    feat_maps_centered = []
    for c in range(y.shape[-1]):
        mean = tf.reduce_mean(tf.gather_nd(feat_maps, tf.where(y[..., c] == 1)), axis=0)
        means.append(mean)

        feat_map_centered = feat_maps - tf.expand_dims(tf.expand_dims(tf.expand_dims(mean, 0), 0), 0)
        feat_map_centered_distr = tf.gather_nd(feat_map_centered, tf.where(y[..., c] == 1))
        feat_map_centered_distr = feat_map_centered_distr.numpy()
        feat_maps_centered.append(feat_map_centered_distr)

        robust_cov = MinCovDet(assume_centered=True).fit(feat_map_centered_distr)
        locs = tf.where((y_hood[..., c] > 0) & (tf.reduce_sum(y, axis=-1) == 0))
        feat_map_centered_dist = tf.gather_nd(feat_map_centered, locs)
        feat_map_centered_dist = feat_map_centered_dist.numpy()

        distances = robust_cov.mahalanobis(feat_map_centered_dist)

        dist_map = tf.zeros_like(y[..., c])
        dist_map = tf.tensor_scatter_nd_update(dist_map, locs, distances)
        plot_img(dist_map, epoch, fig, output_dir, f"dist_map_{c}")

        thresh = np.percentile(distances, p_thresh)
        is_ood = dist_map > thresh
        is_ood = tf.gather_nd(is_ood, locs)
        print(f" num ood: {tf.reduce_sum(tf.cast(is_ood, tf.int32)).numpy()}")
        locs = tf.concat([locs, tf.ones((len(locs), 1), dtype=tf.int64) * c], axis=-1)
        if tf.reduce_sum(tf.cast(is_ood, tf.int32)) > 0:
            ood_map = tf.tensor_scatter_nd_update(ood_map, tf.boolean_mask(locs, is_ood), tf.ones(tf.reduce_sum(tf.cast(is_ood, tf.int32))))

    return ood_map



"""
The function is to update the OOD label map using logits extracted from
the final layer of the regular unet. 
"""
def update_ood(ood_map, y, y_hood, feat_maps, p_thresh, epoch, fig, output_dir):
    means = []
    for c in range(y.shape[-1]):
        means.append(tf.reduce_mean(tf.gather_nd(feat_maps[..., c], tf.where(y[..., c] == 1))))
    feat_maps_centered = feat_maps - tf.expand_dims(tf.expand_dims(tf.expand_dims(means, 0), 0), 0)
    feat_map_centered_distr = tf.gather_nd(feat_maps_centered, tf.where(y == 1))
    feat_map_centered_distr = tf.reshape(feat_map_centered_distr, (-1, 1)).numpy()

    robust_cov = MinCovDet(assume_centered=True).fit(feat_map_centered_distr)
    cov = np.cov(feat_map_centered_distr.T)
    # cov_manual = np.mean(feat_maps_centered ** 2)
    if np.ndim(cov) < 2:
        cov_inv = 1 / cov
    else:
        cov_inv = np.linalg.inv(cov)

    for c in range(y.shape[-1]):
        locs = tf.where((y_hood[..., c] > 0) & (tf.reduce_sum(y, axis=-1) == 0))
        feat_map_centered_dist = tf.gather_nd(feat_maps_centered[..., c], locs)
        feat_map_centered_dist = tf.reshape(feat_map_centered_dist, (-1, 1)).numpy()

        distances = robust_cov.mahalanobis(feat_map_centered_dist)

        dist_map = tf.zeros_like(y[..., c])
        dist_map = tf.tensor_scatter_nd_update(dist_map, locs, distances)
        plot_img(dist_map, epoch, fig, output_dir, f"dist_map")

        thresh = np.percentile(distances, p_thresh)
        is_ood = distances > thresh
        # is_ood = tf.gather_nd(is_ood, locs)
        print(f" num ood: {np.sum(is_ood)}")
        locs = tf.concat([locs, tf.ones((len(locs), 1), dtype=tf.int64) * c], axis=-1)
        if tf.reduce_sum(tf.cast(is_ood, tf.int32)) > 0:
            ood_map = tf.tensor_scatter_nd_update(ood_map, tf.boolean_mask(locs, is_ood), tf.ones(tf.reduce_sum(tf.cast(is_ood, tf.int32))))

    return ood_map


"""
The function is to update the OOD label map using probabilities 
obtained from sigmoids, each sigmoid gives probabilities for a particular class
The last value of the sigm_thresh used was 0.6
"""
def update_ood_sigmoid(ood_map, probs, y, y_hood, sigm_thresh):
    new_idx = tf.where((tf.reduce_max(y_hood, axis=-1) > 0) & (tf.reduce_max(y, axis=-1) == 0) &
                       (tf.reduce_max(probs, axis=-1) < sigm_thresh))
    if len(new_idx) > 0:
        ood_map = tf.tensor_scatter_nd_update(ood_map, new_idx, tf.ones(len(new_idx)))

    return ood_map


"""
Plotting images
"""
def plot_img(img, ep, fig, output_dir, name, cmap=None):
    if ep is None or ep == 0 or (ep + 1) % 1 == 0:
        if np.ndim(img) == 4:
            for i, item in enumerate(img):
                for c in range(item.shape[-1]):
                    im = plt.imshow(item[..., c], cmap=cmap)
                    fig.colorbar(im)
                    if ep is None:
                        plt.savefig(output_dir + f'{name}_{i}_c{c}.jpg')
                    else:
                        plt.savefig(output_dir + f'ep{ep}_{name}_{i}_c{c}.jpg')
                    plt.clf()

        elif np.ndim(img) == 3:
            for i, item in enumerate(img):
                im = plt.imshow(item, cmap=cmap)
                fig.colorbar(im)
                if ep is None:
                    plt.savefig(output_dir + f'{name}_{i}.jpg')
                else:
                    plt.savefig(output_dir + f'ep{ep}_{name}_{i}.jpg')
                plt.clf()

        elif np.ndim(img) == 2:
            im = plt.imshow(img, cmap=cmap)
            fig.colorbar(im)
            if ep is None:
                plt.savefig(output_dir + f'{name}.jpg')
            else:
                plt.savefig(output_dir + f'ep{ep}_{name}.jpg')
            plt.clf()

    return


"""
Calculates the mean and covariance of the multivariate input (not used)
"""
def get_statistics(feat_maps, y):
    means = []
    for c in range(y.shape[-1]):
        means.append(tf.reduce_mean(tf.gather_nd(feat_maps[..., c], tf.where(y[..., c] == 1))))
    means = tf.convert_to_tensor(means)
    cov = tf.reduce_mean((feat_maps - tf.expand_dims(tf.expand_dims(tf.expand_dims(means, 0), 0), 0)) ** 2)

    return means, cov


"""
This calculates rms amplitudes for input seismic images. Implemented in a naive way
using a loop to perform calculations in a sliding window. Can be optimized if needed. 
"""
def rms_amplitudes(data, rms_window):
    data_rms = np.zeros_like(data)[:, :, :, 0]
    data = np.pad(data, ((0, 0), (rms_window // 2, rms_window // 2), (rms_window // 2, rms_window // 2), (0, 0)))
    w_start_x = 0
    w_start_y = 0
    for s in range(data_rms.shape[1] * data_rms.shape[2]):
        data_rms[:, w_start_x, w_start_y] = np.sqrt(np.sum(
            data[:, w_start_x:w_start_x + rms_window, w_start_y:w_start_y + rms_window, 0] ** 2,
            axis=(1, 2)) / rms_window ** 2)

        w_start_x += 1

        if w_start_x == data.shape[1] - rms_window + 1:
            w_start_x = 0
            w_start_y += 1

    return data_rms


"""
Superloss implementation
"""
class SuperLoss:
    def __init__(self, n_classes, lambd):
        self.tau = tf.math.log(tf.cast(n_classes, tf.float32))
        self.lambd = lambd

    @tf.function
    def get_loss(self, loss_in):
        sigma = self.get_sigma(loss_in)
        loss_out = (loss_in - self.tau) * sigma + self.lambd * tf.math.log(sigma) ** 2
        return loss_out

    @tf.function
    def get_sigma(self, loss):
        beta = (loss - self.tau) / self.lambd
        arg = 0.5 * tf.math.maximum(-2. / tf.exp(1.), beta)
        sigma = tf.math.exp(-tfp.math.lambertw(arg))
        return sigma