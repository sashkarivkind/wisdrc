'''
functions that support learning of feature statistics
'''
import tensorflow as tf
import numpy as np


# define function to calculate equal-frequency bins
# taken from: https://www.statology.org/equal-frequency-binning-python/
def equal_freq_bins(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))


def equal_freq_bins_for_batch(feature_map, nbins):
    bin_edges = []
    for feature in feature_map.transpose([3, 0, 1, 2]):
        bin_edges.append(
            equal_freq_bins(np.reshape(feature, [-1]), nbins)
        )
    bin_edges = np.array(bin_edges)
    return bin_edges


def init_bins(n_features, bin_edges):
    """
    TODO: Vectorize.
    See sec. 3.1 in https://arxiv.org/abs/1806.06988 for details on the soft binning procedure.
    """
    # Init bins using super class
    #     super(SoftBinStats, self).init_bins()

    # Define W + b for soft binning
    inner_edges = bin_edges[:, 1:-1]  # soft binning method using n "inner" edges for n+1 intervals
    n_edges = inner_edges.shape[1]
    #         dev = self.bin_edges.device
    W = []
    b = []
    for f in range(n_features):
        w_ = tf.reshape(tf.linspace(1, n_edges + 1, n_edges + 1), [1, -1])
        b_ = tf.math.cumsum(tf.concat([tf.zeros([1]), -inner_edges[f]], 0), 0)
        W.append(w_)
        b.append(b_)
    W = tf.reshape(tf.stack(W), (n_features, 1, n_edges + 1))  # reshape for matmul later
    b = tf.reshape(tf.stack(b), (n_features, 1, n_edges + 1))  # reshape for matmul later
    return W, b


def generate_random_couplings(n, k, discard_diagonal=False):
    M = np.random.uniform(size=(n, n))
    if discard_diagonal:
        raise NotImplemented
    return np.argwhere(M < k / (n ** 2))


def mesh_sum(a, b, axis=-1):
    # take two tensors of the same size
    # let the last dimensio of tensor a be (a1,a2,a3,a4)
    # and of tensor b (b1,b2,b3,b4)
    # returns:
    # a1+b1,... a4+b1,
    # a2+b2 ... a4+b2
    # ...
    # a4+b1  ... a4+b4
    if axis != -1:
        raise NotImplemented

    a_shape = tf.shape(a)
    a_tile_times = tf.concat([a_shape[:-1] * 0 + 1, a_shape[-1:]], axis=0)
    a = tf.tile(a, a_tile_times)

    b_shape = tf.shape(b)
    b_tile_times = tf.concat([b_shape * 0 + 1, b_shape[-1:]], axis=0)
    b_target_shape = tf.concat([b_shape[:-1], [-1]], axis=0)
    b = tf.tile(b[..., tf.newaxis], b_tile_times)
    b = tf.reshape(b, b_target_shape)

    return a + b


def get_soft_batch_counts(x, n_features, bin_edges, norm_inputs=False, tau=0.1, norm_output=True):
    """ See Neural Decision Forests paper for details on the soft binning procedure. """
    # Conv: [B, H, W, C] --> [C, BxHxW], FC: [B, C] --> [C, B]
    W, b = init_bins(n_features, bin_edges)

    x = tf.transpose(x, [3, 0, 1, 2])
    x = tf.reshape(x, [n_features, -1])
    if norm_inputs:
        raise NotImplemented
    #         x = norm_inputs(x)

    x = tf.reshape(x, (n_features, -1, 1))  # reshape for make-shift batch outer prod via matmul

    # Calculate "logits" per sample via batch outer-product.
    # x:[n_features, n_samples, 1] x W:[n_features, 1, n_bins] = [n_features, n_samples, n_bins]
    #     z = tf.matmul(x, W) + b
    z = tf.matmul(x, tf.cast(W, dtype=tf.float32)) + b

    # Calculate soft allocations per sample ("soft" --> sum to 1)
    sft_cs = tf.nn.softmax(z / tau, axis=2)  # [n_features, n_samples, n_bins]

    # Sum over samples to get total soft counts ("soft" --> real number)
    total_sft_cs = tf.reduce_sum(sft_cs, axis=1)

    if norm_output:
        total_sft_cs = total_sft_cs / tf.reduce_sum(total_sft_cs, axis=1, keepdims=True)

    return total_sft_cs


def get_soft_batch_counts_sparse2D(x, n_features, bin_edges, couplings=[], norm_inputs=False, tau=0.1,
                                   norm_output=True):
    """ TODO: REFACTOR, very similar to the 1D binning. maybe turn to class with inherence? """
    # Conv: [B, H, W, C] --> [C, BxHxW], FC: [B, C] --> [C, B]
    W, b = init_bins(n_features, bin_edges)

    x = tf.transpose(x, [3, 0, 1, 2])
    x = tf.reshape(x, [n_features, -1])
    if norm_inputs:
        raise NotImplemented

    x = tf.reshape(x, (n_features, -1, 1))  # reshape for make-shift batch outer prod via matmul

    # Calculate "logits" per sample via batch outer-product.
    # x:[n_features, n_samples, 1] x W:[n_features, 1, n_bins] = [n_features, n_samples, n_bins]
    #     z = tf.matmul(x, W) + b
    z = tf.matmul(x, tf.cast(W, dtype=tf.float32)) + b

    z_couplings = []
    for couple in couplings:
        z_couplings.append(mesh_sum(z[couple[0], ...], z[couple[1], ...]))

    z_couplings = tf.stack(z_couplings, axis=0)
    # Calculate soft allocations per sample ("soft" --> sum to 1)
    sft_cs = tf.nn.softmax(z_couplings / tau, axis=2)  # [n_features, n_samples, n_bins]

    # Sum over samples to get total soft counts ("soft" --> real number)
    total_sft_cs = tf.reduce_sum(sft_cs, axis=1)

    if norm_output:
        total_sft_cs = total_sft_cs / tf.reduce_sum(total_sft_cs, axis=1, keepdims=True)

    return total_sft_cs


def keras_loss_for_soft_binning(ref_feature_hist,
                                n_features,
                                bin_edges,
                                couplings=[],
                                tau=0.1,
                                loss_type='SKLD',
                                mode='1D',
                                custom_name='loss_over_binning'):
    kl = tf.keras.losses.KLDivergence()

    def loss_fun(y_true, y_pred):
        # here y_true is a sham input. the reference distribution is fixed for all batches
        if mode == '1D':
            soft_feature_hist = get_soft_batch_counts(y_pred,
                                                      n_features,
                                                      bin_edges,
                                                      norm_inputs=False,
                                                      tau=tau)
        elif mode == 'sparse2D':
            soft_feature_hist = get_soft_batch_counts_sparse2D(y_pred,
                                                               n_features,
                                                               bin_edges,
                                                               couplings=couplings,
                                                               norm_inputs=False,
                                                               tau=tau)
        else:
            raise NotImplementedError

        if loss_type == 'SKLD':
            loss = 0.5 * (kl(soft_feature_hist, ref_feature_hist) + kl(ref_feature_hist, soft_feature_hist))
        else:
            raise NotImplementedError
        return loss

    loss_fun.__name__ = custom_name
    return loss_fun

