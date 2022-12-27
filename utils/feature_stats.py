'''
functions that support learning of feature statistics
'''
import tensorflow as tf

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


def keras_loss_for_soft_binning(ref_feature_hist, num_features, bin_edges, tau=0.1, loss_type='SKLD'):
    kl = tf.keras.losses.KLDivergence()

    def loss_fun(y_true, y_pred):
        # here y_true is a sham input. the reference distribution is fixed for all batches
        soft_feature_hist = get_soft_batch_counts(y_pred,
                                                  num_features,
                                                  bin_edges,
                                                  norm_inputs=False,
                                                  tau=tau)
        if loss_type == 'SKLD':
            loss = 0.5 * (kl(soft_feature_hist, ref_feature_hist) + kl(ref_feature_hist, soft_feature_hist))
        else:
            raise NotImplementedError
        return loss

    return loss_fun
