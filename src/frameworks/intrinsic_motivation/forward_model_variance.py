"""Forward Model Variance Intrinsic Reward

    Relies on bagging principle:
    > multiple, independent models fit on different subsets of trajectories

    Theory: once a state region has been visited a reasonable number of times
        --> independent model prediction models will converge
    Key: we should get prediction convergence even if high prediciton error
        > Ex: gaussian prediction. In high error environment with large
            number of visits --> model converge to same gaussian
            with high variance
"""
import tensorflow as tf
import tensorflow.linalg as tfalg


# General Continuous space


# TODO: obsolete?
def cov(x: tf.Tensor) -> tf.Tensor:
    # TODO: should we reduce mean across models???
    # x = batch_size x num_model x d
    #   each model is a different sample
    # --> returns: batch_size x d x d covariance matrix
    mu = tf.math.reduce_mean(x, axis=1, keepdims=True)
    dif = x - mu
    return tf.math.reduce_sum(tf.expand_dims(dif, 3) * tf.expand_dims(dif, 2),
                              axis=1)


# TODO: obsolete?
def detcov(x: tf.Tensor) -> tf.Tensor: 
    # x = batch_size x num_model x d
    #   each model is a different sample
    # --> returns: tensor with shape batch_size
    #       = determinant of covariance matrix for each batch sample
    #           where models are viewed as different samples
    #   NOTE: this is likely to be slow for large d
    return tfalg.det(cov(x))


# TGaussian Variance Calcs
# Ideas?
# > KL-Divergence(average gaussian)
# > Average(Pairwise KL-divergence)


def _average_gauss(means: tf.Tensor, variances: tf.Tensor) -> tf.Tensor:
    # returns distro for Y, where Y = (1/N) sum_i [ X_i ]
    #   X_i = samples from the gaussian i
    #
    #   means = batch_size x num_model x d
    #   variances = diagonals of diagonal covariance matrices
    #         = batch_size x num_model x d
    #
    #   returns: mean = batch_size x d; variance (diagonal) = batch_size x d
    #
    # Properties of variance operator?
    # https://en.wikipedia.org/wiki/Variance#Properties
    # > Var(X + a) = Var(X)
    # > Var(aX) = a^2 Var(X)
    # > Var(aX + bY) = a^2Var(X) + b^2Var(Y) + 2abCov(X, Y)
    # > Var(sum[a_i X_i]) = sum[a_i^2 Var(X_i)] + 2 sum [ sum [ a_i a_j Cov(X_i, X_j)]], i!=j
    # Covariance between X_i and X_j?
    # > In this case --> all samples are independent, right? (even though individual distros have covariance?)
    #   YES!
    # --> Cov matrix = diagonal --> sum of sums terms goes to 0 (cuz they're off-diagonal only)
    # --> = sum[a_i^2 Var(X_i)] = sum[(1/N)^2 Var(X_i)] for average of gaussians case
    # Properties of mean operator?
    # > expected value operator is linear: E[aX + bY] = a E[X] + b E[Y]
    N = tf.shape(means)[1]
    mu_y = tf.math.reduce_mean(means, axis=1)
    var_y = tf.math.divide(tf.math.reduce_sum(variances, axis=1),
                           tf.math.pow(tf.cast(N, variances.dtype), 2.))
    return mu_y, var_y

def _kldiv_gauss(mu1: tf.Tensor, mu2: tf.Tensor,
                 var1: tf.Tensor, var2: tf.Tensor) -> tf.Tensor:
    # mu1, mu2 = batch_size x num_model x d
    # var1, var2 = batch_size x num_model x d = diagonal covars
    #               for each batch sample
    #
    # formula
    #   = (1/2) ln | var2 var1^-1 | - d/2 + 1/2 trace(var1 var2^-1)
    #       + 1/2 <(mu2 - mu1) var2^-1 (mu2 - mu1)>
    #
    # for diagonal --> ln | var2 var1^-1 | = ln prod_i (var2_i / var1_i)
    #                                      = sum_i [ ln var2_i - ln var1_i
    #                                      = t1
    #              --> trace(var var2^-1) = sum_i var1_i / var2_i
    #                                     = t3
    #              --> (mu2 - mu1) var2^-1 (mu2 - mu1) = sum((mu2 - mu1)^2 / var2)
    #                                                  = t4
    t1 = tf.math.reduce_sum(tf.math.log(var2) - tf.math.log(var1), axis=1)
    t2 = -1. * tf.cast(tf.shape(mu1)[-1], t1.dtype)
    t3 = tf.math.reduce_sum(tf.math.divide(var1, var2))
    t4 = tf.math.reduce_sum(tf.math.divide(tf.math.pow(mu2 - mu1, 2.),
                                           var2), axis=1)
    return 0.5 * (t1 + t2 + t3 + t4)

def kldiv_ave_gauss(means: tf.Tensor, variances: tf.Tensor) -> tf.Tensor:
    """Average KL Divergence between average gaussian and all gaussians

        NOTE: we do not propagate gradients through the average gaussian
            --> better for kldiv stability cuz division becomes
                division by a constant

    Args:
        means (tf.Tensor): mean of every gaussian
            batch_size x num_gauss x d
        variances (tf.Tensor): variance of every gaussian
            assumed to be diagonal
            batch_size x num_gauss x d

    Returns:
        tf.Tensor: average kldiv(average gauss, all gaussians)
            shape = batch_size
    """
    # > calculate average gaussian
    # > calculate kl divergence between ave gauss and all other gaussians
    # > return average of kldivs
    #
    #   means = batch_size x num_model x d
    #   variances = diagonals of diagonal covariance matrices
    #         = batch_size x num_model x d
    #
    # kldiv between diagonal gaussians?
    # > 

    # --> batch_size x d
    ave_mu, ave_var = _average_gauss(means, variances)
    # --> batch_size x num_model
    kld = _kldiv_gauss(means, tf.stop_gradient(tf.expand_dims(ave_mu, 1)),
                       variances, tf.stop_gradient(tf.expand_dims(ave_var, 1)))
    return tf.math.reduce_mean(kld, axis=1)


# TODO: Categorical Variance Calcs
# > Average (x-entropy from average probs)?
# > Average Pairwise x-entropy?


# TODO: high order funcs
# Q? should errors and variances be computed separately?
# ... probably cuz variances won't be part of the graph
# Q? can we use existing forward model errors?
#   nope: these depend on parallel model signatures
