"""Test forward model variance"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
from unittest import TestCase
from scipy.stats import multivariate_normal
from frameworks.intrinsic_motivation.forward_model_variance import _average_gauss, _kldiv_gauss


class TestGaussUtils(TestCase):

    def test_ave_gauss(self):
        # from definition -->
        # mean of ave gauss = Mean(sum[a_i X_i])
        #   where a_i = 1/N
        # variance of ave guass = Var(sum[a_i X_i])
        #   where a_i = 1/N
        # and X_i = samples from gaussian i
        # here, we do a law of large number test
        rng = npr.default_rng(42)
        num_sample = 15000
        for _ in range(3):
            mu1 = rng.random(2) - 0.5
            mu2 = rng.random(2) - 0.5
            cov1 = np.diag(rng.random(2))
            cov2 = np.diag(rng.random(2))
            # --> num_sample x d
            num_sample = 8000
            samples1 = rng.multivariate_normal(mu1, cov1, size=num_sample)
            samples2 = rng.multivariate_normal(mu2, cov2, size=num_sample)
            mean_sample = (samples1 + samples2) / 2
            empirical_ave_mu = np.mean(mean_sample, axis=0)
            diff = mean_sample - np.mean(mean_sample, axis=0)
            empirical_ave_cov = np.matmul(diff.T, diff) / np.shape(diff)[0]
            exp_mu, exp_var = _average_gauss(tf.constant([[mu1, mu2]]),
                                             tf.constant([[np.diag(cov1), np.diag(cov2)]]))
            # mean comparison:
            mean_err = np.fabs(empirical_ave_mu - exp_mu.numpy())
            self.assertTrue(np.all(mean_err < .05))
            # covariance comparison:
            cov_err = np.fabs(np.diag(empirical_ave_cov) - exp_var.numpy())
            self.assertTrue(np.all(cov_err < .05))

    def test_kldiv_gauss(self):
        # discrete approximationt to kldiv
        # > formula = integral p(x) log [ p(x) / q(x) ] dx
        # iter over support with step size dx --> calc approx
        # 2 dimensional test
        rng = npr.default_rng(42)
        x_support = np.linspace(-4, 4, 40)
        dx = np.mean(x_support[1:] - x_support[:-1])
        (x1, x2) = np.meshgrid(x_support, x_support)
        x2_support = np.vstack((np.reshape(x1, -1), np.reshape(x2, -1))).T
        for _ in range(3):  # for N different distros
            mu1 = rng.random(2) - 0.5
            mu2 = rng.random(2) - 0.5
            cov1 = np.diag(rng.random(2))
            cov2 = np.diag(rng.random(2))
            y1 = multivariate_normal.pdf(x2_support, mu1, cov1)
            y2 = multivariate_normal.pdf(x2_support, mu2, cov2)
            approx_kld = np.sum(y1 * np.log(y1 / y2)) * (dx**2.)
            tmu1 = tf.constant([[mu1]])
            tmu2 = tf.constant([[mu2]])
            tcov1 = tf.constant([[np.diag(cov1)]])
            tcov2 = tf.constant([[np.diag(cov2)]])
            exp_kld = _kldiv_gauss(tmu1, tmu2, tcov1, tcov2)
            kld_err = np.fabs(approx_kld - exp_kld.numpy()[0, 0])
            self.assertTrue(kld_err < .005)


if __name__ == "__main__":
    T = TestGaussUtils()
    T.test_ave_gauss()
    T.test_kldiv_gauss()
