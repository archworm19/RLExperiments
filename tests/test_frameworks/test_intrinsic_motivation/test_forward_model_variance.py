"""Test forward model variance"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
from unittest import TestCase
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


if __name__ == "__main__":
    T = TestGaussUtils()
    T.test_ave_gauss()
