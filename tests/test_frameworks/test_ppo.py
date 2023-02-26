"""Proximal Policy Optimization (PPO) tests"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
from unittest import TestCase
from scipy.stats import multivariate_normal
from frameworks.ppo import (clipped_surrogate_likelihood, advantage_conv, value_conv, package_dataset,
                            _gauss_prob_ratio2, ppo_loss_multiclass, ppo_loss_gauss)


class TestSurrogateLoss(TestCase):

    def test_loss(self):
        # where does gradient exist for surrogate loss?
        # should exist for all ones
        # case 1: prob ratio inside clip boundary --> gradient = advantage
        # case 2: prob ratio outside boundary + performance is better --> gradient = 0
        # case 3: prob ratio outside boundary + performance is worse --> gradient = advantage
        # case 4: prob ratio outside boundary + performance is worse --> gradient = advantage (neg version)
        xs = [tf.ones([5]), tf.ones([5]) * 10, tf.ones([5]) * -10, tf.ones([5]) * 10]
        Ats = [tf.ones([5]) * 2., tf.ones([5]) * 2., tf.ones([5]) * 2., tf.ones([5]) * -2.]
        exp_g = [tf.ones([5]) * 2., tf.ones([5]) * 0., tf.ones([5]) * 2., tf.ones([5]) * -2.]
        for x, At, eg in zip(xs, Ats, exp_g):
            with tf.GradientTape() as g:
                g.watch(x)
                y = clipped_surrogate_likelihood(x, At, 0.2)
            self.assertTrue(tf.math.reduce_all(g.gradient(y, x) == eg))


class TestDatasetGen(TestCase):
    # some quantities are calculated off graph and inserted into dataset
    # > advantage, > value target
    def setUp(self):
        self.V = np.ones((20,))
        self.reward = np.ones((19,))
        self.gamma = 0.9

    def test_advantage(self):
        # advantage calculation testing
        # if lam = 1 --> should become reward sum
        At = advantage_conv(self.V, self.reward, self.gamma, 1.)

        for i in range(19):
            a_i = -1. * self.V[i]
            gfactor = 1.
            for j in range(i, 19):
                a_i += self.reward[j] * gfactor
                gfactor *= self.gamma
            a_i += gfactor * self.V[-1]
            self.assertTrue(np.round(a_i, 4) == np.round(At[i], 4))

    def test_value_target(self):
        # value estimate?
        val = value_conv(self.V, self.reward, self.gamma)
        for i in range(20):
            v_i = 0.
            gfactor = 1.
            for j in range(i, 19):
                v_i += self.reward[j] * gfactor
                gfactor *= self.gamma
            v_i += self.V[-1] * gfactor
            self.assertTrue(np.round(v_i, 4) == np.round(val[i], 4))

    # TODO: termination tests

    def test_pkg_dset(self):
        # test with 2 sequences of different lengths
        s1 = np.zeros((11, 2))
        s2 = np.zeros((6, 2))
        v1 = np.ones((11,))
        v2 = np.ones((6,))
        r1 = np.ones((10,))
        r2 = np.ones((5,))
        a1 = np.ones((10, 3))
        a2 = np.ones((5, 3))
        terminated = [False, True]
        dset = package_dataset([{"s": s1}, {"s": s2}], [v1, v2], [r1, r2], [a1, a2], terminated, 0.9, 1.)
        for v in dset.batch(4):
            self.assertTrue(tf.math.reduce_all(tf.shape(v["s"]) == tf.constant([4, 2], dtype=tf.int32)))
            self.assertTrue(tf.math.reduce_all(tf.shape(v["action"]) == tf.constant([4, 3], dtype=tf.int32)))
            self.assertTrue(tf.math.reduce_all(tf.shape(v["adv"]) == tf.constant([4], dtype=tf.int32)))
            self.assertTrue(tf.math.reduce_all(tf.shape(v["val"]) == tf.constant([4], dtype=tf.int32)))
            break


class TestLosses(TestCase):

    def test_gauss_ratio(self):
        rng = npr.default_rng(42)
        # seems like multivariate_normal can figure out how to use diagonal covar
        N = 10
        mu = rng.random((N, 2))
        var = rng.random((N, 2))
        mu2 = rng.random((N, 2))
        var2 = rng.random((N, 2))
        x = rng.random((N, 2))
        tf_ratio = _gauss_prob_ratio2(tf.constant(x, dtype=tf.float32),
                                    tf.constant(mu, dtype=tf.float32),
                                    1. / tf.constant(var, dtype=tf.float32),
                                    tf.constant(mu2, dtype=tf.float32),
                                    1. / tf.constant(var2, dtype=tf.float32))
        for i in range(N):
            ratio = multivariate_normal.pdf(x[i:i+1], mu[i], var[i]) / multivariate_normal.pdf(x[i:i+1], mu2[i], var2[i])
            diff = np.fabs(tf_ratio[i].numpy() - ratio)
            self.assertTrue(diff < .001)

    def test_multiclass_loss(self):
        # ppo loss multiclass test
        # 3 "models"
        # > base model
        # > model that is very close to base model + improves advantage scale (should be best performance)
        # > model that is very far from base model + improves advantage scale (should be same as best)
        critic_pred = tf.zeros([8], dtype=tf.float32)
        action_np = np.zeros((8, 2))
        action_np[:4, 0] = 1.
        action_np[4:, 1] = 1.
        action = tf.constant(action_np, dtype=tf.float32)
        value_target = tf.zeros([8], dtype=tf.float32)
        advantage = tf.constant([1., -1.] * 2 + [-1., 1.] * 2, dtype=tf.float32)
        eta = 0.2  # step size
        # base model
        pi_base = tf.constant([[0.5, 0.5] for _ in range(8)], dtype=tf.float32)

        # Q? when do we get clipped?
        #   upper: x / base_prob = 1 + eta --> x_up = (1 + eta) * base_prob
        #   lower: x / base_prob = 1 - eta --> x_low = (1 - eta) * base_prob
        # here: base_prob = 0.5
        x_up = (1 + eta) * 0.5
        x_lo = (1 - eta) * 0.5

        # best model
        pi_best = tf.reshape(tf.constant([x_up, x_lo, x_lo, x_up] * 4, dtype=tf.float32), [8, 2])
        # wrong way
        pi_bad = 1. - pi_best
        # should be same as best
        pi_big = tf.reshape(tf.constant([0.99, 0.01, 0.01, 0.99] * 4, dtype=tf.float32), [8, 2])

        losses = []
        for v in [pi_best, pi_bad, pi_big]:
            loss_vf, loss_clip, negent = ppo_loss_multiclass(pi_base, v,
                                    critic_pred,
                                    action,
                                    advantage,
                                    value_target,
                                    eta)
            self.assertTrue(np.shape(loss_vf.numpy()) == (8,))
            losses.append(tf.math.reduce_mean(loss_clip).numpy())
        self.assertTrue(losses[0] < losses[1])
        self.assertTrue(np.round(losses[0], 4) == np.round(losses[2], 4))

    def test_gauss_loss(self):
        # ppo loss gauss
        # repeat the multiloss tests for gaussian space
        rng = npr.default_rng(42)
        critic_pred = tf.zeros([8], dtype=tf.float32)
        action = tf.constant(rng.random((8, 2)), dtype=tf.float32)
        value_target = tf.zeros([8], dtype=tf.float32)
        advantage = tf.constant([1., -1.] * 2 + [-1., 1.] * 2, dtype=tf.float32)
        eta = 0.3  # step size
        # everybody uses same precision
        prec = tf.ones([8, 2], dtype=tf.float32)
        # base model
        mu_base = action + 0.25 * tf.constant(rng.random((8, 2)), dtype=tf.float32)
        # good model ~ move mean towards action if advantage positive (away if neg)
        mu_good = mu_base + (action - mu_base) * advantage[:, None]
        # bad model ~ wrong direction
        mu_bad = mu_base - (action - mu_base) * advantage[:, None]

        loss_good, pr_good = ppo_loss_gauss(mu_base, prec, mu_good, prec,
                                critic_pred, action,
                                advantage, value_target, eta)
        loss_bad, pr_bad = ppo_loss_gauss(mu_base, prec, mu_bad, prec,
                                critic_pred, action,
                                advantage, value_target, eta)
        print(loss_good)
        print(loss_bad)
        self.assertTrue(np.shape(loss_good.numpy()) == (8,))
        self.assertTrue(tf.math.reduce_mean(loss_good) < tf.math.reduce_mean(loss_bad))


if __name__ == "__main__":
    T = TestSurrogateLoss()
    T.test_loss()
    T = TestDatasetGen()
    T.setUp()
    T.test_advantage()
    T.test_value_target()
    T.test_pkg_dset()
    T = TestLosses()
    T.test_gauss_ratio()
    T.test_multiclass_loss()
    T.test_gauss_loss()
