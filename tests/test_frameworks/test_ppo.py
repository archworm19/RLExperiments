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

    def test_advantage_spec(self):
        # test some special cases
        # not termination
        lam = 0.95
        v = np.array([0., 1., 2., -2.])
        r = np.array([2., -2., 1.])
        deltas = np.array([r[0] + self.gamma*v[1] - v[0],
                           r[1] + self.gamma*v[2] - v[1],
                           r[2] + self.gamma*v[3] - v[2]])
        exp_advs = np.array([deltas[0] + self.gamma * lam * deltas[1] + (self.gamma * lam)**2. * deltas[2],
                             deltas[1] + self.gamma * lam * deltas[2],
                             deltas[2]])
        advs = advantage_conv(v, r, self.gamma, lam, terminated=False)
        self.assertTrue(np.all(advs == exp_advs))
        # terminated
        deltas = np.array([r[0] + self.gamma*v[1] - v[0],
                           r[1] + self.gamma*v[2] - v[1],
                           r[2] + 0.*v[3] - v[2]])
        exp_advs = np.array([deltas[0] + self.gamma * lam * deltas[1] + (self.gamma * lam)**2. * deltas[2],
                             deltas[1] + self.gamma * lam * deltas[2],
                             deltas[2]])
        advs = advantage_conv(v, r, self.gamma, lam, terminated=True)
        self.assertTrue(np.all(advs == exp_advs))

    def test_0advantage(self):
        # design a case where there should
        #       be no advantage anywhere
        # case 1: lambda = 1 --> v(t) = sum [gamma*i r(t + i)]
        rng = npr.default_rng(0)
        r = (rng.random(50) - 0.5) * 20.
        vend = rng.random()
        rv2 = np.hstack((r, [vend]))
        gamma_mask = self.gamma**(np.arange(51))
        v_gen = []
        for i in range(50):
            vi = np.sum(rv2[i:] * gamma_mask[:51-i])
            v_gen.append(vi)
        advs = advantage_conv(np.array(v_gen + [vend]), r, self.gamma, 1., terminated=False)
        self.assertTrue(np.amax(np.fabs(advs)) < 1e-7)

        # TODO: test with lambda != 1

    def test_value_target(self):
        # value estimate?
        val = value_conv(self.V[-1], self.reward, self.gamma)
        for i in range(20):
            v_i = 0.
            gfactor = 1.
            for j in range(i, 19):
                v_i += self.reward[j] * gfactor
                gfactor *= self.gamma
            v_i += self.V[-1] * gfactor
            self.assertTrue(np.round(v_i, 4) == np.round(val[i], 4))

    def test_value_spec(self):
        # some value special cases
        # 1: sparse termination
        rew = np.array([1., 0., 0., 0., 0.])
        Vend = 10.
        val = value_conv(Vend, rew, self.gamma, terminated=True)
        self.assertTrue(np.all(val == np.array([1.] + [0.] * 5)))
        # 2: sparse + no termination
        val = value_conv(Vend, rew, self.gamma, terminated=False)
        exp_base = Vend * (self.gamma ** np.arange(6))[::-1]
        exp_base[0] = exp_base[0] + 1.
        self.assertTrue(np.all(val == exp_base))
        # 3: dense + termination
        rew = np.array([1., -1., 1., -1., 1.])
        val = value_conv(-1., rew, 1., terminated=True)
        self.assertTrue(np.all(val == np.array([1., 0., 1., 0., 1., 0.])))
        # 4: dense + no termination
        val = value_conv(-1., rew, 1., terminated=False)
        self.assertTrue(np.all(val == np.array([0., -1., 0., -1., 0., -1.])))

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
        log_std = rng.random((N, 2))
        mu2 = rng.random((N, 2))
        log_std2 = rng.random((N, 2))
        x = rng.random((N, 2))
        tf_ratio = _gauss_prob_ratio2(tf.constant(x, dtype=tf.float32),
                                    tf.constant(mu, dtype=tf.float32),
                                    tf.constant(log_std, dtype=tf.float32),
                                    tf.constant(mu2, dtype=tf.float32),
                                    tf.constant(log_std2, dtype=tf.float32))
        for i in range(N):
            var = np.exp(log_std[i])**2.
            var2 = np.exp(log_std2[i])**2.
            ratio = multivariate_normal.pdf(x[i:i+1], mu[i], var) / multivariate_normal.pdf(x[i:i+1], mu2[i], var2)
            diff = np.fabs(tf_ratio[i].numpy() - ratio)
            self.assertTrue(diff < .00001)

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
            loss_clip, negent = ppo_loss_multiclass(pi_base, v,
                                    action,
                                    advantage,
                                    eta)
            self.assertTrue(np.shape(loss_clip.numpy()) == (8,))
            # entropy is positive bounded --> negent is negative bounded
            self.assertTrue(tf.math.reduce_all(negent < 0))
            losses.append(tf.math.reduce_mean(loss_clip).numpy())
        self.assertTrue(losses[0] < losses[1])
        self.assertTrue(np.round(losses[0], 4) == np.round(losses[2], 4))

    def test_gauss_loss(self):
        # ppo loss gauss
        # repeat the multiloss tests for gaussian space
        rng = npr.default_rng(42)
        action = tf.constant(rng.random((8, 2)), dtype=tf.float32)
        advantage = tf.constant([1., -1.] * 2 + [-1., 1.] * 2, dtype=tf.float32)
        eta = 0.3  # step size
        # everybody uses same log_std
        log_std = tf.ones([8, 2], dtype=tf.float32)
        # base model
        mu_base = action + 0.25 * tf.constant(rng.random((8, 2)), dtype=tf.float32)
        # good model ~ move mean towards action if advantage positive (away if neg)
        mu_good = mu_base + (action - mu_base) * advantage[:, None]
        # bad model ~ wrong direction
        mu_bad = mu_base - (action - mu_base) * advantage[:, None]

        loss_good, _, pr_good = ppo_loss_gauss(mu_base, log_std, mu_good, log_std,
                                               action, advantage, eta)
        loss_bad, _, pr_bad = ppo_loss_gauss(mu_base, log_std, mu_bad, log_std,
                                             action, advantage, eta)
        print(loss_good)
        print(loss_bad)
        self.assertTrue(np.shape(loss_good.numpy()) == (8,))
        self.assertTrue(tf.math.reduce_mean(loss_good) < tf.math.reduce_mean(loss_bad))

        # test (neg)entropy
        # high covariance --> high entropy --> low negentropy
        _, negent_hi, _ = ppo_loss_gauss(mu_base, log_std, mu_good, log_std,
                                          action, advantage, eta)
        _, negent_low, _ = ppo_loss_gauss(mu_base, log_std, mu_good, log_std * 2.,
                                         action, advantage, eta)
        self.assertTrue(tf.math.reduce_all(negent_hi > negent_low))


if __name__ == "__main__":
    T = TestSurrogateLoss()
    T.test_loss()
    T = TestDatasetGen()
    T.setUp()
    T.test_0advantage()
    T.test_advantage_spec()
    T.test_value_spec()
    T.test_advantage()
    T.test_value_target()
    T.test_pkg_dset()
    T = TestLosses()
    T.test_gauss_ratio()
    T.test_multiclass_loss()
    T.test_gauss_loss()
