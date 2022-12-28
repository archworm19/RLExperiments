"""Q learning tests"""
import tensorflow as tf
from typing import List
from unittest import TestCase
from tensorflow.keras.layers import Dense
from frameworks.q_learning import (calc_q_error_sm, _greedy_select, calc_q_error_huber,
                                   calc_q_error_critic, calc_q_error_actor,
                                   _redistribute_weight, _calc_q_from_distro,
                                   calc_q_error_distro_discrete)
from frameworks.layer_signatures import ScalarModel, ScalarStateModel, DistroModel


class AModel(ScalarModel):
    # returns action_t[:, target_idx]

    def __init__(self, target_idx: int):
        super(AModel, self).__init__()
        self.target_idx = target_idx

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]):
        return action_t[:, self.target_idx]


class TestHuber(TestCase):

    def test_huber(self):
        q1 = tf.constant([1., 2., 3., 4., 2.5])
        q2 = tf.constant([3., 2. , 1., 0., 2.0])
        reward = tf.constant([0., 0., 0., 0., 0.])
        err, _ = calc_q_error_huber(q1, q2, reward, 1.)
        target = tf.constant([1.5, 0., 1.5, 3.5, 0.124999])
        self.assertTrue(tf.math.reduce_all(tf.round(100 * err) ==
                                           tf.round(100 * target)))


class TestQL(TestCase):

    def setUp(self) -> None:
        self.model0 = AModel(0)
        self.model1 = AModel(1)

    def test_greedy_select(self):
        state_t1 = tf.constant([[1., 2.],
                                [3., 4.],
                                [1., 2.]])
        max_q, qs =_greedy_select(self.model0, 3, state_t1)
        self.assertTrue(tf.math.reduce_all(max_q == tf.constant([0, 0], tf.int64)))
        self.assertTrue(tf.math.reduce_all(qs == tf.constant([[1., 0., 0.],
                                                              [1., 0., 0.]])))
        max_q, qs =_greedy_select(self.model1, 3, state_t1)
        self.assertTrue(tf.math.reduce_all(max_q == tf.constant([1, 1], tf.int64)))
        self.assertTrue(tf.math.reduce_all(qs == tf.constant([[0., 1., 0.],
                                                              [0., 1., 0.]])))

    def test_double_q(self):
        gamma = 0.8
        batch_size = 4
        reward_t1 = tf.zeros(batch_size)
        state_t = [tf.zeros(batch_size)]
        state_t1 = [tf.zeros(batch_size)]
        action_t = tf.constant([[0, 0, 1] * batch_size], dtype=tf.float32)
        terminatiion = tf.zeros(batch_size)
        num_action = 3

        # same model (within reward = 0)
        # --> max_a[Q(t+1)] = 1 for each sample
        # --> target = 0 + gamma * 1
        # if feed in action = 0 --> error = gamma^2
        qerr, yt = calc_q_error_sm(self.model0, self.model0, self.model0,
                                   action_t,
                                   reward_t1,
                                   state_t,
                                   state_t1,
                                   terminatiion,
                                   num_action,
                                   gamma,
                                   huber=False)
        # 100 = precision
        self.assertTrue(tf.math.reduce_all(tf.round(100 * qerr) ==
                                           tf.round(100 * gamma**2.)))
        self.assertTrue(tf.math.reduce_all(tf.round(100 * yt) ==
                                           tf.round(100 * gamma)))

        # different model (with 0 reward)
        # --> max_a[Q(t+1)] = 0 for each sample
        # --> target = 0 + gamma * 0
        qerr, yt = calc_q_error_sm(self.model0, self.model0, self.model1,
                                   action_t,
                                   reward_t1,
                                   state_t,
                                   state_t1,
                                   terminatiion,
                                   num_action,
                                   gamma)
        self.assertTrue(tf.math.reduce_all(tf.round(100 * qerr) ==
                                           tf.round(100 * 0.)))
        self.assertTrue(tf.math.reduce_all(tf.round(100 * yt) ==
                                           tf.round(100 * 0.)))


class QModel(ScalarModel):

    def __init__(self):
        super(QModel, self).__init__()

    def call(self, action: tf.Tensor, state: List[tf.Tensor]):
        # assumes: state and action space have same dimensionality
        # both are batch_size x ...
        return action + state[0]


class PiModel(ScalarStateModel):
    # returns action_t[:, target_idx]

    def __init__(self):
        super(PiModel, self).__init__()
        self.d = Dense(1)

    def call(self, state_t: List[tf.Tensor]):
        return self.d(tf.expand_dims(state_t[0], 0))[:,0]


class TestQLcont(TestCase):

    def setUp(self) -> None:
        self.Q = QModel()
        self.pi = PiModel()

    def test_positive_control(self):
        # def: Q(pi, s) = r + gamma * Q(pi, s)
        # --> error = d(Q - [r + gamma * Q])
        #           = d(Q * (1 - gamma) - r)
        batch_size = 8
        s = tf.random.uniform([batch_size])
        r = tf.random.uniform([batch_size])
        term = tf.zeros([batch_size])
        gamma = 0.85
        a = self.pi([s])
        Qval = self.Q(self.pi([s]), [s])
        err, Y_t = calc_q_error_critic(self.Q, self.Q, self.pi, a,
                                     r, [s], [s], term, gamma, huber=False)
        target = tf.pow(Qval * (1. - gamma) - r, 2.)
        self.assertTrue(tf.math.reduce_all(tf.round(err * 100) ==
                                           tf.round(target * 100)))

    def test_negative_control(self):
        # positive control condition fails when state t1 is changed
        #   <-- models is operating in a different state
        batch_size = 8
        s = tf.random.uniform([batch_size])
        s_t1 = s + 1.
        r = tf.random.uniform([batch_size])
        term = tf.zeros([batch_size])
        gamma = 0.85
        a = self.pi([s])
        Qval = self.Q(self.pi([s]), [s])
        err, Y_t = calc_q_error_critic(self.Q, self.Q, self.pi, a,
                                       r, [s], [s_t1], term, gamma, huber=False)
        target = tf.pow(Qval * (1. - gamma) - r, 2.)
        self.assertFalse(tf.math.reduce_all(tf.round(err * 100) ==
                                            tf.round(target * 100)))

    def test_actor_grad(self):
        # first: let's see if anything happens
        batch_size = 8
        s = tf.random.uniform([batch_size])
        # build weights:
        _ = self.Q(self.pi([s]), [s])
        Q0 = -1 * calc_q_error_actor(self.Q, self.pi, [s])
        # gradient ascent for a few steps:
        opt = tf.keras.optimizers.SGD(0.1)
        for _ in range(10):
            with tf.GradientTape() as tape:
                loss = calc_q_error_actor(self.Q, self.pi, [s])
            g = tape.gradient(loss, self.pi.trainable_weights)
            opt.apply_gradients(zip(g, self.pi.trainable_weights))
        Qfin = -1 * calc_q_error_actor(self.Q, self.pi, [s])
        self.assertTrue(Qfin > Q0)


class TestDistroQ(TestCase):
    # shared components in distribution-based Q learning

    def test_trivial(self):
        # trivial example: state to same state due to gamma = 1.
        # remember: atoms_static (z) is in reward space
        Vmin = 1.
        Vmax = 3.
        atoms_probs = tf.constant([[0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0]], dtype=tf.float32)
        reward = tf.constant([0., 0.])
        weights = _redistribute_weight(Vmin, Vmax, atoms_probs, reward, 1.)
        target = tf.constant([[0, 0, 1],
                            [1, 0, 0]], dtype=tf.float32)
        assert tf.math.reduce_all(tf.round(100. * weights) ==
                                tf.round(100. * target))
        Q = _calc_q_from_distro(Vmin, Vmax, atoms_probs)
        Q_target = tf.constant([3., 1.])
        assert tf.math.reduce_all(tf.round(100. * Q) ==
                                tf.round(100. * Q_target))

    def test_extreme(self):
        # extreme case: extreme rewards that saturate at Vmin or Vmax
        Vmin = 1.
        Vmax = 3.
        v = 1. / 3.
        atoms_probs = tf.constant([[v, v, v],
                                [v, v, v]], dtype=tf.float32)
        reward = tf.constant([-50., 50.])
        weights = _redistribute_weight(Vmin, Vmax, atoms_probs, reward, 1.)
        target = tf.constant([[1, 0, 0],
                            [0, 0, 1]], dtype=tf.float32)
        assert tf.math.reduce_all(tf.round(100. * weights) ==
                                tf.round(100. * target))
        Q = _calc_q_from_distro(Vmin, Vmax, atoms_probs)
        Q_target = tf.constant([2., 2.])
        assert tf.math.reduce_all(tf.round(100. * Q) ==
                                tf.round(100. * Q_target))


class QModelUniformD(DistroModel):
    # returns constant K

    def __init__(self, num_atoms: int = 5, K: int = 1.):
        super(QModelUniformD, self).__init__()
        self.num_atoms = num_atoms
        self.v = tf.reshape(tf.constant([K] * num_atoms, tf.float32), [1, -1])

    def call(self, action: tf.Tensor, state: List[tf.Tensor]):
        # --> batch_size x num_atoms
        return tf.tile(self.v, [tf.shape(action)[0], 1])


class expQDist(ScalarModel):
    # expectation across a distro model

    def __init__(self, qdist: QModelUniformD, Vmin: float, Vmax: float):
        super(expQDist, self).__init__()
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.qdist = qdist

    def call(self, action: tf.Tensor, state: List[tf.Tensor]):
        v = self.qdist(action, state)
        atom_probs = tf.nn.softmax(v, axis=1)
        return _calc_q_from_distro(self.Vmin, self.Vmax, atom_probs)


class TestDistroQdiscrete(TestCase):
    # distributional approach + discrete control

    def test_q_err(self):
        # same model + rewards + (gamma=1) = 0 -->
        # weights proportional to probs?
        num_atoms = 11
        gamma = 1.
        Q = QModelUniformD(num_atoms)
        Vmin = -10.
        Vmax = 10.
        Qexp = expQDist(Q, Vmin, Vmax)
        action = tf.constant([[1, 0, 0, 0],
                              [0, 1, 0, 0]], tf.float32)
        reward = tf.constant([0., 0.], tf.float32)
        state = [tf.constant([0., 0.], tf.float32)]
        term = tf.constant([0, 0,], tf.float32)
        # 0 representation vector = which atom is active
        #   when reward = 0?
        v0 = [0] * num_atoms
        v0[6] = 1
        vector0 = tf.constant(v0, tf.float32)
        Qerr, weights = calc_q_error_distro_discrete(Q, Qexp, Q,
                                                     Vmin, Vmax,
                                                     action, reward,
                                                     state, state,
                                                     term, 4, gamma,
                                                     vector0)
        weights_target = tf.ones([2, num_atoms]) * (1. / num_atoms)
        self.assertTrue(tf.math.reduce_all(tf.round(100. * weights) ==
                                           tf.round(100. * weights_target)))
        Qerr_target = -1. * tf.math.log(tf.math.divide(tf.ones(2), num_atoms))
        self.assertTrue(tf.math.reduce_all(tf.round(100. * Qerr) ==
                                           tf.round(100. * Qerr_target)))

    def test_shift(self):
        # shift atoms over N using reward (keep gamma constant at 1)
        num_atoms = 11
        gamma = 1.
        Q = QModelUniformD(num_atoms)
        Vmin = -10.
        Vmax = 10.
        Qexp = expQDist(Q, Vmin, Vmax)
        action = tf.constant([[1, 0, 0, 0],
                              [0, 1, 0, 0]], tf.float32)
        state = [tf.constant([0., 0.], tf.float32)]
        term = tf.constant([0, 0,], tf.float32)
        # 0 representation vector = which atom is active
        #   when reward = 0?
        v0 = [0] * num_atoms
        v0[6] = 1
        vector0 = tf.constant(v0, tf.float32)
        for i in range(5):
            z = ((Vmax - Vmin) / (num_atoms - 1)) * i
            reward = tf.constant([z, z], tf.float32)
            Qerr, weights = calc_q_error_distro_discrete(Q, Qexp, Q,
                                                        Vmin, Vmax,
                                                        action, reward,
                                                        state, state,
                                                        term, 4, gamma,
                                                        vector0)
            wt = [0] * i + [1. / num_atoms] * (num_atoms - i)
            wt[-1] = wt[-1] * (i + 1)
            weights_target = tf.constant([wt, wt], tf.float32)
            self.assertTrue(tf.math.reduce_all(tf.round(100. * weights) ==
                                            tf.round(100. * weights_target)))
            # Qerr?
            # -1 * sum [w_i * k] = -1 * k (cuz weights sum to 1)
            # where k = log(1. / num_atoms)
            Qerr_target = -1. * tf.math.log(tf.math.divide(tf.ones(2), num_atoms))
            self.assertTrue(tf.math.reduce_all(tf.round(100. * Qerr) ==
                                            tf.round(100. * Qerr_target)))


if __name__ == "__main__":
    T = TestHuber()
    T.test_huber()
    T = TestQL()
    T.setUp()
    T.test_greedy_select()
    T.test_double_q()
    T = TestQLcont()
    T.setUp()
    T.test_positive_control()
    T.test_negative_control()
    T.test_actor_grad()
    T = TestDistroQ()
    T.test_trivial()
    T.test_extreme()
    T = TestDistroQdiscrete()
    T.test_q_err()
    T.test_shift()
