"""Sub-models used by model builders"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from frameworks.layer_signatures import ScalarModel, ScalarStateModel, DistroModel, DistroStateModel
from arch_layers.simple_networks import DenseNetwork
from typing import List, Tuple


class DenseScalar(Layer):
    # > run all states thru embeddings
    # > concat
    # > run thru dense network 
    def __init__(self,
                 embed_dims: List[int],
                 layer_sizes: List[int],
                 drop_rate: float):
        super(DenseScalar, self).__init__()
        self.d_act = Dense(embed_dims[0])
        self.d_states = [Dense(ed) for ed in embed_dims]
        self.net = DenseNetwork(layer_sizes, 1, drop_rate)

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]):
        x_a = self.d_act(action_t)
        x_s = [dse(s) for dse, s in zip(self.d_states, state_t)]
        yp = self.net(tf.concat([x_a] + x_s, axis=1))
        return yp[:, 0]  # to scalar


class DenseScalarState(Layer):
    # NOTE: unbounded outputs
    # > run all states thru embeddings
    # > concat
    # > run thru dense network 
    def __init__(self,
                 embed_dims: List[int],
                 layer_sizes: List[int],
                 drop_rate: float):
        super(DenseScalarState, self).__init__()
        self.d_states = [Dense(ed) for ed in embed_dims]
        self.net = DenseNetwork(layer_sizes, 1, drop_rate)

    def call(self, state_t: List[tf.Tensor]):
        x_s = [dse(s) for dse, s in zip(self.d_states, state_t)]
        yp = self.net(tf.concat(x_s, axis=1))
        return yp[:, 0]  # to scalar


class DenseScalarPi(Layer):
    # TODO: wut? this isn't actually a scalar model!
    # pi: pi(a | s)
    def __init__(self,
                 output_dims: int,
                 embed_dims: List[int],
                 layer_sizes: List[int],
                 drop_rate: float,
                 bounds: List[Tuple[int]] = None):
        # bounds = list of (lower bound, upper bound) pairs
        super(DenseScalarPi, self).__init__()
        self.d_states = [Dense(ed) for ed in embed_dims]
        self.d_out = Dense(output_dims)
        self.net = DenseNetwork(layer_sizes, 1, drop_rate)
        if bounds is not None:
            assert len(bounds) == output_dims, "dim mismatch"
            self.offsets = tf.constant([bi[0] for bi in bounds], dtype=tf.float32)
            self.ranges = tf.constant([bi[1] - bi[0] for bi in bounds], dtype=tf.float32)
            self.activation = tf.math.sigmoid
        else:
            self.offsets = 0.
            self.ranges = 1.
            self.activation = lambda x: x

    def call(self, state_t: List[tf.Tensor]):
        x_s = [dse(s) for dse, s in zip(self.d_states, state_t)]
        yp = self.net(tf.concat(x_s, axis=1))
        return self.activation(self.d_out(yp)) * self.ranges + self.offsets


class DenseDistro(Layer):
    def __init__(self,
                 embed_dims: List[int],
                 layer_sizes: List[int],
                 drop_rate: float,
                 num_atoms: int = 51):
        # NOTE: use sigmoid_scale to avoid overflows
        super(DenseDistro, self).__init__()
        self.d_act = Dense(embed_dims[0])
        self.d_states = [Dense(ed) for ed in embed_dims]
        self.net = DenseNetwork(layer_sizes, num_atoms, drop_rate)

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]):
        # NOTE: only uses 0th tensor in state_t
        x_a = self.d_act(action_t)
        x_s = [dse(s) for dse, s in zip(self.d_states, state_t)]
        yp = self.net(tf.concat([x_a] + x_s, axis=1))
        return yp


class DenseDiscreteState(Layer):
    # softmaxed output

    def __init__(self,
                 num_action: int,
                 embed_dims: List[int],
                 layer_sizes: List[int],
                 drop_rate: float):
        super(DenseDiscreteState, self).__init__()
        self.d_states = [Dense(ed) for ed in embed_dims]
        self.net = DenseNetwork(layer_sizes, num_action, drop_rate)

    def call(self, state_t: List[tf.Tensor]):
        x_s = [dse(s) for dse, s in zip(self.d_states, state_t)]
        yp = self.net(tf.concat(x_s, axis=1))
        return tf.nn.softmax(yp, axis=1)


class DenseGaussState(Layer):
    # outputs means and log(std_dev) in a batch_size x (2 * action_dim) tensor
    # NOTE: log(std_dev) not a function of input

    def __init__(self,
                 action_bounds: List[Tuple[float]],
                 embed_dims: List[int],
                 layer_sizes: List[int],
                 drop_rate: float,
                 init_log_std_dev: float,
                 scale_log_std_dev: float = 0.1):
        super(DenseGaussState, self).__init__()
        for ab in action_bounds:
            assert ab[1] == -1 * ab[0]
        self.action_bounds = action_bounds
        self.d_states = [Dense(ed) for ed in embed_dims]
        self.net = DenseNetwork(layer_sizes, len(action_bounds), drop_rate)
        # bounding vars
        self._ranges = tf.constant([0.5 * (ab[1] - ab[0]) for ab in action_bounds], dtype=tf.float32)
        self._means = tf.constant([0.5 * (ab[0] + ab[1]) for ab in action_bounds], dtype=tf.float32)
        self._log_std_dev = tf.Variable(init_log_std_dev, dtype=tf.float32)
        self._scale_lsd = scale_log_std_dev

    def call(self, state_t: List[tf.Tensor]):
        # returns batch_size x (2 * action_dims)
        x_s = [dse(s) for dse, s in zip(self.d_states, state_t)]
        yp = self.net(tf.concat(x_s, axis=1))
        # mean
        mu = tf.expand_dims(self._ranges, 0) * tf.math.tanh(yp) + tf.expand_dims(self._means, 0)
        # log std dev
        lsd = tf.ones(tf.shape(mu), dtype=mu.dtype) * tf.cast(self._log_std_dev * self._scale_lsd, mu.dtype)
        return tf.concat([mu, lsd], axis=1)
