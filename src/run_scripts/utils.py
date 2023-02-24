"""Sub-models used by model builders"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
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
    # outputs means and precisions in a batch_size x (2 * action_dim) tensor
    # NOTE: variable precisions are not a function of inputs

    def __init__(self,
                 action_bounds: List[Tuple[float]],
                 embed_dims: List[int],
                 layer_sizes: List[int],
                 drop_rate: float,
                 init_prec: float = 1.,
                 min_prec: float = 0.1,
                 max_prec: float = 10.):  # maximum precision value
        super(DenseGaussState, self).__init__()
        self.action_bounds = action_bounds
        self.d_states = [Dense(ed) for ed in embed_dims]
        self.net = DenseNetwork(layer_sizes, len(action_bounds), drop_rate)
        # bounding vars
        self._ranges = tf.constant([ab[1] - ab[0] for ab in action_bounds], dtype=tf.float32)
        self._mins = tf.constant([ab[0] for ab in action_bounds], dtype=tf.float32)
        # precision:
        # init_prec = max_prec * sigmoid(pi)
        # pi = inv_sigmoid(init_prec / max_prec) = logit(init_prec / max_prec)
        self.max_prec = max_prec
        self.min_prec = min_prec
        v = (init_prec / max_prec)
        self._pi = tf.math.log(tf.math.divide(v, 1. - v))
        self.prec_var = tf.Variable(initial_value=self._pi * tf.ones((len(action_bounds),)))

    def call(self, state_t: List[tf.Tensor]):
        # returns batch_size x (2 * action_dims)
        x_s = [dse(s) for dse, s in zip(self.d_states, state_t)]
        yp = self.net(tf.concat(x_s, axis=1))
        mu = tf.expand_dims(self._ranges, 0) * tf.math.sigmoid(yp) + tf.expand_dims(self._mins, 0)
        prec = self.max_prec * tf.math.sigmoid(self.prec_var) + self.min_prec
        prec = tf.tile(tf.expand_dims(prec, 0), [tf.shape(mu)[0], 1])
        return tf.concat([mu, prec], axis=1)


class DenseForwardModel(Layer):

    def __init__(self, layer_sizes: List[int],
                 embed_dims: int, output_dims: int,
                 num_state: int,
                 drop_rate: float = 0.):
        super(DenseForwardModel, self).__init__()
        self.action_layer = Dense(embed_dims, activation="relu")
        self.state_layers = [Dense(embed_dims, activation="relu") for _ in range(num_state)]
        self.net = DenseNetwork(layer_sizes, output_dims, drop_rate=drop_rate)

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]):
        vs = [sl(st) for sl, st in zip(self.state_layers, state_t)]
        va = self.action_layer(action_t)
        return self.net(tf.concat(vs + [va], axis=1))


class DenseEncoder(Layer):

    def __init__(self, layer_sizes: List[int],
                 embed_dims: int, output_dims: int,
                 num_state: int,
                 drop_rate: float = 0.):
        super(DenseEncoder, self).__init__()
        self.state_layers = [Dense(embed_dims, activation="relu") for _ in range(num_state)]
        self.net = DenseNetwork(layer_sizes, output_dims, drop_rate=drop_rate)

    def call(self, state_t: List[tf.Tensor]):
        vs = [sl(st) for sl, st in zip(self.state_layers, state_t)]
        return self.net(tf.concat(vs, axis=1))
