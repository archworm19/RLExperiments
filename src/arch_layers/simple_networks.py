"""Simple = composed of a few keras layers, laid out serially

    All use keras functional api
"""
import tensorflow as tf
from typing import List
from tensorflow.keras.layers import Layer, Dense, Dropout


class DenseNetwork(Layer):

    def __init__(self, layer_sizes: List[int],
                 output_size: int,
                 drop_rate: float = 0.):
        # use relus
        assert output_size >= 1
        super(DenseNetwork, self).__init__()
        self.layers = [Dense(ls, activation="relu") for ls in layer_sizes]
        self.drops = [Dropout(drop_rate) for _ in layer_sizes]
        self.output_layer = Dense(output_size, activation="linear")

    def call(self, x: tf.Tensor, train_mode: bool = False):
        for l, d in zip(self.layers, self.drops):
            x = d(l(x), training=train_mode)
        return self.output_layer(x)
