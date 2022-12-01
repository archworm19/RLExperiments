"""I always do this cuz I don't like the keras default fit method
    I prefer to have a loss tensor in the model output

    Assumptions:
        1. all of x should be run through model
        2. model output = dictionary D
            and D[loss_name] = tensor representing the loss
"""
import tensorflow as tf
import keras
from tensorflow.keras import Model


class CustomModel(Model):

    def __init__(self, loss_name: str, *args, **kwargs):
        super(CustomModel, self).__init__(*args, **kwargs)
        self.loss_name = loss_name
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    def train_step(self, x):
        # NOTE: inputs list will be fed in for x
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            loss = y_pred[self.loss_name]
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # Update metrics
        self.total_loss_tracker.update_state(loss)
        return {"train_loss": self.total_loss_tracker.result()}
