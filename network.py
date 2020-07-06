import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class NoiseSmearingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(NoiseSmearingLayer, self).__init__(name="NoiseSmearingLayer")

    def build(self, input_shape):
        self.mean = tf.Variable(
            initial_value=np.zeros([int(input_shape[-2]), 1], dtype=np.float32), trainable=True, name="mean", shape=[int(input_shape[-2]), 1]
        )
        self.width = tf.Variable(
            initial_value=np.ones(shape=[int(input_shape[-2]), 1], dtype=np.float32), trainable=True, name="width", shape=[int(input_shape[-2]), 1]
        )

    def call(self, input, noise):
        """
        important to only connect layers and tensors
        in this function and not declare any properties
        """
        if noise is None:
            return (input + self.mean) * self.width

        else:

            shifted_noise = noise + self.mean
            scaled_noise = shifted_noise * self.width
            smeared_input = input * scaled_noise

            return smeared_input

    def get_config(self):
        """
        this needs to be implemente in order to
        be able to save and restore the layer
        """
        config = {}
        base_config = super(NoiseSmearingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_network(n_particles, batch_size=256):
    input_particles = tf.keras.layers.Input(shape=(batch_size, n_particles, 4))
    input_noise = tf.keras.layers.Input(shape=(batch_size, n_particles, 4))

    smearing_layer = NoiseSmearingLayer()
    output = smearing_layer(input_particles, input_noise)

    network = tf.keras.Model(inputs=[input_particles, input_noise], outputs=[output])
    return network
