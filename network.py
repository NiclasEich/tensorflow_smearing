import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class NoiseSmearingLayer(tf.keras.layers.Layer):
    def __init__(self, built_in_noise=True):
        self.built_in_noise = built_in_noise
        super(NoiseSmearingLayer, self).__init__(name="NoiseSmearingLayer")

    def build(self, input_shape):
        self.mean = tf.Variable(
            initial_value=np.zeros([int(input_shape[-2]), 1], dtype=np.float32),
            trainable=True,
            name="mean",
            shape=[int(input_shape[-2]), 1],
        )
        self.width = tf.Variable(
            initial_value=np.ones(shape=[int(input_shape[-2]), 1], dtype=np.float32),
            trainable=True,
            name="width",
            shape=[int(input_shape[-2]), 1],
        )

        if self.built_in_noise is True:
            # print("INPUT_SHAPE", "\n"*10, input_shape)
            self.noise = tf.random.normal(input_shape, 0.0, 1.0)

    def call(self, input, noise=None):
        """
        important to only connect layers and tensors
        in this function and not declare any properties
        """
        if self.built_in_noise is True:
            if noise is not None:
                raise NotImplementedError(
                    "This functionality is not implemented! Either define a layer without internal noise or don't pass a noise vector"
                )
            noise = self.noise

        shifted_noise = noise + self.mean
        scaled_noise = shifted_noise * self.width
        smeared_input = input * scaled_noise

        return smeared_input

    def get_config(self):
        """
        this needs to be implemente in order to
        be able to save and restore the layer
        """
        config = {"build_in_noise": self.built_in_noise}
        base_config = super(NoiseSmearingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_network(n_particles, batch_size=256, built_in_noise=True):
    input_particles = tf.keras.layers.Input(shape=(n_particles, 4), batch_size=batch_size)

    smearing_layer = NoiseSmearingLayer(built_in_noise=built_in_noise)

    inp_tensors = [input_particles]

    if built_in_noise is True:
        output = smearing_layer(input_particles)
    else:
        input_noise = tf.keras.layers.Input(shape=(n_particles, 4), batch_size=batch_size)
        output = smearing_layer(input_particles, input_noise)
        inp_tensors += input_noise

    network = tf.keras.Model(inputs=inp_tensors, outputs=[output])
    return network
