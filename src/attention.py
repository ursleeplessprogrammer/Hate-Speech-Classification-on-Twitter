"""Sequence attention layer (program section 4.18)."""
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class SeqAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(1,),
                                 initializer="uniform", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        scores = K.tanh(K.dot(inputs, self.W) + self.b)
        weights = K.softmax(scores, axis=1)
        return inputs * weights

    def compute_output_shape(self, input_shape):
        return input_shape
