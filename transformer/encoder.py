import tensorflow as tf


from .module import *

class Encoder:

    def __init__(self, k_dim, v_dim, model_dim, num_heads, dropout, use_conv, num_layer):

        self.num_layer = num_layer
        self.self_att = Multi_head_attention(k_dim, v_dim, model_dim, num_heads, dropout)
        self.add_and_normalize = Add_and_normalize(dropout)
        if use_conv:
            self.ffn = Conv_layer(model_dim, dropout)
        else:
            self.ffn = Feed_forward_layer(model_dim, dropout)


    def build(self, inputs):

        output = self._encoder_block(inputs)

        return output

    def _encoder_block(self, inputs):

        for i in range(self.num_layer):
            with tf.variable_scope('encoder_{}-th_layer'.format(i)):
                att_output = self.self_att.build(inputs, inputs, inputs)
                inputs = self.add_and_normalize.build(inputs, att_output)

                ffn_output = self.ffn.build(inputs)
                inputs = self.add_and_normalize.build(inputs, ffn_output)

        return inputs
