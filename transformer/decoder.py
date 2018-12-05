import tensorflow as tf


from .module import *
1
class Decoder:

    def __init__(self, k_dim, v_dim, model_dim, num_heads, dropout, use_conv, num_layer):

        self.num_layer = num_layer
        self.self_att = Multi_head_attention(k_dim, v_dim, model_dim, num_heads, dropout, masked = True)
        self.multi_att = Multi_head_attention(k_dim, v_dim, model_dim, num_heads, dropout)
        self.add_and_normalize = Add_and_normalize(dropout)
        if use_conv:
            self.ffn = Conv_layer(model_dim, dropout)
        else:
            self.ffn = Feed_forward_layer(model_dim, dropout)

    def build(self, enc_outputs, dec_inputs):

        multi_att_output = self._decoder_block(enc_outputs, dec_inputs)

        return multi_att_output

    def _decoder_block(self, enc_outputs, dec_inputs):

        for i in range(self.num_layer):
            with tf.variable_scope('decoder_{}-th_layer'.format(i)):
                self_att_output = self.self_att.build(dec_inputs, dec_inputs, dec_inputs)
                dec_inputs = self.add_and_normalize.build(dec_inputs, self_att_output)

                att_output = self.multi_att.build(enc_outputs, enc_outputs, self_att_output)
                dec_inputs = self.add_and_normalize.build(dec_inputs, att_output)

                ffn_output = self.ffn.build(dec_inputs)
                dec_inputs = self.add_and_normalize.build(dec_inputs, ffn_output)

        return dec_inputs
