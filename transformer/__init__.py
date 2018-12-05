import tensorflow as tf

from tensorflow.keras.layers import Dense

from .encoder import Encoder
from .decoder import Decoder


class Graph:

    def __init__(self, mode, params):

        self.is_pred = (mode == tf.estimator.ModeKeys.PREDICT)
        self.max_len = params['max_len']
        self.k_dim = params['k_dim']
        self.v_dim = params['v_dim']
        self.embedding_matrix = params['embedding_matrix']
        self.model_dim = params['model_dim']
        self.num_heads = params['num_heads']
        self.use_conv = params['use_conv']
        self.num_layer = params['num_layer']
        self.dropout = params['dropout_rate']
        with tf.variable_scope('linear', reuse = tf.AUTO_REUSE):
            self.linear = Dense(params['vocab_size'])



    def build(self, encoder_inputs, decoder_inputs):


        with tf.variable_scope('encoder', reuse = tf.AUTO_REUSE):
            encoder = Encoder(k_dim = self.k_dim,
                            v_dim = self.v_dim,
                            model_dim = self.model_dim,
                            num_heads = self.num_heads,
                            dropout = self.dropout,
                            use_conv = self.use_conv,
                            num_layer = self.num_layer)

            encoder_embed = self._make_embed(encoder_inputs)
            encoder_outputs = encoder.build(encoder_embed)

        with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):
            decoder = Decoder(k_dim = self.k_dim,
                            v_dim = self.v_dim,
                            model_dim = self.model_dim,
                            num_heads = self.num_heads,
                            dropout = self.dropout,
                            use_conv = self.use_conv,
                            num_layer = self.num_layer)

            decoder_embed = self._make_embed(decoder_inputs)
            decoder_outputs = decoder.build(encoder_outputs, decoder_embed)

        with tf.variable_scope('linear', reuse = tf.AUTO_REUSE):
            output = self.linear(decoder_outputs)

        if not self.is_pred:
            prediction = tf.argmax(output, axis=2, output_type=tf.int32)
            return output, prediction

        else:
            decoder_inputs = self._make_pred_decoder_inputs(prediction, decoder_inputs, 1)
            for i in range(self.max_len):
                with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE):

                    decoder_embed = self._make_embed(decoder_inputs)
                    decoder_outputs = decoder.build(encoder_outputs, decoder_embed)

                with tf.variable_scope('linear', reuse = tf.AUTO_REUSE):
                    output = self.linear(decoder_outputs)

                prediction = tf.argmax(output, axis=2, output_type=tf.int32)
                decoder_inputs = self._make_pred_decoder_inputs(prediction, decoder_inputs, i+2)

            return output, prediction

    def _make_embed(self, input):

        embed_input = tf.nn.embedding_lookup(ids = input, params = self.embedding_matrix)

        return embed_input


    def _make_pred_decoder_inputs(self, input_token, output_token, idx):

        if idx > self.max_len:
            return 0

        left = tf.slice(input_token, [0,0], [-1, idx])
        right = tf.slice(output_token, [0, idx], [-1, 1])
        right = tf.cast(right, tf.int32)
        zero = tf.zeros_like(input_token, dtype= tf.int32)
        zero_slice = tf.slice(zero, [0, 0], [-1, self.max_len - idx])

        new_input = tf.concat((left,right,zero_slice), axis=1)

        return new_input
