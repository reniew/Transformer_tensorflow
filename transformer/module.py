import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Conv1D


class Multi_head_attention:

    def __init__(self,
                k_dim,
                v_dim,
                model_dim,
                num_heads,
                dropout,
                masked=False):

        self.num_heads = num_heads
        self.dropout = Dropout(dropout)
        self.masked = masked
        self.W_Q = Dense(units = k_dim)
        self.W_K = Dense(units = k_dim)
        self.W_V = Dense(units = v_dim)
        self.linear = Dense(units = model_dim)

        assert k_dim % num_heads == 0, 'Dimension can not devided by number of head'
        assert v_dim % num_heads == 0, 'Dimension can not devided by number of head'


    def build(self, queries, keys, values):

        with tf.variable_scope('multi_head_attention', reuse = tf.AUTO_REUSE):

            queries, keys, values = self._make_qkv(queries, keys, values)
            queries_i, keys_i, values_i = self._split_head(queries, keys, values)
            outputs = self._scaled_dot_attention(queries_i, keys_i, values_i)
            outputs = self._reshape_outputs(outputs)
            outputs = self.linear(outputs)

        return self.dropout(outputs)

    def _make_qkv(self, queries, keys, values):

        queries = self.W_Q(queries)
        keys = self.W_K(keys)
        values = self.W_V(values)

        return queries, keys, values



    def _split_head(self, queries, keys, values):

        queries_i = tf.stack(tf.split(value = queries,
                                    num_or_size_splits = self.num_heads,
                                    axis = -1))
        keys_i = tf.stack(tf.split(value = keys,
                                    num_or_size_splits = self.num_heads,
                                    axis = -1))
        values_i = tf.stack(tf.split(value = values,
                                    num_or_size_splits = self.num_heads,
                                    axis = -1))

        return queries_i, keys_i, values_i

    def _scaled_dot_attention(self, queries, keys, values):

        q_dot_k = tf.matmul(queries, tf.transpose(keys, [0,1,3,2]))
        q_dot_k = q_dot_k/(keys.shape.as_list()[-1] ** 0.5)

        if self.masked:
            masks = tf.linalg.LinearOperatorLowerTriangular(q_dot_k).to_dense()
            pad = tf.ones_like(masks) * -(2**32 +1)
            q_dot_k = tf.where(tf.equal(masks, 0), pad, q_dot_k)

        q_dot_k = tf.nn.softmax(q_dot_k)
        outputs = tf.matmul(q_dot_k, values)

        return outputs

    def _reshape_outputs(self, outputs):
        outputs = tf.transpose(outputs, [1,2,0,3])
        shape_out = outputs.get_shape().as_list()
        outputs = tf.reshape(outputs, shape = [-1, shape_out[1], shape_out[2]*shape_out[3]])

        return outputs


class Feed_forward_layer:

    def __init__(self, num_units, dropout):
        self.dense1 = Dense(num_units*4, activation = tf.nn.relu)
        self.dense2 = Dense(num_units)
        self.dropout = Dropout(dropout)

    def build(self, inputs):
        with tf.variable_scope('feed_forward_layer', reuse=tf.AUTO_REUSE):
            out = self.dense1(inputs)
            out = self.dense2(out)
        return out

class Conv_layer:

    def __init__(self, num_units, dropout):
        self.conv1 = Conv1D(filters = num_units*4, kernel_size = 1, activation = tf.nn.relu)
        self.conv2 = Conv1D(filters = num_units, kernel_size = 1)
        self.dropout = Dropout(dropout)

    def build(self, inputs):
        with tf.variable_scope('conv_layer', reuse = tf.AUTO_REUSE):
            out = self.conv1(inputs)
            out = self.conv2(out)
        return out

class Add_and_normalize:

    def __init__(self, dropout):
        self.dropout = Dropout(dropout)
        self.eps = 1e-6

    def build(self, x, fx):

        with tf.variable_scope('add_and_normalize', reuse = tf.AUTO_REUSE):

            output = tf.add(fx, self.dropout(x))
            output = self._layer_norm(output)

        return output


    def _layer_norm(self, inputs):

        beta = tf.get_variable('beta', initializer=tf.zeros_like(inputs[0,0,:]))
        gamma = tf.get_variable('beta', initializer=tf.ones_like(inputs[0,0,:]))
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims = True)
        output = (inputs - mean)/( (variance ** 0.5) +self.eps)

        return gamma*output + beta


def positional_encoding(sentence_len, model_dim, dtype = tf.float32):
    '''
    Positional Encoding
    paper: https://arxiv.org/abs/1706.03762

    Arg
        sentence_len: integer.
        model_dim: integer. dimension used by model
        dtype: tensorflow data type Default is 'tf.float32'

    Return
        positional_2d: shape is [sentence_len, model_dim]
    '''

    positional_1d = np.array([pos/10000**(2*i/model_dim) for pos in range(sentence_len) for i in range(model_dim)])
    positional_1d[::2] = np.sin(positional_1d[::2])
    positional_1d[1::2] = np.cos(positional_1d[1::2])

    positional_2d = positional_1d.reshape([sentence_len, model_dim])
    positional_2d = tf.cast(positional_2d, dtype)


    return tf.convert_to_tensor(positional_2d)
