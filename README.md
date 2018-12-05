# Transformer

Implementation Transformer model introduced "Attention is All You Need"

# Dependencies

* Python >= 3.6
* Tensorflow >= 1.8
* gensim
* numpy

# Datasets

Crawled koeran poem dataset

* Total 1034 poem
* Total 17066 sequnece to trainable

# Config

```python
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size')
tf.app.flags.DEFINE_integer('epoch', 1, 'epoch')
tf.app.flags.DEFINE_integer('model_dim', 128, 'model dim')
tf.app.flags.DEFINE_integer('embedding_dim', 128, 'embedding dim')
tf.app.flags.DEFINE_integer('k_dim', 128, 'key dim')
tf.app.flags.DEFINE_integer('v_dim', 128, 'value dim')
tf.app.flags.DEFINE_integer('num_heads', 4, 'num heads')
tf.app.flags.DEFINE_integer('num_layer', 2, 'num layer')
tf.app.flags.DEFINE_boolean('use_conv', False, 'use conv')
tf.app.flags.DEFINE_float('dropout_rate', 0.5, 'dropout rate')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_string('data_path', './data/poem_data.json', 'data path')
tf.app.flags.DEFINE_string('embedding_path', './data/embedding_matrix.npy', 'embeding path')
tf.app.flags.DEFINE_string('check_point', './check_point', 'chech_point')
```

# Usage

Install requirements

```
pip install -r requirements.txt
```

Training model

```
python main.py
```


# Predict

For prediction, you should put one sentence used by input


```
python predict.py <input-sentence>
```


# Reference Paper

* [Attention is All You Need](https://arxiv.org/pdf/1706.03762)
