# TensorFlow.js Example: Sequence-to-Sequence English-French Translation

This demo shows how to perform sequence-to-sequence prediction using the Layers
API of TensorFlow.js.

It demonstrates loading a pretrained model hosted at a URL, using
`tf.loadLayersModel()`

## Training Demo

The training data was 149,861 English-French sentence pairs available from [http://www.manythings.org/anki](http://www.manythings.org/anki).

### JavaScript/TypeScript Version

To train the demo in JavaScript, do

```sh
yarn train ${DATA_PATH}
```

The model was trained in Node.js with Tensorflow.js, which the model code is converted from Python to TypeScript by @[huan](https://github.com/huan) based on the [translation.py](https://github.com/tensorflow/tfjs-examples/blob/master/translation/python/translation.py) example.

### Python Version

```sh
python python/translation.py ${DATA_PATH}
```

The model was trained in Python Keras, based on the [lstm_seq2seq](https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py)
example.

## LANCH DEMO

To launch the demo, do

```sh
yarn
yarn watch
```

[See this example live!](https://storage.googleapis.com/tfjs-examples/translation/dist/index.html)
