# TensorFlow.js Example: Sequence-to-Sequence English-French Translation

This demo shows how to perform sequence-to-sequence prediction using the Layers
API of TensorFlow.js.

It demonstrates loading a pretrained model hosted at a URL, using
`tf.loadLayersModel()`

The model was trained in Python Keras, based on the [lstm_seq2seq](https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py)
example.  The training data was 149,861 English-French sentence pairs available
from [http://www.manythings.org/anki](http://www.manythings.org/anki).

To launch the demo, do

```sh
yarn
yarn watch
```

[See this example live!](https://storage.googleapis.com/tfjs-examples/translation/dist/index.html)
