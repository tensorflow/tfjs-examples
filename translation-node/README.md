# TensorFlow.js Example: Sequence-to-Sequence English-French Translation with Node.js

This demo shows how to perform sequence-to-sequence prediction using the Layers
API of TensorFlow.js under Node.js.

It demonstrates loading a local pretrained model in Node.js, using
`tf.loadModel()`

The model was trained in Python Keras, based on the [lstm_seq2seq](https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py)
example.  The training data was 149,861 English-French sentence pairs available
from [http://www.manythings.org/anki](http://www.manythings.org/anki).

Prepare the node environment:
```sh
$ npm install
# Or
$ yarn
```

To launch the demo, do

```sh
node main.js
```

If you are running on a Linux system that is [CUDA compatible](https://www.tensorflow.org/install/install_linux), try installing the GPU
package and replacing the require statement:

```sh
$ npm install @tensorflow/tfjs-node-gpu
# Or
$ yarn add @tensorflow/tfjs-node-gpu
```

After installing the package, replace `require('@tensorflow/tfjs-node')` with `require('@tensorflow/tfjs-node-gpu');` in main.js
