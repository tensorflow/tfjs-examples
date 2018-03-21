# TensorFlow.js Example: Iris Classification

This demo shows how to perform classification on the
[classic Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set)
using the Layers API of TensorFlow.js.

It demonstrates loading a pretrained model hosted at a URL, using `tf.loadModel()`

The model consists of two `Dense` layers: one with a `relu` activation followed
by another with a `softmax` activation.

To launch the demo, do

```sh
yarn
yarn watch
```
