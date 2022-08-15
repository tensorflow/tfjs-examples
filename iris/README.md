# TensorFlow.js Example: Iris Classification

This demo shows how to perform classification on the
[classic Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set)
using the Layers API of TensorFlow.js.

It demonstrates ways to create a model:
* Loading a pretrained model hosted at a URL, using `tf.loadLayersModel()`
* Creating and training a model from scratch in the browser.

This demo also shows how to use the `callbacks` field of the `Model.fit()`
configuration to perform real-time visualization of training progress.

The model consists of two `Dense` layers: one with a `relu` activation followed
by another with a `softmax` activation.

To launch the demo, do

```sh
yarn
yarn watch
```

[See this example live!](https://storage.googleapis.com/tfjs-examples/iris/dist/index.html)
