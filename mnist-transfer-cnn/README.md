# TensorFlow.js Example: MNIST CNN Transfer Learning Demo

This demo shows how to perform transfer learning using the Layers API of
TensorFlow.js.

It follows the procedure outlined in the Keras
[mnist_transfer_cnn](https://github.com/keras-team/keras/blob/master/examples/mnist_transfer_cnn.py)
example.

 * A simple convnet was trained in Python Keras on only the first 5 digits [0..4] from the MNIST dataset.  The resulting model is hosted at a URL and loaded into TensorFlow.js using
`tf.loadLayersModel()`.
 * The convolutional layers are frozen, and the dense layers are fine-tuned in the browser to classify the digits [5..9].

To launch the demo, do

```sh
yarn
yarn watch
```

[See this example live!](https://storage.googleapis.com/tfjs-examples/mnist-transfer-cnn/dist/index.html)
