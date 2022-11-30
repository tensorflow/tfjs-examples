# TensorFlow.js Example: MNIST using tfjs-core

This example shows you how to train MNIST only using the [TensorFlow.js core]
(https://github.com/tensorflow/tfjs-core) API (not using the layers API).

Note: currently the entire dataset of MNIST images is stored in a PNG image we have
sprited, and the code in `data.js` is responsible for converting it into
`Tensor`s.

[See this example live!](https://storage.googleapis.com/tfjs-examples/mnist-core/dist/index.html)
