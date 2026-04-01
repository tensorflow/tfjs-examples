# TensorFlow.js Example: MNIST using tfjs-core

This example shows you how to train MNIST on a web worker using WebGL only using the [TensorFlow.js core]
(https://github.com/tensorflow/tfjs-core) library (not using the layers API).

Note: currently the entire dataset of MNIST images is stored in a PNG image we have
sprited, and the code in `data.js` is responsible for converting it into
`Tensor`s. This will become much simpler in the near future.

[See this example live!](https://storage.googleapis.com/tfjs-examples/mnist-core-webworker/dist/index.html)
