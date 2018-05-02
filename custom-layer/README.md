# TensorFlow.js Example: Custom Layer

This example demonstrates how to write a custom layer for tfjs-layers.

We build a custom activation layer called 'Antirectifier' which outputs two
channels for each input, one with just the positive signal, and one with just
the negative signal.  The example shows exercises the `call` and
`computeOutputShape` overides of the `Layer`.

We then use the custom layer to train and evaluate an MNIST model, comparing it
to the stock MNIST model from the example titled `mnist`.

Note: currently the entire dataset of MNIST images is stored in a PNG image we
have sprited, and the code in `data.js` is responsible for converting it into
`Tensor`s. This will become much simpler in the near future.

[See this example
live!](https://storage.googleapis.com/tfjs-examples/custom-layer/dist/index.html)
