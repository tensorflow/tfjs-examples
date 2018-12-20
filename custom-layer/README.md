# TensorFlow.js Example: Custom Layer

This example demonstrates how to write a custom layer for tfjs-layers.

We build a custom activation layer called 'Antirectifier' which outputs two
channels for each input, one with just the positive signal, and one with just
the negative signal.  The example shows exercises the `call` and
`computeOutputShape` overides of the `Layer`.

[See this example
live!](https://storage.googleapis.com/tfjs-examples/custom-layer/dist/index.html)
