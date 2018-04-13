# TensorFlow.js Example: Custom Layer

This example demonstrates how to write custom layers for tfjs-layers.

We build a custom activation layer called 'Antirectifier',
which modifies the shape of the tensor that passes through it.
We need to specify a constructor and three methods: `build`, `call`, and `getConfig`.

We then use the custom layer to train and evaluate an mnist model, comparing it to a more stock mnist model.

Note: currently the entire dataset of MNIST images is stored in a PNG image we have
sprited, and the code in `data.js` is responsible for converting it into
`Tensor`s. This will become much simpler in the near future.

[See this example live!](https://storage.googleapis.com/tfjs-examples/custom-layer/dist/index.html)
