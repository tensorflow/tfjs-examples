# tfjs-tsne example: MNIST

This example shows usage of the <a href="https://github.com/tensorflow/tfjs-tsne">tfjs-tsne</a> library to do dimensionality reduction on the <a href="https://en.wikipedia.org/wiki/MNIST_database">MNIST data set</a>. This examples takes 10000 images of digits, resize them to 10x10px and use T-SNE to organize them in two dimensions.

Note that tfjs-tsne requires WebGL 2 support and thus will not work on certain devices, mobile devices especially. Currently it best works on desktop devices.

```sh
yarn
yarn watch
```

[See this example live!](https://storage.googleapis.com/tfjs-examples/tsne-mnist-canvas/dist/index.html)
