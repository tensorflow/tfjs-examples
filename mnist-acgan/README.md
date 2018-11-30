# TensorFlow.js Example: ACGAN on the MNIST Dataset

## What is this example about.

This example trains an Auxiliary Classifier Generative Adversarial Network
(ACGAN) on the MNIST dataset.

For background of ACGAN, see:
 - Augustus Odena, Christopher Olah, Jonathon Shlens. (2017) "Conditional
   image synthesis with auxiliary classifier GANs"
   https://arxiv.org/abs/1610.09585

This example of TensorFlow.js runs simultaneously in two different environments:
 - Training in the Node.js environment. During the long-running training process,
   a checkpoint of the generator will be saved to the disk at the end of every
   epoch.
 - Demonstration of generation in the browser. The demo webpage will load
   the checkpoints saved from the training process and use it to generate
   fake MNIST images in the browser.
 
## How to use this example

To start the training,
 
```sh
yarn
yarn train
```
 
To start the demo in the browser, do in a separate terminal:
 
```sh
yarn
yarn watch
```

### Training the model on CUDA GPUs using tfjs-node-gpu
 
It is recommended to use tfjs-node-gpu to train the model on a CUDA-enabled GPU,
as the convolution heavy operations run several times faster a GPU than on the
CPU with tfjs-node.

By default, the [trainig script](./gan.js) runs on the CPU using tfjs-node. To
run it on the GPU, repace the line 

```js
require('@tensorflow/tfjs-node');
```

with

```js
require('@tensorflow/tfjs-node-gpu');
```
