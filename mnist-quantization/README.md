# TensorFlow.js Example: Effects of Weight Quantization

## Running the demo

```sh
npm install
# Or
yarn
```

Run the training script:
```sh
yarn train
```

If you are running on a Linux system that is [CUDA compatible](https://www.tensorflow.org/install/install_linux), try installing the GPU
package and replacing the require statement:

```sh
$ npm install @tensorflow/tfjs-node-gpu
# Or
$ yarn add @tensorflow/tfjs-node-gpu
```

After installing the package, replace `require('@tensorflow/tfjs-node')` with `require('@tensorflow/tfjs-node-gpu');` in main.js
