# TensorFlow.js Example: Effects of Weight Quantization

This demo on quantization consists of three examples:
1. mnist
2. fashion-mnist
3. autompg

They demonstrate different effects of the same quantization technique
on different problems.

## Running the MNIST demo

In preparation, do:

```sh
yarn
```

Run the training script:
```sh
yarn train-mnist
```

If you are running on a Linux system that is [CUDA compatible](https://www.tensorflow.org/install/install_linux), try installing the GPU:

```sh
yarn train-mnist --gpu
```

To perform quantization on the model saved in the `yarn train` step
and evaluate the effects on the model's test accuracy, do:

```
yarn quantize-and-evaluate-mnist
```

## Running the AutoMPG demo


