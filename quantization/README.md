# TensorFlow.js Example: Effects of Weight Quantization

This demo on quantization consists of three examples:
1. housing: this demo evaluates the effect of quantization on the accuracy
   of a multi-layer perceptron regression model.
2. mnist: this demo evaluates the effect of quantization on the accuracy
   of a relatively small deep convnet on the MNIST handwritten digits
   dataset. Without quantization, the convnet can achieve close-to-perfect
   (i.e., ~99.5%) accuracy.
3. fashion-mnist: this demo evaluates the effect of quantization on the
   accuracy of another small deep convnet traind or a problem slightly harder
   than MNIST. In particular, it is based on the Fashion MNIST dataset. The
   original, non-quantized model has an accuracy of 92%-93%.
4. MobileNetV2: this demo evaluates quantized and non-quantizd versions of
   MobeilNetV2 (width = 1.0) on a sample of 1000 images from the
   [ImageNet](http://www.image-net.org/) dataset. This subset is based on the
   sampling done by https://github.com/ajschumacher/imagen.

In the first three cases, quantizing the weights to 16 or 8 bits does not
have any significant effect on the accuracy. In the MobileNetV2 case, however,
quantizing the weights to 8 bits leads to a significant deterioration in
accuracy, as measured by the top-1 and top-5 accuracies.

Exapmle results are listed in the table below:

| Dataset and Mdoel      | Original (no-quantization) | 16-bit quantization | 8-bit quantization |
| ---------------------- | -------------------------- | ------------------- | ------------------ |
| housing: multi-layer regressor  |  loss=0.311984    | loss=0.311983       | loss=0.312780      |
| MNIST: convnet         | accuracy=0.9952            | accuracy=0.9952     | accuracy=0.9952    |
| Fashion MNIST: convnet | accuracy=0.922             | accuracy=0.922      | accuracy=0.9211    |
| MobileNetV2            | top-1 accuracy=0.618; top-5 accuracy=0.788 | top-1 accuracy=0.624; top-5 accuracy=0.789 | top-1 accuracy=0.280; top-5 accuracy=0.490 |

They demonstrate different effects of the same quantization technique
on different problems.

## Running the housing quantization demo

In preparation, do:

```sh
yarn
```

To run the train and save the model from scratch, do:
```sh
yarn train-housing
```

If you are running on a Linux system that is [CUDA compatible](https://www.tensorflow.org/install/install_linux), try installing the GPU:

```sh
yarn train-housing --gpu
```

To perform quantization on the model saved in the `yarn train` step
and evaluate the effects on the model's test accuracy, do:

```
yarn quantize-and-evaluate-housing
```

## Running the MNIST quantization demo

In preparation, do:

```sh
yarn
```

To run the train and save the model from scratch, do:
```sh
yarn train-mnist
```

or with CUDA acceleration:

```sh
yarn train-mnist --gpu
```

To perform quantization on the model saved in the `yarn train` step
and evaluate the effects on the model's test accuracy, do:

```
yarn quantize-and-evaluate-mnist
```

## Running the Fashion-MNIST quantization demo

In preparation, do:

```sh
yarn
```

To run the train and save the model from scratch, do:
```sh
yarn train-fashion-mnist
```

or with CUDA acceleration:

```sh
yarn train-fashion-mnist --gpu
```

To perform quantization on the model saved in the `yarn train` step
and evaluate the effects on the model's test accuracy, do:

```
yarn quantize-and-evaluate-fashion-mnist
```

## Running the MobileNetV2 quantization demo

Unlike the previous three demos, the MobileNetV2 demo doesn't involve
a model training step. Instead, the model is loaded as a Keras application
and converted to the TensorFlow.js format for quantization and evaluation.

The non-quantized and quantized versions of MobileNetV2 are evaluated
on a sample of 1000 images from the [ImageNet](http://www.image-net.org/)
dataset. The image files are downloaded from the hosted location on the
web. This subset is based on the sampling done by
https://github.com/ajschumacher/imagen.

All these steps can be performed with a single command:

```sh
yarn quantize-and-evaluate-MobileNetV2
```
