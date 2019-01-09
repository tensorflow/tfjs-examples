# TensorFlow.js Example: Visualizing Convnet Filters

## Description

This TensorFlow.js example demonstrates some techniques of visualizing
the internal workings of a convolutional neural network (convnet), including:

- Finding what convolutional layers' filters are sensitive to after
  training: calculating maximally-activating input image for
  convolutional filters through gradient ascent in the input space.
- Getting the internal activation of a convnet by uisng the
  functional model API of TensorFlow.js
- Finding which part of an input image is most relevant to the
  classification decision made by a convnet (VGG16 in this case),
  using the gradient-based class activation map (CAM) approach.

## How to use this demo

Run the command:
```sh
yarn visualize
```

This will automatically

1. install the necessary Python dependencies. If the required
   Python package (keras, tensorflow and tensorflowjs) are already installed,
   this step will be a no-op. However, to prevent this step from
   modifying your global Python environment, you may run this demo from
   a [virtualenv](https://virtualenv.pypa.io/en/latest/) or
   [pipenv](https://pipenv.readthedocs.io/en/latest/).
2. download and convert the VGG16 model to TensorFlow.js format
3. launch a Node.js script to load the converted model and compute
   the maximally-activating input images for the convnet's filters
   using gradient ascent in the input space and save them as image
   files under the `dist/filters` directory
4. launch a Node.js script to calculate the internal convolutional
   layers' activations and th gradient-based class activation
   map (CAM) and save them as image files under the
   `dist/activation` directory.
5. compile and launch the web view using parcel

Step 3 and 4 (especially step 3) involve relatively heavy computation
and is best done usnig tfjs-node-gpu instead of the default
tfjs-node. This requires that a CUDA-enabled GPU and the necessary
driver and libraries are installed on your system.

Assuming those prerequisites are met, do:

```sh
yarn visualize --gpu
```

You may also increase the number of filters to visualize per convolutional
layer from the default 8 to a larger value, e.g., 32:

```sh
yarn visualize --gpu --filters 32
```

The default image used for the internal-activation and CAM visualization is
"owl.jpg". You can switch to another image by using the "--image" flag, e.g.,

```sh
yarn visualize --image dog.jpg
```
