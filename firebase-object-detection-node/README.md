# TensorFlow.js Example: Running a TensorFlow SavedModel in Node.js

This demo demonstrates how to run a TensorFlow SavedModel in Node.js without using [tfjs-converter](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter) to convert the model.

This example runs the [Mobile Object Localizer](https://tfhub.dev/google/object_detection/mobile_object_localizer_v1/1) model, and hosts the model through Firebase Cloud Functions. Before getting started, please finish first three steps of the firebae Guide to ensure the machine has required setup.

To launch the demo, do

```sh
cd functions
npm install
firebase serve
```

Open the website at: http://localhost:5000/

It will take several seconds to load and warm up the model. Once the page is loaded, you can upload an image and test.

By default, the training uses tfjs-node, which runs on the CPU.
If you have a CUDA-enabled GPU and have the CUDA and CuDNN libraries
set up properly on your system, you can run the training on the GPU
by replacing the tfjs-node package with tfjs-node-gpu.
