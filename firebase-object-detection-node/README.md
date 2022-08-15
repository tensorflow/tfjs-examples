# TensorFlow.js Example: Running a TensorFlow SavedModel in Node.js

This demo demonstrates how to run a TensorFlow SavedModel in Node.js natively without using [tfjs-converter](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter) to convert the model. Native execution of TensorFlow SavedModel in Node.js supports more ops and better performance compared with converted model.

This example runs a object detection model in Node.js, and hosts the model through Cloud Functions for Firebase. Before getting started, please complete the first three steps of the [Firebase Guide](https://firebase.google.com/docs/functions/get-started). For more information about the Tensorflow object detection API, check out the README in [tensorflow/object_detection](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md).


To launch the demo, complete the following steps:

1. Go to file `.firebaserc` and update the value `projects.default` with your own Firebase project ID.

2. Run the following commands:

```sh
cd functions
npm install
firebase serve
```
3. Open the website link displayed in the log. The link has the following format:

  http://localhost:5001/[firebase-project-id]/us-central1/app

It will take several seconds to load and warm up the model. Once the page is loaded, you can upload an JPEG image and test. Following is an example output:

![example output](test_result.png)

By default, the inference uses `tfjs-node`, which runs on the CPU.
If you have a CUDA-enabled GPU and have the CUDA and CuDNN libraries
set up properly on your system, you can run the inference on the GPU
by replacing the tfjs-node package with tfjs-node-gpu in the `package.json` file and `functions/index.js` file.

Please note: Node.js versions 10, 12, 14, and 16 are supported.
See [Set runtime options](https://firebase.google.com/docs/functions/manage-functions#set_nodejs_version0)
for important information regarding ongoing support for these versions of Node.js.
