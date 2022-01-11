# TensorFlow.js Example: Simple Object Detection

This example illustrates how to train a model to perform simple object
detection in TensorFlow.js. It includes the full workflow:

- Generation of synthetic images and labels for training and testing
- Creation of a model for the object-detection task based on a pretrained
  computer-vision model (MobileNet)
- Training of the model in Node.js using [tfjs-node](https://github.com/tensorflow/tfjs-node)
- Transfering the model from the Node.js environment into the browser
  through saving and loading
- Performing inference with the loaded model in the browser and visualizing
  the inference results.

## How to use this example

First, train the model using Node.js:

```sh
yarn
yarn train
```

Then, run the model in the browser:

```sh
yarn watch
```

## Adjusting training parameters from the "yarn train" command line

The `yarn train` command stores all training examples in memory. Hence,
it is possible for the process to run out of memory and crash if there
are too many training examples generated. In the meantime, having a large
number of training examples benefits the accuracy of the model after
training. The default number of examples is 2000. You can adjust the number
of examples by using the `--numExamples` flag of the `yarn train` command.
For example, the hosted model is trained with the 20000 examples, using
the command line:

```sh
yarn train \
    --numExamples 20000 \
    --initialTransferEpochs 100 \
    --fineTuningEpochs 200
```

See `train.js` for other adjustable parameters.

### Using CUDA GPU for Training

Note that by default, the model is trained using the CPU version of tfjs-node.
If you machine is equipped with a CUDA(R) GPU, you may switch to using
tfjs-node-gpu, which will significantly shorten the training time. Specifically,
add the `--gpu` flag to the command above, i.e.,

```sh
yarn train --gpu \
    --numExamples 20000 \
    --initialTransferEpochs 100 \
    --fineTuningEpochs 200
```

[See this example live!](https://storage.googleapis.com/tfjs-examples/simple-object-detection/dist/index.html)

### Monitoring model training with TensorBoard

The Node.js-based training script allows you to log the loss values to
[TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard).
Compared to printing loss values to the console, which the
training script performs by default, logging to tensorboard has the following
advantages:

 1. Persistence of the loss values, so you can have a copy of the training
   history available even if the system crashes in the middle of the training
   for some reason, while logs in consoles are more ephemeral.
2. Visualizing the loss values as curves makes the trends easier to see (e.g.,
   see the screenshot below).

To do this in this example, add the flag `--logDir` to the `yarn train`
command, followed by the directory to which you want the logs to
be written, e.g.,

 ```sh
yarn train
    --numExamples 20000 \
    --initialTransferEpochs 100 \
    --fineTuningEpochs 200 \
    --logDir /tmp/simple-object-detection-logs
```

 Then install tensorboard and start it by pointing it to the log directory:

 ```sh
# Skip this step if you have already installed tensorboard.
pip install tensorboard
 tensorboard --logdir /tmp/simple-object-detection-logs
```

 tensorboard will print an HTTP URL in the terminal. Open your browser and
navigate to the URL to view the loss curves in the Scalar dashboard of
TensorBoard.
