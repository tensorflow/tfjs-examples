# TensorFlow.js Example: Date Conversion Through an LSTM-Attention Model

[See this example live!](https://storage.googleapis.com/tfjs-examples/date-conversion-attention/dist/index.html)

## Overview

This example shows how to use TensorFlow.js to train a model based on
long short-term memory (LSTM) and the attention mechanism to achieve
a task of converting various commonly seen date formats (e.g., 01/18/2019,
18JAN2019, 18-01-2019) to the ISO date format (i.e., 2019-01-18).

We demonstrate the full machine-learning workflow, consisting of
data engineering, server-side model training, client-side inference,
model visualization, and unit testing in this example.

The training data is synthesized programmatically.

## Model training in Node.js

For efficiency, the training of the model happens outside the browser
in Node.js, using tfjs-node or tfjs-node-gpu.

To run the training job, do

```sh
yarn
yarn train
```

By default, the training uses tfjs-node, which runs on the CPU.
If you have a CUDA-enabled GPU and have the CUDA and CuDNN libraries
set up properly on your system, you can run the training on the GPU
by:

```sh
yarn
yarn train --gpu
```

### Monitoring model training with TensorBoard

The Node.js-based training script allows you to log the loss values to
[TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard).
Compared to printing loss values to the console, which the
training script performs by default, logging to tensorboard has the following
advantanges:

1. Persistence of the loss values, so you can have a copy of the training
   history available even if the system crashes in the middle of the training
   for some reason, while logs in consoles are more ephemeral.
2. Visualizing the loss values as curves makes the trends easier to see (e.g.,
   see the screenshot below).

![date-conversion attention model training: TensorBoard example](./date-conversion-attention-tensorboard-example.png)

A detailed tensorboard training log is hosted and viewable at this
[TensorBoard.dev link](https://tensorboard.dev/experiment/CqhZhKlNSgimJbnIwvbmnw/#scalars).

To do this in this example, add the flag `--logDir` to the `yarn train`
command, followed by the directory to which you want the logs to
be written, e.g.,

```sh
yarn train --logDir /tmp/date-conversion-attention-logs
```

Then install tensorboard and start it by pointing it to the log directory:

```sh
# Skip this step if you have already installed tensorboard.
pip install tensorboard

tensorboard --logdir /tmp/date-conversion-attention-logs
```

tensorboard will print an HTTP URL in the terminal. Open your browser and
navigate to the URL to view the loss curves in the Scalar dashboard of
TensorBoard.

## Using the model in the browser

To see the trained model in action in the browser, do:

```sh
yarn
yarn watch
```

### Visualization of the attention mechanism

In the page opened by the `yarn watch` command, you can generate
random input date strings by clicking the "Random" button. A converted
date string will appear in the output text box each time a new input
date string is entered. You may also manually enter a date in the input-date
box. But make sure that the date falls into a range between the years 1950
and 2050, as this is the range of dates that the model is trained on.
See [date_format.js](./date_format.js) for more details.

In addition to converting the date and showing the output, the page visualizes
the attention matrix used by the trained model to convert the input date string
to the output one in (e.g., see the image below).

![Attention Matrix](./attention_matrix.png)

Each column of the attention matrix corresponds to a character in the input
string and each column corresponds to a character in the output string.
A darker color in a cell of the attention matrix
indicates a greater attention paid by the model to the corresponding input
character when generating the corresponding output character.

## Running unit tests

The data and model code in this example are covered by unit tests.
To run the unit tests:

```sh
cd ../
yarn
cd date-conversion-attention
yarn
yarn test
```
