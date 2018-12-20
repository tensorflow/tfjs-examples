/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

import * as data from './data';
import * as loader from './loader';
import * as ui from './ui';

let model;

/**
 * Train a `tf.Model` to recognize Iris flower type.
 *
 * @param xTrain Training feature data, a `tf.Tensor` of shape
 *   [numTrainExamples, 4]. The second dimension include the features
 *   petal length, petalwidth, sepal length and sepal width.
 * @param yTrain One-hot training labels, a `tf.Tensor` of shape
 *   [numTrainExamples, 3].
 * @param xTest Test feature data, a `tf.Tensor` of shape [numTestExamples, 4].
 * @param yTest One-hot test labels, a `tf.Tensor` of shape
 *   [numTestExamples, 3].
 * @returns The trained `tf.Model` instance.
 */
async function trainModel(xTrain, yTrain, xTest, yTest) {
  ui.status('Training model... Please wait.');

  const params = ui.loadTrainParametersFromUI();

  // Define the topology of the model: two dense layers.
  const model = tf.sequential();
  model.add(tf.layers.dense(
      {units: 10, activation: 'sigmoid', inputShape: [xTrain.shape[1]]}));
  model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
  model.summary();

  const optimizer = tf.train.adam(params.learningRate);
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const trainLogs = [];
  const lossContainer = document.getElementById('lossCanvas');
  const accContainer = document.getElementById('accuracyCanvas');
  const beginMs = performance.now();
  // Call `model.fit` to train the model.
  const history = await model.fit(xTrain, yTrain, {
    epochs: params.epochs,
    validationData: [xTest, yTest],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // Plot the loss and accuracy values at the end of every training epoch.
        const secPerEpoch =
            (performance.now() - beginMs) / (1000 * (epoch + 1));
        ui.status(`Training model... Approximately ${
            secPerEpoch.toFixed(4)} seconds per epoch`)
        trainLogs.push(logs);
        tfvis.show.history(lossContainer, trainLogs, ['loss', 'val_loss'])
        tfvis.show.history(accContainer, trainLogs, ['acc', 'val_acc'])
        calculateAndDrawConfusionMatrix(model, xTest, yTest);
      },
    }
  });
  const secPerEpoch = (performance.now() - beginMs) / (1000 * params.epochs);
  ui.status(
      `Model training complete:  ${secPerEpoch.toFixed(4)} seconds per epoch`);
  return model;
}

/**
 * Run inference on manually-input Iris flower data.
 *
 * @param model The instance of `tf.Model` to run the inference with.
 */
async function predictOnManualInput(model) {
  if (model == null) {
    ui.setManualInputWinnerMessage('ERROR: Please load or train model first.');
    return;
  }

  // Use a `tf.tidy` scope to make sure that WebGL memory allocated for the
  // `predict` call is released at the end.
  tf.tidy(() => {
    // Prepare input data as a 2D `tf.Tensor`.
    const inputData = ui.getManualInputData();
    const input = tf.tensor2d([inputData], [1, 4]);

    // Call `model.predict` to get the prediction output as probabilities for
    // the Iris flower categories.

    const predictOut = model.predict(input);
    const logits = Array.from(predictOut.dataSync());
    const winner = data.IRIS_CLASSES[predictOut.argMax(-1).dataSync()[0]];
    ui.setManualInputWinnerMessage(winner);
    ui.renderLogitsForManualInput(logits);
  });
}

/**
 * Draw confusion matrix.
 */
async function calculateAndDrawConfusionMatrix(model, xTest, yTest) {
  const [preds, labels] = tf.tidy(() => {
    const preds = model.predict(xTest).argMax(-1);
    const labels = yTest.argMax(-1);
    return [preds, labels];
  });

  const confMatrixData = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = document.getElementById('confusion-matrix');
  tfvis.render.confusionMatrix(
      {values: confMatrixData, labels: data.IRIS_CLASSES},
      container,
      {shadeDiagonal: true},
  );

  tf.dispose([preds, labels]);
}

/**
 * Run inference on some test Iris flower data.
 *
 * @param model The instance of `tf.Model` to run the inference with.
 * @param xTest Test data feature, a `tf.Tensor` of shape [numTestExamples, 4].
 * @param yTest Test true labels, one-hot encoded, a `tf.Tensor` of shape
 *   [numTestExamples, 3].
 */
async function evaluateModelOnTestData(model, xTest, yTest) {
  ui.clearEvaluateTable();

  tf.tidy(() => {
    const xData = xTest.dataSync();
    const yTrue = yTest.argMax(-1).dataSync();
    const predictOut = model.predict(xTest);
    const yPred = predictOut.argMax(-1);
    ui.renderEvaluateTable(
        xData, yTrue, yPred.dataSync(), predictOut.dataSync());
    calculateAndDrawConfusionMatrix(model, xTest, yTest);
  });

  predictOnManualInput(model);
}

const HOSTED_MODEL_JSON_URL =
    'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json';

/**
 * The main function of the Iris demo.
 */
async function iris() {
  const [xTrain, yTrain, xTest, yTest] = data.getIrisData(0.15);

  const localLoadButton = document.getElementById('load-local');
  const localSaveButton = document.getElementById('save-local');
  const localRemoveButton = document.getElementById('remove-local');

  document.getElementById('train-from-scratch')
      .addEventListener('click', async () => {
        model = await trainModel(xTrain, yTrain, xTest, yTest);
        await evaluateModelOnTestData(model, xTest, yTest);
        localSaveButton.disabled = false;
      });

  if (await loader.urlExists(HOSTED_MODEL_JSON_URL)) {
    ui.status('Model available: ' + HOSTED_MODEL_JSON_URL);
    const button = document.getElementById('load-pretrained-remote');
    button.addEventListener('click', async () => {
      ui.clearEvaluateTable();
      model = await loader.loadHostedPretrainedModel(HOSTED_MODEL_JSON_URL);
      await predictOnManualInput(model);
      localSaveButton.disabled = false;
    });
  }

  localLoadButton.addEventListener('click', async () => {
    model = await loader.loadModelLocally();
    await predictOnManualInput(model);
  });

  localSaveButton.addEventListener('click', async () => {
    await loader.saveModelLocally(model);
    await loader.updateLocalModelStatus();
  });

  localRemoveButton.addEventListener('click', async () => {
    await loader.removeModelLocally();
    await loader.updateLocalModelStatus();
  });

  await loader.updateLocalModelStatus();

  ui.status('Standing by.');
  ui.wireUpEvaluateTableCallbacks(() => predictOnManualInput(model));
}

iris();
