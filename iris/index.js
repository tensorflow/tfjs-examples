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

import * as data from './data';
import * as loader from './loader';
import * as ui from './ui';

// TODO(cais): Remove in favor of tf.confusionMatrix once it's available.
//   https://github.com/tensorflow/tfjs/issues/771
/**
 * Calcualte the confusion matrix.
 *
 * @param {tf.Tensor} labels The target labels, assumed to be 0-based integers
 *   for the categories. The shape is `[numExamples]`, where
 *   `numExamples` is the number of examples included.
 * @param {tf.Tensor} predictions The predicted probabilities, assumed to be
 *   0-based integers for the categories. Must have the same shape as `labels`.
 * @param {number} numClasses Number of all classes, if not provided,
 *   will calculate from both `labels` and `predictions`.
 * @return {tf.Tensor} The confusion matrix as a 2D tf.Tensor. The value at row
 *   `r` and column `c` is the number of times examples of actual class `r` were
 *   predicted as class `c`.
 */
function confusionMatrix(labels, predictions, numClasses) {
  tf.util.assert(
      numClasses == null || numClasses > 0 && Number.isInteger(numClasses),
      `If provided, numClasses must be a positive integer, ` +
          `but got ${numClasses}`);
  tf.util.assert(
      labels.rank === 1,
      `Expected the rank of labels to be 1, but got ${labels.rank}`);
  tf.util.assert(
      predictions.rank === 1,
      `Expected the rank of predictions to be 1, ` +
          `but got ${predictions.rank}`);
  tf.util.assert(
      labels.shape[0] === predictions.shape[0],
      `Mismatch in the number of examples: ` +
      `${labels.shape[0]} vs. ${predictions.shape[0]}`);

  if (numClasses == null) {
    // If numClasses is not provided, determine it.
    const labelClasses = labels.max().get();
    const predictionClasses = predictions.max().get();
    numClasses = (labelClasses > predictionClasses ?
        labelClasses : predictionClasses) + 1;
  }

  return tf.tidy(() => {
    const oneHotLabels = tf.oneHot(labels, numClasses);
    const oneHotPredictions = tf.oneHot(predictions, numClasses);
    return oneHotLabels.transpose().matMul(oneHotPredictions);
  });
}

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

  const lossValues = [];
  const accuracyValues = [];
  // Call `model.fit` to train the model.
  const history = await model.fit(xTrain, yTrain, {
    epochs: params.epochs,
    validationData: [xTest, yTest],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // Plot the loss and accuracy values at the end of every training epoch.
        ui.plotLosses(lossValues, epoch, logs.loss, logs.val_loss);
        ui.plotAccuracies(accuracyValues, epoch, logs.acc, logs.val_acc);
        calculateAndDrawConfusionMatrix(model, xTest, yTest);
      },
    }
  });

  ui.status('Model training complete.');
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
function calculateAndDrawConfusionMatrix(model, xTest, yTest) {
  tf.tidy(() => {
    const predictOut = model.predict(xTest);
    const yPred = predictOut.argMax(-1);

    const confusionMat = confusionMatrix(yTest.argMax(-1), yPred);
    ui.drawConfusionMatrix(confusionMat);
  });
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
