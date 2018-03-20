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

import {getIrisData, IRIS_CLASSES, IRIS_NUM_CLASSES} from './data';
import {clearEvaluateTable, getManualInputData, loadTrainParametersFromUI, plotAccuracies, plotLosses, renderEvaluateTable, renderLogitsForManualInput, setManualInputWinnerMessage, status, wireUpEvaluateTableCallbacks} from './ui';

let model;

/**
 * Load pretrained model stored at a remote URL.
 *
 * @return An instance of `tf.Model` with model topology and weights loaded.
 */
async function loadHostedPretrainedModel() {
  const HOSTED_MODEL_JSON_URL =
      'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json';
  status('Loading pretrained model from ' + HOSTED_MODEL_JSON_URL);
  try {
    model = await tf.loadModel(HOSTED_MODEL_JSON_URL);
    status('Done loading pretrained model.');
  } catch (err) {
    status('Loading pretrained model failed.');
  }
}

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
  status('Training model... Please wait.');

  const params = loadTrainParametersFromUI();

  // Define the topology of the model: two dense layers.
  const model = tf.sequential();
  model.add(tf.layers.dense(
      {units: 10, activation: 'sigmoid', inputShape: [xTrain.shape[1]]}));
  model.add(tf.layers.dense({units: 3, activation: 'softmax'}));

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
        plotLosses(lossValues, epoch, logs.loss, logs.val_loss);
        plotAccuracies(accuracyValues, epoch, logs.acc, logs.val_acc);

        // Await web page DOM to refresh for the most recently plotted values.
        await tf.nextFrame();
      },
    }
  });

  status('Model training complete.');
  return model;
}

/**
 * Run inference on manually-input Iris flower data.
 *
 * @param model The instance of `tf.Model` to run the inference with.
 */
async function predictOnManualInput(model) {
  if (model == null) {
    setManualInputWinnerMessage('ERROR: Please load or train model first.');
    return;
  }

  // Use a `tf.tidy` scope to make sure that WebGL memory allocated for the is
  // released at the end.
  tf.tidy(() => {
    // Prepare input data as a 2D `tf.Tensor`.
    const inputData = getManualInputData();
    const input = tf.tensor2d([inputData], [1, 4]);

    // Call `model.predict` to get the prediction output as probabilities for
    // the the Iris flower categories.

    const predictOut = model.predict(input);
    const logits = Array.from(predictOut.dataSync());
    const winner = IRIS_CLASSES[predictOut.argMax(-1).dataSync()[0]];
    setManualInputWinnerMessage(winner);
    renderLogitsForManualInput(logits);
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
  clearEvaluateTable();

  tf.tidy(() => {
    const xData = xTest.dataSync();
    const yTrue = yTest.argMax(-1).dataSync();
    const predictOut = model.predict(xTest);
    const yPred = predictOut.argMax(-1);
    renderEvaluateTable(xData, yTrue, yPred.dataSync(), predictOut.dataSync());
  });

  predictOnManualInput(model);
}

/**
 * The main function of the Iris demo.
 */
async function iris() {
  const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);

  document.getElementById('train-from-scratch')
      .addEventListener('click', async () => {
        model = await trainModel(xTrain, yTrain, xTest, yTest);
        evaluateModelOnTestData(model, xTest, yTest);
      });

  document.getElementById('load-pretrained')
      .addEventListener('click', async () => {
        clearEvaluateTable();
        await loadHostedPretrainedModel();
        predictOnManualInput(model);
      });

  wireUpEvaluateTableCallbacks(() => predictOnManualInput(model));
}

iris();
