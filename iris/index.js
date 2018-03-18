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

async function loadHostedPretrainedModel() {
  const HOSTED_MODEL_JSON_URL =
      'https://storage.googleapis.com/tfjs-models/tfjs-layers/iris_v1/model.json';
  status('Loading pretrained model from ' + HOSTED_MODEL_JSON_URL);
  try {
    model = await tf.loadModel(HOSTED_MODEL_JSON_URL);
    status('Done loading pretrained model.');
  } catch (err) {
    status('Loading pretrained model failed.');
  }
}

async function trainModel(xTrain, yTrain, xTest, yTest) {
  status('Training model... Please wait.');

  const params = loadTrainParametersFromUI();

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
  const history = await model.fit(xTrain, yTrain, {
    epochs: params.epochs,
    validationData: [xTest, yTest],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        lossValues.push({'epoch': epoch, 'loss': logs['loss'], 'set': 'train'});
        lossValues.push(
            {'epoch': epoch, 'loss': logs['val_loss'], 'set': 'validation'});
        plotLosses(lossValues);

        accuracyValues.push(
            {'epoch': epoch, 'accuracy': logs['acc'], 'set': 'train'});
        accuracyValues.push(
            {'epoch': epoch, 'accuracy': logs['val_acc'], 'set': 'validation'});
        plotAccuracies(accuracyValues);

        await tf.nextFrame();
      },
    }
  });

  status('Model training complete.');
  return model;
}

async function predictOnManualInput(model) {
  if (model == null) {
    setManualInputWinnerMessage('ERROR: Please load or train model first.');
    return;
  }

  const inputData = getManualInputData();
  const input = tf.tensor2d([inputData], [1, 4]);
  const predictOut = await model.predict(input);
  const logits = Array.from(predictOut.dataSync());
  const winner = IRIS_CLASSES[predictOut.argMax(-1).dataSync()[0]];
  setManualInputWinnerMessage(winner);
  renderLogitsForManualInput(logits);
  input.dispose();
  predictOut.dispose();
}

async function evaluateModelOnTestData(model, xTest, yTest) {
  clearEvaluateTable();
  const xData = xTest.dataSync();
  const yTrue = yTest.argMax(-1).dataSync();
  const predictOut = await model.predict(xTest);
  // const logits = Array.from(predictOut);  // TOD(cais): Remove
  const yPred = predictOut.argMax(-1);

  renderEvaluateTable(xData, yTrue, yPred.dataSync(), predictOut.dataSync());
  predictOnManualInput(model);

  yPred.dispoe();
  predictOut.dispose();
}

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
