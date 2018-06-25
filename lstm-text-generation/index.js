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

/**
 * TensorFlow.js Example: LSTM Text Generation.
 *
 * Inspirations come from:
 *
 * - https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
 * - Andrej Karpathy. "The Unreasonable Effectiveness of Recurrent Neural Networks"
 *   http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 */

import * as tf from '@tensorflow/tfjs';

import {NeuralNetworkTextGenerator} from './nn-text-generator';

const TEXT_DATA_URLS = {
  'nietzsche': 'https://storage.googleapis.com/tfjs-examples/lstm-text-generation/data/nietzsche.txt',
  'tfjs-code': 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.11.7/dist/tf.js'
}

// UI controls.
const testText = document.getElementById('test-text');
const createOrLoadModelButton = document.getElementById('create-or-load-model');
const deleteModelButton = document.getElementById('delete-model');
const trainModelButton = document.getElementById('train-model');
const generateTextButton = document.getElementById('generate-text');

const appStatus = document.getElementById('app-status');
const loadTextDataButton = document.getElementById('load-text-data');
const textDataSelect = document.getElementById('text-data-select');

const examplesPerEpochInput = document.getElementById('examples-per-epoch');
const batchSizeInput = document.getElementById('batch-size');
const epochsInput = document.getElementById('epochs');
const learningRateInput = document.getElementById('learning-rate');

const generateLengthInput = document.getElementById('generate-length');
const temperatureInput = document.getElementById('temperature');
const generatedTextInput = document.getElementById('generated-text');

const modelAvailableInfo = document.getElementById('model-available');

const sampleLen = 40;
const sampleStep = 3;

// Module-global instance of NeuralNetworkTextGenerator.
let textGenerator;

/**
 * Refresh the status of locally saved model (in IndexedDB).
 */
async function refreshLocalModelStatus() {
  const modelInfo = await textGenerator.checkStoredModelStatus();
  if (modelInfo == null) {
    modelAvailableInfo.value =
        `No locally saved model for "${textGenerator.modelIdentifier()}"`;
    createOrLoadModelButton.textContent = 'Create model';
    deleteModelButton.disabled = true;
  } else {
    modelAvailableInfo.value = `Saved @ ${modelInfo.dateSaved.toISOString()}`;
    createOrLoadModelButton.textContent = 'Load model';
    deleteModelButton.disabled = false;
  }
  createOrLoadModelButton.disabled = false;
}

async function generateText() {
  try {
    if (textGenerator == null) {
      throw new Error('Load text data set first.');
    }
    const generateLength = Number.parseInt(generateLengthInput.value);
    const temperature = Number.parseFloat(temperatureInput.value);
    if (!(temperature > 0 && temperature <= 1)) {
      throw new Error(`Invalid temperature: ${temperature}`);
    }
    generatedTextInput.value = '';
    logStatus('Generating text...');

    let charCount = 0;
    const sentence = await textGenerator.generateText(
        generateLength,
        temperature,
        async char => {
          generatedTextInput.value += char;
          charCount++;
          logStatus(
              `Generating text: ${charCount}/${generateLength} complete...`);
          await tf.nextFrame();
        });
    generatedTextInput.value = sentence;
    logStatus('Done generating text.');
    return sentence;
  } catch (err) {
    logStatus(`ERROR: Failed to generate text: ${err.message}`);
  }
}

function logStatus(message) {
  appStatus.textContent = message;
}

/**
 * Wire up UI callbacks.
 */

loadTextDataButton.addEventListener('click', async () => {
  textDataSelect.disabled = true;
  loadTextDataButton.disabled = true;
  const modelIdentifier = textDataSelect.value;
  const url = TEXT_DATA_URLS[modelIdentifier];
  try {
    logStatus(`Loading text data from URL: ${url} ...`);
    const response = await fetch(url);
    const textString = await response.text();
    testText.value = textString;
    logStatus(
        `Done loading text data ` +
        `(length=${(textString.length / 1024).toFixed(1)}k). ` +
        `Next, please load or create model.`);
  } catch (err) {
    logStatus('Failed to load text data: ' + err.message);
  }

  if (testText.value.length === 0) {
    throw new Error("ERROR: Empty text data.");
  }
  textGenerator = new NeuralNetworkTextGenerator(
      modelIdentifier, testText.value, sampleLen, sampleStep);
  await refreshLocalModelStatus();
});


createOrLoadModelButton.addEventListener('click', async () => {
  createOrLoadModelButton.disabled = true;
  if (textGenerator == null) {
    createOrLoadModelButton.disabled = false;
    throw new Error('Load text data set first.');
  }

  logStatus('Creating or loading model... Please wait.');
  await textGenerator.createModel(true);
  logStatus(
      'Done creating or loading model. ' +
      'Now you can train the model or use it to generate text.');
  trainModelButton.disabled = false;
  generateTextButton.disabled = false;
});

deleteModelButton.addEventListener('click', async () => {
  if (textGenerator == null) {
    throw new Error('Load text data set first.');
  }
  if (confirm(`Are you sure you want to delete the model ` +
      `'${textGenerator.modelIdentifier()}'?`)) {
    console.log(await textGenerator.removeModel());
    await refreshLocalModelStatus();
  }
});

trainModelButton.addEventListener('click', async () => {
  if (textGenerator == null) {
    throw new Error('Load text data set first.');
  }

  const numEpochs = Number.parseInt(epochsInput.value);
  const examplesPerEpoch = Number.parseInt(examplesPerEpochInput.value);
  const batchSize = Number.parseInt(batchSizeInput.value);
  const learningRate = Number.parseFloat(learningRateInput.value);
  if (!(learningRate > 0)) {
    createOrLoadModelButton.disabled = false;
    throw new Error(`Invalid learning rate: ${learningRate}`);
  }
  createOrLoadModelButton.disabled = true;
  deleteModelButton.disabled = true;
  trainModelButton.disabled = true;
  generateTextButton.disabled = true;

  textGenerator.compileModel(learningRate);
  await textGenerator.fitModel(
      numEpochs, examplesPerEpoch, batchSize,
      () => {
        logStatus('Starting model training...');
      },
      null,
      // async () => {
      //   await generateText();
      // },
      (progress, examplesPerSec) =>
        logStatus(
           `Model training: ${(progress * 1e2).toFixed(1)}% complete... ` +
           `(${examplesPerSec.toFixed(0)} examples/s)`));
  console.log(await textGenerator.saveModel());
  await refreshLocalModelStatus();

  await generateText();
  createOrLoadModelButton.disabled = false;
  deleteModelButton.disabled = false;
  trainModelButton.disabled = false;
  generateTextButton.disabled = false;
});

generateTextButton.addEventListener('click', async () => {
  if (textGenerator == null) {
    throw new Error('Load text data set first.');
  }
  trainModelButton.disabled = true;
  generateTextButton.disabled = true;
  await generateText();
  trainModelButton.disabled = false;
  generateTextButton.disabled = false;
});

