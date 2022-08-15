
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

import {TEXT_DATA_URLS, TextData} from './data';
import {SaveableLSTMTextGenerator} from './index';

// UI controls.
const testText = document.getElementById('test-text');
const createOrLoadModelButton = document.getElementById('create-or-load-model');
const deleteModelButton = document.getElementById('delete-model');
const trainModelButton = document.getElementById('train-model');
const generateTextButton = document.getElementById('generate-text');

const appStatus = document.getElementById('app-status');
const textGenerationStatus = document.getElementById('text-generation-status');
const loadTextDataButton = document.getElementById('load-text-data');
const textDataSelect = document.getElementById('text-data-select');

const lstmLayersSizesInput = document.getElementById('lstm-layer-sizes');

const examplesPerEpochInput = document.getElementById('examples-per-epoch');
const batchSizeInput = document.getElementById('batch-size');
const epochsInput = document.getElementById('epochs');
const validationSplitInput = document.getElementById('validation-split');
const learningRateInput = document.getElementById('learning-rate');

const generateLengthInput = document.getElementById('generate-length');
const temperatureInput = document.getElementById('temperature');
const seedTextInput = document.getElementById('seed-text');
const generatedTextInput = document.getElementById('generated-text');

const modelAvailableInfo = document.getElementById('model-available');

const sampleLen = 40;
const sampleStep = 3;

// Module-global instance of TextData.
let textData;

// Module-global instance of SaveableLSTMTextGenerator.
let textGenerator;

function logStatus(message) {
  appStatus.textContent = message;
}

let batchLossValues;
let epochLossValues;

/**
 * A function to call when a training process starts.
 */
export function onTrainBegin() {
  batchLossValues = [];
  epochLossValues = [];
  logStatus('Starting model training...');
}

/**
 * A function to call when a batch is competed during training.
 *
 * @param {number} loss Loss value of the current batch.
 * @param {number} progress Total training progress, as a number between 0
 *   and 1.
 * @param {number} examplesPerSec The training speed in the batch, in examples
 *   per second.
 */
export function onTrainBatchEnd(logs, progress, examplesPerSec) {
  logStatus(
      `Model training: ${(progress * 1e2).toFixed(1)}% complete... ` +
      `(${examplesPerSec.toFixed(0)} examples/s)`);
  batchLossValues.push(logs);
  const container = document.getElementById('batch-loss-canvas');
  tfvis.show.history(container, batchLossValues, ['loss'], {
    height: 300,
    zoomToFit: true,
    xLabel: 'Batch',
  });
}

export function onTrainEpochEnd(logs) {
  epochLossValues.push(logs);
  const container = document.getElementById('epoch-loss-canvas');
  tfvis.show.history(container, epochLossValues, ['loss', 'val_loss'], {
    height: 300,
    zoomToFit: true,
    xLabel: 'Epoch',
  });
}

/**
 * A function to call when text generation begins.
 *
 * @param {string} seedSentence: The seed sentence being used for text
 *   generation.
 */
export function onTextGenerationBegin() {
  generatedTextInput.value = '';
  logStatus('Generating text...');
}

/**
 * A function to call each time a character is obtained during text generation.
 *
 * @param {string} char The just-generated character.
 */
export async function onTextGenerationChar(char) {
  generatedTextInput.value += char;
  generatedTextInput.scrollTop = generatedTextInput.scrollHeight;
  const charCount = generatedTextInput.value.length;
  const generateLength = parseInt(generateLengthInput.value);
  const status = `Generating text: ${charCount}/${generateLength} complete...`;
  logStatus(status);
  textGenerationStatus.textContent = status;
  await tf.nextFrame();
}

export function setUpUI() {
  /**
   * Refresh the status of locally saved model (in IndexedDB).
   */
  async function refreshLocalModelStatus() {
    const modelInfo = await textGenerator.checkStoredModelStatus();
    if (modelInfo == null) {
      modelAvailableInfo.innerText =
          `No locally saved model for "${textGenerator.modelIdentifier()}".`;
      createOrLoadModelButton.textContent = 'Create model';
      deleteModelButton.disabled = true;
      enableModelParameterControls();
    } else {
      modelAvailableInfo.innerText =
          `Saved @ ${modelInfo.dateSaved.toISOString()}`;
      createOrLoadModelButton.textContent = 'Load model';
      deleteModelButton.disabled = false;
      disableModelParameterControls();
    }
    createOrLoadModelButton.disabled = false;
  }

  function disableModelButtons() {
    createOrLoadModelButton.disabled = true;
    deleteModelButton.disabled = true;
    trainModelButton.disabled = true;
    generateTextButton.disabled = true;
  }

  function enableModelButtons() {
    createOrLoadModelButton.disabled = false;
    deleteModelButton.disabled = false;
    trainModelButton.disabled = false;
    generateTextButton.disabled = false;
  }

  /**
   * Use `textGenerator` to generate random text, show the characters on the
   * screen as they are generated one by one.
   */
  async function generateText() {
    try {
      disableModelButtons();

      if (textGenerator == null) {
        logStatus('ERROR: Please load text data set first.');
        return;
      }
      const generateLength = parseInt(generateLengthInput.value);
      const temperature = parseFloat(temperatureInput.value);
      if (!(generateLength > 0)) {
        logStatus(
            `ERROR: Invalid generation length: ${generateLength}. ` +
            `Generation length must be a positive number.`);
        enableModelButtons();
        return;
      }
      if (!(temperature > 0 && temperature <= 1)) {
        logStatus(
            `ERROR: Invalid temperature: ${temperature}. ` +
            `Temperature must be a positive number.`);
        enableModelButtons();
        return;
      }

      let seedSentence;
      let seedSentenceIndices;
      if (seedTextInput.value.length === 0) {
        // Seed sentence is not specified yet. Get it from the data.
        [seedSentence, seedSentenceIndices] = textData.getRandomSlice();
        seedTextInput.value = seedSentence;
      } else {
        seedSentence = seedTextInput.value;
        if (seedSentence.length < textData.sampleLen()) {
          logStatus(
              `ERROR: Seed text must have a length of at least ` +
              `${textData.sampleLen()}, but has a length of ` +
              `${seedSentence.length}.`);
          enableModelButtons();
          return;
        }
        seedSentence = seedSentence.slice(
            seedSentence.length - textData.sampleLen(), seedSentence.length);
        seedSentenceIndices = textData.textToIndices(seedSentence);
      }

      const sentence = await textGenerator.generateText(
          seedSentenceIndices, generateLength, temperature);
      generatedTextInput.value = sentence;
      const status = 'Done generating text.';
      logStatus(status);
      textGenerationStatus.value = status;

      enableModelButtons();

      return sentence;
    } catch (err) {
      logStatus(`ERROR: Failed to generate text: ${err.message}, ${err.stack}`);
    }
  }

  function disableModelParameterControls() {
    lstmLayersSizesInput.disabled = true;
  }

  function enableModelParameterControls() {
    lstmLayersSizesInput.disabled = false;
  }

  function updateModelParameterControls(lstmLayerSizes) {
    lstmLayersSizesInput.value = lstmLayerSizes;
  }

  function updateTextInputParameters() {
    Object.keys(TEXT_DATA_URLS).forEach(key => {
      var opt = document.createElement('option');
      opt.value = key;
      opt.innerHTML = TEXT_DATA_URLS[key].needle;
      textDataSelect.appendChild(opt);
    });
  }

  function hashCode(str) {
    let hash = 5381, i = str.length;
    while (i) {
      hash = (hash * 33) ^ str.charCodeAt(--i);
    }
    return hash >>> 0;
  }

  /**
   * Initialize UI state.
   */

  disableModelParameterControls();

  /**
   * Update Text Inputs
   */
  updateTextInputParameters();

  /**
   * Wire up UI callbacks.
   */

  loadTextDataButton.addEventListener('click', async () => {
    textDataSelect.disabled = true;
    loadTextDataButton.disabled = true;
    let dataIdentifier = textDataSelect.value;
    const url = TEXT_DATA_URLS[dataIdentifier].url;
    if (testText.value.length === 0) {
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
        logStatus('ERROR: Empty text data.');
        return;
      }
    } else {
      dataIdentifier = hashCode(testText.value);
    }
    textData =
        new TextData(dataIdentifier, testText.value, sampleLen, sampleStep);
    textGenerator = new SaveableLSTMTextGenerator(textData);
    await refreshLocalModelStatus();
  });

  createOrLoadModelButton.addEventListener('click', async () => {
    createOrLoadModelButton.disabled = true;
    if (textGenerator == null) {
      createOrLoadModelButton.disabled = false;
      logStatus('ERROR: Please load text data set first.');
      return;
    }

    if (await textGenerator.checkStoredModelStatus()) {
      // Load locally-saved model.
      logStatus('Loading model from IndexedDB... Please wait.');
      await textGenerator.loadModel();
      updateModelParameterControls(textGenerator.lstmLayerSizes());
      logStatus(
          'Done loading model from IndexedDB. ' +
          'Now you can train the model further or use it to generate text.');
    } else {
      // Create model from scratch.
      logStatus('Creating model... Please wait.');
      const lstmLayerSizes = lstmLayersSizesInput.value.trim().split(',').map(
          s => parseInt(s));

      // Sanity check on the LSTM layer sizes.
      if (lstmLayerSizes.length === 0) {
        logStatus('ERROR: Invalid LSTM layer sizes.');
        return;
      }
      for (let i = 0; i < lstmLayerSizes.length; ++i) {
        const lstmLayerSize = lstmLayerSizes[i];
        if (!(lstmLayerSize > 0)) {
          logStatus(
              `ERROR: lstmLayerSizes must be a positive integer, ` +
              `but got ${lstmLayerSize} for layer ${i + 1} ` +
              `of ${lstmLayerSizes.length}.`);
          return;
        }
      }

      await textGenerator.createModel(lstmLayerSizes);
      logStatus(
          'Done creating model. ' +
          'Now you can train the model or use it to generate text.');
    }

    trainModelButton.disabled = false;
    generateTextButton.disabled = false;
  });

  deleteModelButton.addEventListener('click', async () => {
    if (textGenerator == null) {
      logStatus('ERROR: Please load text data set first.');
      return;
    }
    if (confirm(
            `Are you sure you want to delete the model ` +
            `'${textGenerator.modelIdentifier()}'?`)) {
      console.log(await textGenerator.removeModel());
      await refreshLocalModelStatus();
    }
  });

  trainModelButton.addEventListener('click', async () => {
    if (textGenerator == null) {
      logStatus('ERROR: Please load text data set first.');
      return;
    }

    const numEpochs = parseInt(epochsInput.value);
    if (!(numEpochs > 0)) {
      logStatus(`ERROR: Invalid number of epochs: ${numEpochs}`);
      return;
    }
    const examplesPerEpoch = parseInt(examplesPerEpochInput.value);
    if (!(examplesPerEpoch > 0)) {
      logStatus(`ERROR: Invalid examples per epoch: ${examplesPerEpoch}`);
      return;
    }
    const batchSize = parseInt(batchSizeInput.value);
    if (!(batchSize > 0)) {
      logStatus(`ERROR: Invalid batch size: ${batchSize}`);
      return;
    }
    const validationSplit = parseFloat(validationSplitInput.value);
    if (!(validationSplit >= 0 && validationSplit < 1)) {
      logStatus(`ERROR: Invalid validation split: ${validationSplit}`);
      return;
    }
    const learningRate = parseFloat(learningRateInput.value);
    if (!(learningRate > 0)) {
      logStatus(`ERROR: Invalid learning rate: ${learningRate}`);
      return;
    }

    textGenerator.compileModel(learningRate);
    disableModelButtons();
    await textGenerator.fitModel(
        numEpochs, examplesPerEpoch, batchSize, validationSplit);
    console.log(await textGenerator.saveModel());
    await refreshLocalModelStatus();
    enableModelButtons();

    await generateText();
  });

  generateTextButton.addEventListener('click', async () => {
    if (textGenerator == null) {
      logStatus('ERROR: Load text data set first.');
      return;
    }
    await generateText();
  });
}
