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
import embed from 'vega-embed';

const TEXT_DATA_URLS = {
  'nietzsche': 'https://storage.googleapis.com/tfjs-examples/lstm-text-generation/data/nietzsche.txt',
  'tfjs-code': 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.11.7/dist/tf.js'
}

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

let textGenerator;

loadTextDataButton.addEventListener('click', async () => {
  textDataSelect.disabled = true;
  loadTextDataButton.disabled = true;
  const modelIdentifier = textDataSelect.value;
  const url = TEXT_DATA_URLS[modelIdentifier];
  try {
    appStatus.textContent = `Loading text data from URL: ${url} ...`;
    const response = await fetch(url);
    const textString = await response.text();
    testText.value = textString;
    appStatus.textContent =
        `Done loading text data (length=${(textString.length / 1024).toFixed(1)}k). ` +
        `Next, please load or create model.`;
  } catch (err) {
    appStatus.textContent = 'Failed to load text data: ' + err.message;
  }

  if (testText.value.length === 0) {
    throw new Error("ERROR: Empty text data.");
  }
  textGenerator = new NeuralNetworkTextGenerator(
      modelIdentifier, testText.value, sampleLen, sampleStep);
  await refreshLocalModelStatus();
});

async function refreshLocalModelStatus() {
  const modelInfo = await textGenerator.checkStoredModelStatus();
  if (modelInfo == null) {
    modelAvailableInfo.value =
        `No locally saved model for "${textGenerator.modelIdentifier()}"`;
    createOrLoadModelButton.textContent = 'Create model';
    deleteModelButton.disabled = true;
  } else {
    modelAvailableInfo.value = `Saved @ ${modelInfo.dateSaved}`;
    createOrLoadModelButton.textContent = 'Load model';
    deleteModelButton.disabled = false;
  }
  createOrLoadModelButton.disabled = false;
}

/**
 * Randomly shuffle an Array.
 * @param {Array} array
 * @returns {Array} Shuffled array.
 */
function shuffle(array) {
  // Origin of the code:
  // https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
  let currentIndex = array.length;
  let temporaryValue;
  let randomIndex;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {
    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }
  return array;
}

/**
 * Draw one sample from a multinomial distribution.
 * @param {number[]} probs Probabilities. Assumed to sum to 1.
 * @returns {number} A zero-based sample index.
 */
function sampleOneFromMultinomial(probs) {
  const score = Math.random();
  let cumProb = 0;
  const n = probs.length;
  for (let i = 0; i < n; ++i) {
    if (score >= cumProb && score < cumProb + probs[i]) {
      return i;
    }
    cumProb += probs[i];
  }
  return n - 1;
}

class NeuralNetworkTextGenerator {
  constructor(modelIdentifier, textString, sampleLen, sampleStep) {
    if (!modelIdentifier) {
      throw new Error('Model identifier is not provided.');
    }

    this._modelIdentifier = modelIdentifier;
    this._MODEL_SAVE_PATH_PREFIX = 'indexeddb://lstm-text-generation';
    this._modelSavePath =
        `${this._MODEL_SAVE_PATH_PREFIX}/${this._modelIdentifier}`;

    this._textString = textString;
    this._textLen = textString.length;
    this._sampleLen = sampleLen;
    this._sampleStep = sampleStep;

    this._getCharSet();
    this._textToIndices();
    this._generateExampleBeginIndices();
  }

  modelIdentifier() {
    return this._modelIdentifier;
  }

  textLen() {
    return this._textLen;
  }

  charSetSize() {
    return this._charSetSize;
  }

  nextDataEpoch(numExamples) {
    const xsBuffer = new tf.TensorBuffer([
        numExamples, this._sampleLen, this._charSetSize]);
    const ysBuffer  = new tf.TensorBuffer([numExamples, this._charSetSize]);
    for (let i = 0; i < numExamples; ++i) {
      const beginIndex = this._exampleBeginIndices[
          this._examplePosition % this._exampleBeginIndices.length];
      for (let j = 0; j < this._sampleLen; ++j) {
        xsBuffer.set(1, i, j, this._indices[beginIndex + j]);
      }
      ysBuffer.set(1, i, this._indices[beginIndex + this._sampleLen]);
      this._examplePosition++;
    }
    return [xsBuffer.toTensor(), ysBuffer.toTensor()];
  }

  /**
   * Generate text using the LSTM model that this object possesses.
   *
   * @param {number} length Length of the text to generate, in number of
   *   characters.
   * @param {number} temperature Temperature parameter. Must be a number > 0.
   * @param {(char: string) => void} characterCallback Action to take when a
   *   character is generated.
   * @returns {string} The generated text.
   */
  async generateText(length, temperature, characterCallback) {
    if (this.model == null) {
      throw new Error('Create model first.');
    }

    const startIndex =
        Math.round(Math.random() * (this._textLen - this._sampleLen - 1));
    let generated = '';
    const sentence =
        this._textString.slice(startIndex, startIndex + this._sampleLen);
    console.log(`Generating with seed: "${sentence}"`);
    let sentenceIndices = Array.from(
        this._indices.slice(startIndex, startIndex + this._sampleLen));

    while (generated.length < length) {
      const inputBuffer =
          new tf.TensorBuffer([1, this._sampleLen, this._charSetSize]);
      for (let i = 0; i < this._sampleLen; ++i) {
        inputBuffer.set(1, 0, i, sentenceIndices[i]);
      }
      const input = inputBuffer.toTensor();
      const output = this.model.predict(input);
      const winnerIndex = this._sample(output.dataSync(), temperature);
      const winnerChar = this._charSet[winnerIndex];

      if (characterCallback != null) {
        await characterCallback(winnerChar);
      }

      generated += winnerChar;
      sentenceIndices = sentenceIndices.slice(1);
      sentenceIndices.push(winnerIndex);

      input.dispose();
      output.dispose();
    }
    return generated;
  }

  async createModel(loadModelIfAvailable) {
    let createFromScratch = false;
    if (loadModelIfAvailable === true) {
      const modelsInfo = await tf.io.listModels();
      if (this._modelSavePath in modelsInfo) {
        console.log(`Loading existing model...`);
        this.model = await tf.loadModel(this._modelSavePath);
        console.log(`Loaded model from ${this._modelSavePath}`);
      } else {
        console.log(
            `Cannot find model at ${this._modelSavePath}. ` +
            `Creating model from scratch.`);
        createFromScratch = true;
      }
    }

    if (createFromScratch) {
      this.model = tf.sequential();
      this.model.add(tf.layers.lstm({
        units: 128,
        returnSequences: false,
        inputShape: [this._maxLen, this._charSetSize]
      }));
      this.model.add(tf.layers.dense({
        units: this._charSetSize,
        activation: 'softmax'
      }));
    }
  }

  compileModel(learningRate) {
    const optimizer = tf.train.rmsprop(learningRate);
    this.model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
    console.log(`Compiled model with learning rate ${learningRate}`);
    this.model.summary();
  }

  async removeModel() {
    if (await this.checkStoredModelStatus() == null) {
      throw new Error(
          'Cannot remove locally saved model because it does not exist.');
    }
    return await tf.io.removeModel(this._modelSavePath);
  }

  async saveModel() {
    if (this.model == null) {
      throw new Error('Cannot save model before creating model.');
    } else {
      return await this.model.save(this._modelSavePath);
    }
  }

  async checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels();
    return modelsInfo[this._modelSavePath];
  }

  async fitModel(numEpochs,
                 examplesPerEpoch,
                 batchSize,
                 trainBeginCallback,
                 trainEpochCallback,
                 trainBatchCallback) {
    const lossValues = [];
    let batchCount = 0;
    const batchesPerEpoch = examplesPerEpoch / batchSize;
    const totalBatches = numEpochs * batchesPerEpoch;
    trainBeginCallback();
    await tf.nextFrame();
    let t = new Date().getTime();
    for (let i = 0; i < numEpochs; ++i) {
      const [xs, ys] =  this.nextDataEpoch(examplesPerEpoch);
      await this.model.fit(xs, ys, {
        epochs: 1,
        batchSize: batchSize,
        callbacks: {
          onTrainEnd: async () => {
            if (trainEpochCallback != null) {
              await trainEpochCallback();
            }
          },
          onBatchEnd: async (batch, logs) => {
            console.log(`batch ${batch + 1}, logs = ${JSON.stringify(logs)}`);
            lossValues.push({'batch': ++batchCount, 'loss': logs.loss});
            embed(
              '#loss-canvas', {
                '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
                'data': {'values': lossValues},
                'mark': 'line',
                'encoding': {
                  'x': {'field': 'batch', 'type': 'ordinal'},
                  'y': {'field': 'loss', 'type': 'quantitative'},
                },
                'width': 300,
              },
              {});
            if (trainBatchCallback != null) {
              const t1 = new Date().getTime();
              const examplesPerSec = batchSize / ((t1 - t) / 1e3);
              t = t1;
              trainBatchCallback(batchCount / totalBatches, examplesPerSec);
            }
            await tf.nextFrame();
          },
          onEpochEnd: async (epochs, log) => {
            console.log(
                `epoch ${i + 1}/${numEpochs}; ` +
                `log = ${JSON.stringify(log)}`);
            await tf.nextFrame();
          },
        }
      });
      xs.dispose();
      ys.dispose();
    }
  }

  _sample(preds, temperature) {
    const logPreds = preds.map(pred => Math.log(pred) / temperature);
    const expPreds = logPreds.map(logPred => Math.exp(logPred));
    let sumExpPreds = 0;
    for (const expPred of expPreds) {
      sumExpPreds += expPred;
    }
    preds = expPreds.map(expPred => expPred / sumExpPreds);
    // Treat preds a the probabilites of a multinomial distribution and
    // randomly draw a sample from the distribution.
    return sampleOneFromMultinomial(preds);
  }

  /**
   * Get the set of unique characters from text.
   */
  _getCharSet() {
    this._charSet = [];
    for (let i = 0; i < this._textLen; ++i) {
      if (this._charSet.indexOf(this._textString[i]) === -1) {
        this._charSet.push(this._textString[i]);
      }
    }
    this._charSetSize = this._charSet.length;
  }

  /**
   * Convert text string to integers.
   */
  _textToIndices() {
    this._indices = new Uint16Array(this._textLen);
    for (let i = 0; i < this._textLen; ++i) {
      this._indices[i] = this._charSet.indexOf(this._textString[i]);
    }
  }

  /**
   * Generate the example-begin indices; shuffle them randomly.
   */
  _generateExampleBeginIndices() {
    // Prepare beginning indices of examples.
    const exampleBeginIndices = [];
    for (let i = 0;
        i < this._textLen - this._sampleLen - 1;
        i += this._sampleStep) {
      exampleBeginIndices.push(i);
    }

    // Randomly shuffle the beginning indices.
    this._exampleBeginIndices = shuffle(exampleBeginIndices);
    this._examplePosition = 0;
  }
};

createOrLoadModelButton.addEventListener('click', async () => {
  createOrLoadModelButton.disabled = true;
  if (textGenerator == null) {
    createOrLoadModelButton.disabled = false;
    throw new Error('Load text data set first.');
  }

  appStatus.textContent = 'Creating or loading model... Please wait.';
  await textGenerator.createModel(true);
  appStatus.textContent =
      'Done creating or loading model. ' +
      'Now you can train the model or use it to generate text.';
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
  trainModelButton.disabled = true;
  generateTextButton.disabled = true;

  textGenerator.compileModel(learningRate);
  await textGenerator.fitModel(
      numEpochs, examplesPerEpoch, batchSize,
      () => {
        appStatus.textContent = 'Model training is starting...';
      },
      null,
      // async () => {
      //   await generateText();
      // },
      (progress, examplesPerSec) => {
        appStatus.textContent =
           `Model training: ${(progress * 1e2).toFixed(1)}% complete... ` +
           `(${examplesPerSec.toFixed(0)} examples/s)`
      });
  console.log(await textGenerator.saveModel());
  await refreshLocalModelStatus();

  await generateText();
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
    appStatus.textContent = 'Generating text...';

    const sentence = await textGenerator.generateText(
        generateLength,
        temperature,
        async char => {
          generatedTextInput.value += char;
          await tf.nextFrame();
        });
    generatedTextInput.value = sentence;
    appStatus.textContent = 'Done generating text.';
    return sentence;
  } catch (err) {
    appStatus.textContent = `Failed to generate text: ${err.message}`;
  }
}
