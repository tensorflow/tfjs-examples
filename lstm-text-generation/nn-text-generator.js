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
import embed from 'vega-embed';

/**
 * Randomly shuffle an Array.
 *
 * Based on:
 *   https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
 *
 * @param {Array} array Input array.
 * @returns {Array} Shuffled array.
 */
function shuffle(array) {
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
 *
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

/**
 * Class that manages the neural network-based text generation.
 *
 * This class manages the following:
 *
 * - Converting training data (as a string) into one hot-encoded vectors.
 * - The creation, training, saving and loading of a LSTM model, written with
 *   the tf.layers API.
 * - Generating random text based on the LSTM model.
 */
export class NeuralNetworkTextGenerator {
  /**
   * Constructor of NeuralNetworkTextGenerator.
   *
   * @param {string} modelIdentifier An identifier for this instance of
   *   NeuralNetworkTextGenerator. Used during saving and loading of model.
   * @param {string} textString The training test data.
   * @param {number} sampleLen Length of each training example, i.e., the input
   *   sequence length expected by the LSTM model.
   * @param {number} sampleStep How many characters to skip when going from one
   *   example of the training data (in `textString`) to the next.
   */
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

  /**
   * Get model identifier.
   * @returns {string} The model identifier.
   */
  modelIdentifier() {
    return this._modelIdentifier;
  }

  /**
   * Get length of the training text data.
   * @returns {number} Length of training text data.
   */
  textLen() {
    return this._textLen;
  }

  /**
   * Get the size of the character set.
   * @returns {number} Size of the character set, i.e., how many unique
   *   characters there are in the training text data.
   */
  charSetSize() {
    return this._charSetSize;
  }

  lstmLayerSizes() {
    if (this.model == null) {
      throw new Error('Create model first.');
    }
    const numLSTMLayers = this.model.layers.length - 1;
    const layerSizes = [];
    for (let i = 0; i < numLSTMLayers; ++i) {
      layerSizes.push(this.model.layers[i].units);
    }
    return layerSizes.length === 1 ? layerSizes[0] : layerSizes;
  }

  /**
   *
   * @param {number} numExamples Number examples to generate.
   * @returns {[tf.Tensor, tf.Tensor]} `xs` and `ys` Tensors.
   *   `xs` has the shape of `[numExamples, this.sampleLen, this.charSetSize]`.
   *   `ys` has the shape of `[numExamples, this.charSetSize]`.
   */
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

  /**
   * Create LSTM model.
   *
   * @param {number | number[]} lstmLayerSizes Sizes of the LSTM layers, as a
   *   number or an non-empty array of numbers.
   * @param {bool} loadModelIfAvailable Whether to load model from IndexedDB if
   *   locally-saved model artifacts exist there.
   */
  async createModel(lstmLayerSizes, loadModelIfAvailable) {
    let createFromScratch;
    if (loadModelIfAvailable === true) {
      const modelsInfo = await tf.io.listModels();
      if (this._modelSavePath in modelsInfo) {
        console.log(`Loading existing model...`);
        this.model = await tf.loadModel(this._modelSavePath);
        console.log(`Loaded model from ${this._modelSavePath}`);
        createFromScratch = false;
      } else {
        console.log(
            `Cannot find model at ${this._modelSavePath}. ` +
            `Creating model from scratch.`);
        createFromScratch = true;
      }
    } else {
      createFromScratch = true;
    }

    if (createFromScratch) {
      if (!Array.isArray(lstmLayerSizes)) {
        lstmLayerSizes = [lstmLayerSizes];
      }
      if (lstmLayerSizes.length === 0) {
        throw new Error(
            'lstmLayerSizes must be a number or a non-empty array of numbers.');
      }

      this.model = tf.sequential();
      for (let i = 0; i < lstmLayerSizes.length; ++i) {
        const lstmLayerSize = lstmLayerSizes[i];
        if (!(lstmLayerSize > 0)) {
          throw new Error(
              `lstmLayerSizes must be a positive integer, ` +
              `but got ${lstmLayerSize}`);
        }
        this.model.add(tf.layers.lstm({
          units: lstmLayerSize,
          returnSequences: i < lstmLayerSizes.length - 1,
          inputShape: i === 0 ? [this._maxLen, this._charSetSize] : undefined
        }));
      }
      this.model.add(tf.layers.dense({
        units: this._charSetSize,
        activation: 'softmax'
      }));
    }
  }

  /**
   * Compile model for training.
   *
   * @param {number} learningRate The learning rate to use during training.
   */
  compileModel(learningRate) {
    const optimizer = tf.train.rmsprop(learningRate);
    this.model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
    console.log(`Compiled model with learning rate ${learningRate}`);
    this.model.summary();
  }

  /**
   * Remove the locally saved model from IndexedDB.
   */
  async removeModel() {
    if (await this.checkStoredModelStatus() == null) {
      throw new Error(
          'Cannot remove locally saved model because it does not exist.');
    }
    return await tf.io.removeModel(this._modelSavePath);
  }

  /**
   * Save the model in IndexedDB.
   */
  async saveModel() {
    if (this.model == null) {
      throw new Error('Cannot save model before creating model.');
    } else {
      return await this.model.save(this._modelSavePath);
    }
  }

  /**
   * Check the status of locally saved model.
   *
   * @returns If the locally saved model exists, the model info as a JSON
   *   object. Else, `undefined`.
   */
  async checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels();
    return modelsInfo[this._modelSavePath];
  }

  /**
   * Train the LSTM model that this object possesses.
   *
   * @param {number} numEpochs Number of epochs to train the model for.
   * @param {number} examplesPerEpoch Number of epochs to use in each training
   *   epochs.
   * @param {number} batchSize Batch size to use during training.
   * @param {() => void} trainBeginCallback A callback to invoke when the
   *   training begins.
   * @param {() => void} trainEpochCallback A callback to invoke at the end
   *   of every training epoch.
   * @param {() => void} trainBatchCallback A callback to invoke at the end
   *   of every training batch.
   */
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
            lossValues.push({'batch': ++batchCount, 'loss': logs.loss});
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

  /**
   * Sample from probabilities.
   *
   * @param {Float32Array} preds Predicted probabilities, of length
   *   `this.charSetSize`.
   * @param {number} temperature Temperature (i.e., a measure of randomness or
   *   diversity) to use during sampling. Number be a number > 0.
   */
  _sample(preds, temperature) {
    if (!(temperature > 0)) {
      throw new Error(`Invalid value in temperature: ${temperature}`);
    }
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
