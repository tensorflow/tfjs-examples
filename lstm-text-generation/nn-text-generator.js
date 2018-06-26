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

import {TextData} from './data';
import {sampleOneFromMultinomial} from './utils';

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
   * @param {TextData} textData An instance of `TextData`.
   */
  constructor(textData) {
    this._textData = textData;
    this._charSetSize = textData.charSetSize();
    this._sampleLen = textData.sampleLen();
    this._textLen = textData.textLen();

    this._modelIdentifier = textData.dataIdentifier();
    this._MODEL_SAVE_PATH_PREFIX = 'indexeddb://lstm-text-generation';
    this._modelSavePath =
        `${this._MODEL_SAVE_PATH_PREFIX}/${this._modelIdentifier}`;
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
          inputShape: i === 0 ? [this._sampleLen, this._charSetSize] : undefined
        }));
      }
      this.model.add(tf.layers.dense({
        units: this._charSetSize,
        activation: 'softmax'
      }));
    }
  }

  // TODO(cais): Refactor into createOrLoadModel and createModel.

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
      const [xs, ys] =  this._textData.nextDataEpoch(examplesPerEpoch);
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
    if (!(temperature > 0)) {
      throw new Error(`Invalid value in temperature: ${temperature}`);
    }
    const temperatureScalar = tf.scalar(temperature);

    let generated = '';
    let [sentence, sentenceIndices] = this._textData.getRandomSlice();

    while (generated.length < length) {
      const inputBuffer = new tf.TensorBuffer([1, this._sampleLen, this._charSetSize]);
      for (let i = 0; i < this._sampleLen; ++i) {
        inputBuffer.set(1, 0, i, sentenceIndices[i]);
      }
      const input = inputBuffer.toTensor();
      const output = this.model.predict(input);
      const winnerIndex = this._sample(tf.squeeze(output), temperatureScalar);
      const winnerChar = this._textData.getFromCharSet(winnerIndex);

      if (characterCallback != null) {
        await characterCallback(winnerChar);
      }

      generated += winnerChar;
      sentenceIndices = sentenceIndices.slice(1);
      sentenceIndices.push(winnerIndex);

      input.dispose();
      output.dispose();
    }
    temperatureScalar.dispose();
    return generated;
  }

  /**
   * Get model identifier.
   * @returns {string} The model identifier.
   */
  modelIdentifier() {
    return this._modelIdentifier;
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
   * Sample from probabilities.
   *
   * @param {tf.Tensor} preds Predicted probabilities, as a 1D `tf.Tensor` of
   *   shape `[this._charSetSize]`.
   * @param {tf.Tensor} temperature Temperature (i.e., a measure of randomness
   *   or diversity) to use during sampling. Number be a number > 0, as a Scalar
   *   `tf.Tensor`.
   * @returns {number} The 0-based index for the randomly-drawn sample, in the
   *   range of [0, this._charSetSize - 1].
   */
  _sample(preds, temperature) {
    return tf.tidy(() => {
      const logPreds = tf.div(tf.log(preds), temperature);
      const expPreds = tf.exp(logPreds);
      const sumExpPreds = tf.sum(expPreds);
      preds = tf.div(expPreds, sumExpPreds);
      // Treat preds a the probabilites of a multinomial distribution and
      // randomly draw a sample from the distribution.
      // TODO(cais): Investigate why tf.multinomial gives different results.
      //   When the difference is resolved, use tf.multinomial here.
      return sampleOneFromMultinomial(preds.dataSync());
    });
  }
};
