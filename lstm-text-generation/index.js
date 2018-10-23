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
 * Inspiration comes from:
 *
 * -
 * https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
 * - Andrej Karpathy. "The Unreasonable Effectiveness of Recurrent Neural
 * Networks" http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 */

import * as tf from '@tensorflow/tfjs';

import {TextData} from './data';
import {onTextGenerationBegin, onTextGenerationChar, onTrainBatchEnd, onTrainBegin, onTrainEpochEnd, setUpUI} from './ui';
import {sample} from './utils';

/**
 * Class that manages LSTM-based text generation.
 *
 * This class manages the following:
 *
 * - Creating and training a LSTM model, written with the tf.layers API, to
 *   predict the next character given a sequence of input characters.
 * - Generating random text using the LSTM model.
 */
export class LSTMTextGenerator {
  /**
   * Constructor of NeuralNetworkTextGenerator.
   *
   * @param {TextData} textData An instance of `TextData`.
   */
  constructor(textData) {
    this.textData_ = textData;
    this.charSetSize_ = textData.charSetSize();
    this.sampleLen_ = textData.sampleLen();
    this.textLen_ = textData.textLen();
  }

  /**
   * Create LSTM model from scratch.
   *
   * @param {number | number[]} lstmLayerSizes Sizes of the LSTM layers, as a
   *   number or an non-empty array of numbers.
   */
  createModel(lstmLayerSizes) {
    if (!Array.isArray(lstmLayerSizes)) {
      lstmLayerSizes = [lstmLayerSizes];
    }

    this.model = tf.sequential();
    for (let i = 0; i < lstmLayerSizes.length; ++i) {
      const lstmLayerSize = lstmLayerSizes[i];
      this.model.add(tf.layers.lstm({
        units: lstmLayerSize,
        returnSequences: i < lstmLayerSizes.length - 1,
        inputShape: i === 0 ? [this.sampleLen_, this.charSetSize_] : undefined
      }));
    }
    this.model.add(
        tf.layers.dense({units: this.charSetSize_, activation: 'softmax'}));
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
   * Train the LSTM model.
   *
   * @param {number} numEpochs Number of epochs to train the model for.
   * @param {number} examplesPerEpoch Number of epochs to use in each training
   *   epochs.
   * @param {number} batchSize Batch size to use during training.
   * @param {number} validationSplit Validation split to be used during the
   *   training epochs.
   */
  async fitModel(numEpochs, examplesPerEpoch, batchSize, validationSplit) {
    let batchCount = 0;
    const batchesPerEpoch = examplesPerEpoch / batchSize;
    const totalBatches = numEpochs * batchesPerEpoch;

    onTrainBegin();
    await tf.nextFrame();

    let t = new Date().getTime();
    for (let i = 0; i < numEpochs; ++i) {
      const [xs, ys] = this.textData_.nextDataEpoch(examplesPerEpoch);
      await this.model.fit(xs, ys, {
        epochs: 1,
        batchSize: batchSize,
        validationSplit,
        callbacks: {
          onBatchEnd: async (batch, logs) => {
            // Calculate the training speed in the current batch, in # of
            // examples per second.
            const t1 = new Date().getTime();
            const examplesPerSec = batchSize / ((t1 - t) / 1e3);
            t = t1;
            onTrainBatchEnd(
                logs.loss, ++batchCount / totalBatches, examplesPerSec);
          },
          onEpochEnd: async (epoch, logs) => {
            onTrainEpochEnd(logs.val_loss);
          },
        }
      });
      xs.dispose();
      ys.dispose();
    }
  }

  /**
   * Generate text using the LSTM model.
   *
   * @param {number[]} sentenceIndices Seed sentence, represented as the
   *   indices of the constituent characters.
   * @param {number} length Length of the text to generate, in number of
   *   characters.
   * @param {number} temperature Temperature parameter. Must be a number > 0.
   * @returns {string} The generated text.
   */
  async generateText(sentenceIndices, length, temperature) {
    onTextGenerationBegin();
    const temperatureScalar = tf.scalar(temperature);

    let generated = '';
    while (generated.length < length) {
      // Encode the current input sequence as a one-hot Tensor.
      const inputBuffer =
          new tf.TensorBuffer([1, this.sampleLen_, this.charSetSize_]);
      for (let i = 0; i < this.sampleLen_; ++i) {
        inputBuffer.set(1, 0, i, sentenceIndices[i]);
      }
      const input = inputBuffer.toTensor();

      // Call model.predict() to get the probability values of the next
      // character.
      const output = this.model.predict(input);

      // Sample randomly based on the probability values.
      const winnerIndex = sample(tf.squeeze(output), temperatureScalar);
      const winnerChar = this.textData_.getFromCharSet(winnerIndex);
      await onTextGenerationChar(winnerChar);

      generated += winnerChar;
      sentenceIndices = sentenceIndices.slice(1);
      sentenceIndices.push(winnerIndex);

      input.dispose();
      output.dispose();
    }
    temperatureScalar.dispose();
    return generated;
  }
};

/**
 * A subclass of LSTMTextGenerator that supports model saving and loading.
 *
 * The model is saved to and loaded from browser's IndexedDB.
 */
export class SaveableLSTMTextGenerator extends LSTMTextGenerator {
  /**
   * Constructor of NeuralNetworkTextGenerator.
   *
   * @param {TextData} textData An instance of `TextData`.
   */
  constructor(textData) {
    super(textData);
    this.modelIdentifier_ = textData.dataIdentifier();
    this.MODEL_SAVE_PATH_PREFIX_ = 'indexeddb://lstm-text-generation';
    this.modelSavePath_ =
        `${this.MODEL_SAVE_PATH_PREFIX_}/${this.modelIdentifier_}`;
  }

  /**
   * Get model identifier.
   *
   * @returns {string} The model identifier.
   */
  modelIdentifier() {
    return this.modelIdentifier_;
  }

  /**
   * Create LSTM model if it is not saved locally; load it if it is.
   *
   * @param {number | number[]} lstmLayerSizes Sizes of the LSTM layers, as a
   *   number or an non-empty array of numbers.
   */
  async loadModel(lstmLayerSizes) {
    const modelsInfo = await tf.io.listModels();
    if (this.modelSavePath_ in modelsInfo) {
      console.log(`Loading existing model...`);
      this.model = await tf.loadModel(this.modelSavePath_);
      console.log(`Loaded model from ${this.modelSavePath_}`);
    } else {
      throw new Error(
          `Cannot find model at ${this.modelSavePath_}. ` +
          `Creating model from scratch.`);
    }
  }

  /**
   * Save the model in IndexedDB.
   *
   * @returns ModelInfo from the saving, if the saving succeeds.
   */
  async saveModel() {
    if (this.model == null) {
      throw new Error('Cannot save model before creating model.');
    } else {
      return await this.model.save(this.modelSavePath_);
    }
  }

  /**
   * Remove the locally saved model from IndexedDB.
   */
  async removeModel() {
    if (await this.checkStoredModelStatus() == null) {
      throw new Error(
          'Cannot remove locally saved model because it does not exist.');
    }
    return await tf.io.removeModel(this.modelSavePath_);
  }

  /**
   * Check the status of locally saved model.
   *
   * @returns If the locally saved model exists, the model info as a JSON
   *   object. Else, `undefined`.
   */
  async checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels();
    return modelsInfo[this.modelSavePath_];
  }

  /**
   * Get a representation of the sizes of the LSTM layers in the model.
   *
   * @returns {number | number[]} The sizes (i.e., number of units) of the
   *   LSTM layers that the model contains. If there is only one LSTM layer, a
   *   single number is returned; else, an Array of numbers is returned.
   */
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
}

setUpUI();
