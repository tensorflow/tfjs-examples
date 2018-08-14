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
import * as utils from './utils';

// Boston Housing data constants:
const TRAIN_DATA = 'train-data';
const TRAIN_TARGET = 'train-target';
const TEST_DATA = 'test-data';
const TEST_TARGET = 'test-target';

/** Helper class to handle loading training and test data. */
export class BostonHousingDataset {
  constructor() {
    // Arrays to hold the data.
    this.trainFeatures = null;
    this.trainLabels = null;
    this.testFeatures = null;
    this.testLabels = null;
    // Metadata
    this.trainSize = 0;
    this.testSize = 0;
  }

  get numFeatures() {
    return this.trainFeatures[0].length;
  }

  /** Loads training and test data. */
  async loadData() {
    [this.trainFeatures, this.trainLabels, this.testFeatures, this.testLabels] =
        await Promise.all([
          utils.loadCsv(TRAIN_DATA), utils.loadCsv(TRAIN_TARGET),
          utils.loadCsv(TEST_DATA), utils.loadCsv(TEST_TARGET)
        ]);

    let {dataset: trainDataset, vectorMeans, vectorStddevs} =
        utils.normalizeDataset(this.trainFeatures);

    this.trainFeatures = trainDataset;

    let {dataset: testDataset} = utils.normalizeDataset(
        this.testFeatures, false, vectorMeans, vectorStddevs);

    this.testFeatures = testDataset;

    this.trainSize = this.trainFeatures.length;
    this.testSize = this.testFeatures.length;

    shuffle(this.trainFeatures, this.trainLabels);
    shuffle(this.testFeatures, this.testLabels);
  }

  getTrainData() {
    const dataShape = [this.trainSize, this.numFeatures];
    const targetShape = [this.trainSize, 1];

    const trainData =
        Float32Array.from([].concat.apply([], this.trainFeatures));
    const trainTarget =
        Float32Array.from([].concat.apply([], this.trainLabels));

    return {
      data: tf.tensor2d(trainData, dataShape),
      target: tf.tensor1d(trainTarget).reshape(targetShape)
    };
  }

  getTestData() {
    const dataShape = [this.testSize, this.numFeatures];
    const targetShape = [this.testSize, 1];

    const testData = Float32Array.from([].concat.apply([], this.testFeatures));
    const testTarget = Float32Array.from([].concat.apply([], this.testLabels));

    return {
      data: tf.tensor2d(testData, dataShape),
      target: tf.tensor1d(testTarget).reshape(targetShape)
    };
  }
}

/**
 * Shuffles data and label (maintaining alignment) using Fisher-Yates algorithm.
 */
function shuffle(data, label) {
  let counter = data.length;
  let temp = 0;
  let index = 0;
  while (counter > 0) {
    index = (Math.random() * counter) | 0;
    counter--;
    // data:
    temp = data[counter];
    data[counter] = data[index];
    data[index] = temp;
    // label:
    temp = label[counter];
    label[counter] = label[index];
    label[index] = temp;
  }
};
