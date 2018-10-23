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

// Website Phishing data constants:
const TRAIN_DATA = 'train-data';
const TRAIN_TARGET = 'train-target';
const TEST_DATA = 'test-data';
const TEST_TARGET = 'test-target';

/** Helper class to handle loading training and test data. */
export class WebsitePhishingDataset {
  constructor() {
    this.dataset = null;
    this.trainSize = 0;
    this.testSize = 0;
    this.trainBatchIndex = 0;
    this.testBatchIndex = 0;

    this.NUM_FEATURES = 30;
    this.NUM_CLASSES = 2;
  }

  get numFeatures() {
    return this.NUM_FEATURES;
  }

  /** Loads training and test data. */
  async loadData() {
    this.dataset = await Promise.all([
      utils.loadCsv(TRAIN_DATA), utils.loadCsv(TRAIN_TARGET),
      utils.loadCsv(TEST_DATA), utils.loadCsv(TEST_TARGET)
    ]);

    let {dataset: trainDataset, vectorMeans, vectorStddevs} =
        utils.normalizeDataset(this.dataset[0]);

    this.dataset[0] = trainDataset;

    let {dataset: testDataset} = utils.normalizeDataset(
        this.dataset[2], false, vectorMeans, vectorStddevs);

    this.dataset[2] = testDataset;

    this.trainSize = this.dataset[0].length;
    this.testSize = this.dataset[2].length;

    utils.shuffle(this.dataset[0], this.dataset[1]);
    utils.shuffle(this.dataset[2], this.dataset[3]);
  }

  getTrainData() {
    const dataShape = [this.trainSize, this.NUM_FEATURES];

    const trainData = Float32Array.from([].concat.apply([], this.dataset[0]));
    const trainTarget = Float32Array.from([].concat.apply([], this.dataset[1]));

    return {
      data: tf.tensor2d(trainData, dataShape),
      target: tf.tensor1d(trainTarget)
    };
  }

  getTestData() {
    const dataShape = [this.testSize, this.NUM_FEATURES];

    const testData = Float32Array.from([].concat.apply([], this.dataset[2]));
    const testTarget = Float32Array.from([].concat.apply([], this.dataset[3]));

    return {
      data: tf.tensor2d(testData, dataShape),
      target: tf.tensor1d(testTarget)
    };
  }
}
