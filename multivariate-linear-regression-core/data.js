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
    this.dataset = null;
    this.trainSize = 0;
    this.testSize = 0;
    this.trainBatchIndex = 0;
    this.testBatchIndex = 0;

    this.NUM_FEATURES = 13;
  }

  get num_features() {
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

    console.log(this.dataset);

    this.trainSize = this.dataset[0].length;
    this.testSize = this.dataset[2].length;

    // Shuffle training and test data:
    utils.shuffle(this.dataset[0], this.dataset[1]);
    utils.shuffle(this.dataset[2], this.dataset[3]);
  }

  /** Resets training data batches. */
  resetTraining() {
    this.trainBatchIndex = 0;
  }

  /** Resets test data batches. */
  resetTest() {
    this.testBatchIndex = 0;
  }

  /** Returns true if the training data has another batch. */
  hasMoreTrainingData() {
    return this.trainBatchIndex < this.trainSize;
  }

  /** Returns true if the test data has another batch. */
  hasMoreTestData() {
    return this.testBatchIndex < this.testSize;
  }

  /**
   * Returns an object with training data and target for a given batch size.
   */
  nextTrainBatch(batchSize) {
    return this.generateBatch(true, batchSize);
  }

  /**
   * Returns an object with test data and target for a given batch size.
   */
  nextTestBatch(batchSize) {
    return this.generateBatch(false, batchSize);
  }

  generateBatch(isTrainingData, batchSize) {
    let batchIndexMax;
    let size;
    let dataIndex;
    let targetIndex;
    if (isTrainingData) {
      batchIndexMax = this.trainBatchIndex + batchSize > this.trainSize ?
          this.trainSize :
          batchSize + this.trainBatchIndex;
      size = batchIndexMax - this.trainBatchIndex;
      dataIndex = 0;
      targetIndex = 1;
    } else {
      batchIndexMax = this.testBatchIndex + batchSize > this.testSize ?
          this.testSize :
          batchSize + this.testBatchIndex;
      size = batchIndexMax - this.testBatchIndex;
      dataIndex = 2;
      targetIndex = 3;
    }

    const dataShape = [size, this.NUM_FEATURES];
    const data = new Float32Array(tf.util.sizeFromShape(dataShape));

    const targetShape = [size, 1];
    const target = new Float32Array(tf.util.sizeFromShape(targetShape));

    let dataOffset = 0;
    let targetOffset = 0;
    let batchIndex =
        isTrainingData ? this.trainBatchIndex : this.testBatchIndex;
    while ((isTrainingData ? this.trainBatchIndex : this.testBatchIndex) <
           batchIndexMax) {
      data.set(this.dataset[dataIndex][batchIndex], dataOffset);
      target.set(this.dataset[targetIndex][batchIndex], targetOffset);

      batchIndex =
          isTrainingData ? ++this.trainBatchIndex : ++this.testBatchIndex;
      dataOffset += this.NUM_FEATURES;
      targetOffset += 1;
    }

    return {
      data: tf.tensor2d(data, dataShape),
      target: tf.tensor1d(target).reshape(targetShape)
    };
  }
}
