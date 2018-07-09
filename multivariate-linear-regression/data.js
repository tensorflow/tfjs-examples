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

const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const https = require('https');
const zlib = require('zlib');
const csv = require('csv-parser')

// Boston Housing data constants:
const BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/boston-housing/';
const TRAIN_DATA = 'train-data';
const TRAIN_TARGET = 'train-target';
const TEST_DATA = 'test-data';
const TEST_TARGET = 'test-target';
const NUM_FEATURES = 13;

// Downloads a test file only once and returns the csv
async function loadCsv(filename) {
  return new Promise(resolve => {
    const url = `${BASE_URL}${filename}.gz`;
    const csvName = `${filename}.csv`;
    if (fs.existsSync(csvName)) {
      resolve(readCsv(csvName));
      return;
    }
    const file = fs.createWriteStream(filename);
    console.log(`  * Downloading from: ${url}`);
    https.get(url, (response) => {
      const unzip = zlib.createGunzip();
      response.pipe(unzip).pipe(file);
      unzip.on('end', () => {
        resolve(readCsv(filename));
      });
    });
  });
}

async function readCsv(filename) {
  return new Promise(resolve => {
    const result = [];
    fs.createReadStream(filename)
        .pipe(csv())
        .on('data',
            data => {
              const output = Object.keys(data).sort().map(key => data[key]);
              result.push(output)
            })
        .on('end', () => resolve(result));
  });
}


// Shuffles data and label using Fisher-Yates algorithm.
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
}

/** Helper class to handle loading training and test data. */
class BostonHousingDataset {
  constructor() {
    this.dataset = null;
    this.trainSize = 0;
    this.testSize = 0;
    this.trainBatchIndex = 0;
    this.testBatchIndex = 0;
  }

  /** Loads training and test data. */
  async loadData() {
    this.dataset = await Promise.all([
      loadCsv(TRAIN_DATA), loadCsv(TRAIN_TARGET), loadCsv(TEST_DATA),
      loadCsv(TEST_TARGET)
    ]);

    this.trainSize = this.dataset[0].length;
    this.testSize = this.dataset[2].length;

    // Shuffle training and test data:
    shuffle(this.dataset[0], this.dataset[1]);
    shuffle(this.dataset[2], this.dataset[3]);
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
    return this._generateBatch(true, batchSize);
  }

  /**
   * Returns an object with test data and target for a given batch size.
   */
  nextTestBatch(batchSize) {
    return this._generateBatch(false, batchSize);
  }

  _generateBatch(isTrainingData, batchSize) {
    let batchIndexMax;
    let size;
    let dataIndex;
    let targetIndex;
    if (isTrainingData) {
      batchIndexMax = this.trainBatchIndex + batchSize > this.trainSize ?
          this.trainSize - this.trainBatchIndex :
          batchSize + this.trainBatchIndex;
      size = batchIndexMax - this.trainBatchIndex;
      dataIndex = 0;
      targetIndex = 1;
    } else {
      batchIndexMax = this.testBatchIndex + batchSize > this.testSize ?
          this.testSize - this.testBatchIndex :
          batchSize + this.testBatchIndex;
      size = batchIndexMax - this.testBatchIndex;
      dataIndex = 2;
      targetIndex = 3;
    }

    const dataShape = [size, NUM_FEATURES];
    const data = new Float32Array(tf.util.sizeFromShape(dataShape));

    const targetShape = [size, 1];
    const target = new Float32Array(tf.util.sizeFromShape(targetShape));

    let dataOffset = 0;
    let targetOffset = 0;
    while ((isTrainingData ? this.trainBatchIndex : this.testBatchIndex) <
           batchIndexMax) {
      data.set(this.dataset[dataIndex][this.trainBatchIndex], dataOffset);
      target.set(this.dataset[targetIndex][this.trainBatchIndex], targetOffset);

      isTrainingData ? this.trainBatchIndex++ : this.testBatchIndex++;
      dataOffset += NUM_FEATURES;
      targetOffset += 1;
    }

    return {data: tf.tensor2d(data, dataShape), target: tf.tensor1d(target)};
  }
}

module.exports = new BostonHousingDataset();
