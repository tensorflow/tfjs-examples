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
const assert = require('assert');
const fs = require('fs');
const https = require('https');
const zlib = require('zlib');

// MNIST data constants:
const BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/';
const TRAIN_IMAGES_FILE = 'train-images-idx3-ubyte';
const TRAIN_LABELS_FILE = 'train-labels-idx1-ubyte';
const TEST_IMAGES_FILE = 't10k-images-idx3-ubyte';
const TEST_LABELS_FILE = 't10k-labels-idx1-ubyte';
const IMAGE_HEADER_BYTES = 16;
const IMAGE_DIMENSION_SIZE = 28;
const IMAGE_FLAT_SIZE = IMAGE_DIMENSION_SIZE * IMAGE_DIMENSION_SIZE;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;
const LABEL_FLAT_SIZE = 10;

async function downloadFile(filename) {
  return new Promise((resolve) => {
    const url = `${BASE_URL}${filename}.gz`;
    if (fs.existsSync(filename)) {
      return resolve();
    }
    const file = fs.createWriteStream(filename);
    console.log(`  * Downloading from: ${url}`);
    https.get(url, (response) => {
      const unzip = zlib.createGunzip();
      response.pipe(unzip).pipe(file);
      unzip.on('end', resolve);
    });
  });
}

function loadHeaderValues(buffer, headerLength) {
  const headerValues = [];
  for (let i = 0; i < headerLength / 4; i++) {
    // Header data is stored in-order (aka BE)
    headerValues[i] = buffer.readUInt32BE(i * 4);
  }
  return headerValues;
}

async function loadImages(filename) {
  await downloadFile(filename);
  return new Promise((resolve) => {
    const buffer = fs.readFileSync(filename);

    const headerBytes = IMAGE_HEADER_BYTES;
    const recordBytes = IMAGE_DIMENSION_SIZE * IMAGE_DIMENSION_SIZE;

    const headerValues = loadHeaderValues(buffer, headerBytes);
    assert.equal(headerValues[0], 2051);  // magic number for images
    assert.equal(headerValues[2], IMAGE_DIMENSION_SIZE);
    assert.equal(headerValues[3], IMAGE_DIMENSION_SIZE);

    const downsize = 1.0 / 255.0;

    const images = [];
    let index = headerBytes;
    while (index < buffer.byteLength) {
      const array = new Float32Array(recordBytes);
      for (let i = 0; i < recordBytes; i++) {
        array[i] = buffer.readUInt8(index++) * downsize;
      }
      images.push(array);
    }

    assert.equal(images.length, headerValues[1]);
    resolve(images);
  });
}

async function loadLabels(filename) {
  await downloadFile(filename);
  return new Promise((resolve) => {
    const buffer = fs.readFileSync(filename);

    const headerBytes = LABEL_HEADER_BYTES;
    const recordBytes = LABEL_RECORD_BYTE;

    const headerValues = loadHeaderValues(buffer, headerBytes);
    assert.equal(headerValues[0], 2049);  // magic number for labels

    const labels = [];
    let index = headerBytes;
    while (index < buffer.byteLength) {
      const array = new Uint8Array(recordBytes);
      for (let i = 0; i < recordBytes; i++) {
        array[i] = buffer.readUInt8(index++);
      }
      labels.push(array);
    }

    assert.equal(labels.length, headerValues[1]);
    resolve(labels);
  });
}

/** Helper class to handle loading training and test data. */
class MnistDataset {
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
      loadImages(TRAIN_IMAGES_FILE), loadLabels(TRAIN_LABELS_FILE),
      loadImages(TEST_IMAGES_FILE), loadLabels(TEST_LABELS_FILE)
    ]);
    this.trainSize = this.dataset[0].length;
    this.testSize = this.dataset[2].length;
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
   * Returns an object with training images and labels for a given batch size.
   */
  nextTrainBatch(batchSize) {
    return this._generateBatch(true, batchSize);
  }

  /**
   * Returns an object with test images and labels for a given batch size.
   */
  nextTestBatch(batchSize) {
    return this._generateBatch(false, batchSize);
  }

  _generateBatch(isTrainingData, batchSize) {
    let batchIndexMax;
    let size;
    if (isTrainingData) {
      batchIndexMax = this.trainBatchIndex + batchSize > this.trainSize ?
          this.trainSize - this.trainBatchIndex :
          batchSize + this.trainBatchIndex;
      size = batchIndexMax - this.trainBatchIndex;
    } else {
      batchIndexMax = this.testBatchIndex + batchSize > this.testSize ?
          this.testSize - this.testBatchIndex :
          batchSize + this.testBatchIndex;
      size = batchIndexMax - this.testBatchIndex;
    }

    // Only create one big array to hold batch of images.
    const imagesShape = [size, 28, 28, 1];
    const images = new Float32Array(tf.util.sizeFromShape(imagesShape));

    const labelsShape = [size, 1];
    const labels = new Int32Array(tf.util.sizeFromShape(labelsShape));

    let imageOffset = 0;
    let labelOffset = 0;
    while ((isTrainingData ? this.trainBatchIndex : this.testBatchIndex) <
           batchIndexMax) {
      if (isTrainingData) {
        images.set(this.dataset[0][this.trainBatchIndex], imageOffset);
        labels.set(this.dataset[1][this.trainBatchIndex], labelOffset);
        this.trainBatchIndex++;
      } else {
        images.set(this.dataset[2][this.testBatchIndex], imageOffset);
        labels.set(this.dataset[3][this.testBatchIndex], labelOffset);
        this.testBatchIndex++;
      }

      imageOffset += IMAGE_FLAT_SIZE;
      labelOffset += 1;
    }

    return {
      image: tf.tensor4d(images, imagesShape),
      label: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat()
    };
  }
}

module.exports = new MnistDataset();