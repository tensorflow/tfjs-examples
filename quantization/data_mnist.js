/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
 * Provides methods and classes that support loading data from
 * both MNIST and Fashion MNIST datasets.
 */

import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as http from 'http';
import * as https from 'https';
import * as path from 'path';
import * as util from 'util';
import * as zlib from 'zlib';

const exists = util.promisify(fs.exists);
const mkdir = util.promisify(fs.mkdir);
const readFile = util.promisify(fs.readFile);
const rename = util.promisify(fs.rename);

// Shared specs for the MNIST and Fashion MNIST datasets.
const IMAGE_HEADER_MAGIC_NUM = 2051;
const IMAGE_HEADER_BYTES = 16;
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
const LABEL_HEADER_MAGIC_NUM = 2049;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;
const LABEL_FLAT_SIZE = 10;

// Downloads a test file only once and returns the buffer for the file.
export async function fetchOnceAndSaveToDiskWithBuffer(
    baseURL, destDir, filename) {

  return new Promise(async (resolve, reject) => {
    const url = `${baseURL}${filename}.gz`;
    const localPath = path.join(destDir, filename);
    if (await exists(localPath)) {
      resolve(readFile(localPath));
      return;
    }
    const file = fs.createWriteStream(filename);
    console.log(`  * Downloading from: ${url}`);
    let httpModule;
    if (url.indexOf('https://') === 0) {
      httpModule = https;
    } else if (url.indexOf('http://') === 0) {
      httpModule =  http;
    } else {
      return reject(`Unrecognized protocol in URL: ${url}`);
    }

    httpModule.get(url, (response) => {
      const unzip = zlib.createGunzip();
      response.pipe(unzip).pipe(file);
      unzip.on('end', async () => {
        await rename(filename, localPath);
        resolve(readFile(localPath));
      });
    });
  });
}

function loadHeaderValues(buffer, headerLength) {
  const headerValues = [];
  for (let i = 0; i < headerLength / 4; i++) {
    // Header data is stored in-order (aka big-endian)
    headerValues[i] = buffer.readUInt32BE(i * 4);
  }
  return headerValues;
}

async function loadImages(baseURL, destDir, filename) {
  const buffer =
      await fetchOnceAndSaveToDiskWithBuffer(baseURL, destDir, filename);

  const headerBytes = IMAGE_HEADER_BYTES;
  const recordBytes = IMAGE_HEIGHT * IMAGE_WIDTH;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  tf.util.assert(
      headerValues[0] === IMAGE_HEADER_MAGIC_NUM,
      () => `Image file header doesn't match expected magic num.`);
  tf.util.assert(
      headerValues[2] === IMAGE_HEIGHT,
      () => `Value in file header (${headerValues[2]}) doesn't ` +
      `match the expected image height ${IMAGE_HEIGHT}`);
  tf.util.assert(
      headerValues[3] === IMAGE_WIDTH,
      () => `Value in file header (${headerValues[3]}) doesn't ` +
      `match the expected image height ${IMAGE_WIDTH}`);

  const images = [];
  let index = headerBytes;
  while (index < buffer.byteLength) {
    const array = new Float32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      // Normalize the pixel values into the 0-1 interval, from
      // the original 0-255 interval.
      array[i] = buffer.readUInt8(index++) / 255;
    }
    images.push(array);
  }

  tf.util.assert(
      images.length === headerValues[1],
      () => `Actual images length (${images.length} doesn't match ` +
      `value in header (${headerValues[1]})`);
  return images;
}

async function loadLabels(baseURL, destDir, filename) {
  const buffer =
      await fetchOnceAndSaveToDiskWithBuffer(baseURL, destDir, filename);

  const headerBytes = LABEL_HEADER_BYTES;
  const recordBytes = LABEL_RECORD_BYTE;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  tf.util.assert(
      headerValues[0] === LABEL_HEADER_MAGIC_NUM,
      () => `Label file header doesn't match expected magic num.`);

  const labels = [];
  let index = headerBytes;
  while (index < buffer.byteLength) {
    const array = new Int32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      array[i] = buffer.readUInt8(index++);
    }
    labels.push(array);
  }

  tf.util.assert(
      labels.length === headerValues[1],
      () => `Actual labels length (${images.length} doesn't match ` +
      `value in header (${headerValues[1]})`);
  return labels;
}

/** Helper class to handle loading training and test data. */
export class MnistDataset {
  // MNIST data constants:
  constructor() {
    this.dataset = null;
    this.trainSize = 0;
    this.testSize = 0;
    this.trainBatchIndex = 0;
    this.testBatchIndex = 0;
  }

  getBaseUrlAndFilePaths() {
    return {
      baseUrl: 'https://storage.googleapis.com/cvdf-datasets/mnist/',
      destDir: 'data-mnist',
      trainImages: 'train-images-idx3-ubyte',
      trainLabels: 'train-labels-idx1-ubyte',
      testImages: 't10k-images-idx3-ubyte',
      testLabels: 't10k-labels-idx1-ubyte'
    }
  }

  /** Loads training and test data. */
  async loadData() {
    const baseUrlAndFilePaths = this.getBaseUrlAndFilePaths();
    const baseUrl = baseUrlAndFilePaths.baseUrl;
    const destDir = baseUrlAndFilePaths.destDir;
    if (!(await exists(destDir))) {
      await mkdir(destDir);
    }

    this.dataset = await Promise.all([
      loadImages(baseUrl, destDir, baseUrlAndFilePaths.trainImages),
      loadLabels(baseUrl, destDir, baseUrlAndFilePaths.trainLabels),
      loadImages(baseUrl, destDir, baseUrlAndFilePaths.testImages),
      loadLabels(baseUrl, destDir, baseUrlAndFilePaths.testLabels)
    ]);
    this.trainSize = this.dataset[0].length;
    this.testSize = this.dataset[2].length;
  }

  getTrainData() {
    return this.getData_(true);
  }

  getTestData() {
    return this.getData_(false);
  }

  getData_(isTrainingData) {
    let imagesIndex;
    let labelsIndex;
    if (isTrainingData) {
      imagesIndex = 0;
      labelsIndex = 1;
    } else {
      imagesIndex = 2;
      labelsIndex = 3;
    }
    const size = this.dataset[imagesIndex].length;
    tf.util.assert(
        this.dataset[labelsIndex].length === size,
        `Mismatch in the number of images (${size}) and ` +
            `the number of labels (${this.dataset[labelsIndex].length})`);

    // Only create one big array to hold batch of images.
    const imagesShape = [size, IMAGE_HEIGHT, IMAGE_WIDTH, 1];
    const images = new Float32Array(tf.util.sizeFromShape(imagesShape));
    const labels = new Int32Array(tf.util.sizeFromShape([size, 1]));

    let imageOffset = 0;
    let labelOffset = 0;
    for (let i = 0; i < size; ++i) {
      images.set(this.dataset[imagesIndex][i], imageOffset);
      labels.set(this.dataset[labelsIndex][i], labelOffset);
      imageOffset += IMAGE_FLAT_SIZE;
      labelOffset += 1;
    }

    return {
      images: tf.tensor4d(images, imagesShape),
      labels: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat()
    };
  }
}

export class FashionMnistDataset extends MnistDataset {
  getBaseUrlAndFilePaths() {
    return {
      baseUrl: 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',
      destDir: 'data-fashion-mnist',
      trainImages: 'train-images-idx3-ubyte',
      trainLabels: 'train-labels-idx1-ubyte',
      testImages: 't10k-images-idx3-ubyte',
      testLabels: 't10k-labels-idx1-ubyte'
    }
  }
}
