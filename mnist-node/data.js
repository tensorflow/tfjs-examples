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

const assert = require('assert');
const fs = require('fs');
const https = require('https');
const zlib = require('zlib');

// MNIST data constants:
const BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/';
const TRAIN_IMAGES_FILE = 'train-images-idx3-ubyte';
const TRAIN_LABELS_FILE = 'train-labels-idx1-ubyte';
const NUM_TRAIN_EXAMPLES = 60000;
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
      console.log('file exists: ', filename);
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
    assert.equal(headerValues[1], NUM_TRAIN_EXAMPLES);
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
    assert.equal(headerValues[1], NUM_TRAIN_EXAMPLES);

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

// TODO(kreeger): Doc me.
class MnistDataset {
  constructor() {
    this.dataset = null;
    this.batchIndex = 0;
  }

  // TODO(kreeger): Doc me.
  async loadData() {
    this.dataset = await Promise.all(
        [loadImages(TRAIN_IMAGES_FILE), loadLabels(TRAIN_LABELS_FILE)]);
    console.log('... loaded');
  }
}

module.exports = MnistDataset;