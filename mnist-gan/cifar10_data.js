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

const fs = require('fs');
const path = require('path');
const util = require('util');
const jimp = require('jimp');

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const readFile = util.promisify(fs.readFile);

const IMAGE_H = 32;
const IMAGE_W = 32;
const IMAGE_CHANNELS = 3;
const IMAGE_BYTES = (1 + IMAGE_H * IMAGE_W * IMAGE_CHANNELS);

async function readDataBatch(filePath) {
  const buffer = await readFile(filePath);
  if (buffer.byteLength % IMAGE_BYTES !== 0) {
    throw new Error(
        `Non-integer number of images found in file ${filePath}: ` +
        `${buffer.byteLength} is not divisible by ${IMAGE_BYTES}`);
  }
  return tf.tidy(() => {
    const numImages = buffer.byteLength / IMAGE_BYTES;
    let p = 0;
    const labels = [];
    const imageArrays = [];
    for (let i = 0; i < numImages; ++i) {
      labels.push(buffer[p++]);
      const imageArray = [];
      for (let j = 0; j < IMAGE_BYTES - 1; ++j) {
        imageArray.push(buffer[p++]);
      }
      imageArrays.push(imageArray);
    }
    const y = tf.tensor2d(labels, [numImages, 1], 'float32');
    const x = tf.tensor2d(
                    imageArrays,
                    [numImages, IMAGE_H * IMAGE_W * IMAGE_CHANNELS], 'float32')
                  .reshape([numImages, IMAGE_H, IMAGE_W, IMAGE_CHANNELS]);
    return {x, y};
  });
}

async function readCifar10Data(baseDir) {
  const numBatches = 5;
  const xs = [];
  const ys = [];
  for (let b = 0; b < numBatches; ++b) {
    const filePath = path.join(baseDir, `data_batch_${b + 1}.bin`);
    console.log(`Loading data from ${filePath}...`);
    const {x: batchX, y: batchY} = await readDataBatch(filePath);
    xs.push(batchX);
    ys.push(batchY);
  }
  const x = tf.concat(xs, 0);
  const y = tf.concat(ys, 0);
  tf.dispose([xs, ys]);
  return {x, y};
}

(async function() {
  const dirPath = '/home/cais/Downloads/cifar-10-batches-bin';
  const {x, y} = await readCifar10Data(dirPath);
  console.log(x.shape);
  console.log(y.shape);
})();
