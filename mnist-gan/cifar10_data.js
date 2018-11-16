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
const https = require('https');
const os = require('os');
const path = require('path');
const targz = require('targz');
const util = require('util');

const tf = require('@tensorflow/tfjs');

const CIFAR10_BINARY_DATA_URL =
    'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz';

const readFile = util.promisify(fs.readFile);

const IMAGE_H = 32;
const IMAGE_W = 32;
const IMAGE_CHANNELS = 3;
const IMAGE_BYTES = (1 + IMAGE_H * IMAGE_W * IMAGE_CHANNELS);

// Downloads a test file only once and returns the buffer for the file.
async function maybeDownloadAndExtract(downloadURL) {
  return new Promise((resolve, reject) => {
    const baseName = path.basename(downloadURL);
    const baseNameNoExt = baseName.slice(0, baseName.indexOf('.'));
    const destDir = path.join(os.tmpdir(), baseNameNoExt);
    if (fs.existsSync(destDir) && fs.readdirSync(destDir).length > 0) {
      resolve(path.join(destDir, fs.readdirSync(destDir)[0]));
      return;
    }

    const tarGzDestPath = path.join(os.tmpdir(), baseName);
    const tarGzFile = fs.createWriteStream(tarGzDestPath);
    https.get(downloadURL, async response => {
      console.log(`* Downloading from: ${downloadURL} --> ${tarGzDestPath}`);
      const stream = response.pipe(tarGzFile);

      stream.on('finish', () => {
        // Extract the tar ball.
        console.log(`* Extracting tar ball: ${tarGzDestPath} --> ${destDir}`);
        targz.decompress({src: tarGzDestPath, dest: destDir}, function(err) {
          if (err) {
            reject(err);
            return;
          } else {
            console.log(fs.readdirSync(destDir));
            // Remove the temporary downloaded .tar.gz file.
            fs.unlinkSync(tarGzDestPath);
            resolve(path.join(destDir, fs.readdirSync(destDir)[0]));
            return;
          }
        });
      });

      stream.on('error', err => {
        reject(err);
        return;
      });
    });
  });
}

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

async function loadCifar10Data() {
  const dataDir = await maybeDownloadAndExtract(CIFAR10_BINARY_DATA_URL);

  const numBatches = 5;
  const xs = [];
  const ys = [];
  for (let b = 0; b < numBatches; ++b) {
    const filePath = path.join(dataDir, `data_batch_${b + 1}.bin`);
    console.log(`* Reading data from file: ${filePath} ...`);
    const {x: batchX, y: batchY} = await readDataBatch(filePath);
    xs.push(batchX);
    ys.push(batchY);
  }
  // TODO(cais): Shuffle? DO NOT SUBMIT.
  const x = tf.concat(xs, 0);
  const y = tf.concat(ys, 0);
  tf.dispose([xs, ys]);
  return {x, y};
}

module.exports = {loadCifar10Data};