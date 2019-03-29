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

const tf = require('@tensorflow/tfjs');
const assert = require('assert');
const fs = require('fs');
const util = require('util');
const jimp = require('jimp');
const terminalImage = require('terminal-image');

const readFile = util.promisify(fs.readFile);

const DATASET_PATH = './dataset';
const TRAIN_IMAGES_FILE = 'train-images-idx3-ubyte';
const IMAGE_HEADER_MAGIC_NUM = 2051;
const IMAGE_HEADER_BYTES = 16;
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_CHANNELS = 1;
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS;


/**
 * Read the header of the dataset file
 *
 * @param {Buffer} buffer
 * @param {number} headerLength
 *
 * @returns {number[]} MNIST data header values
 */
function loadHeaderValues(buffer, headerLength) {
  const headerValues = [];
  for (let i = 0; i < headerLength / 4; i++) {
    // Header data is stored in-order (aka big-endian)
    headerValues[i] = buffer.readUInt32BE(i * 4);
  }
  return headerValues;
}

/**
 * Load the images from the given file and normalize the data to 0-1 range.
 *
 * Input file should be in the MNIST/FashionMNSIT file format
 *
 * @param {string} filepath
 *
 * @returns {Float32Array[]} an array of images represented as typed arrays.
 */
async function loadImages(filepath) {
  if (!fs.existsSync(filepath)) {
    console.log(`Data File: ${filepath} does not exist.
      Please see the README for instructions on how to download it`);
    process.exit(1);
  }

  const buffer = await readFile(filepath)

  const headerBytes = IMAGE_HEADER_BYTES;
  const recordBytes = IMAGE_HEIGHT * IMAGE_WIDTH;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  assert.equal(headerValues[0], IMAGE_HEADER_MAGIC_NUM);
  assert.equal(headerValues[2], IMAGE_HEIGHT);
  assert.equal(headerValues[3], IMAGE_WIDTH);

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

  assert.equal(images.length, headerValues[1]);
  tf.util.shuffle(images);
  return images;
}

/**
 * Take an array of images (represented as typedarrays) and return
 * a tensor representing them.
 *
 * @param {Float32Array[]} imagesData
 *
 * @returns {Tensor3d} tensor of input images
 */
function batchImages(imagesData) {
  const numImages = imagesData.length;
  const flat = [];
  for (let i = 0; i < numImages; i++) {
    const image = imagesData[i];
    for (let j = 0; j < image.length; j++) {
      flat.push(image[j]);
    }
  }

  const batchedTensor =
      tf.tensor3d(flat, [numImages, IMAGE_WIDTH, IMAGE_HEIGHT], 'float32');

  return batchedTensor;
}

/**
 * Convert an image represented as a typed array to a JIMP object.
 *
 * @param {Float32Array} imageData
 *
 * @returns {Promise[Jimp]} Jimp object representing image.
 */
async function arrayToJimp(imageData) {
  const bufferLen = IMAGE_HEIGHT * IMAGE_WIDTH * 4;
  const buffer = new Uint8Array(bufferLen);

  let index = 0;
  for (let i = 0; i < IMAGE_HEIGHT; ++i) {
    for (let j = 0; j < IMAGE_WIDTH; ++j) {
      const inIndex = (i * IMAGE_WIDTH + j);
      const val = imageData[inIndex] * 255;
      buffer.set([Math.floor(val)], index++);
      buffer.set([Math.floor(val)], index++);
      buffer.set([Math.floor(val)], index++);
      buffer.set([255], index++);
    }
  }

  return new Promise((resolve, reject) => {
    new jimp(
        {data: buffer, width: IMAGE_WIDTH, height: IMAGE_HEIGHT},
        (err, img) => {
          if (err) {
            reject(err);
          } else {
            resolve(img);
          }
        });
  });
}

/**
 * Preview an image on the console.
 *
 * @param {Float32Array} imageData
 */
async function previewImage(imageData) {
  const imageAsJimp = await arrayToJimp(imageData);
  const pngBuffer = await imageAsJimp.getBufferAsync(jimp.MIME_PNG);
  console.log(await terminalImage.buffer(pngBuffer));
}

module.exports = {
  DATASET_PATH,
  TRAIN_IMAGES_FILE,
  IMAGE_FLAT_SIZE,
  loadImages,
  previewImage,
  batchImages,
};
