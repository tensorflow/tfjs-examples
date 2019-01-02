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

/**
 * Utility functions for the visualize-convnet demo.
 */

const jimp = require('jimp');
const tf = require('@tensorflow/tfjs');

/**
 * Read an image file as a TensorFlow.js tensor.
 *
 * Image resizing is performed with tf.image.resizeBilinear.
 *
 * @param {string} filePath Path to the input image file.
 * @param {number} height Desired height of the output image tensor, in pixels.
 * @param {number} width Desired width of the output image tensor, in pixels.
 * @return {tf.Tensor4D} The read float32-type tf.Tensor of shape
 *   `[1, height, width, 3]`
 */
async function readImageTensorFromFile(filePath, height, width) {
  return new Promise((resolve, reject) => {
    jimp.read(filePath, (err, image) => {
      if (err) {
        reject(err);
      } else {
        const h = image.bitmap.height;
        const w = image.bitmap.width;
        const buffer = tf.buffer([1, h, w, 3], 'float32');
        image.scan(0, 0, w, h, function(x, y, index) {
          buffer.set(image.bitmap.data[index], 0, y, x, 0);
          buffer.set(image.bitmap.data[index + 1], 0, y, x, 1);
          buffer.set(image.bitmap.data[index + 2], 0, y, x, 2);
        });
        resolve(tf.tidy(
            () => tf.image.resizeBilinear(buffer.toTensor(), [height, width])));
      }
    });
  });
}

/**
 * Write an image tensor to a image file.
 *
 * @param {tf.Tensor} imageTensor The image tensor to write to file.
 *   Assumed to be an int32-type tensor with value in the range 0-255.
 * @param {string} filePath Destination file path.
 */
async function writeImageTensorToFile(imageTensor, filePath) {
  const imageH = imageTensor.shape[1];
  const imageW = imageTensor.shape[2];
  const imageData = imageTensor.dataSync();

  const bufferLen = imageH * imageW * 4;
  const buffer = new Uint8Array(bufferLen);
  let index = 0;
  for (let i = 0; i < imageH; ++i) {
    for (let j = 0; j < imageW; ++j) {
      const inIndex = 3 * (i * imageW + j);
      buffer.set([imageData[inIndex]], index++);
      buffer.set([imageData[inIndex + 1]], index++);
      buffer.set([imageData[inIndex + 2]], index++);
      buffer.set([255], index++);
    }
  }

  return new Promise((resolve, reject) => {
    new jimp(
        {data: new Buffer(buffer), width: imageW, height: imageH},
        (err, img) => {
          if (err) {
            reject(err);
          } else {
            img.write(filePath);
            resolve();
          }
        });
  });
}

module.exports = {
  readImageTensorFromFile,
  writeImageTensorToFile
};
