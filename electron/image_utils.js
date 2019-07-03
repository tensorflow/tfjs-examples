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

import * as jimp from 'jimp';
import * as tf from '@tensorflow/tfjs';

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
export async function readImageAsTensor(filePath, height, width) {
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
        resolve(tf.tidy(() => tf.image.resizeBilinear(
            buffer.toTensor(), [height, width]).div(255)));
      }
    });
  });
}

export async function readImageAsBase64(filePath) {
  let mimeType;
  if (filePath.toLowerCase().endsWith('.png')) {
    mimeType = jimp.MIME_PNG;
  } else if (filePath.toLowerCase().endsWith('.bmp')) {
    mimeType = jimp.MIME_BMP;
  } else if (filePath.toLowerCase().endsWith('jpg') ||
             filePath.toLowerCase().endsWith('jpeg')) {
    mimeType = jimp.MIME_JPEG;
  } else {
    throw new Error(`Unsupported image file extension name in: ${filePath}`);
  }
  return new Promise((resolve, reject) => {
    jimp.read(filePath, async (err, image) => {
      if (err) {
        reject(err);
      } else {
        resolve(await image.getBase64Async(mimeType));
      }
    });
  });
}