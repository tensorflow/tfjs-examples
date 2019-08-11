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

import * as fs from 'fs';
import * as path from 'path';

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

export const IMAGE_EXTENSION_NAMES = ['jpg', 'jpeg', 'png'];

/**
 * Recursively find all image files with matching extension names.
 *
 * @param {string} dirPath Path to a directory to perform the
 *   recursive search in.
 * @return {string[]} An array of full paths to all the image files
 *   under the directory.
 */
export function findImagesFromDirectoriesRecursive(dirPath) {
  const imageFilePaths = [];
  const items = fs.readdirSync(dirPath);
  for (const item of items) {
    const fullPath = path.join(dirPath, item);
    if (fs.lstatSync(fullPath).isDirectory()) {
      // NOTE: This doesn't follow symlinks.
      try {
        imageFilePaths.push(...findImagesFromDirectoriesRecursive(fullPath));
      } catch (err) {}
    } else {
      let extMatch = false;
      for (const extName of IMAGE_EXTENSION_NAMES) {
        if (item.toLowerCase().endsWith(extName)) {
          extMatch = true;
          break;
        }
      }
      if (extMatch) {
        imageFilePaths.push(fullPath);
      }
    }
  }
  return imageFilePaths;
}
