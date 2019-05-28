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

import * as tf from '@tensorflow/tfjs';

// Convert an image vector (length 784) representing an MNIST image into
// a human-friendly text representation.
//
// Args:
//   imageVector: An Array of Numbers of length `imageSize * imageSize`.
//
// Returns:
//   A String representing the image.
export function imageVectorToText(imageVector, imageSize) {
  if (imageVector.length !== imageSize * imageSize) {
    throw new Error(
        'Incorrect length of image vector (expected ' + imageSize * imageSize +
        '; got ' + imageVector.length + ')');
  }
  let text = '';
  for (let i = 0; i < imageSize * imageSize; ++i) {
    if (i % imageSize === 0 && i > 0) {
      text += '\n';
    }
    const numString = imageVector[i].toString();
    text +=
        ' '.repeat(numString.length < 4 ? 4 - numString.length : 0) + numString;
  }
  return text;
}

// Convert a text representation of an MNIST image into an deeplearn Tensor4D
// of shape [1, imageSize, imageSize, 1].
//
// Args:
//   text: A String representing the MNIST image.
//
// Returns:
//   A Tensor4D instance representing the image, in a size-1 batch.
//     Shape: [1, imageSize, imageSize, 1].
export function textToImageArray(text, imageSize) {
  // Split into rows.
  const pixels = [];
  const rows = text.split('\n');
  for (const row of rows) {
    const tokens = row.split(' ');
    for (const token of tokens) {
      if (token.length > 0) {
        pixels.push(Number.parseInt(token) / 255);
      }
    }
  }
  if (pixels.length !== imageSize * imageSize) {
    throw new Error(
        'Incorrect length of image vector (expected ' + imageSize * imageSize +
        '; got ' + pixels.length + ')');
  }
  return tf.tensor4d(pixels, [1, imageSize, imageSize, 1]);
}

export function indexToOneHot(index, numClasses) {
  const oneHot = [];
  for (let i = 0; i < numClasses; ++i) {
    oneHot.push(i === index ? 1 : 0);
  }
  return oneHot;
}

export function convertDataToTensors(data, numClasses) {
  const numExamples = data.length;
  const imgRows = data[0].x.length;
  const imgCols = data[0].x[0].length;
  const xs = [];
  const ys = [];
  data.map(example => {
    xs.push(example.x);
    ys.push(this.indexToOneHot(example.y, numClasses));
  });
  let xsTensor = tf.reshape(
      tf.tensor3d(xs, [numExamples, imgRows, imgCols]),
      [numExamples, imgRows, imgCols, 1]);
  xsTensor = tf.mul(tf.scalar(1 / 255), xsTensor);
  const ysTensor = tf.tensor2d(ys, [numExamples, numClasses]);
  return {x: xsTensor, y: ysTensor};
}
