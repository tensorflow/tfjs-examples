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

/**
 * Calculates the mean and standard deviation of each column of a data array.
 *
 * @param {Tensor2d} data Dataset from which to calculate the mean and
 *                        std of each column independently.
 *
 * @returns {Object} Contains the mean and standarddeviation of each vector
 *                   column.
 */
export const determineMeanAndStddev =
    (data) => {
      const means = tf.mean(data, 0);
      const zeroMean = tf.sub(data, means);
      const stddevs = tf.sqrt(tf.mean(tf.mul(zeroMean, zeroMean), 0));
      return {means, stddevs};
    }

/**
 * Given expected mean and standard deviation, normalizes a dataset by
 * subtracting the mean and dividing by the standard deviation.
 *
 * @param {Tensor2d} data
 * @param {Tensor1d} means
 * @param {Tensor1d} stddevs
 */
export const normalizeTensor = (data, means, stddevs) => {
  return tf.div(tf.sub(data, means), stddevs);
}
