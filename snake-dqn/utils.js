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

import * as tf from '@tensorflow/tfjs';

/**
 * Generate a random integer >= min and < max.
 *
 * @param {number} min Lower bound, inclusive.
 * @param {number} max Upper bound, exclusive.
 * @return {number} The random integers.
 */
export function getRandomInteger(min, max) {
  // Note that we don't reuse the implementation in the more generic
  // `getRandomIntegers()` (plural) below, for performance optimization.
  return Math.floor((max - min) * Math.random()) + min;
}

/**
 * Generate a given number of random integers >= min and < max.
 *
 * @param {number} min Lower bound, inclusive.
 * @param {number} max Upper bound, exclusive.
 * @param {number} numIntegers Number of random integers to get.
 * @return {number[]} The random integers.
 */
export function getRandomIntegers(min, max, numIntegers) {
  const output = [];
  for (let i = 0; i < numIntegers; ++i) {
    output.push(Math.floor((max - min) * Math.random()) + min);
  }
  return output;
}


export function assertPositiveInteger(x, name) {
  if (!Number.isInteger(x)) {
    throw new Error(
        `Expected ${name} to be an integer, but received ${x}`);
  }
  if (!(x > 0)) {
    throw new Error(
        `Expected ${name} to be a positive number, but received ${x}`);
  }
}
