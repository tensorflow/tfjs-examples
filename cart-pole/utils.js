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
 * Calculate the mean of an Array of numbers.
 *
 * @param {number[]} xs
 * @returns {number} The arithmetic mean of `xs`
 */
export function mean(xs) {
  return sum(xs) / xs.length;
}

/**
 * Calculate the sum of an Array of numbers.
 *
 * @param {number[]} xs
 * @returns {number} The sum of `xs`.
 * @throws Error if `xs` is empty.
 */
export function sum(xs) {
  if (xs.length === 0) {
    throw new Error('Expected xs to be a non-empty Array.');
  } else {
    return xs.reduce((x, prev) => prev + x);
  }
}
