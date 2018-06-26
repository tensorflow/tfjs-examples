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
 * Draw one sample from a multinomial distribution.
 *
 * @param {number[]} probs Probabilities. Assumed to sum to 1.
 * @returns {number} A zero-based sample index.
 */
export function sampleOneFromMultinomial(probs) {
  const score = Math.random();
  let cumProb = 0;
  const n = probs.length;
  for (let i = 0; i < n; ++i) {
    if (score >= cumProb && score < cumProb + probs[i]) {
      return i;
    }
    cumProb += probs[i];
  }
  return n - 1;
}
