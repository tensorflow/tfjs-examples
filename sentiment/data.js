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
import * as fs from 'fs';
import {padSequences} from './sequence_utils';

const PAD_CHAR = 0;
const OOV_CHAR = 2;
const INDEX_FROM = 3;

function loadFeatures(filePath, numWords, maxLen) {
  const buffer = fs.readFileSync(filePath);
  const numBytes = buffer.byteLength;

  let sequences = [];
  let seq = [];
  let index = 0;

  while (index < numBytes) {
    const value = buffer.readInt32LE(index);
    if (value === 1) {
      // A new sequence has started.
      if (index > 0) {
        sequences.push(seq);
      }
      seq = [];
    } else {
      // Sequence continues.
      seq.push(value >= numWords ? OOV_CHAR : value);
    }
    index += 4;
  }
  if (seq.length > 0) {
    sequences.push(seq);
  }
  const paddedSequences =
      padSequences(sequences, maxLen, 'pre', 'pre', PAD_CHAR);
  return tf.tensor2d(
      paddedSequences, [paddedSequences.length, maxLen], 'int32');
}

function loadTargets(filePath) {
  const buffer = fs.readFileSync(filePath);
  const numBytes = buffer.byteLength;

  let ys = [];
  for (let i = 0; i < numBytes; ++i) {
    ys.push(buffer.readUInt8(i));
  }
  return tf.tensor2d(ys, [ys.length, 1], 'float32');
}

export function loadData(pathPrefix, numWords, len) {
  const trainFeaturePath = `${pathPrefix}_train_data.bin`;
  const xTrain = loadFeatures(trainFeaturePath, numWords, len);
  const testFeaturePath = `${pathPrefix}_test_data.bin`;
  const xTest = loadFeatures(testFeaturePath, numWords, len);
  const trainTargetsPath = `${pathPrefix}_train_targets.bin`;
  const yTrain = loadTargets(trainTargetsPath);
  const testTargetsPath = `${pathPrefix}_test_targets.bin`;
  const yTest = loadTargets(testTargetsPath);

  tf.util.assert(
      xTrain.shape[0] === yTrain.shape[0],
      `Mismatch in number of examples between xTrain and yTrain`);
  tf.util.assert(
      xTest.shape[0] === yTest.shape[0],
      `Mismatch in number of examples between xTest and yTest`);
  return {xTrain, yTrain, xTest, yTest};
}
