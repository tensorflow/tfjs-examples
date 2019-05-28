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

import * as dateFormat from './date_format';
import {generateDataForTraining} from './train';

describe('generateBatchesForTraining', () => {
  it('generateDataForTraining', () => {
    const {
      trainEncoderInput,
      trainDecoderInput,
      trainDecoderOutput,
      valEncoderInput,
      valDecoderInput,
      valDecoderOutput,
      testDateTuples
    } = generateDataForTraining(0.5, 0.25);
    const numTrain = trainEncoderInput.shape[0];
    const numVal = valEncoderInput.shape[0];
    expect(numTrain / numVal).toBeCloseTo(2);
    expect(trainEncoderInput.shape).toEqual(
        [numTrain, dateFormat.INPUT_LENGTH]);
    expect(trainDecoderInput.shape).toEqual(
        [numTrain, dateFormat.OUTPUT_LENGTH]);
    expect(trainDecoderOutput.shape).toEqual(
        [numTrain, dateFormat.OUTPUT_LENGTH, dateFormat.OUTPUT_VOCAB.length]);
    expect(valEncoderInput.shape).toEqual(
        [numVal, dateFormat.INPUT_LENGTH]);
    expect(valDecoderInput.shape).toEqual(
        [numVal, dateFormat.OUTPUT_LENGTH]);
    expect(valDecoderOutput.shape).toEqual(
        [numVal, dateFormat.OUTPUT_LENGTH, dateFormat.OUTPUT_VOCAB.length]);
    expect(testDateTuples[0].length).toEqual(3);
    expect(testDateTuples[testDateTuples.length - 1].length).toEqual(3);
  });
});
