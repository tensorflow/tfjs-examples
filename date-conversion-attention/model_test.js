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
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';
import * as tmp from 'tmp';

import * as dateFormat from './date_format';
import {createModel, runSeq2SeqInference} from './model';

require('@tensorflow/tfjs-node');

describe('Model', () => {
  it('Created model can train', async () => {
    const inputVocabSize = 16;
    const outputVocabSize = 8;
    const inputLength = 6;
    const outputLength = 5;
    const model =
        createModel(inputVocabSize, outputVocabSize, inputLength, outputLength);
    expect(model.inputs.length).toEqual(2);
    expect(model.inputs[0].shape).toEqual([null, inputLength]);
    expect(model.inputs[1].shape).toEqual([null, outputLength]);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([
      null, outputLength, outputVocabSize
    ]);

    const numExamples = 3;
    const encoderInputs = tf.ones([numExamples, inputLength]);
    const decoderInputs = tf.ones([numExamples, outputLength]);
    const decoderOutputs =
        tf.randomUniform([numExamples, outputLength, outputVocabSize]);
    const history = await model.fit(
        [encoderInputs, decoderInputs], decoderOutputs, {epochs: 2});
    expect(history.history.loss.length).toEqual(2);
  });

  it('Model save-load roundtrip', async () => {
    const inputVocabSize = 16;
    const outputVocabSize = 8;
    const inputLength = 6;
    const outputLength = 5;
    const model =
        createModel(inputVocabSize, outputVocabSize, inputLength, outputLength);
    const numExamples = 3;
    const encoderInputs = tf.ones([numExamples, inputLength]);
    const decoderInputs = tf.ones([numExamples, outputLength]);
    const y = model.predict([encoderInputs, decoderInputs]);

    const saveDir = tmp.dirSync();
    await model.save(`file://${saveDir.name}`);
    const modelPrime =
        await tf.loadLayersModel(`file://${saveDir.name}/model.json`);
    const yPrime = modelPrime.predict([encoderInputs, decoderInputs]);
    expectArraysClose(yPrime, y);
  });

  it('seq2seq inference', async () => {
    const model = createModel(
        dateFormat.INPUT_VOCAB.length, dateFormat.OUTPUT_VOCAB.length,
        dateFormat.INPUT_LENGTH, dateFormat.OUTPUT_LENGTH);

    const numTensors0 = tf.memory().numTensors;
    const {outputStr} = await runSeq2SeqInference(model, '2019/01/18');
    // Assert no memory leak.
    expect(tf.memory().numTensors).toEqual(numTensors0);
    expect(outputStr.length).toEqual(dateFormat.OUTPUT_LENGTH);
  });
});
