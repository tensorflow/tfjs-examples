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

import * as tf from '@tensorflow/tfjs-node';
import * as shelljs from 'shelljs';
import * as tmp from 'tmp';

import {buildModel} from "./train";
import {writeEmbeddingMatrixAndLabels} from "./embedding";

describe('writeEmbeddingMatrixAndLabels', () => {
  let tmpDir;

  beforeEach(() => {
    tmpDir = tmp.dirSync().name;
  });

  afterEach(() => {
    if (fs.existsSync(tmpDir)) {
      shelljs.rm('-rf', tmpDir);
    }
  });

  it('writeEmbeddingMatrixAndLabels', async () => {
    const maxLen = 5;
    const vocabSize = 4;
    const embeddingSize = 8;
    const model = buildModel('cnn', maxLen, vocabSize, embeddingSize);
    expect(model.inputs.length).toEqual(1);
    expect(model.inputs[0].shape).toEqual([null, maxLen]);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, 1]);

    model.compile({
      loss: 'binaryCrossentropy',
      optimizer: 'rmsprop',
      metrics: ['acc']
    });
    const xs = tf.ones([2, maxLen])
    const ys = tf.ones([2, 1]);
    const history = await model.fit(xs, ys, {epochs: 2, batchSize: 2});
    expect(history.history.loss.length).toEqual(2);
    expect(history.history.acc.length).toEqual(2);

    const predictOuts = model.predict(xs);
    expect(predictOuts.shape).toEqual([2, 1]);
    const values = predictOuts.arraySync();
    expect(values[0][0]).toBeGreaterThanOrEqual(0);
    expect(values[0][0]).toBeLessThanOrEqual(1);
    expect(values[1][0]).toBeGreaterThanOrEqual(0);
    expect(values[1][0]).toBeLessThanOrEqual(1);

    const wordIndex = {
      'foo': 1,
      'bar': 2,
      'baz': 3,
      'qux': 4
    };
    await writeEmbeddingMatrixAndLabels(model, `${tmpDir}/embed`, wordIndex, 0);
    const vectorFileContent =
        fs.readFileSync(`${tmpDir}/embed_vectors.tsv`, {encoding: 'utf-8'})
        .trim().split('\n');
    expect(vectorFileContent.length).toEqual(4);
    const labelsFileContent =
        fs.readFileSync(`${tmpDir}/embed_labels.tsv`, {encoding: 'utf-8'})
        .trim().split('\n');
    expect(labelsFileContent.length).toEqual(4);
  });
});
