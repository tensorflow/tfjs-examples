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

const tf = require('@tensorflow/tfjs-node');
const shelljs = require('shelljs');
const tmp = require('tmp');
const fs = require('fs');
const createModel = require('./model');

let tempDir;

describe('Model', () => {
  beforeEach(() => {
    tempDir = tmp.dirSync();
  });

  afterEach(() => {
    if (fs.existsSync(tempDir)) {
      shelljs.rm('-rf', tempDir);
    }
  });

  it('Created model can train', async () => {
    const inputLength = 6;
    const outputLength = 1;
    const model = createModel([inputLength]);
    expect(model.inputs.length).toEqual(1);
    expect(model.inputs[0].shape).toEqual([null, inputLength]);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, outputLength]);

    const numExamples = 3;
    const inputFeature = tf.ones([numExamples, inputLength]);
    const inputLabel = tf.ones([numExamples, outputLength]);
    const history = await model.fit(inputFeature, inputLabel, {epochs: 2});
    expect(history.history.loss.length).toEqual(2);
  });

  it('Model save-load roundtrip', async () => {
    const inputLength = 6;
    const model = createModel([inputLength]);
    const numExamples = 3;
    const feature = tf.ones([numExamples, inputLength]);
    const y = model.predict(feature);

    await model.save(`file://${tempDir.name}`);
    const modelPrime =
        await tf.loadLayersModel(`file://${tempDir.name}/model.json`);
    const yPrime = modelPrime.predict([feature]);
    tf.test_util.expectArraysClose(yPrime, y);
  });
});
