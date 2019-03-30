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

const {encoder, decoder, vae, vaeLoss} = require('./model');

describe('Encoder', () => {
  it('Constructor and predict() call', () => {
    const opts = {
      originalDim: 100,
      intermediateDim: 10,
      latentDim: 2
    };
    const enc = encoder(opts);
    expect(enc.inputs.length).toEqual(1);
    expect(enc.inputs[0].shape).toEqual([null, 100]);
    expect(enc.layers[1].outputShape).toEqual([null, 10]);
    expect(enc.outputs.length).toEqual(3);
    expect(enc.outputs[0].shape).toEqual([null, 2]);
    expect(enc.outputs[1].shape).toEqual([null, 2]);
    expect(enc.outputs[2].shape).toEqual([null, 2]);

    // Run a tensor input through the predict() method.
    const numExamples = 4;
    xs = tf.randomUniform([numExamples, 100]);
    const outs = enc.predict(xs);
    expect(outs.length).toEqual(3);  // zMean, zLogVar and z.
    expect(outs[0].shape).toEqual([numExamples, 2]);
    expect(outs[1].shape).toEqual([numExamples, 2]);
    expect(outs[2].shape).toEqual([numExamples, 2]);
  });
});

describe('Decoder', () => {
  it('Constructor and predict() call', () => {
    const opts = {
      originalDim: 100,
      intermediateDim: 10,
      latentDim: 2
    };
    const dec = decoder(opts);
    expect(dec.inputs.length).toEqual(1);
    expect(dec.inputs[0].shape).toEqual([null, 2]);
    expect(dec.layers.length).toEqual(2);
    expect(dec.layers[0].outputShape).toEqual([null, 10]);
    expect(dec.outputs.length).toEqual(1);
    expect(dec.outputs[0].shape).toEqual([null, 100]);

    // Run a tensor input through the predict() method.
    const numExamples = 4;
    xs = tf.randomUniform([numExamples, 2]);
    const outs = dec.predict(xs);
    expect(outs.shape).toEqual([numExamples, 100]);
  });
});
