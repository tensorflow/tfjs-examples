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

import * as tf from '@tensorflow/tfjs-node';

import {createDeepQNetwork, copyWeights} from "./dqn";

describe('createDeepQNetwork', () => {
  it('createDeepQNetwork', () => {
    const h = 9;
    const w = 9;
    const numActions = 4;
    const model = createDeepQNetwork(h, w, numActions);

    expect(model.inputs.length).toEqual(1);
    expect(model.inputs[0].shape).toEqual([null, h, w, 2]);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, numActions]);
  });

  it('Invalid h and/or w leads to Error', () => {
    expect(() => createDeepQNetwork(0, 10, 4)).toThrowError(/height/);
    expect(() => createDeepQNetwork('10', 10, 4)).toThrowError(/height/);
    expect(() => createDeepQNetwork(null, 10, 4)).toThrowError(/height/);
    expect(() => createDeepQNetwork(undefined, 10, 4)).toThrowError(/height/);
    expect(() => createDeepQNetwork(10.8, 10, 4)).toThrowError(/height/);
    expect(() => createDeepQNetwork(10, 0, 4)).toThrowError(/width/);
    expect(() => createDeepQNetwork(10, '10', 4)).toThrowError(/width/);
    expect(() => createDeepQNetwork(10, null, 4)).toThrowError(/width/);
    expect(() => createDeepQNetwork(10, undefined, 4)).toThrowError(/width/);
    expect(() => createDeepQNetwork(10, 10.8, 4)).toThrowError(/width/);
  });

  it('Invali numActions leads to Error', () => {
    expect(() => createDeepQNetwork(10, 10, 0)).toThrowError(/numActions/);
    expect(() => createDeepQNetwork(10, 10, 1)).toThrowError(/numActions/);
    expect(() => createDeepQNetwork(10, 10, '4')).toThrowError(/numActions/);
    expect(() => createDeepQNetwork(10, 10, null)).toThrowError(/numActions/);
    expect(() => createDeepQNetwork(10, 10, undefined)).toThrowError(/numActions/);
  });
});

describe('copyWeights', () => {
  it('copyWeights', async () => {
    const h = 9;
    const w = 9;
    const numActions = 4;
    const onlineNetwork = createDeepQNetwork(h, w, numActions);
    const targetNetwork = createDeepQNetwork(h, w, numActions);
    onlineNetwork.compile({
      loss: 'meanSquaredError',
      optimizer: tf.train.sgd(0.1)
    });

    // Initially, the two networks should have different values in their
    // weights.
    const conv1Weights0 = onlineNetwork.layers[0].getWeights();
    const conv1Weights1 = targetNetwork.layers[0].getWeights();
    expect(conv1Weights0.length).toEqual(conv1Weights1.length);
    // The 1st weight is the 1st conv layer's kernel.
    expect(conv1Weights0[0].sub(conv1Weights1[0]).abs().mean().arraySync())
        .toBeGreaterThan(0);

    const conv2Weights0 = onlineNetwork.layers[2].getWeights();
    const conv2Weights1 = targetNetwork.layers[2].getWeights();
    expect(conv2Weights0.length).toEqual(conv2Weights1.length);
    // The 1st weight is the 2nd conv layer's kernel.
    expect(conv2Weights0[0].sub(conv2Weights1[0]).abs().mean().arraySync())
        .toBeGreaterThan(0);

    copyWeights(targetNetwork, onlineNetwork);

    // After the copying, all the weights should be equal between the two
    // networks.
    const onlineWeights1 = onlineNetwork.getWeights();
    const targetWeights1 = targetNetwork.getWeights();
    expect(onlineWeights1.length).toEqual(targetWeights1.length);
    for (let i = 0; i < onlineWeights1.length; ++i) {
      expect(onlineWeights1[i].sub(targetWeights1[i]).abs().mean().arraySync())
          .toEqual(0);
    }

    // Modifying source network weight should not change target network weight.
    const xs =
        tf.randomUniform([4].concat(onlineNetwork.inputs[0].shape.slice(1)));
    const ys =
        tf.randomUniform([4].concat(onlineNetwork.outputs[0].shape.slice(1)));
    await onlineNetwork.fit(xs, ys, {epochs: 1});

    const onlineWeights2 = onlineNetwork.getWeights();
    const targetWeights2 = targetNetwork.getWeights();
    expect(onlineWeights2.length).toEqual(targetWeights2.length);
    for (let i = 0; i < onlineWeights1.length; ++i) {
      // Verify that the target network's weights haven't changed from before,
      // even though the online network's weights have.
      expect(onlineWeights2[0].sub(targetWeights2[0]).abs().mean().arraySync())
          .toBeGreaterThan(0);
      expect(targetWeights2[0].sub(targetWeights1[0]).abs().mean().arraySync())
          .toEqual(0);
    }
  });

  it('Copy from trainble source to untrainble dest works', () => {
    // Covers https://github.com/tensorflow/tfjs/issues/1807.
    const h = 9;
    const w = 9;
    const numActions = 4;
    const srcNetwork = createDeepQNetwork(h, w, numActions);
    const destNetwork = createDeepQNetwork(h, w, numActions);

    destNetwork.trainable = false;
    copyWeights(destNetwork, srcNetwork);
    expect(destNetwork.trainable).toEqual(false);
    expect(srcNetwork.trainable).toEqual(true);
  });
});
