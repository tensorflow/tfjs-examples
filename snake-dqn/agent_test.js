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

import {SnakeGameAgent} from "./agent";
import {SnakeGame} from "./snake_game";

describe('SnakeGameAgent', () => {
  it('playStep', () => {
    const game = new SnakeGame({
      height: 9,
      width: 9,
      numFruits: 1,
      initLen: 2
    });
    const agent = new SnakeGameAgent(game, {
      replayBufferSize: 100,
      epsilonInit: 1,
      epsilonFinal: 0.1,
      epsilonNumFrames: 10,
      batchSize: 32,
      learningRate: 1e-3
    });

    const numGames = 40;
    let bufferIndex = 0;
    for (let n = 0; n < numGames; ++n) {
      // At the beginnig of a game, the cumulative reward ought to be 0.
      expect(agent.cumulativeReward_).toEqual(0);
      let out = null;
      let outPrev = null;
      for (let m = 0; m < 10; ++m) {
        const currentState = agent.game.getState();
        out = agent.playStep();
        // Check the content of the replay buffer.
        expect(agent.replayMemory.buffer[bufferIndex % 100][0])
            .toEqual(currentState);
        expect(agent.replayMemory.buffer[bufferIndex % 100][1])
            .toEqual(out.action);

        expect(agent.replayMemory.buffer[bufferIndex % 100][2]).toEqual(
            outPrev == null ? out.cumulativeReward :
            out.cumulativeReward - outPrev.cumulativeReward);
        expect(agent.replayMemory.buffer[bufferIndex % 100][3]).toEqual(out.done);
        expect(agent.replayMemory.buffer[bufferIndex % 100][4])
            .toEqual(out.done ? undefined : agent.game.getState());
        bufferIndex++;
        if (out.done) {
          break;
        }
        outPrev = out;
      }
      agent.reset();
    }
  });

  it('trainOnReplayBatch', () => {
    const game = new SnakeGame({
      height: 9,
      width: 9,
      numFruits: 1,
      initLen: 2
    });
    const replayBufferSize = 1000;
    const agent = new SnakeGameAgent(game, {
      gamma: 0.95,
      replayBufferSize,
      epsilonInit: 1,
      epsilonFinal: 0.1,
      epsilonNumFrames: 1000,
      batchSize: 512,
      learningRate: 1e-2
    });

    const oldOnlineWeights =
        agent.onlineNetwork.getWeights().map(x => x.dataSync());
    const oldTargetWeights =
        agent.targetNetwork.getWeights().map(x => x.dataSync());

    for (let i = 0; i < replayBufferSize; ++i) {
      agent.playStep();
    }
    // Burn-in run for memory leak check below.
    agent.trainOnReplayBatch();

    const numTensors0 = tf.memory().numTensors;
    agent.trainOnReplayBatch();
    expect(tf.memory().numTensors).toEqual(numTensors0);

    const newOnlineWeights =
        agent.onlineNetwork.getWeights().map(x => x.dataSync());
    const newTargetWeights =
        agent.targetNetwork.getWeights().map(x => x.dataSync());

    // Verify that the online network's weights are updated.
    for (let i = 0; i < oldOnlineWeights.length; ++i) {
      expect(tf.tensor1d(newOnlineWeights[i])
          .sub(tf.tensor1d(oldOnlineWeights[i]))
          .abs().max().arraySync()).toBeGreaterThan(0);
    }
    // Verify that the target network's weights have not changed.
    for (let i = 0; i < oldOnlineWeights.length; ++i) {
      expect(tf.tensor1d(newTargetWeights[i])
          .sub(tf.tensor1d(oldTargetWeights[i]))
          .abs().max().arraySync()).toEqual(0);
    }
  });

  fit('train', () => {
    const game = new SnakeGame({
      height: 9,
      width: 9,
      numFruits: 1,
      initLen: 2
    });
    const replayBufferSize = 1e4;
    const agent = new SnakeGameAgent(game, {
      gamma: 0.99,
      replayBufferSize,
      epsilonInit: 1,
      epsilonFinal: 0.05,
      epsilonNumFrames: 1e5,
      batchSize: 64,
      learningRate: 1e-3
    });

    const cumulativeRewardThreshold = 10;
    const copyPerFrame = 1000;
    agent.train(cumulativeRewardThreshold, copyPerFrame);
  });
});
