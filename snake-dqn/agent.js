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

import {copyWeights, createDeepQNetwork} from './dqn';
import {getRandomAction, SnakeGame, NUM_ACTIONS, ALL_ACTIONS, getStateTensor} from './snake_game';
import {ReplayMemory} from './replay_memory';

export class SnakeGameAgent {
  /**
   * Constructor of SnakeGameAgent.
   *
   * @param {SnakeGame} game A game object.
   * @param {object} config The configuration object with the following keys:
   *   - `gamma` {number} reward discount rate. Should be a number >= 0 and <= 1.
   *   - `replayBufferSize` {number} Size of the replay memory. Must be a
   *     positive integer.
   *   - `epsilonInit` {number} Initial value of epsilon (for the epsilon-
   *     greedy algorithm). Must be >= 0 and <= 1.
   *   - `epsilonFinal` {number} The final value of epsilon. Must be >= 0 and
   *     <= 1.
   *   - `epsilonNumFrames` {number} The # of frames over which the value of
   *     `epsilon` decreases from `episloInit` to `epsilonFinal`, via a linear
   *     schedule.
   *   - `batchSize` {number} Batch size for training.
   *   - `learningRate` {number} Learning rate for training.
   */
  constructor(game, config) {
    this.game = game;

    this.gamma = config.gamma;
    this.epsilonInit = config.epsilonInit;
    this.epsilonFinal = config.epsilonFinal;
    this.epislonNumFrames = config.epsilonNumFrames;
    this.epsilonIncrement_ = (this.epsilonFinal - this.epsilonInit) /
        this.epislonNumFrames;
    this.batchSize = config.batchSize;

    // TODO(cais): Check to make sure that `batchSize` is <= `replayBufferSize`.

    this.onlineNetwork =
        createDeepQNetwork(game.height,  game.width, NUM_ACTIONS);
    this.targetNetwork =
        createDeepQNetwork(game.height,  game.width, NUM_ACTIONS);
    // Freeze taget network: it's weights are updated only through copying from
    // the online network.
    this.targetNetwork.trainable = false;

    this.optimizer = tf.train.adam(config.learningRate);

    this.replayBufferSize = config.replayBufferSize;
    this.replayMemory = new ReplayMemory(config.replayBufferSize);
    this.frameCount = 0;
    this.reset();
  }

  reset() {
    this.cumulativeReward_ = 0;
    this.game.reset();
  }

  /**
   * Play one step of the game.
   *
   * @returns {number | null} If this step leads to the end of the game,
   *   the total reward from the game as a plain number. Else, `null`.
   */
  playStep() {
    const epsilon = this.frameCount >= this.epislonNumFrames ?
        this.epsilonFinal :
        this.epsilonInit + this.epsilonIncrement_  * this.frameCount;
    this.frameCount++;

    // The epsilon-greedy algorithm.
    let action;
    const state = this.game.getState();
    if (Math.random() < epsilon) {
      // Pick an action at random.
      action = getRandomAction();
    } else {
      // Greedily pick an action based on online DQN output.
      tf.tidy(() => {
        const stateTensor =
            getStateTensor(state, this.game.height, this.game.width)
        action = ALL_ACTIONS[
            this.onlineNetwork.predict(stateTensor).argMax(-1).dataSync()[0]];
      });
    }

    const {state: nextState, reward, done} = this.game.step(action);

    this.replayMemory.append([state, action, reward, done, nextState]);

    this.cumulativeReward_ += reward;
    const output = {
      action,
      cumulativeReward: this.cumulativeReward_,
      done
    };
    if (done) {
      this.reset();
    }
    return output;
  }

  /**
   * TODO(cais): Doc string.
   */
  trainOnReplayBatch() {
    // Get a batch of examples from the replay buffer.
    const batch = this.replayMemory.sample(this.batchSize);
    const lossFunction = () => tf.tidy(() => {
      const stateTensor = getStateTensor(
          batch.map(example => example[0]), this.game.height, this.game.width);
      const actionTensor = tf.tensor1d(
          batch.map(example => example[1]), 'int32');
      const qs = this.onlineNetwork.predict(
          stateTensor).mul(tf.oneHot(actionTensor, NUM_ACTIONS)).sum(-1);

      const rewardTensor = tf.tensor1d(batch.map(example => example[2]));
      const nextStateTensor = getStateTensor(
          batch.map(example => example[4]), this.game.height, this.game.width);
      const nextMaxQTensor =
          this.targetNetwork.predict(nextStateTensor).max(-1);
      const doneMask = tf.scalar(1).sub(
          tf.tensor1d(batch.map(example => example[3])).asType('float32'));
      const targetQs =
          rewardTensor.add(nextMaxQTensor.mul(doneMask).mul(this.gamma));
      return tf.losses.meanSquaredError(targetQs, qs);
    });

    // TODO(cais): Remove the second argument when `variableGrads()` obeys the
    // trainable flag.
    const grads =
        tf.variableGrads(lossFunction, this.onlineNetwork.getWeights());
    this.optimizer.applyGradients(grads.grads);
    tf.dispose(grads);
  }

  /**
   *
   * @param {*} cumulativeRewardThreshold
   */
  train(cumulativeRewardThreshold, copyPerFrame) {
    for (let i = 0; i < this.replayBufferSize; ++i) {
      this.playStep();
    }

    while (true) {
      // console.log('Calling trainOnReplayBatch()');  // DEBUG
      this.trainOnReplayBatch();
      const {cumulativeReward, done} = this.playStep();
      if (done) {
        console.log(`Frame #${this.frameCount}: ` +
            `cumulativeReward = ${cumulativeReward}`);
        if (cumulativeReward > cumulativeRewardThreshold) {
          // TODO(cais): Save online network.
          break;
        }
      }
      if (this.frameCount % copyPerFrame === 0) {
        console.log('Copying weights from online network to target network');
        copyWeights(this.targetNetwork, this.onlineNetwork);
      }
    }
  }
}
