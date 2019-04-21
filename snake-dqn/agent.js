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

import {createDeepQNetwork} from './dqn';
import {getRandomAction, SnakeGame, NUM_ACTIONS, ALL_ACTIONS, getStateTensor} from './snake_game';
import {ReplayMemory} from './replay_memory';

export class SnakeGameAgent {
  /**
   * Constructor of SnakeGameAgent.
   *
   * @param {SnakeGame} game A game object.
   * @param {object} config The configuration object with the following keys:
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
    this.game_ = game;

    this.epsilonInit_ = config.epsilonInit;
    this.epsilonFinal_ = config.epsilonFinal;
    this.epislonNumFrames_ = config.epsilonNumFrames;
    this.epsilonIncrement_ = (this.epsilonFinal_ - this.epsilonInit_) /
        this.epislonNumFrames_;

    // TODO(cais): Check to make sure that `batchSize` is <= `replayBufferSize`.

    this.onlineNetwork_ =
        createDeepQNetwork(game.height,  game.width, NUM_ACTIONS);
    this.targetNetwork_ =
        createDeepQNetwork(game.height,  game.width, NUM_ACTIONS);

    this.replayMemory_ = new ReplayMemory(config.replayBufferSize);
    this.frameCount_ = 0;
    this.reset();
  }

  reset() {
    this.cumulativeReward_ = 0;
    this.game_.reset();
  }

  /**
   * Play one step of the game.
   *
   * @returns {number | null} If this step leads to the end of the game,
   *   the total reward from the game as a plain number. Else, `null`.
   */
  playStep() {
    const epsilon = this.frameCount_ >= this.epislonNumFrames_ ?
        this.epsilonFinal_ :
        this.epsilonInit_ + this.epsilonIncrement_  * this.frameCount_;
    this.frameCount_++;

    // The epsilon-greedy algorithm.
    let action;
    const state = this.game_.getState();
    if (Math.random() < epsilon) {
      // Pick an action at random.
      action = getRandomAction();
    } else {
      // Greedily pick an action based on online DQN output.
      tf.tidy(() => {
        const stateTensor =
            getStateTensor(state, this.game_.height, this.game_.width)
            .expandDims(0);
        action = ALL_ACTIONS[
            this.onlineNetwork_.predict(stateTensor).argMax(-1).dataSync()[0]];
      });
    }

    const {state: newState, reward, done} = this.game_.step(action);

    this.replayMemory_.append([state, action, reward, done, newState]);

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
  async trainOnReplayBatch() {
    throw new Error('Not implemented yet.');
  }
}

