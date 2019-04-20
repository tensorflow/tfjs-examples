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

import {createDeepQNetwork} from './dqn';
import {getRandomAction, SnakeGame, NUM_ACTIONS} from './snake_game';
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
   */
  constructor(game, config) {
    this.game_ = game;

    this.epsilonInit_ = config.epsilonInit;
    this.epsilonFinal_ = config.epsilonFinal;
    this.epislonNumFrames_ = config.epsilonNumFrames;
    this.epsilonIncrement_ = (this.epsilonFinal_ - this.epsilonInit_) /
        this.epislonNumFrames_;

    this.onlineNetwork_ =
        createDeepQNetwork(game.height,  game.width, NUM_ACTIONS);
    this.targetNetwork_ =
        createDeepQNetwork(game.height,  game.width, NUM_ACTIONS);

    this.replayMemory_ = new ReplayMemory(config.replayBufferSize);
    this.frameCount_ = 0;
    this.state_ = game.reset();
  }

  /**
   * Play one step of the game.
   *
   * @returns {number | null} If this step leads to the end of the game,
   *   the total reward from the game as a plain number. Else, `null`.
   */
  playStep() {
    const epsilon = this.frameCount_ >= this.epislonNumFrames ?
        this.epsilonFinal_ :
        this.epsilonInit_ + this.epsilonIncrement_  * this.frameCount_;

    // The epsilon-greedy algorithm.
    if (Math.random() < epsilon) {
      // Pick an action at random.
      const action = getRandomAction();
    } else {
      // Greedily pick an action.
    }
  }
}

