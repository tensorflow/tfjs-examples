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

import {assertPositiveInteger, getRandomInteger} from './utils';

const DEFAULT_HEIGHT = 16;
const DEFAULT_WIDTH = 16;
const DEFAULT_NUM_FRUITS = 1;
const DEFAULT_INIT_LEN = 4;

// TODO(cais): Tune these parameters.
export const NO_FRUIT_REWARD = -0.2;
export const FRUIT_REWARD = 10;
export const DEATH_REWARD = -10;
// TODO(cais): Explore adding a "bad fruit" with a negative reward.

export const ACTION_GO_STRAIGHT = 0;
export const ACTION_TURN_LEFT = 1;
export const ACTION_TURN_RIGHT = 2;

export const ALL_ACTIONS = [ACTION_GO_STRAIGHT, ACTION_TURN_LEFT, ACTION_TURN_RIGHT];
export const NUM_ACTIONS = ALL_ACTIONS.length;

/**
 * Generate a random action among all possible actions.
 *
 * @return {0 | 1 | 2} Action represented as a number.
 */
export function getRandomAction() {
  return getRandomInteger(0, NUM_ACTIONS);
}

export class SnakeGame {
  /**
   * Constructor of SnakeGame.
   *
   * @param {object} args Configurations for the game. Fields include:
   *   - height {number} height of the board (positive integer).
   *   - width {number} width of the board (positive integer).
   *   - numFruits {number} number of fruits present on the screen
   *     at any given step.
   *   - initLen {number} initial length of the snake.
   */
  constructor(args) {
    if (args == null) {
      args = {};
    }
    if (args.height == null) {
      args.height = DEFAULT_HEIGHT;
    }
    if (args.width == null) {
      args.width = DEFAULT_WIDTH;
    }
    if (args.numFruits == null) {
      args.numFruits = DEFAULT_NUM_FRUITS;
    }
    if (args.initLen == null) {
      args.initLen = DEFAULT_INIT_LEN;
    }

    assertPositiveInteger(args.height, 'height');
    assertPositiveInteger(args.width, 'width');
    assertPositiveInteger(args.numFruits, 'numFruits');
    assertPositiveInteger(args.initLen, 'initLen');

    this.height_ = args.height;
    this.width_ = args.width;
    this.numFruits_ = args.numFruits;
    this.initLen_ = args.initLen;

    this.reset();
  }

  /**
   * Reset the state of the game.
   *
   * @return {object} Initial state of the game.
   *   See the documentation of `getState()` for details.
   */
  reset() {
    this.initializeSnake_();
    this.fruitSquares_ = null;
    this.makeFruits_();
    return this.getState();
  }

  /**
   * Perform a step of the game.
   *
   * @param {0 | 1 | 2 | 3} action The action to take in the current step.
   *   The meaning of the possible values:
   *     0 - left
   *     1 - top
   *     2 - right
   *     3 - bottom
   * @return {object} Object with the following keys:
   *   - `reward` {number} the reward value.
   *     - 0 if no fruit is eaten in this step
   *     - 1 if a fruit is eaten in this step
   *   - `state` New state of the game after the step.
   *   - `fruitEaten` {boolean} Whether a fruit is easten in this step.
   *   - `done` {boolean} whether the game has ended after this step.
   *     A game ends when the head of the snake goes off the board or goes
   *     over its own body.
   */
  step(action) {
    const [headY, headX] = this.snakeSquares_[0];

    // Calculate the coordinates of the new head and check whether it has
    // gone off the board, in which case the game will end.
    let done;
    let newHeadY;
    let newHeadX;

    this.updateDirection_(action);
    if (this.snakeDirection_ === 'l') {
      newHeadY = headY;
      newHeadX = headX - 1;
      done = newHeadX < 0;
    } else if (this.snakeDirection_ === 'u') {
      newHeadY = headY - 1;
      newHeadX = headX;
      done = newHeadY < 0
    } else if (this.snakeDirection_ === 'r') {
      newHeadY = headY;
      newHeadX = headX + 1;
      done = newHeadX >= this.width_;
    } else if (this.snakeDirection_ === 'd') {
      newHeadY = headY + 1;
      newHeadX = headX;
      done = newHeadY >= this.height_;
    }

    // Check if the head goes over the snake's body, in which case the
    // game will end.
    for (let i = 1; i < this.snakeSquares_.length; ++i) {
      if (this.snakeSquares_[i][0] === newHeadY &&
          this.snakeSquares_[i][1] === newHeadX) {
        done = true;
      }
    }

    let fruitEaten = false;
    if (done) {
      return {reward: DEATH_REWARD, done, fruitEaten};
    }

    // Update the position of the snake.
    this.snakeSquares_.unshift([newHeadY, newHeadX]);

    // Check if a fruit is eaten.
    let reward = NO_FRUIT_REWARD;
    for (let i = 0; i < this.fruitSquares_.length; ++i) {
      const fruitYX = this.fruitSquares_[i];
      if (fruitYX[0] === newHeadY && fruitYX[1] === newHeadX) {
        reward = FRUIT_REWARD;
        fruitEaten = true;
        this.fruitSquares_.splice(i, 1);
        this.makeFruits_();
        break;
      }
    }
    if (!fruitEaten) {
      // Pop the tail off if and only if the snake didn't eat a fruit in this
      // step.
      this.snakeSquares_.pop();
    }

    const state = this.getState();
    return {reward, state, done, fruitEaten};
  }

  updateDirection_(action) {
    if (this.snakeDirection_ === 'l') {
      if (action === ACTION_TURN_LEFT) {
        this.snakeDirection_ = 'd';
      } else if (action === ACTION_TURN_RIGHT) {
        this.snakeDirection_ = 'u';
      }
    } else if (this.snakeDirection_ === 'u') {
      if (action === ACTION_TURN_LEFT) {
        this.snakeDirection_ = 'l';
      } else if (action === ACTION_TURN_RIGHT) {
        this.snakeDirection_ = 'r';
      }
    } else if (this.snakeDirection_ === 'r') {
      if (action === ACTION_TURN_LEFT) {
        this.snakeDirection_ = 'u';
      } else if (action === ACTION_TURN_RIGHT) {
        this.snakeDirection_ = 'd';
      }
    } else if (this.snakeDirection_ === 'd') {
      if (action === ACTION_TURN_LEFT) {
        this.snakeDirection_ = 'r';
      } else if (action === ACTION_TURN_RIGHT) {
        this.snakeDirection_ = 'l';
      }
    }
  }

  /**
   * Get the current direction of the snake.
   *
   * @returns {'l' | 'u' | 'r' | 'd'} Current direction of the snake.
   */
  get snakeDirection() {
    return this.snakeDirection_;
  }

  initializeSnake_() {
    /**
     * @private {Array<[number, number]>} Squares currently occupied by the
     * snake.
     *
     * Each element is a length-2 array representing the [y, x] coordinates of
     * the square. The array is ordered such that the first element is the
     * head of the snake and the last one is the tail.
     */
    this.snakeSquares_ = [];

    // Currently, the snake will start from a completely-straight and
    // horizontally-posed state.
    const y = getRandomInteger(0, this.height_);
    let x = getRandomInteger(this.initLen_ - 1, this.width_);
    for (let i = 0; i < this.initLen_; ++i) {
      this.snakeSquares_.push([y, x - i]);
    }

    /**
     * Current snake direction {'l' | 'u' | 'r' | 'd'}.
     *
     * Currently, the snake will start from a completely-straight and
     * horizontally-posed state. The initial direction is always right.
     */
    this.snakeDirection_ = 'r';
  }

  /**
   * Generate a number of new fruits at a random locations.
   *
   * The number of frtuis created is such that the total number of
   * fruits will be equal to the numFruits specified during the
   * construction of this object.
   *
   * The fruits will be created at unoccupied squares of the board.
   */
  makeFruits_() {
    if (this.fruitSquares_ == null) {
      this.fruitSquares_ = [];
    }
    const numFruits = this.numFruits_ - this.fruitSquares_.length;
    if (numFruits <= 0) {
      return;
    }

    const emptyIndices = [];
    for (let i = 0; i < this.height_; ++i) {
      for (let j = 0; j < this.width_; ++j) {
	      emptyIndices.push(i * this.width_ + j);
      }
    }

    // Remove the squares occupied by the snake from the empty indices.
    const occupiedIndices = [];
    this.snakeSquares_.forEach(yx => {
      occupiedIndices.push(yx[0] * this.width_ + yx[1]);
    });
    occupiedIndices.sort((a, b) => a - b);  // TODO(cais): Possible optimization?
    for (let i = occupiedIndices.length - 1; i >= 0; --i) {
      emptyIndices.splice(occupiedIndices[i], 1);
    }

    for (let i = 0; i < numFruits; ++i) {
      const fruitIndex = emptyIndices[getRandomInteger(0, emptyIndices.length)];
      const fruitY = Math.floor(fruitIndex / this.width_);
      const fruitX = fruitIndex % this.width_;
      this.fruitSquares_.push([fruitY, fruitX]);
      if (numFruits > 1) {
	      emptyIndices.splice(emptyIndices.indexOf(fruitIndex), 1);
      }
    }
  }

  get height() {
    return this.height_;
  }

  get width() {
    return this.width_;
  }

  /**
   * Get plain JavaScript representation of the game state.
   *
   * @return An object with two keys:
   *   - s: {Array<[number, number]>} representing the squares occupied by
   *        the snake. The array is ordered in such a way that the first
   *        element corresponds to the head of the snake and the last
   *        element corresponds to the tail.
   *   - f: {Array<[number, number]>} representing the squares occupied by
   *        the fruit(s).
   */
  getState() {
    return {
      "s": this.snakeSquares_.slice(),
      "f": this.fruitSquares_.slice()
    }
  }
}

/**
 * Get the current state of the game as an image tensor.
 *
 * @param {object | object[]} state The state object as returned by
 *   `SnakeGame.getState()`, consisting of two keys: `s` for the snake and
 *   `f` for the fruit(s). Can also be an array of such state objects.
 * @param {number} h Height.
 * @param {number} w With.
 * @return {tf.Tensor} A tensor of shape [numExamples, height, width, 2] and
 *   dtype 'float32'
 *   - The first channel uses 0-1-2 values to mark the snake.
 *     - 0 means an empty square.
 *     - 1 means the body of the snake.
 *     - 2 means the haed of the snake.
 *   - The second channel uses 0-1 values to mark the fruits.
 *   - `numExamples` is 1 if `state` argument is a single object or an
 *     array of a single object. Otherwise, it will be equal to the length
 *     of the state-object array.
 */

export function getStateTensor(state, h, w) {
  if (!Array.isArray(state)) {
    state = [state];
  }
  const numExamples = state.length;
  // TODO(cais): Maintain only a single buffer for efficiency.
  const buffer = tf.buffer([numExamples, h, w, 2]);

  for (let n = 0; n < numExamples; ++n) {
    if (state[n] == null) {
      continue;
    }
    // Mark the snake.
    state[n].s.forEach((yx, i) => {
      buffer.set(i === 0 ? 2 : 1, n, yx[0], yx[1], 0);
    });

    // Mark the fruit(s).
    state[n].f.forEach(yx => {
      buffer.set(1, n, yx[0], yx[1], 1);
    });
  }
  return buffer.toTensor();
}
