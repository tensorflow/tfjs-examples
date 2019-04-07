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

const DEFAULT_HEIGHT = 16;
const DEFAULT_WIDTH = 16;
const DEFAULT_NUM_FRUITS = 1;
const DEFAULT_INIT_LEN = 4;

export class SnakeGame {
  /**
   * Constructor of SnakeGame.
   *
   * @param {object} args Configurations for the game. Fields include:
   *   - height {number} height of the board (positive integer)
   *   - width {number} width of the board (positive integer)
   *   - numFruits {number} number of fruits that present on the screen
   *     at any given time
   *   - initLen {number} initial length of the snake
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

    this.height_ = args.height;
    this.width_ = args.width;
    this.numFruits_ = args.numFruits;
    this.initLen_ = args.initLen;

    assertPositiveInteger(args.height, 'height');
    assertPositiveInteger(args.width, 'width');
    assertPositiveInteger(args.numFruits, 'numFruits');
    assertPositiveInteger(args.initLen, 'initLen');

    this.initializeSnake_();
    this.makeFruits_();
  }

  /**
   * Perform a step of the game.
   *
   * @param {0 | 1 | 2 | 3} action The action to take in the current step.
   *   The meaning of the possible values:
 rtui  *     0 - left
   *     1 - top
   *     2 - right
   *     3 - bottom
   */
  step(action) {

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

    // Currently, the snake will start from a completely-straight state.

    const y = getRandomInteger(0, this.height_);
    let x = getRandomInteger(this.initLen_ - 1, this.width_);
    this.snakeSquares_.push([y, x]);
    for (let i = 1; i < this.initLen_; ++i) {
      this.snakeSquares_.push([y, x - i]);
    }
  }

  /**
   * Generate a given number of new fruits at a random locations.
   *
   * The fruits will be created at unoccupied squares of the board.
   */
  makeFruits_() {
    if (this.fruitSquares_ == null) {
      this.fruitSquares_ = [];
    }
    const numFruits = this.numFruits_ - this.fruitSquares_.length;
    console.log(`numFruits = ${numFruits}`);

    const emptyIndices = [];
    for (let i = 0; i < this.height_; ++i) {
      for (let j = 0; j < this.width_; ++j) {
	emptyIndices.push(i * this.width_ + j);
      }
    }

    // Remove the squares occupied by the snake from the empty indices.
    this.snakeSquares_.forEach(yx => {
      const index = yx[0] * this.width_ + yx[1];
      // TODO(cais): Possible optimization?
      emptyIndices.splice(emptyIndices.indexOf(index), 1);
    });

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
   * Get the current state of the game as an image tensor.
   *
   * @return {tf.Tensor} A tensor of shape [height, width, 2] and dtype
   *   'float32'
   *   - The first channel uses 0-1-2 values to mark the snake.
   *     - 0 means an empty square.
   *     - 1 means the body of the snake.
   *     - 2 means the haed of the snake.
   *   - The second channel uses 0-1 values to mark the fruits.
   */
  getStateTensor() {
    // TODO(cais): Maintain only a single buffer for efficiency.
    const buffer = tf.buffer([this.height_, this.width_, 2]);

    // Mark the snake.
    this.snakeSquares_.forEach((yx, i) => {
      buffer.set(i === 0 ? 2 : 1, yx[0], yx[1], 0);
    });

    // Mark the fruit(s).
    this.fruitSquares_.forEach(yx => {
      buffer.set(1, yx[0], yx[1], 1);
    });

    return buffer.toTensor();
  }
}


export function getRandomInteger(min, max) {
  return Math.floor((max - min) * Math.random()) + min;
}

function assertPositiveInteger(x, name) {
  if (!Number.isInteger(x)) {
    throw new Error(
        `Expected ${name} to be an integer, but received ${x}`);
  }
  if (!(x > 0)) {
    throw new Error(
        `Expected ${name} to be a positive number, but received ${x}`);
  }
}
