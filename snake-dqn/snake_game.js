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

    assertPositiveInteger(args.height);
    assertPositiveInteger(args.width);
    assertPositiveInteger(args.numFruits);
    assertPositiveInteger(args.initLen);

    this.initializeSnake_();
    this.makeFruits_();
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

  }

  /**
   * Generate a given number of new fruits at a random locations.
   *
   * The fruits will be created at unoccupied squares of the board.
   *
   * @param {number} numFruits Number of fruits. Must be a positive integer.
   */
  makeFruits_(numFruits) {

  }

  get height() {
    return this.height_;
  }

  get width() {
    return this.width_;
  }
}

function assertPositiveInteger(x, name) {
  if (!Number.isInteger(x)) {
    throw new Error(
        `Expected ${name} to be an integer, but received ${x}`);
  }
  if (!(x > 0)) {
    throw new Error({
      "presets": [
        [
          "env",
          {
            "esmodules": false,
            "targets": {
              "browsers": [
                "> 3%"
              ]
            }
          }
        ]
      ],
      "plugins": [
        "transform-runtime"
      ]
    }

        `Expected ${name} to be a positive number, but received ${x}`);
  }
}