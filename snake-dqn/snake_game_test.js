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

import {expectArraysClose} from '../test_util';

import {ACTION_GO_STRAIGHT, ACTION_TURN_LEFT, ACTION_TURN_RIGHT, DEATH_REWARD, FRUIT_REWARD, getRandomAction, getStateTensor, NO_FRUIT_REWARD, SnakeGame} from './snake_game';

describe('getRandomAction', () => {
  it('getRandomAction', () => {
    for (let i = 0; i < 40; ++i) {
      const action = getRandomAction();
      expect([ACTION_GO_STRAIGHT, ACTION_TURN_LEFT, ACTION_TURN_RIGHT].indexOf(
                 action))
          .not.toEqual(-1);
    }
  });
});

function manhanttanDistance(xy1, xy2) {
  return Math.abs(xy1[0] - xy2[0]) + Math.abs(xy1[1] - xy2[1]);
}

describe('SnakeGame', () => {
  it('SnakeGame constructor: no-arg', () => {
    const game = new SnakeGame();
    expect(game.height).toEqual(16);
    expect(game.width).toEqual(16);
  });

  it('SnakeGame constructor: with args', () => {
    const game = new SnakeGame({height: 12, width: 14});
    expect(game.height).toEqual(12);
    expect(game.width).toEqual(14);
  });

  it('Invalid height and/or width leads to Error', () => {
    expect(() => new SnakeGame({height: -12}))
        .toThrowError(/Expected height to be a positive number/);
    expect(() => new SnakeGame({height: 1.2}))
        .toThrowError(/Expected height to be an integer/);
    expect(() => new SnakeGame({width: -12}))
        .toThrowError(/Expected width to be a positive number/);
    expect(() => new SnakeGame({width: 1.2}))
        .toThrowError(/Expected width to be an integer/);
  });

  it('getState: 1 fruit', async () => {
    // Since the board generation process is random, run the testing
    // for multiple iterations to increase the likelihood of finding
    // nondeterministic testing failures.
    for (let i = 0; i < 40; ++i) {
      const game = new SnakeGame({height: 4, width: 4, initLen: 2});
      const {s, f} = game.getState();
      expect(s).toEqual(game.snakeSquares_);
      expect(f).toEqual(game.fruitSquares_);
      expect(s.length).toBeGreaterThanOrEqual(2);
      for (let i = 0; i < s.length; ++i) {
        expect(s[i].length).toEqual(2);
        const [y, x] = s[i];
        expect(Number.isInteger(y));
        expect(y).toBeGreaterThanOrEqual(0);
        expect(y).toBeLessThan(4);
        expect(Number.isInteger(x));
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x).toBeLessThan(4);
        // Verify that the snake squares are consecutive (i.e.,
        // with a Manhattan distance of 1.
        if (i > 0) {
          expect(manhanttanDistance(s[i - 1], s[i])).toEqual(1);
        }
      }
      expect(f.length).toEqual(1);
      f.forEach(item => {
        expect(item.length).toEqual(2);
      });

      // Check that the snake and fruit are non-overlapping.
      const sIndices = s.map(yx => yx[0] * 4 + yx[1]);
      const fIndices = f.map(yx => yx[0] * 4 + yx[1]);
      expect(sIndices.indexOf(fIndices[0])).toEqual(-1);
    }
  });

  it('getState: 2 fruits', async () => {
    for (let i = 0; i < 40; ++i) {
      const game =
          new SnakeGame({height: 5, width: 5, initLen: 2, numFruits: 2});

      const {s, f} = game.getState();
      expect(s).toEqual(game.snakeSquares_);
      expect(f).toEqual(game.fruitSquares_);
      expect(s.length).toBeGreaterThanOrEqual(2);
      for (let i = 0; i < s.length; ++i) {
        expect(s[i].length).toEqual(2);
        const [y, x] = s[i];
        expect(Number.isInteger(y));
        expect(y).toBeGreaterThanOrEqual(0);
        expect(y).toBeLessThan(5);
        expect(Number.isInteger(x));
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x).toBeLessThan(5);
        // Verify that the snake squares are consecutive (i.e.,
        // with a Manhattan distance of 1.
        if (i > 0) {
          expect(manhanttanDistance(s[i - 1], s[i])).toEqual(1);
        }
      }
      expect(f.length).toEqual(2);
      f.forEach(item => expect(item.length).toEqual(2));

      // Check that the snake and fruit are non-overlapping.
      const sIndices = s.map(yx => yx[0] * 5 + yx[1]);
      const fIndices = f.map(yx => yx[0] * 5 + yx[1]);
      for (let i = 0; i < fIndices.length; ++i) {
        expect(sIndices.indexOf(fIndices[i])).toEqual(-1);
      }

      // Check that the two fruits are non-overlapping.
      expect(sIndices[0] !== sIndices[1]).toEqual(true);
    }
  });

  it('step: not done, no fruit eaten', () => {
    const game = new SnakeGame({height: 5, width: 5, initLen: 2});

    // Manually set the positions of the snake and the fruit for testing.
    game.snakeSquares_ = [[4, 1], [4, 0]];
    game.fruitSquares_ = [[0, 4]];

    let out = game.step(ACTION_GO_STRAIGHT);
    let reward = out.reward;
    let done = out.done;
    let fruitEaten = out.fruitEaten;
    expect(game.snakeDirection).toEqual('r');
    expect(reward).toEqual(NO_FRUIT_REWARD);
    expect(done).toEqual(false);
    expect(fruitEaten).toEqual(false);
    expect(game.snakeSquares_).toEqual([[4, 2], [4, 1]]);
    expect(game.fruitSquares_).toEqual([[0, 4]]);

    out = game.step(ACTION_TURN_LEFT);
    reward = out.reward;
    done = out.done;
    fruitEaten = out.fruitEaten;
    expect(game.snakeDirection).toEqual('u');
    expect(reward).toEqual(NO_FRUIT_REWARD);
    expect(done).toEqual(false);
    expect(fruitEaten).toEqual(false);
    expect(game.snakeSquares_).toEqual([[3, 2], [4, 2]]);
    expect(game.fruitSquares_).toEqual([[0, 4]]);

    out = game.step(ACTION_TURN_RIGHT);
    reward = out.reward;
    done = out.done;
    fruitEaten = out.fruitEaten;
    expect(reward).toEqual(NO_FRUIT_REWARD);
    expect(game.snakeDirection).toEqual('r');
    expect(done).toEqual(false);
    expect(fruitEaten).toEqual(false);
    expect(game.snakeSquares_).toEqual([[3, 3], [3, 2]]);
    expect(game.fruitSquares_).toEqual([[0, 4]]);

    out = game.step(ACTION_TURN_RIGHT);
    reward = out.reward;
    done = out.done;
    fruitEaten = out.fruitEaten;
    expect(game.snakeDirection).toEqual('d');
    expect(reward).toEqual(NO_FRUIT_REWARD);
    expect(done).toEqual(false);
    expect(fruitEaten).toEqual(false);
    expect(game.snakeSquares_).toEqual([[4, 3], [3, 3]]);
    expect(game.fruitSquares_).toEqual([[0, 4]]);
  });

  it('step: goes off left edge of board', () => {
    const game = new SnakeGame({height: 5, width: 5, initLen: 2});
    // Manually set the positions of the snake and the fruit for testing.
    game.snakeSquares_ = [[4, 0], [4, 1]];
    game.fruitSquares_ = [[0, 4]];

    const {reward, done} = game.step(ACTION_GO_STRAIGHT);
    expect(reward).toEqual(DEATH_REWARD);
    expect(done).toEqual(true);
  });

  it('step: goes off top edge of board', () => {
    const game = new SnakeGame({height: 5, width: 5, initLen: 2});
    game.snakeSquares_ = [[0, 0], [1, 0]];
    game.snakeDirection_ = 'u';
    game.fruitSquares_ = [[0, 4]];

    const {reward, done} = game.step(ACTION_GO_STRAIGHT);
    expect(reward).toEqual(DEATH_REWARD);
    expect(done).toEqual(true);
  });

  it('step: goes off right edge of board', () => {
    const game = new SnakeGame({height: 5, width: 5, initLen: 2});
    game.snakeSquares_ = [[3, 4], [3, 3]];
    game.fruitSquares_ = [[0, 4]];

    const {reward, done} = game.step(ACTION_GO_STRAIGHT);
    expect(reward).toEqual(DEATH_REWARD);
    expect(done).toEqual(true);
  });

  it('step: goes off bottom edge of board', () => {
    const game = new SnakeGame({height: 5, width: 5, initLen: 2});
    game.snakeSquares_ = [[4, 2], [4, 3]];
    game.snakeDirection_ = 'l';
    game.fruitSquares_ = [[0, 4]];

    const {reward, done} = game.step(ACTION_TURN_LEFT);
    expect(reward).toEqual(DEATH_REWARD);
    expect(done).toEqual(true);
  });

  it('step: bumps into own body 1', () => {
    const game = new SnakeGame({height: 5, width: 5, initLen: 4});
    game.snakeSquares_ = [[2, 3], [3, 3], [3, 4], [2, 4]];
    game.snakeDirection_ = 'u';
    game.fruitSquares_ = [[0, 4]];

    const {reward, done} = game.step(ACTION_TURN_RIGHT);
    expect(game.snakeDirection).toEqual('r');
    expect(reward).toEqual(DEATH_REWARD);
    expect(done).toEqual(true);
  });

  it('step: fruit eaten', () => {
    const game = new SnakeGame({height: 5, width: 5, initLen: 4});
    game.snakeSquares_ = [[2, 3], [3, 3], [3, 4], [2, 4]];
    game.snakeDirection_ = 'u';
    game.fruitSquares_ = [[1, 3]];

    const {reward, done, fruitEaten} = game.step(ACTION_GO_STRAIGHT);
    expect(game.snakeDirection).toEqual('u');
    expect(reward).toEqual(FRUIT_REWARD);
    expect(fruitEaten).toEqual(true);
    expect(done).toEqual(false);
    // The snake ate a fruit. Its length should grow from 4 to 5.
    const expectedSnakeSquares = [[1, 3], [2, 3], [3, 3], [3, 4], [2, 4]];
    expect(game.snakeSquares_).toEqual(expectedSnakeSquares);

    // Check that a new fruit has been generated.
    expect(game.fruitSquares_.length).toEqual(1);
    expect(game.fruitSquares_[0].length).toEqual(2);
    const [fruitY, fruitX] = game.fruitSquares_[0];
    expectedSnakeSquares.forEach(snakeYX => {
      expect(snakeYX[0] === fruitY && snakeYX[1] === fruitX).toEqual(false);
    });
  });

  it('reset after game over', () => {
    const game = new SnakeGame({height: 5, width: 5, initLen: 4});
    game.snakeSquares_ = [[2, 3], [3, 3], [3, 4], [2, 4]];
    game.snakeDirection_ = 'u';
    game.fruitSquares_ = [[0, 4]];

    const {reward, done} = game.step(ACTION_TURN_RIGHT);
    expect(game.snakeDirection).toEqual('r');
    expect(reward).toEqual(DEATH_REWARD);
    expect(done).toEqual(true);

    const {s, f} = game.reset();
    expect(s.length).toEqual(4);
    expect(f.length).toEqual(1);
    const [headY, headX] = s[0];
    // Find a valid direction to step in.
    // This works because the snake currently always starts from
    // a straight horizontal pose.
    const action = headY > 0 ? ACTION_TURN_LEFT : ACTION_TURN_RIGHT;
    const out = game.step(action);
    expect(out.done).toEqual(false);
  });

  it('reset after eating fruits and then dies', () => {
    const game = new SnakeGame({height: 5, width: 5, initLen: 3});
    game.snakeSquares_ = [[1, 3], [1, 2], [1, 1]];
    game.fruitSquares_ = [[1, 4]];

    let out = game.step(ACTION_GO_STRAIGHT);
    expect(game.snakeDirection).toEqual('r');
    expect(out.reward).toEqual(FRUIT_REWARD);
    expect(out.done).toEqual(false);
    expect(game.snakeSquares_).toEqual([[1, 4], [1, 3], [1, 2], [1, 1]]);
    expect(game.fruitSquares_.length).toEqual(1);

    game.fruitSquares_ = [[0, 4]];
    out = game.step(ACTION_TURN_LEFT);
    expect(game.snakeDirection).toEqual('u');
    expect(out.reward).toEqual(FRUIT_REWARD);
    expect(out.done).toEqual(false);
    expect(game.snakeSquares_).toEqual([
      [0, 4], [1, 4], [1, 3], [1, 2], [1, 1]
    ]);
    expect(game.fruitSquares_.length).toEqual(1);

    out = game.step(ACTION_GO_STRAIGHT);
    expect(game.snakeDirection).toEqual('u');
    expect(out.reward).toEqual(DEATH_REWARD);
    expect(out.done).toEqual(true);

    const {s, f} = game.reset();
    expect(s.length).toEqual(3);
    expect(f.length).toEqual(1);
  });
});

describe('getStateTensor', () => {
  it('1 frame, 1 fruit', () => {
    const h = 4;
    const w = 4;
    const state = {
      s: [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [2, 1]],
      f: [[2, 2]]
    };
    const tensor = getStateTensor(state, h, w);
    expect(tensor.shape).toEqual([1, 4, 4, 2]);
    expect(tensor.dtype).toEqual('float32');
    const [snakeTensor, fruitTensor] = tensor.squeeze(0).unstack(-1);

    expectArraysClose(
        snakeTensor,
        tf.tensor2d([[2, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]));
    expectArraysClose(
        fruitTensor,
        tf.tensor2d([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]));
  });

  it('2 fruits', () => {
    const h = 5;
    const w = 5;
    const state = {
      s: [[4, 2], [4, 1], [4, 0], [3, 0], [2, 0], [2, 1], [2, 2]],
      f: [[4, 4], [0, 0]]
    };
    const tensor = getStateTensor([state], h, w);
    expect(tensor.shape).toEqual([1, 5, 5, 2]);
    expect(tensor.dtype).toEqual('float32');
    const [snakeTensor, fruitTensor] = tensor.squeeze(0).unstack(-1);

    expectArraysClose(snakeTensor, tf.tensor2d([
      [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0],
      [1, 1, 2, 0, 0]
    ]));
    expectArraysClose(fruitTensor, tf.tensor2d([
      [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1]
    ]));
  });

  // TODO: fix the following two tests.
  xit('2 examples, both defined', () => {
    const h = 4;
    const w = 4;
    const state1 = {
      s: [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [2, 1]],
      f: [[2, 2]]
    };
    const state2 = {s: [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]], f: [[3, 3]]};
    const tensor = getStateTensor([state1, state2], h, w);
    expect(tensor.shape).toEqual([2, 4, 4, 2]);
    tensor.print();
    tensor.gather(0).gather(0, -1).print();
    expectArraysClose(
        tensor.gather(0).gather(0, -1),
        tf.tensor2d([[2, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]));
    expectArraysClose(
        tensor.gather(0).gather(1, -1),
        tf.tensor2d([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]));
    expectArraysClose(
        tensor.gather(1).gather(0, -1),
        tf.tensor2d([[2, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]]));
    expectArraysClose(
        tensor.gather(1).gather(1, -1),
        tf.tensor2d([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]));
  });

  xit('2 examples, one undefined', () => {
    const h = 4;
    const w = 4;
    const state1 = undefined;
    const state2 = {s: [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]], f: [[3, 3]]};
    const tensor = getStateTensor([state1, state2], h, w);
    expect(tensor.shape).toEqual([2, 4, 4, 2]);

    expectArraysClose(tensor.gather(0).gather(0, -1), tf.zeros([4, 4]));
    expectArraysClose(tensor.gather(0).gather(1, -1), tf.zeros([4, 4]));
    expectArraysClose(
        tensor.gather(1).gather(0, -1),
        tf.tensor2d([[2, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]]));
    expectArraysClose(
        tensor.gather(1).gather(1, -1),
        tf.tensor2d([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]));
  });
});
