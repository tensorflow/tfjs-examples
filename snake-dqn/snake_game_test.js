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

import {getRandomInteger, SnakeGame} from "./snake_game";
import {expectArraysClose} from "@tensorflow/tfjs-core/dist/test_util";

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
    expect(() => new SnakeGame({height: -12})).toThrowError(
	/Expected height to be a positive number/);
    expect(() => new SnakeGame({height: 1.2})).toThrowError(
	/Expected height to be an integer/);
    expect(() => new SnakeGame({width: -12})).toThrowError(
	/Expected width to be a positive number/);
    expect(() => new SnakeGame({width: 1.2})).toThrowError(
	/Expected width to be an integer/);
  });

  it('getStateTensor: 1 fruit', async () => {
    const game = new SnakeGame({height: 5, width: 5, initLen: 2});
    const x = game.getStateTensor();
    expect(x.dtype).toEqual('float32');
    expect(x.shape).toEqual([5, 5, 2]);
    const [snake, fruits] = x.unstack(-1);
    expectArraysClose(snake.sum(), 3);
    expectArraysClose(snake.max(), 2);
    expectArraysClose(snake.min(), 0);
    expectArraysClose(fruits.sum(), 1);
    expectArraysClose(fruits.max(), 1);
    expectArraysClose(fruits.min(), 0);

    // Check that the snake and fruit are non-overlapping.
    const fruitIndex = await fruits.flatten().argMax().array();
    expect((await snake.flatten().data())[fruitIndex]).toEqual(0);
  });

  it('getStateTensor: 2 fruits', async () => {
    const game = new SnakeGame({
      height: 5,
      width: 5,
      initLen: 2,
      numFruits: 2
    });
    const x = game.getStateTensor();
    expect(x.dtype).toEqual('float32');
    expect(x.shape).toEqual([5, 5, 2]);
    const [snake, fruits] = x.unstack(-1);
    expectArraysClose(snake.sum(), 3);
    expectArraysClose(snake.max(), 2);
    expectArraysClose(snake.min(), 0);
    expectArraysClose(fruits.sum(), 2);  // Two fruits.
    expectArraysClose(fruits.max(), 1);
    expectArraysClose(fruits.min(), 0);

    // Check that the snake and fruit are non-overlapping.
    const isFruit = await fruits.flatten().equal(1).array();
    for (let i = 0; i < isFruit.length; ++i) {
      if (isFruit[i] === 1) {
        expect((await snake.flatten().data())[i]).toEqual(0);
      }
    }
  });
});

describe('getRandomInteger()', () => {
  it('max > min', () => {
    const values = [];
    for (let i = 0; i < 10; ++i) {
      const v = getRandomInteger(3, 6);
      expect(v).toBeGreaterThanOrEqual(3);
      expect(v).toBeLessThan(6);
      if (values.indexOf(v) === -1) {
	values.push(v);
      }
    }
    expect(values.length).toBeGreaterThan(1);
  });
});
