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
import * as game from './game';
import * as human_agent from './human_agent';

describe('HumanAgent', () => {
  it('can be created', () => {
    new human_agent.HumanAgent();
  });

  it('parseAnswers array format', () => {
    const board = new game.Board();
    const agent = new human_agent.HumanAgent();
    expect(agent._parseAnswer('[0, 0]', board)).toEqual(0);
    expect(agent._parseAnswer('[1, 0]', board)).toEqual(1);
    expect(agent._parseAnswer('[0, 1]', board)).toEqual(8);
    expect(agent._parseAnswer('[9, 9]', board))
        .toEqual(game.INVALID_BOARD_MOVE);
  });

  it('parseAnswers pair format', () => {
    const board = new game.Board();
    const agent = new human_agent.HumanAgent();
    expect(agent._parseAnswer('0, 0', board)).toEqual(0);
    expect(agent._parseAnswer('1, 0', board)).toEqual(1);
    expect(agent._parseAnswer('0, 1', board)).toEqual(8);
    expect(agent._parseAnswer('9, 9', board)).toEqual(game.INVALID_BOARD_MOVE);
  });

  it('parseAnswers no comma format', () => {
    const board = new game.Board();
    const agent = new human_agent.HumanAgent();
    expect(agent._parseAnswer('0 0', board)).toEqual(0);
    expect(agent._parseAnswer('1 0', board)).toEqual(1);
    expect(agent._parseAnswer('0 1', board)).toEqual(8);
    expect(agent._parseAnswer('9 9', board)).toEqual(game.INVALID_BOARD_MOVE);
  });

  it('parseAnswers fast format', () => {
    const board = new game.Board();
    const agent = new human_agent.HumanAgent();
    expect(agent._parseAnswer('00', board)).toEqual(0);
    expect(agent._parseAnswer('10', board)).toEqual(1);
    expect(agent._parseAnswer('01', board)).toEqual(8);
    expect(agent._parseAnswer('99', board)).toEqual(game.INVALID_BOARD_MOVE);
  });
});
