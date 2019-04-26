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

const boardConfig66 = {
  width: 6,
  height: 6,
  nInRow: 4
};
const boardConfig33 = {
  width: 3,
  height: 3,
  nInRow: 3
};
const boardConfig88 = {
  width: 8,
  height: 8,
  nInRow: 5
};
const boardConfig93 = {
  width: 9,
  height: 3,
  nInRow: 3
};

describe('Board', () => {
  const boardConfigs =
      [boardConfig66, boardConfig88, boardConfig33, boardConfig93];

  it('is constructable with config', () => {
    for (let boardConfig of boardConfigs) {
      const board = new game.Board(boardConfig);
      expect(board.width).toEqual(boardConfig.width);
      expect(board.height).toEqual(boardConfig.height);
      expect(board.nInRow).toEqual(boardConfig.nInRow);
    }
  });

  it('is constructable with default config', () => {
    const board = new game.Board();
    expect(board.width).toEqual(game.DEFAULT_BOARD_SIZE);
    expect(board.height).toEqual(game.DEFAULT_BOARD_SIZE);
    expect(board.nInRow).toEqual(game.DEFAULT_N_IN_ROW);
  });

  it('will init', () => {
    for (let boardConfig of boardConfigs) {
      const board = new game.Board(boardConfig);
      board.initBoard();
      expect(Object.keys(board.availables).length)
          .toEqual(board.width * board.height);
      expect(board.lastMove).toEqual(game.LAST_MOVE_SENTINEL);
      expect(board.currentPlayerIndex).toEqual(0);
      expect(Object.keys(board.states).length).toEqual(0);
    }
  });

  it('can init with player 2 first', () => {
    const board = new game.Board();
    board.initBoard(1);
    expect(board.currentPlayerIndex).toEqual(1);
  });

  it('convert moveToLocation', () => {
    const board = new game.Board(boardConfig66);
    const loc0 = board.moveToLocation(0);
    const loc1 = board.moveToLocation(1);
    const loc6 = board.moveToLocation(6);
    const loc34 = board.moveToLocation(34);
    const loc36 = board.moveToLocation(36);
    expect(loc0).toEqual({x: 0, y: 0});
    expect(loc1).toEqual({x: 1, y: 0});
    expect(loc6).toEqual({x: 0, y: 1});
    expect(loc34).toEqual({x: 4, y: 5});
    expect(loc36).toEqual(game.INVALID_BOARD_LOCATION);
  });

  it('convert locationToMove', () => {
    const board = new game.Board(boardConfig88);
    const loc0_0 = {x: 0, y: 0};
    const loc1_0 = {x: 1, y: 0};
    const loc0_1 = {x: 0, y: 1};
    const loc4_5 = {x: 4, y: 5};
    const locInvalid = {x: -1, y: 0};
    expect(board.locationToMove(loc0_0)).toEqual(0);
    expect(board.locationToMove(loc1_0)).toEqual(1);
    expect(board.locationToMove(loc0_1)).toEqual(8);
    expect(board.locationToMove(loc4_5)).toEqual(44);
    expect(board.locationToMove(locInvalid)).toEqual(game.INVALID_BOARD_MOVE);
  });

  it('doMove changes currentPlayer', () => {
    const board = new game.Board(boardConfig66);
    board.initBoard();
    expect(board.currentPlayerIndex === 0);
    board.doMove(0);
    expect(board.currentPlayerIndex === 1);
    board.doMove(1);
    expect(board.currentPlayerIndex === 0);
    board.doMove(2);
    expect(board.currentPlayerIndex === 1);
    // Re-init:  second player goes first.
    board.initBoard(1);
    expect(board.currentPlayerIndex === 1);
    board.doMove(0);
    expect(board.currentPlayerIndex === 1);
  });

  it('doMove adds moves', () => {
    const board = new game.Board(boardConfig66);
    board.initBoard();
    board.doMove(0);
    board.doMove(1);
    board.doMove(2);
    expect(Object.keys(board.states).length).toEqual(3);
    expect(board.states[0]).toEqual(0);
    expect(board.states[1]).toEqual(1);
    expect(board.states[2]).toEqual(0);
  });

  it('hasAWinner horizontal', () => {
    const board = new game.Board(boardConfig66);
    board.initBoard();
    for (let i = 0; i < board.nInRow; i++) {
      board.states[board.locationToMove({x: i, y: 0})] = 0;
    }
    expect(board.hasAWinner()).toEqual({win: true, winner: 0});
  });

  it('hasAWinner vertical', () => {
    const board = new game.Board(boardConfig66);
    board.initBoard();
    for (let i = 0; i < board.nInRow; i++) {
      board.states[board.locationToMove({x: 0, y: i})] = 0;
    }
    expect(board.hasAWinner()).toEqual({win: true, winner: 0});
  });

  it('hasAWinner diagonal \\', () => {
    const board = new game.Board(boardConfig66);
    board.initBoard();
    for (let i = 0; i < board.nInRow; i++) {
      board.states[board.locationToMove({x: 5 - i, y: i})] = 0;
    }
    expect(board.hasAWinner()).toEqual({win: true, winner: 0});
  });


  it('hasAWinner diagonal //', () => {
    const board = new game.Board(boardConfig66);
    board.initBoard();
    for (let i = 0; i < board.nInRow; i++) {
      board.states[board.locationToMove({x: i, y: i})] = 0;
    }
    expect(board.hasAWinner()).toEqual({win: true, winner: 0});
  });

  // Other hasAWinner tests hack the board state.  Let's test one using the
  // doMove api directly.
  it('hasAWinner horizontal using doMove', () => {
    const board = new game.Board(boardConfig66);
    board.initBoard();
    board.doMove(0);
    board.doMove(9);
    board.doMove(1);
    board.doMove(10);
    board.doMove(2);
    board.doMove(11);
    board.doMove(3);
    board.doMove(12);
    board.doMove(4);
    expect(board.hasAWinner()).toEqual({win: true, winner: 0});
  });

  it('gameEnd not finished', () => {
    const board = new game.Board(boardConfig66);
    board.initBoard();
    expect(board.gameEnd()).toEqual({win: false, winner: game.NO_WIN_SENTINEL});
  });

  it('gameEnd winner', () => {
    const board = new game.Board(boardConfig66);
    board.initBoard();
    for (let i = 0; i < board.nInRow; i++) {
      board.states[board.locationToMove({x: i, y: i})] = 0;
    }
    expect(board.gameEnd()).toEqual({win: true, winner: 0});
  });

  it('gameEnd tie', () => {
    const board = new game.Board(boardConfig66);
    board.initBoard();
    board.availables = {};
    expect(board.gameEnd()).toEqual({win: true, winner: game.NO_WIN_SENTINEL});
  });

  it('currentStateTensor is zeros for empty board', () => {
    const board = new game.Board(boardConfig66);
    board.initBoard();
    const state = board.currentStateTensor();
    tf.test_util.expectArraysClose(state, tf.zeros([4, 6, 6]));
  });

  it('currentStateTensor is as expected for board with some moves', () => {
    const board = new game.Board(boardConfig66);
    board.initBoard();
    board.doMove(board.locationToMove({x: 0, y: 0}));
    board.doMove(board.locationToMove({x: 1, y: 1}));
    board.doMove(board.locationToMove({x: 2, y: 2}));
    const state = board.currentStateTensor();
    const expectedBuffer = tf.buffer([3, 6, 6]);
    // player 1's moves.
    expectedBuffer.set(1.0, 0, 0, 0);
    expectedBuffer.set(1.0, 0, 2, 2);
    // player 2's moves.
    expectedBuffer.set(1.0, 1, 1, 1);
    // last move.
    expectedBuffer.set(1.0, 2, 2, 2);
    // next player field.
    const expectedState =
        tf.concat([expectedBuffer.toTensor(), tf.ones([1, 6, 6])]);
    tf.test_util.expectArraysClose(state, expectedState);
  });

  it('currentStateTensor does not leak', () => {
    const board = new game.Board(boardConfig66);
    board.initBoard();
    board.doMove(board.locationToMove({x: 0, y: 0}));
    const tensorsBefore = tf.memory().numTensors;
    const state = board.currentStateTensor();
    const tensorsAfter = tf.memory().numTensors;
    expect(state).not.toBeNull();
    expect(tensorsAfter).toEqual(tensorsBefore + 1);
  });
});



describe('GameObject', () => {
  it('is creatable', () => {
    new game.Game(new game.Board());
  });

  it('asAsciiArt empty game', () => {
    const myGame = new game.Game(new game.Board());
    const myArt = myGame.asAsciiArt();
    const expectedArt = `
  01234567
0 --------
1 --------
2 --------
3 --------
4 --------
5 --------
6 --------
7 --------`;
    expect(myArt).toContain(expectedArt);
  });

  it('asAsciiArt after some moves', () => {
    const board = new game.Board(boardConfig66);
    board.initBoard();
    const myGame = new game.Game(board);
    let expectedArt = `
  012345
0 ------
1 ------
2 ------
3 ------
4 ------
5 ------`;
    expect(myGame.asAsciiArt()).toContain(expectedArt);
    board.doMove(board.locationToMove({x: 1, y: 2}));
    board.doMove(board.locationToMove({x: 3, y: 4}));
    expectedArt = `
  012345
0 ------
1 ------
2 -X----
3 ------
4 ---O--
5 ------`;
    expect(myGame.asAsciiArt()).toContain(expectedArt);
  });
});
