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

/**
 * Translated from code written by Junxiao Song
 * https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/game.py
 * and published under the MIT License, copied here:
 *
 * MIT License
 *
 * Copyright (c) 2017 junxiaosong
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


// The value of the tf object will be set dynamically, depending on whether
// the CPU (tfjs-node) or GPU (tfjs-node-gpu) backend is used. This is why
// `let` is used in lieu of the more conventional `const` here.
const tf = require('@tensorflow/tfjs');

const INVALID_BOARD_MOVE = null;
const INVALID_BOARD_LOCATION = null;
const DEFAULT_BOARD_SIZE = 8;
const DEFAULT_N_IN_ROW = 5;
const LAST_MOVE_SENTINEL = -1;
const NO_WIN_SENTINEL = -1;

/**
 * Returns an object with keys for integer values up to but not including N.
 * Value for each key is null.
 */
function createAvailableKeys(N) {
  const myObj = {};
  for (let i = 0; i < N; i++) {
    myObj[i] = null;
  }
  return myObj;
}

/**
 * Encapusulates the concept of a game board and the current game state.
 * Includes pieces placed, which player's turn it is, and game parameters
 * like the size of the board.
 */
class Board {
  /**
   * boardConfig
   *  .width = the integer size of the board along the x dimension.
   *  .height = the integer size of the board along the y dimension.
   *  .nInRow = how many pieces are needed to be in a row to constitute a win.
   */
  constructor(boardConfig = {}) {
    this.width = boardConfig.width || DEFAULT_BOARD_SIZE;
    this.height = boardConfig.height || DEFAULT_BOARD_SIZE;
    // States is an object that contains the all the completed moves on the
    // board.  Keys are the move position.  The value at the position
    // corresponds to which player made the move.
    this.states = {};
    this.nInRow = boardConfig.nInRow || DEFAULT_N_IN_ROW;
    this.players = [0, 1];
    this.currentPlayer = null;
  }

  /**
   * Resets the board to empty of pieces.
   * Sets the next player to be the starting player.
   * 
   * @param startPlayer: Which player should go first, 0 or 1.
   */
  initBoard(startPlayer = 0) {
    if (this.width < this.nInRow) {
      throw new Error(`Width (${this.width}) can not be less than winning row size ${
          this.nInRow}.`);
    }
    if (this.height < this.nInRow) {
      throw new Error(`Height (${this.height}) can not be less than winning row size ${
          this.nInRow}.`);
    }
    this.currentPlayer = this.players[startPlayer];
    // Keep available moves as keys in an object.
    this.availables = createAvailableKeys(this.width * this.height);
    this.states = {};
    this.lastMove = LAST_MOVE_SENTINEL;
  }

  /** Returns true if the provided move is not already occupied. */
  isAvailable(move) {
    // Would be undefined if not available.
    this.availables[move] === null;
  }

  /**
   * Converts index of position to a board location.
   *
   * e.g. : given a board like:
   * 6 7 8
   * 3 4 5
   * 0 1 2
   *
   * The location of 5 is {y: 1, x: 2}
   *
   * Returns undefined on invalid location.
   * */
  moveToLocation(move) {
    if (move < 0 | move >= this.width * this.height | !Number.isInteger(move)) { 
      return INVALID_BOARD_MOVE;
    }
    return {y: Math.floor(move / this.width), x: move % this.width};
  }

  /**
   * Given a board location, return the position index.
   *
   * Returns INVALID_BOARD_LOCATION on invalid location.
   * @param {x: number, y:number} location
   */
  locationToMove(location) {
    if (location.x < 0 | location.x >= this.width | location.y < 0 |
        location.y >= this.height | !Number.isInteger(location.x) | 
        !Number.isInteger(location.y)) {
      return INVALID_BOARD_LOCATION;
    }
    return location.x + location.y * this.width;
  }

  /**
   * Registers the current player's move. Updates relevant book keeping.
   * Does not check whether the move is valid.  The user should check for
   * validity before calling this if there is a chance the move is invalid.
   *
   * @param {number} move position of current player's move, in single number
   *     format.
   */
  doMove(move) {
    this.states[move] = this.currentPlayer;
    delete this.availables[move];
    this.currentPlayer = (this.currentPlayer === this.players[0]) ?
        this.players[1] :
        this.players[0];
    this.lastMove = move;
  }

  // TODO(bileschi): Move this to be part of 'Game' not 'Board'
  /**
   * Indicates whether the board has a winner.
   *
   * @returns {(boolean, integer)} If the game has been won returns
   * {win: True, winner: $Player} indicating which player has won the game.  Otherwise
   * returns {win: False, winner: -1};
   */
  hasAWinner() {
    const moved = Object.keys(this.states);
    // Premature optimization? Uncomment below to see if it makes any speed
    // diff.
    /*
    if (moved.length < (this.nInRow * 2 - 1)) {
      return {win: false, winner: -1};
    } */
    moved.forEach(mString => {
      const m = Number.parseInt(mString);
      const loc = this.moveToLocation(m);
      const player = this.states[m];
      // Check if this is the leftmost piece in a horizontal win
      if (loc.x >= 0 && loc.x <= this.width - this.nInRow + 1) {
        let winner = player;
        for (let i = 1; i < this.nInRow; i++) {
          const offsetMove = m + i;
          if (this.states[offsetMove] != player) {
            winner = null;
            break;
          }
        }
        if (winner != null) {
          return {win: true, winner: winner};
        }
      }
      // Check if this is the bottom-most piece in a vertical win.
      if (loc.y >= 0 && loc.y <= this.height - this.nInRow + 1) {
        let winner = player;
        for (let i = 1; i < this.nInRow; i++) {
          const offsetMove = m + this.width * i;
          if (this.states[offsetMove] != player) {
            winner = null;
            break;
          }
        }
        if (winner != null) {
          return {win: true, winner};
        }
      }
      // Check if this is the bottom-left piece in a diagonal win like /.
      if (loc.y >= 0 && loc.y <= this.height - this.nInRow + 1 && loc.x >= 0 &&
          loc.x <= this.width - this.nInRow + 1) {
        let winner = player;
        for (let i = 1; i < this.nInRow; i++) {
          const offsetMove = m + (this.width + 1) * i;
          if (this.states[offsetMove] != player) {
            winner = null;
            break;
          }
        }
        if (winner != null) {
          return {win: true, winner};
        }
      }
      // Check if this is the top-left piece in a diagonal win like \.
      if (loc.y < this.height && loc.y >= this.nInRow - 1 && loc.x >= 0 &&
          loc.x <= this.width - this.nInRow + 1) {
        let winner = player;
        for (let i = 1; i < this.nInRow; i++) {
          const offsetMove = m + (1 - this.width) * i;
          if (this.states[offsetMove] != player) {
            winner = null;
            break;
          }
        }
        if (winner != null) {
          return {win: true, winner};
        }
      }
    });
    return {win: false, winner: NO_WIN_SENTINEL};
  }

  // TODO(bileschi): Move this to be part of 'Game' not 'Board'
  /**
   * The game is over if either there is a winner, or there are no moves left
   * to make.
   * @returns {{win: boolean, winner: integer}} If the game has been won returns
   *   {win: True, winner: $Player} indicating which player has won the game.
   *   If the game is a tie, returns {win: True, winner: -1}
   *   Otherwise returns {win: False, winner: -1};
   */
  gameEnd() {
    const [win, winner] = this.hasAWinner();
    if (win) {
      return {win: true, winner: winner};
    } else if (Object.keys(this.availables).length == 0) {
      return {win: true, winner: NO_WIN_SENTINEL};
    } else {
      return {win: false, winner: NO_WIN_SENTINEL};
    }
  }

  /**
   * Returns the board state from the perspective of the current player.
   *
   * Channel 0 is your opponent's pieces.
   * Channel 1 is your pieces.
   * Channel 2 is the position of the previous move.
   * Channel 3 alternates ones and zeros, depending on which player is next.
   * @returns tf.Tensor with shape [3, width, height] 
   */
  currentStateTensor() {
    const boardBuffer = tf.buffer([3, this.width, this.height], 'float32');
    // Set channel 0 and channel 1.
    for (const [move, movePlayer] of Object.entries(this.states)) {
      const playerIndex = movePlayer === this.currentPlayer;
      const loc = this.moveToLocation(move);
      boardBuffer.set(1.0, playerIndex, loc.x, loc.y);
    }
    if (this.lastMove !== LAST_MOVE_SENTINEL) {
      // Set last move channel.
      const lastLoc = this.moveToLocation(this.lastMove);
      boardBuffer.set(1.0, 2, lastLoc.x, lastLoc.y);
    }
    // Set channel 3 to which player's turn it is, assuming alternate
    // players.
    const lastChannelFillVal =
        (this.currentPlayer === this.players[1]) ? 1.0 : 0.0;
    const playerTensor =
        tf.fill([1, this.width, this.height], lastChannelFillVal);
    return tf.concat([boardBuffer.toTensor(), playerTensor], 0);
  }
}

class Game {
  constructor(board, gameConfig = {}) {
    this.board = board;
  }

  _rowAsAsciiArt(iRow) {
    let rowText = iRow + ' ';
    for (let iCol = 0; iCol < this.board.width; iCol++) {
      const move = this.board.locationToMove({x: iCol, y: iRow});
      const playerMoveVal = this.board.states[move];
      switch (playerMoveVal) {
        case this.board.players[0]:
          rowText += 'X';
          break;
        case this.board.players[1]:
          rowText += 'O';
          break;
        default:
          rowText += '-';
      }
    }
    return rowText;
  }

  _colIndexRow(width) {
    let rowText = '  ';
    for (let i = 0; i < width; i++) {
      rowText += i;
    }
    return rowText;
  }

  /**
   * Returns a string representation of the Game.
   */
  asAsciiArt() {
    const textRows = [];
    textRows.push(`player ${this.board.players[0]} with X`);
    textRows.push(`player ${this.board.players[1]} with O`);

    textRows.push(this._colIndexRow(this.board.width));
    for (let iRow = 0; iRow < this.board.height; iRow++) {
      textRows.push(this._rowAsAsciiArt(iRow));
    }
    return textRows.join('\n');
  }

  /**
   * Begins the game, with agent1 and agent2 representing players.
   * @param agent1: agent acting as player 1
   * @param agent2: agent acting as player 2
   * @param startPlayer: 0 if player 1 goes first, 1 otherwise.
   * @param isShown: true to print the game state out to the console.
   * @returns {{win: boolean, winner: integer}} If the game has been won returns
   *   {win: True, winner: $Player} indicating which player has won the game.
   *   If the game is a tie, returns {win: True, winner: -1}
   *   Otherwise returns {win: False, winner: -1};
 
   */
  startPlay(agent1, agent2, startPlayer = 0, isShown = true) {
    if (startPlayer !== 0 && startPlayer !== 1) {
      throw new Error(
          'start_player should be either 0 (agent1 first) ' +
          'or 1 (agent2 first)');
    }
    self.board.initBoard(startPlayer);
    const [player1Index, player2Index] = board.players;
    agent1.setPlayerIndex(player1Index);
    agent2.setPlayerIndex(player2Index);
    const agents = { p1: agent1, p2: agent2 };
    if (isShown) {
      console.log(this.asAsciiArt());
    }
    while (true) {
      const currentPlayer = this.board.currentPlayer;
      const currentAgent = agents[currentPlayer];
      const move = currentAgent.getAction(this.board);
      this.board.doMove(move);
      if (isShown) {
        console.log(this.asAsciiArt());
      }
      const [end, winner] = this.board.gameEnd();
      if (end) {
        if (isShown) {
          if (winner !== NO_WIN_SENTINEL) {
            console.log('Game end.  Winner is ', agents[winner]);
          } else {
            console.log('Game end.  Tie.');
          }
        }
        return winner;
      }
    }
  }
}

module.exports = {
  Board,
  Game,
  DEFAULT_BOARD_SIZE,
  DEFAULT_N_IN_ROW,
  LAST_MOVE_SENTINEL,
  INVALID_BOARD_MOVE,
  INVALID_BOARD_LOCATION,
  NO_WIN_SENTINEL
};
