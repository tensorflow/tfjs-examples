/**
 * @fileoverview Description of this file.
 */

import * as readlineSync from 'readline-sync';
import * as game from './game';


/**
 * Agent-adapter to the gomoku game which is controlled by a human player via
 * command line input at the console.
 */
export class HumanAgent {
  constructor() {
    this.playerIndex = null;
  }

  setPlayerIndex(p) {
    this.playerIndex = p;
  }

  // Answer string should be a pair of numbers inside brackets like "[0, 3]".
  // Dropping the brackets is also acceptable, like "0, 3".
  // The answerString may also drop the comma, like "0 3".
  _parseAnswer(answerString, board) {
    // If answer is empty return invalid.
    if (!answerString) {
      return game.INVALID_BOARD_MOVE;
    }
    // If the answer does not include a comma, place one at the first space.
    if (answerString.indexOf(',') === -1) {
      answerString = answerString.replace(' ', ',');
    }

    // If the first character is not '[', assume it is num, num pair.
    let arr;
    if (answerString[0] !== '[') {
      answerString = '[' + answerString + ']';
    }
    // Assume the string can be parsed into an array.
    try {
      arr = JSON.parse(answerString);
    } catch (err) {
      console.log(err);
      return game.INVALID_BOARD_MOVE;
    }
    if (arr.length !== 2) {
      return game.INVALID_BOARD_MOVE;
    }
    return board.locationToMove({x: arr[0], y: arr[1]});
  }

  /**
   * Requests input from the user at the console, in [x, y] format.  If the
   * input is a valid move for the current board, then the move is returned.  If
   * the input is invalid, then ask again.
   *
   * Returns the selected move as an integer.
   *
   * @param {Board} board the current state of the board of play.
   * @returns {Number} move the position of the move.
   */
  getAction(board) {
    // TODO(bileschi): Make more general. I.e., accept input from a web
    // interface.
    const answer =
        readlineSync.question(`Your move player ${this.playerIndex}: `);
    console.log(`you answered ${answer}`);
    const move = this._parseAnswer(answer, board);
    console.log(`move is  ${move}`);
    if (move === game.INVALID_BOARD_LOCATION || !board.isAvailable(move)) {
      console.log(`invalid move ${move}`);
      console.log(move === game.INVALID_BOARD_LOCATION);
      console.log(board.isAvailable(move));
      move = this.getAction(board);
    };
    console.log('DONE WAITING FOR MOVE');
    return move;
  }

  toString() {
    return `Human ${this.playerIndex}`;
  }
};
