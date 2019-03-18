/**
 * @fileoverview Description of this file.
 */

const readline = require('readline');
const game = require('./game');

/**
 * TODO(bileschi): describe this class.
 */
class HumanAgent {
  constructor() {
    this.playerIndex = null;
  }

  setPlayerIndex(p) {
    this.playerIndex = p
  }

  _parseAnswer(answerString, board) {
    // If answer is empty return invalid.
    if (!answerString) {
      return game.INVALID_BOARD_MOVE;
    }
    // If the first character is not '[', assume it is a num, num pair.
    let arr;
    if (answerString[0] !== '[') {
      answerString = '[' + answerString + ']';
    }
    // Assume the string can be parsed into an array.
    try {
      arr = JSON.parse(answerString);
    } catch {
      return game.INVALID_BOARD_MOVE;
    }
    if (arr.length !== 2) {
      return game.INVALID_BOARD_MOVE;
    }
    return board.locationToMove({ x: arr[0], y: arr[1] });
  }

  getAction(board) {
    const rl = readline.createInterface(
      { input: process.stdin, output: process.stdout });
    let move = game.INVALID_BOARD_MOVE;
    rl.question(`Your move player ${this.playerIndex}: `, (answer) => {
      console.log(`you answered ${answer}`);
      rl.close();
    });
    move = _parseAnswer(answer);
    if (move === game.INVALID_BOARD_LOCATION || !board.isAvailable(move)) {
      console.log(`invalid move ${move}`);
      move = this.getAction(board);
    }
    return move;
  }

  toString() {
    return `Human ${this.playerIndex}`;
  }
}

module.exports = {
  HumanAgent,
};

