// TODO(bileschi): DO NOT SUBMIT
// TODO(bileschi): manage LICENSING
// https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/game.py


// The value of the tf object will be set dynamically, depending on whether
// the CPU (tfjs-node) or GPU (tfjs-node-gpu) backend is used. This is why
// `let` is used in lieu of the more conventional `const` here.
let tf = require('@tensorflow/tfjs');

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

// TODO(bileschi): Add docstring.
class Board {
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

  // TODO(bileschi): Add docstring.
  initBoard(startPlayer = 0) {
    if (this.width < this.nInRow) {
      throw `Width (${this.width}) can not be less than winning row size ${
          this.nInRow}.`
    }
    if (this.height < this.nInRow) {
      throw `Height (${this.height}) can not be less than winning row size ${
          this.nInRow}.`
    }
    this.currentPlayer = this.players[startPlayer];
    // Keep available moves as keys in an object.
    this.availables = createAvailableKeys(this.width * this.height);
    this.states = {};
    this.lastMove = LAST_MOVE_SENTINEL;
  }

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
    if (move < 0 | move >= this.width * this.height) {
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
        location.y >= this.height) {
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
   *
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
   * [True, $Player] indicating which player has won the game.  Otherwise
   * returns [False, -1];
   */
  hasAWinner() {
    const moved = Object.keys(this.states);
    // Premature optimization? Uncomment below to see if it makes any speed
    // diff.
    /*
    if (moved.length < (this.nInRow * 2 - 1)) {
      return [false, -1];
    } */
    for (let mString of moved) {
      const m = parseInt(mString);
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
          return [true, winner];
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
          return [true, winner];
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
          return [true, winner];
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
          return [true, winner];
        }
      }
    }
    return [false, NO_WIN_SENTINEL];
  }

  // TODO(bileschi): Move this to be part of 'Game' not 'Board'
  /**
   * The game is over if either there is a winner, or there are no moves left
   * to make.
   * @returns {(boolean, integer)} If the game has been won returns
   *   [True, $Player] indicating which player has won the game.
   *   If the game is a tie, returns [True, -1]
   *   Otherwise returns [False, -1];
   */
  gameEnd() {
    const [win, winner] = this.hasAWinner();
    if (win) {
      return [true, winner];
    } else if (Object.keys(this.availables).length == 0) {
      return [true, NO_WIN_SENTINEL];
    } else {
      return [false, NO_WIN_SENTINEL];
    }
  }

  /**
   * Returns the board state from the perspective of the current player.
   *
   * Channel 0 is your opponent's pieces.
   * Channel 1 is your pieces.
   * Channel 2 is the position of the previous move.
   * Channel 3 alternates ones and zeros, depending on which player is next.
   */
  currentState() {
    const boardBuffer = tf.buffer([3, this.width, this.height], 'float32');
    // Set channel 0 and channel 1.
    for (let [move, movePlayer] of Object.entries(this.states)) {
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
