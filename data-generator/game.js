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
 * This file implements a two player card game similar to a much simplified game
 * of Poker. The cards range in value from 1 to GAME_STATE.max_card_value, and
 * each player receives three cards, drawn uniformly from the integers in the
 * range [1, GAME_STATE.max_card_value]. One player wins by having a better
 * hand.
 *
 *    Any triple (three of the same card) beats any double.
 *    Any double (two of the same card) beats any single.
 *    If both players have a triple, the higher triple wins.
 *    If both players have a double, the higher double wins.
 *    If neither player has a double, the player with the highest
 *       individual number wins.
 *    Ties are settled randomly, 50/50.
 */

// Global exported state tracking the rules of the game and how many games have
// been played so far. Individual fields may be read / changed by external
// control.
export const GAME_STATE = {
  // The number of times the game has been played. Useful
  // for illustrating how many simulations it takes to train the model.
  num_simulations_so_far: 0,
  // Constants defining the range of card values.
  min_card_value: 1,
  max_card_value: 13,
  // Controls how many cards per hand.  Controlable from a UI element.
  num_cards_per_hand: 3
};

/**
 * Returns a random integer in the range [GAME_STATE.min_card_value,
 * GAME_STATE.max_card_value]
 */
export function getRandomDigit() {
  return Math.floor(
      GAME_STATE.min_card_value + Math.random() * GAME_STATE.max_card_value);
}

/**
 * Yields a randomly generated array of three elements.  Each element
 * is an integer randomly selected within the range [1, 9].  Hands are returned
 * in sorted order.
 */
export function randomHand() {
  const hand = [];
  for (let i = 0; i < GAME_STATE.num_cards_per_hand; i++) {
    hand.push(getRandomDigit());
  }
  return hand.sort((a, b) => a - b);
}

/**
 * This function produces an array indicating the largest face value for
 * every possible group size (1 card, 1 pair, 3-of-a-kind, 4-of-a-kind, etc.),
 * in the hand.  Zeros indicate there are no groups of that size.
 *
 * E.g., if this function returns [9, 3, 0, 0, 0, 0] it indicates that
 *   - The highest card is a 9
 *   - The highest double is of a 3
 *   - There are no triples etc.
 *
 * This could result if the hand were, e.g,  [1, 2, 2, 3, 3, 9]
 *
 * @param {number[]} hand A sorted integer array of length
 *     GAME_STATE.num_cards_per_hand
 * @returns {number[]} An array of the face value of the largest value for each
 *     group size.  Zero indicates there are no examples of that group size in
 *     the hand, if, for instance, there are no triples.
 */
export function handScoringHelper(hand) {
  // Create an array of all zeros of the appropriate size.
  const faceValOfEachGroup = [];
  for (let i = 0; i < GAME_STATE.num_cards_per_hand; i++) {
    faceValOfEachGroup.push(0);
  }
  let runLength = 0;
  let prevVal = 0;
  for (let i = 0; i < GAME_STATE.num_cards_per_hand; i++) {
    const card = hand[i];
    if (card == prevVal) {
      runLength += 1;
    } else {
      prevVal = card;
      runLength = 1;
    }
    faceValOfEachGroup[runLength - 1] = card;
  }
  return faceValOfEachGroup;
}

/**
 * Returns 1 if hand1 beats hand2, in terms of hand value.
 * Returns 0 if hand1 is less than hand2.
 * In the event of a tie, return 0 or 1 randomly with even odds.
 * @param {number[]} hand1 ordered list of numbers representing Player 1's hand.
 * @param {number[]} hand2 ordered list of numbers representing Player 2's hand.
 * @returns {number} 1 or 0 indicating Player 1's win or loss, respectively.
 */
export function compareHands(hand1, hand2) {
  const handScore1 = handScoringHelper(hand1);
  const handScore2 = handScoringHelper(hand2);
  // In descending order of group size, decide if one hand is better.
  for (let group = GAME_STATE.num_cards_per_hand - 1; group >= 0; group--) {
    if (handScore1[group] > handScore2[group]) {
      return 1;
    }
    if (handScore1[group] < handScore2[group]) {
      return 0;
    }
  }
  // Break a tie by flipping a fair coin.
  if (Math.random() > 0.5) {
    return 1;
  }
  return 0;
}

/**
 * Returns an object representing one complete play of the game.
 * Generates two random hands, and the value of the hand comparison.
 * Returns [hand1, hand2, whetherHand1Wins.
 */
export function generateOnePlay() {
  const player1Hand = randomHand();
  const player2Hand = randomHand();
  const player1Win = compareHands(player1Hand, player2Hand);
  GAME_STATE.num_simulations_so_far++;
  return {player1Hand, player2Hand, player1Win};
}
