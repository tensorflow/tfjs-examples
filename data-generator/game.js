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

import {resolveTripleslashReference} from 'typescript';

/**
 * This file implements a two player card game similar to a much simplified game
 * of Poker. The cards range in value from 1 to MAX_CARD_VALUE, and each player
 * receives three cards, drawn uniformly from the integers in the range [1,
 * MAX_CARD_VALUE]. One player wins by having a better hand.
 *
 *    Any triple (three of the same card) beats any double.
 *    Any double (two of the same card) beats any single.
 *    If both players have a triple, the higher triple wins.
 *    If both players have a double, the higher double wins.
 *    If neither player has a double, the player with the highest
 *       individual number wins.
 *    Ties are settled randomly, 50/50.
 */


// Global exported count of the number of times the game has been played. Useful
// for illustrating how many simulations it takes to train the model.
export let NUM_SIMULATIONS_SO_FAR = 0;

// Constants defining the range of card values.
export const MIN_CARD_VALUE = 1;
export const MAX_CARD_VALUE = 13;
export const NUM_CARDS_PER_HAND = 3;

/**
 * Returns a random integer in the range [MIN_CARD_VALUE, MAX_CARD_VALUE]
 */
function getRandomDigit() {
  return Math.floor(MIN_CARD_VALUE + Math.random() * MAX_CARD_VALUE);
}

/**
 * Yields a randomly generated array of three elements.  Each element
 * is an integer randomly selected within the range [1, 9].  Hands are returned
 * in sorted order.
 */
export function randomHand() {
  const hand = [];
  for (let i = 0; i < NUM_CARDS_PER_HAND; i++) {
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
 * @param {number[]} hand A sorted integer array of length NUM_CARDS_PER_HAND
 * @returns {number[]} An array of the face value of the largest value for each
 *     group size.  Zero indicates there are no examples of that group size in
 *     the hand, if, for instance, there are no triples.
 */
export function handScoringHelper(hand) {
  const faceValOfEachGroup = new Array(NUM_CARDS_PER_HAND).fill(0);
  let runLength = 0;
  let prevVal = 0;
  for (let i = 0; i < NUM_CARDS_PER_HAND; i++) {
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
 */
export function compareHands(hand1, hand2) {
  const handScore1 = handScoringHelper(hand1);
  const handScore2 = handScoringHelper(hand2);
  // In descending order of group size, decide if one hand is better.
  for (let group = NUM_CARDS_PER_HAND - 1; group >= 0; group--) {
    if (handScore1[group] > handScore2[group]) {
      return 1;
    }
    if (handScore1[group] < handScore2[group]) {
      return 0;
    }
  }
  // Tie.
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
  NUM_SIMULATIONS_SO_FAR++;
  return {player1Hand, player2Hand, player1Win};
}
