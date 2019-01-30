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
 * of Poker. The cards range in value from 1 to 9, and each player receives
 * three cards, drawn uniformly from the integers in the range [1, 9].
 * One player wins by having a better hand.
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
export const MAX_CARD_VALUE = 9;
export const NUM_CARDS_PER_HAND = 3;

/**
 * Returns a random integer in the range [MIN_CARD_VALUE, MAX_CARD_VALUE]
 */
function getRandomDigit() {
  return Math.floor(MIN_CARD_VALUE + Math.random() * MAX_CARD_VALUE);
}

/**
 * Yields a randomly generated array of three elements.  Each element
 * is an integer randomly selected within the range [1, 9].
 */
export function randomHand() {
  return [getRandomDigit(), getRandomDigit(), getRandomDigit()];
}

/**
 * Returns the face value of any matching triple in the hand, or zero if no
 * triple exists.
 */
function tripleVal(hand) {
  if ((hand[0] === hand[1]) && (hand[1] === hand[2])) {
    return hand[0];
  }
  return 0;
}

/**
 * Returns the face value of any matching pair in the hand, or zero if no pair
 * exists.
 */
function doubleVal(hand) {
  if (hand[0] === hand[1]) {
    return hand[0];
  }
  if (hand[0] === hand[2]) {
    return (MAX_CARD_VALUE + 1) * hand[0];
  }
  if (hand[1] === hand[2]) {
    return (MAX_CARD_VALUE + 1) * hand[1];
  }
  return 0;
}

/**
 * Returns the highest number in the hand.
 */
function singleVal(hand) {
  return Math.max(...hand);
}

/**
 * The value of a hand is calculated as follows:
 *   100 times the tripled element, if there is one.
 *   + 10 times the paired element, if there is one.
 *   + the value of the largest element of the hand.
 *
 * This assures the rules are met, assuming the highest card value is
 *   less than 10, and the lowest is greater than 0.
 *
 * For instance:
 *   [1, 9, 1] -> 19
 *   [2, 8, 4] -> 8
 *   [4, 9, 1] -> 9
 *   [8, 8, 8] -> 888
 *
 * @param hand An array of three integers in the range [1, 9]
 */
export function handVal(hand) {
  return 100 * tripleVal(hand) + 10 * doubleVal(hand) + singleVal(hand);
}

/**
 * Returns 1 if hand1 beats hand2, in terms of hand value.
 * Returns 0 if hand1 is less than hand2.
 * In the event of a tie, return 0 or 1 randomly with even odds.
 */
export function compareHands(hand1, hand2) {
  const hv1 = handVal(hand1);
  const hv2 = handVal(hand2);
  if (hv1 > hv2) {
    return 1;
  }
  if (hv1 < hv2) {
    return 0;
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
  const result = compareHands(player1Hand, player2Hand);
  NUM_SIMULATIONS_SO_FAR++;
  return [player1Hand, player2Hand, result];
}
