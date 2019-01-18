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

const tf = require('@tensorflow/tfjs');

const MONTH_NAMES_FULL = [
  'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
  'September', 'October', 'November', 'December'
];
const MONTH_NAMES_3LETTER =
    MONTH_NAMES_FULL.map(name => name.slice(0, 3).toUpperCase());

const MIN_DATE = new Date('1950-01-01').getTime();
const MAX_DATE = new Date('2050-01-01').getTime();

export const INPUT_LENGTH = 10   // Maximum length of all input formats.
export const OUTPUT_LENGTH = 10  // Length of 'YYYY-MM-DD'.

// Use "\n" for padding for both input and output. It has to be at the
// beginning so that `mask_zero=True` can be used in the keras model.
export const INPUT_VOCAB = '\n0123456789/-' +
    MONTH_NAMES_3LETTER.join('')
        .split('')
        .filter(function(item, i, ar) {
          return ar.indexOf(item) === i;
        })
        .join('');

// OUTPUT_VOCAB includes an start-of-sequence (SOS) token, represented as
// '\t'.
export const OUTPUT_VOCAB = '\n\t0123456789-';

export const START_CODE = 1;

/**
 * Generate a random date.
 *
 * @return {[number, number, number]} Year as an integer, month as an
 *   integer >= 1 and <= 12, day as an integer >= 1.
 */
export function generateRandomDateTuple() {
  const date = new Date(Math.random() * (MAX_DATE - MIN_DATE) + MIN_DATE);
  return [date.getFullYear(), date.getMonth() + 1, date.getDate()];
}

function toTwoDigitString(num) {
  return num < 10 ? `0${num}` : `${num}`;
}

export function dateTupleToDDMMMYYYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dayStr}${monthStr}${dateTuple[0]}`;
}


export function dateTupleToMMSlashDDSlashYYYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${monthStr}/${dayStr}/${dateTuple[0]}`;
}

export function dateTupleToMMSlashDDSlashYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr}/${dayStr}/${yearStr}`;
}

export function dateTupleToMMDDYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr}${dayStr}${yearStr}`;
}

export function dateTupleToYYYYDashMMDashDD(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dateTuple[0]}-${monthStr}-${dayStr}`;
}

export function encodeInputDateStrings(dateStrings) {
  const n = dateStrings.length;
  const x = tf.buffer([n, INPUT_LENGTH], 'float32');
  for (let i = 0; i < n; ++i) {
    for (let j = 0; j < INPUT_LENGTH; ++j) {
      if (j < dateStrings[i].length) {
        const char = dateStrings[i][j];
        const index = INPUT_VOCAB.indexOf(char);
        if (index === -1) {
          throw new Error(`Unknown char: ${char}`);
        }
        x.set(index, i, j);
      }
    }
  }
  return x.toTensor();
}

export function encodeOutputDateStrings(dateStrings, oneHot = false) {
  const n = dateStrings.length;
  const x =
      oneHot ? tf.buffer([n, OUTPUT_LENGTH, OUTPUT_VOCAB.length], 'float32') :
      tf.buffer([n, OUTPUT_LENGTH], 'float32');
  for (let i = 0; i < n; ++i) {
    tf.util.assert(
        dateStrings[i].length === OUTPUT_LENGTH,
        `Date string is not in ISO format: "${dateStrings[i]}"`);
    for (let j = 0; j < OUTPUT_LENGTH; ++j) {
      const char = dateStrings[i][j];
      const index = OUTPUT_VOCAB.indexOf(char);
      if (index === -1) {
        throw new Error(`Unknown char: ${char}`);
      }
      if (oneHot) {
        x.set(1, i, j, index);
      } else {
        x.set(index, i, j);
      }
    }
  }
  return x.toTensor();
}
