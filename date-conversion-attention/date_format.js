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
 * Date formats and conversion utility functions.
 *
 * This file is used for the training of the date-conversion model and
 * date conversions based on the trained model.
 *
 * It contains functions that generate random dates and represent them in
 * several different formats such as (2019-01-20 and 20JAN19).
 * It also contains functions that convert the text representation of
 * the dates into one-hot `tf.Tensor` representations.
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

export const INPUT_LENGTH = 12   // Maximum length of all input formats.
export const OUTPUT_LENGTH = 10  // Length of 'YYYY-MM-DD'.

// Use "\n" for padding for both input and output. It has to be at the
// beginning so that `mask_zero=True` can be used in the keras model.
export const INPUT_VOCAB = '\n0123456789/-., ' +
    MONTH_NAMES_3LETTER.join('')
        .split('')
        .filter(function(item, i, ar) {
          return ar.indexOf(item) === i;
        })
        .join('');

// OUTPUT_VOCAB includes an start-of-sequence (SOS) token, represented as
// '\t'. Note that the date strings are represented in terms of their
// constituent characters, not words or anything else.
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

/** Date format such as 01202019. */
export function dateTupleToDDMMMYYYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dayStr}${monthStr}${dateTuple[0]}`;
}

/** Date format such as 01/20/2019. */
export function dateTupleToMMSlashDDSlashYYYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${monthStr}/${dayStr}/${dateTuple[0]}`;
}

/** Date format such as 1/20/2019. */
export function dateTupleToMSlashDSlashYYYY(dateTuple) {
  return `${dateTuple[1]}/${dateTuple[2]}/${dateTuple[0]}`;
}

/** Date format such as 01/20/19. */
export function dateTupleToMMSlashDDSlashYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr}/${dayStr}/${yearStr}`;
}

/** Date format such as 1/20/19. */
export function dateTupleToMSlashDSlashYY(dateTuple) {
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${dateTuple[1]}/${dateTuple[2]}/${yearStr}`;
}

/** Date format such as 012019. */
export function dateTupleToMMDDYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr}${dayStr}${yearStr}`;
}

/** Date format such as JAN 20 19. */
export function dateTupleToMMMSpaceDDSpaceYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr} ${dayStr} ${yearStr}`;
}

/** Date format such as JAN 20 2019. */
export function dateTupleToMMMSpaceDDSpaceYYYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${monthStr} ${dayStr} ${dateTuple[0]}`;
}

/** Date format such as JAN 20, 19. */
export function dateTupleToMMMSpaceDDCommaSpaceYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr} ${dayStr}, ${yearStr}`;
}

/** Date format such as JAN 20, 2019. */
export function dateTupleToMMMSpaceDDCommaSpaceYYYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${monthStr} ${dayStr}, ${dateTuple[0]}`;
}

/** Date format such as 20-01-2019. */
export function dateTupleToDDDashMMDashYYYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dayStr}-${monthStr}-${dateTuple[0]}`;
}

/** Date format such as 20-1-2019. */
export function dateTupleToDDashMDashYYYY(dateTuple) {
  return `${dateTuple[2]}-${dateTuple[1]}-${dateTuple[0]}`;
}

/** Date format such as 20.01.2019. */
export function dateTupleToDDDotMMDotYYYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dayStr}.${monthStr}.${dateTuple[0]}`;
}

/** Date format such as 20.1.2019. */
export function dateTupleToDDotMDotYYYY(dateTuple) {
  return `${dateTuple[2]}.${dateTuple[1]}.${dateTuple[0]}`;
}

/** Date format such as 2019.01.20. */
export function dateTupleToYYYYDotMMDotDD(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dateTuple[0]}.${monthStr}.${dayStr}`;
}

/** Date format such as 2019.1.20. */
export function dateTupleToYYYYDotMDotD(dateTuple) {
  return `${dateTuple[0]}.${dateTuple[1]}.${dateTuple[2]}`;
}

/** Date format such as 20190120. */
export function dateTupleToYYYYMMDD(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dateTuple[0]}${monthStr}${dayStr}`;
}

/** Date format such as 2019-1-20. */
export function dateTupleToYYYYDashMDashD(dateTuple) {
  return `${dateTuple[0]}-${dateTuple[1]}-${dateTuple[2]}`;
}

/**
 * Date format such as 2019-01-20
 * (i.e.,  the ISO format and the conversion target).
 * */
export function dateTupleToYYYYDashMMDashDD(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dateTuple[0]}-${monthStr}-${dayStr}`;
}

export const INPUT_FNS = [
  dateTupleToDDMMMYYYY,
  dateTupleToMMDDYY,
  dateTupleToMMSlashDDSlashYY,
  dateTupleToMMSlashDDSlashYYYY,
  dateTupleToMSlashDSlashYYYY,
  dateTupleToDDDashMMDashYYYY,
  dateTupleToDDashMDashYYYY,
  dateTupleToMMMSpaceDDSpaceYY,
  dateTupleToMSlashDSlashYY,
  dateTupleToMMMSpaceDDSpaceYYYY,
  dateTupleToMMMSpaceDDCommaSpaceYY,
  dateTupleToMMMSpaceDDCommaSpaceYYYY,
  dateTupleToDDDotMMDotYYYY,
  dateTupleToDDotMDotYYYY,
  dateTupleToYYYYDotMMDotDD,
  dateTupleToYYYYDotMDotD,
  dateTupleToYYYYMMDD,
  dateTupleToYYYYDashMDashD,
  dateTupleToYYYYDashMMDashDD
];  // TODO(cais): Add more formats if necessary.

/**
 * Encode a number of input date strings as a `tf.Tensor`.
 *
 * The encoding is a sequence of one-hot vectors. The sequence is
 * padded at the end to the maximum possible length of any valid
 * input date strings. The padding value is zero.
 *
 * @param {string[]} dateStrings Input date strings. Each element of the array
 *   must be one of the formats listed above. It is okay to mix multiple formats
 *   in the array.
 * @returns {tf.Tensor} One-hot encoded characters as a `tf.Tensor`, of dtype
 *   `float32` and shape `[numExamples, maxInputLength]`, where `maxInputLength`
 *   is the maximum possible input length of all valid input date-string formats.
 */
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

/**
 * Encode a number of output date strings as a `tf.Tensor`.
 *
 * The encoding is a sequence of integer indices.
 *
 * @param {string[]} dateStrings An array of output date strings, must be in the
 *   ISO date format (YYYY-MM-DD).
 * @returns {tf.Tensor} Integer indices of the characters as a `tf.Tensor`, of
 *   dtype `int32` and shape `[numExamples, outputLength]`, where `outputLength`
 *   is the length of the standard output format (i.e., `10`).
 */
export function encodeOutputDateStrings(dateStrings, oneHot = false) {
  const n = dateStrings.length;
  const x = tf.buffer([n, OUTPUT_LENGTH], 'int32');
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
      x.set(index, i, j);
    }
  }
  return x.toTensor();
}
