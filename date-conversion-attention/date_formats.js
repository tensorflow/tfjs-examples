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

const MONTH_NAMES_FULL = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'];
const MONTH_NAMES_3LETTER =
    MONTH_NAMES_FULL.map(name => name.slice(0, 3).toUpperCase());

const MIN_DATE = new Date('1950-01-01').getTime();
const MAX_DATE = new Date('2050-01-01').getTime();

// randomInt(min, max) {
//   return Math.floor(Math.random() * (max - min) + min);
// }

/**
 * Generate a random date.
 * 
 * @return {[number, number, number]} Year as an integer, month as an
 *   integer >= 1 and <= 12, day as an integer >= 1.
 */
function generateRandomDateTuple() {
  const date = new Date(Math.random() * (MAX_DATE - MIN_DATE) + MIN_DATE);
  return [date.getFullYear(), date.getMonth() + 1, date.getDate()];
}

function toTwoDigitString(num) {
  return num < 10 ? `0${num}` : `${num}`;
}

function dateTupleToDDMMMYYYY(dateTuple) {
    const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
    const dayStr = toTwoDigitString(dateTuple[2]);
    return `${dayStr}${monthStr}${dateTuple[0]}`;
  }
  

function dateTupleToMMSlashDDSlashYYYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${monthStr}/${dayStr}/${dateTuple[0]}`;
}

function dateTupleToMMSlashDDSlashYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr}/${dayStr}/${yearStr}`;
}

function dateTupleToMMDDYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr}${dayStr}${yearStr}`;
}

function dateTupleToYYYYDashMMDashDD(dateTuple) {
    const monthStr = toTwoDigitString(dateTuple[1]);
    const dayStr = toTwoDigitString(dateTuple[2]);
    return `${dateTuple[0]}-${monthStr}-${dayStr}`;
  }

module.exports = {
  dateTupleToDDMMMYYYY,
  dateTupleToMMSlashDDSlashYYYY,
  dateTupleToMMSlashDDSlashYY,
  dateTupleToMMDDYY,
  dateTupleToYYYYDashMMDashDD,
  generateRandomDateTuple
};
