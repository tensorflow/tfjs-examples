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

import * as tf from '@tensorflow/tfjs';
import * as dateFormat from './date_format';

describe('Date formats', () => {
  it('generateRandomDateTuple', () => {
    for (let i = 0; i < 100; ++i) {
      const [year, month, day] = dateFormat.generateRandomDateTuple();
      expect(Number.isInteger(year)).toEqual(true);
      expect(year).toBeGreaterThanOrEqual(1950);
      expect(year).toBeLessThan(2050);
      expect(Number.isInteger(month)).toEqual(true);
      expect(month).toBeGreaterThanOrEqual(1);
      expect(month).toBeLessThan(13);
      expect(Number.isInteger(day)).toEqual(true);
      expect(day).toBeGreaterThanOrEqual(1);
      expect(day).toBeLessThan(32);
    }
  });

  it('DDMMMYYYY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToDDMMMYYYY(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^[0-3]\d[A-Z][A-Z][A-Z][1-2]\d\d\d$/);
    }
  });

  it('MM/DD/YYYY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToMMSlashDDSlashYYYY(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^[0-1]\d\/[0-3]\d\/[1-2]\d\d\d$/);
    }
  });

  it('M/D/YYYY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToMSlashDSlashYYYY(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^\d{1,2}\/\d{1,2}\/[1-2]\d\d\d$/);
    }
  });

  it('MM/DD/YY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToMMSlashDDSlashYY(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^[0-1]\d\/[0-3]\d\/\d\d$/);
    }
  });

  it('M/D/YY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToMSlashDSlashYY(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^\d{1,2}\/\d{1,2}\/\d\d$/);
    }
  });

  it('MMDDYY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToMMDDYY(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^[0-1]\d[0-3]\d\d\d$/);
    }
  });

  it('MMMSpaceDDSpaceYY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToMMMSpaceDDSpaceYY(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(
          /^[A-Z][A-Z][A-Z] [0-3][0-9] [0-9][0-9]$/);
    }
  });

  it('MMMSpaceDDSpaceYYYY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToMMMSpaceDDSpaceYYYY(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(
          /^[A-Z][A-Z][A-Z] [0-3][0-9] [0-9][0-9][0-9][0-9]$/);
    }
  });

  it('MMMSpaceDDCommaSpaceYY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToMMMSpaceDDCommaSpaceYY(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(
          /^[A-Z][A-Z][A-Z] [0-3][0-9], [0-9][0-9]$/);
    }
  });

  it('MMMSpaceDDCommaSpaceYYYY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToMMMSpaceDDCommaSpaceYYYY(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(
          /^[A-Z][A-Z][A-Z] [0-3][0-9], [0-9][0-9][0-9][0-9]$/);
    }
  });

  it('MM-DD-YYYY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToDDDashMMDashYYYY(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^[0-3]\d-[0-1]\d-[1-2]\d\d\d$/);
    }
  });

  it('M-D-YYYY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToDDashMDashYYYY(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^\d{1,2}-\d{1,2}-[1-2]\d\d\d$/);
    }
  });

  it('YYYY.MM.DD', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToYYYYDotMMDotDD(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^[1-2]\d\d\d\.[0-1]\d\.[0-3]\d$/);
    }
  });

  it('YYYY.M.D', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToYYYYDotMMDotDD(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^[1-2]\d\d\d\.\d{1,2}\.\d{1,2}$/);
    }
  });

  it('DD.MM.YYYY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToDDDotMMDotYYYY(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^[0-3]\d\.[0-1]\d\.[1-2]\d\d\d$/);
    }
  });

  it('D.M.YYYY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToDDotMDotYYYY(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^\d{1,2}\.\d{1,2}\.[1-2]\d\d\d$/);
    }
  });

  it('YYYYMMDD', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToYYYYMMDD(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^[1-2]\d\d\d[0-1]\d[0-3]\d$/);
    }
  });

  it('YYYY-M-D', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToYYYYDashMDashD(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^[1-2]\d\d\d-\d{1,2}-\d{1,2}$/);
    }
  });

  it('YYYY-MM-DD', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormat.dateTupleToYYYYDashMMDashDD(
          dateFormat.generateRandomDateTuple());
      expect(str).toMatch(/^[1-2]\d\d\d-[0-1]\d-[0-3]\d$/);
    }
  });

  it('Encode input string', () => {
    const str1 = dateFormat.dateTupleToDDMMMYYYY(
        dateFormat.generateRandomDateTuple());
    const str2 = dateFormat.dateTupleToMMSlashDDSlashYYYY(
        dateFormat.generateRandomDateTuple());
    const str3 = dateFormat.dateTupleToMMSlashDDSlashYY(
        dateFormat.generateRandomDateTuple());
    const encoded = dateFormat.encodeInputDateStrings([str1, str2, str3]);
    expect(encoded.min().dataSync()[0]).toEqual(0);
    expect(encoded.max().dataSync()[0]).toBeLessThan(
        dateFormat.INPUT_VOCAB.length);

    const values = encoded.dataSync();
    let decodedStr = '';
    for (let i = 0; i < dateFormat.INPUT_LENGTH; ++i) {
      decodedStr += dateFormat.INPUT_VOCAB[values[i]];
    }
    expect(decodedStr.trim()).toEqual(str1);

    decodedStr = '';
    for (let i = 0; i < dateFormat.INPUT_LENGTH; ++i) {
      decodedStr += dateFormat.INPUT_VOCAB[values[i + dateFormat.INPUT_LENGTH]];
    }
    expect(decodedStr.trim()).toEqual(str2);

    decodedStr = '';
    for (let i = 0; i < dateFormat.INPUT_LENGTH; ++i) {
      decodedStr +=
          dateFormat.INPUT_VOCAB[values[i + dateFormat.INPUT_LENGTH * 2]];
    }
    expect(decodedStr.trim()).toEqual(str3);
  });

  it('Encode output string', () => {
    const str1 = '2000-01-02';
    const str2 = '1983-08-30';
    const encoded = dateFormat.encodeOutputDateStrings([str1, str2]);
    expect(encoded.shape).toEqual([2, dateFormat.OUTPUT_LENGTH]);

    const values = encoded.dataSync();
    let decodedStr = '';
    for (let i = 0; i < dateFormat.OUTPUT_LENGTH; ++i) {
      decodedStr += dateFormat.OUTPUT_VOCAB[values[i]];
    }
    expect(decodedStr.trim()).toEqual(str1);

    decodedStr = '';
    for (let i = 0; i < dateFormat.OUTPUT_LENGTH; ++i) {
      decodedStr +=
          dateFormat.OUTPUT_VOCAB[values[i + dateFormat.OUTPUT_LENGTH]];
    }
    expect(decodedStr.trim()).toEqual(str2);
  });

  it('Encode output string: oneHot', () => {
    const str1 = '2000-01-02';
    const str2 = '1983-08-30';
    const encoded = tf.oneHot(
        dateFormat.encodeOutputDateStrings([str1, str2]),
        dateFormat.OUTPUT_VOCAB.length);
    expect(encoded.shape).toEqual(
        [2, dateFormat.OUTPUT_LENGTH, dateFormat.OUTPUT_VOCAB.length]);
    expect(encoded.min().dataSync()[0]).toEqual(0);
    expect(encoded.max().dataSync()[0]).toEqual(1);

    const values = encoded.argMax(-1).dataSync();

    let decodedStr = '';
    for (let i = 0; i < dateFormat.OUTPUT_LENGTH; ++i) {
      decodedStr += dateFormat.OUTPUT_VOCAB[values[i]];
    }
    expect(decodedStr.trim()).toEqual(str1);

    decodedStr = '';
    for (let i = 0; i < dateFormat.OUTPUT_LENGTH; ++i) {
      decodedStr +=
          dateFormat.OUTPUT_VOCAB[values[i + dateFormat.OUTPUT_LENGTH]];
    }
    expect(decodedStr.trim()).toEqual(str2);
  });
});
