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

const dateFormats = require('./date_formats');

describe('Date formats', () => {
  it('generateRandomDateTuple', () => {
    for (let i = 0; i < 100; ++i) {
      const [year, month, day] = dateFormats.generateRandomDateTuple();
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
      const str = dateFormats.dateTupleToDDMMMYYYY(
          dateFormats.generateRandomDateTuple());
      expect(str).toMatch(/[0-3]\d[A-Z][A-Z][A-Z][1-2]\d\d\d/);
    }
  });

  it('MM/DD/YYYY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormats.dateTupleToMMSlashDDSlashYYYY(
          dateFormats.generateRandomDateTuple());
      expect(str).toMatch(/[0-1]\d\/[0-3]\d\/[1-2]\d\d\d/);
    }
  });

  it('MM/DD/YY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormats.dateTupleToMMSlashDDSlashYY(
          dateFormats.generateRandomDateTuple());
      expect(str).toMatch(/[0-1]\d\/[0-3]\d\/\d\d/);
    }
  });

  it('MMDDYY', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormats.dateTupleToMMDDYY(
          dateFormats.generateRandomDateTuple());
      expect(str).toMatch(/[0-1]\d[0-3]\d\d\d/);
    }
  });

  it('YYYY-MM-DD', () => {
    for (let i = 0; i < 10; ++i) {
      const str = dateFormats.dateTupleToYYYYDashMMDashDD(
          dateFormats.generateRandomDateTuple());
      expect(str).toMatch(/[1-2]\d\d\d-[0-1]\d-[0-3]\d/);
    }
  });
});
