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
 * Execute all unit tests in the current directory.
 */
function runTests(specFiles) {
  // tslint:disable-next-line:no-require-imports
  const jasmineConstructor = require('jasmine');

  Error.stackTraceLimit = Infinity;

  process.on('unhandledRejection', e => {
    throw e;
  });

  const runner = new jasmineConstructor();
  runner.loadConfig({spec_files: specFiles, random: false});
  runner.execute();
}

function expectArraysClose(actual, expected) {
  const actualValues = actual instanceof Array ? actual : actual.dataSync();
  const expectedValues = expected instanceof Array ? expected : expected.dataSync();

  expect(actualValues.length).toBe(expectedValues.length);
  const PRECISION = 3; // corresponds to 1e-3.
  for (let i = 0; i < actualValues.length; i++) {
    expect(actualValues[i]).toBeCloseTo(expectedValues[i], PRECISION);
  }
}

module.exports = {runTests, expectArraysClose};
