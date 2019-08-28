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

const tf = require('@tensorflow/tfjs-node');
const createDataset = require('./data');

describe('Dataset', () => {
  it('Created dataset and numOfColumns', async () => {
    const csvPath = 'file://./test_data.csv';
    const datasetObj = await createDataset(csvPath);
    const dataset = datasetObj.dataset;
    const row = await dataset.take(1).toArray();
    const numOfColumns = datasetObj.numOfColumns;
    expect(numOfColumns).toBe(8);
    const features = row[0].xs;
    const label = row[0].ys;
    expect(features.length).toBe(8);
    expect(features[0] === 0 || features[0] === 1 || features[0] === 2)
        .toBeTruthy();
    for (let i = 1; i < 8; i++) {
      expect(features[i]).toBeLessThan(1);
      expect(features[i]).toBeGreaterThan(0);
    }
    expect(Number.isInteger(label[0])).toBeTruthy();
    expect(label[0]).toBeGreaterThanOrEqual(2);
    expect(label[0]).toBeLessThanOrEqual(16);
  });
});
