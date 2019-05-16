/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {JenaWeatherData} from "./data";

global.fetch = require('node-fetch');

describe('JenaWeatherData', () => {
  it('construct, load and basic public methods', async () => {
    const dataset = new JenaWeatherData();
    await dataset.load();

    expect(dataset.getDataColumnNames().length).toEqual(14);
    expect(dataset.getDataColumnNames()[0]).toEqual('p (mbar)');
    expect(dataset.getDataColumnNames()[1]).toEqual('T (degC)');

    expect(new Date(dataset.getTime(0)).getTime()).toBeGreaterThan(0);

    const columnData = dataset.getColumnData('T (degC)', false, true, 0, 30, 3);
    expect(columnData.length).toEqual(10);

    const func = dataset.getNextBatchFunction(
        true, 1000, 100, 32, 10, 0, 10000, true, false);
    for (let i = 0; i < 2; ++i) {
      const item = func.next();
      expect(item.done).toEqual(false);
      expect(item.value.xs.shape).toEqual([32, 100, 14]);
    }
  });
});
