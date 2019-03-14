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

import {loadData} from "./data";

describe('loadData', () => {
  it('multihot = false', async () => {
    const numWords = 10;
    const len = 5;
    const multihot = false;
    const data = await loadData(numWords, len, multihot);

    expect(data.xTrain.shape.length).toEqual(2);
    expect(data.yTrain.shape.length).toEqual(2);
    expect(data.xTrain.shape[0]).toEqual(data.yTrain.shape[0]);
    expect(data.xTrain.shape[1]).toEqual(5);
    expect(data.yTrain.shape[1]).toEqual(1);
    expect(data.xTest.shape.length).toEqual(2);
    expect(data.yTest.shape.length).toEqual(2);
    expect(data.xTest.shape[0]).toEqual(data.yTest.shape[0]);
    expect(data.xTest.shape[1]).toEqual(5);
    expect(data.yTest.shape[1]).toEqual(1);
  });

  it('multihot = true', async () => {
    const numWords = 10;
    const len = 5;
    const multihot = true;
    const data = await loadData(numWords, len, multihot);

    expect(data.xTrain.shape.length).toEqual(2);
    expect(data.yTrain.shape.length).toEqual(2);
    expect(data.xTrain.shape[0]).toEqual(data.yTrain.shape[0]);
    expect(data.xTrain.shape[1]).toEqual(10);
    expect(data.yTrain.shape[1]).toEqual(1);
    expect(data.xTest.shape.length).toEqual(2);
    expect(data.yTest.shape.length).toEqual(2);
    expect(data.xTest.shape[0]).toEqual(data.yTest.shape[0]);
    expect(data.xTest.shape[1]).toEqual(10);
    expect(data.yTest.shape[1]).toEqual(1);
  });
});

