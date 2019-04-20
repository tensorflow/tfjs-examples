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

import {createDeepQNetwork} from "./dqn";

describe('createDeepQNetwork', () => {
  it('createDeepQNetwork', () => {
    const h = 9;
    const w = 9;
    const numActions = 4;
    const model = createDeepQNetwork(h, w, numActions);

    expect(model.inputs.length).toEqual(1);
    expect(model.inputs[0].shape).toEqual([null, h, w, 2]);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, numActions]);
  });

  it('Invalid h and/or w leads to Error', () => {
    expect(() => createDeepQNetwork(0, 10, 4)).toThrowError(/height/);
    expect(() => createDeepQNetwork('10', 10, 4)).toThrowError(/height/);
    expect(() => createDeepQNetwork(null, 10, 4)).toThrowError(/height/);
    expect(() => createDeepQNetwork(undefined, 10, 4)).toThrowError(/height/);
    expect(() => createDeepQNetwork(10.8, 10, 4)).toThrowError(/height/);
    expect(() => createDeepQNetwork(10, 0, 4)).toThrowError(/width/);
    expect(() => createDeepQNetwork(10, '10', 4)).toThrowError(/width/);
    expect(() => createDeepQNetwork(10, null, 4)).toThrowError(/width/);
    expect(() => createDeepQNetwork(10, undefined, 4)).toThrowError(/width/);
    expect(() => createDeepQNetwork(10, 10.8, 4)).toThrowError(/width/);
  });

  it('Invali numActions leads to Error', () => {
    expect(() => createDeepQNetwork(10, 10, 0)).toThrowError(/numActions/);
    expect(() => createDeepQNetwork(10, 10, 1)).toThrowError(/numActions/);
    expect(() => createDeepQNetwork(10, 10, '4')).toThrowError(/numActions/);
    expect(() => createDeepQNetwork(10, 10, null)).toThrowError(/numActions/);
    expect(() => createDeepQNetwork(10, 10, undefined)).toThrowError(/numActions/);
  });
});
