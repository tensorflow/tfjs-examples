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

import {padSequences} from "./sequence_utils";

describe('padSequences', () => {
  it('post pad, post truncate', () => {
    const sequences = [[10, 20, 30], [5, 15], [], [1, 2, 3, 4, 5, 6]];
    const output = padSequences(sequences, 4, 'post', 'post');
    expect(output).toEqual(
        [[10, 20, 30, 0], [5, 15, 0, 0], [0, 0, 0, 0], [1, 2, 3, 4]]);
  });

  it('post pad, pre trucnate', () => {
    const sequences = [[10, 20, 30], [5, 15], [], [1, 2, 3, 4, 5, 6]];
    const output = padSequences(sequences, 4, 'post', 'pre');
    expect(output).toEqual(
        [[10, 20, 30, 0], [5, 15, 0, 0], [0, 0, 0, 0], [3, 4, 5, 6]]);
  });

  it('pre pad, post trucnate', () => {
    const sequences = [[10, 20, 30], [5, 15], [], [1, 2, 3, 4, 5, 6]];
    const output = padSequences(sequences, 4, 'pre', 'post');
    expect(output).toEqual(
        [[0, 10, 20, 30], [0, 0, 5, 15], [0, 0, 0, 0], [1, 2, 3, 4]]);
  });

  it('pre pad, pre trucnate', () => {
    const sequences = [[10, 20, 30], [5, 15], [], [1, 2, 3, 4, 5, 6]];
    const output = padSequences(sequences, 4, 'pre', 'pre');
    expect(output).toEqual(
        [[0, 10, 20, 30], [0, 0, 5, 15], [0, 0, 0, 0], [3, 4, 5, 6]]);
  });

  it('custom padding character', () => {
    const sequences = [[10, 20, 30], [5, 15], [], [1, 2, 3, 4, 5, 6]];
    const output = padSequences(sequences, 4, 'pre', 'pre', 42);
    expect(output).toEqual(
        [[42, 10, 20, 30], [42, 42, 5, 15], [42, 42, 42, 42], [3, 4, 5, 6]]);
  });
});
