
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

import * as game from './game';
import * as ui from './ui';

export function generateRows(numRows) {
  const rows = [];
  for (let i = 0; i < numRows; i++) {
    rows.push(game.generateOnePlay());
  }
  return rows;
}

/** Sets up handlers for the user affordences, including all buttons. */
document.addEventListener('DOMContentLoaded', async () => {
  console.log('content loaded');
  document.getElementById('generateData')
      .addEventListener('click', async () => {
        ui.updateSampleRowOutput(generateRows(ui.getBatchSize()));
      }, false);
});
