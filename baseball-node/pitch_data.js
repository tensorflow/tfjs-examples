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

const fs = require('fs');
const path = require('path');

/**
 * Loads a JSON training file and the content to a Pitch array.
 */
function loadPitchData(filename) {
  const pitches = [];
  const content =
      fs.readFileSync(path.join('data', filename), 'utf-8').split('\n');
  for (let i = 0; i < content.length; ++i) {
    if (content[i].length > 0) {
      pitches.push(JSON.parse(content[i]));
    }
  }
  return pitches;
}

//
// TODO(kreeger): Let's convert this data to CSV - need a test helper script for
// this. Remove normalized constants?
//

class PitchData {
  constructor(trainingDataFilename, testDataFilename) {
    this.trainingDataFilename = trainingDataFilename;
    this.testDataFilename = testDataFilename;
  }

  /**
   * Loads training and test data.
   */
  async loadData() {
    //
    // TODO(kreeger): Let's use the tf.data API!!
    //
  }
}

// TODO(kreeger): Fix data download location
module.exports = {loadPitchData};
