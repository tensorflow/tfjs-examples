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

const fs = require('fs');
const dsv = require('d3-dsv');


function tokenizeSentence(input) {
  return input.split(/\b/).map(t => t.trim()).filter(t => t.length !== 0);
}

function loadCSV(path) {
  return dsv.csvParse(fs.readFileSync(path, {encoding: 'utf8'}));
}

function loadJSON(path) {
  return JSON.parse(fs.readFileSync(path, {encoding: 'utf8'}));
}

const TAGS = ['TOK', 'LOC', '__PAD__'];
TAGS.PAD_IDX = 2;

module.exports = {
  tokenizeSentence,
  loadCSV,
  loadJSON,
  TAGS,
};
