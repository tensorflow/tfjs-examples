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

function flatMap(array, func) {
  const result = [];
  for (const el of array) {
    const mapped = func(el);
    for (const innerEl of mapped) {
      result.push(innerEl);
    }
  }
  return result;
}

function chunk(array, chunkSize) {
  const result = [];
  let batch = [];
  for (let index = 0; index < array.length; index++) {
    const element = array[index];
    batch.push(element);
    if (batch.length === chunkSize) {
      result.push(batch);
      batch = [];
    }
  }
  if (batch.length > 0) {
    result.push(batch);
  }
  return result;
}

function unique(array) {
  return Array.from(new Set(array));
}


const TAGS = ['TOK', 'LOC', '__PAD__'];
TAGS.PAD_IDX = 2;

module.exports = {
  tokenizeSentence,
  loadCSV,
  loadJSON,
  flatMap,
  unique,
  chunk,
  TAGS,
};
