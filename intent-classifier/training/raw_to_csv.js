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
 * Reads the original data files from Snips/NLU and converts them to a simpler
 * CSV format.
 */

const fs = require('fs');
const path = require('path');

const dsv = require('d3-dsv');
const argparse = require('argparse');
const {loadJSON} = require('./util');

function parseRecords(data) {
  const result = [];
  const intentStr = Object.keys(data)[0];
  const entries = data[intentStr];
  entries.forEach(entry => {
    const data = entry.data;
    const queryText = data.reduce((memo, curr) => {
      return memo.concat(curr.text);
    }, '');

    result.push({
      query: queryText,
      intent: intentStr,
    });
  });

  return result;
}

function writeCSV(data, path) {
  const csvStr = dsv.csvFormat(data, ['query', 'intent']);
  fs.writeFileSync(path, csvStr, {encoding: 'utf8'});
}

function run(outPath) {
  const DATA_FILES = [
    'train_AddToPlaylist_full.json',
    'train_GetWeather_full.json',
    'train_PlayMusic_full.json',
  ];

  const DATA_PATH = './data/raw/';

  let allRecords = [];
  DATA_FILES.forEach(fp => {
    const fullPath = path.resolve(__dirname, path.join(DATA_PATH, fp));
    const loaded = loadJSON(fullPath);
    const parsed = parseRecords(loaded);
    allRecords = allRecords.concat(parsed);
  });

  console.log(`Parsed ${allRecords.length} records`);
  writeCSV(allRecords, outPath);
}


const parser = new argparse.ArgumentParser();
const defaultOutPath = path.resolve(__dirname, './data/intents.csv');

parser.addArgument('--outPath', {
  type: 'string',
  defaultValue: defaultOutPath,
});

const args = parser.parseArgs();
run(args.outPath);
