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
 * Takes input JSON data files with queries and intents, tokenizes the queries
 * and tags each token. Then writes out the tagged tokens and intents. The
 * output of this script will be an NDJSON file where each line has the
 * following format:
 *  [ [{token: string, tag: string}, ...]  , intent: string ]
 *
 */

const fs = require('fs');
const path = require('path');
const ndjson = require('ndjson');
const argparse = require('argparse');
const {tokenizeSentence, loadJSON, TAGS} = require('./util');

/**
 * Writes an array to an NDJSON file.
 *
 * @param {string} path
 * @param {array} collection
 */
function writeNDJson(path, collection) {
  const serialize = ndjson.serialize();
  fd = fs.openSync(path, 'w');
  serialize.on('data', line => {
    fs.appendFileSync(fd, line, 'utf8');
  });

  for (const item of collection) {
    serialize.write(item);
  }
}

/**
 * These are the mappings of 'entities' in the source data to the tags
 * we want to use in our classifier.
 */
const entityToTag = {
  'geographic_poi': TAGS[1],
  'state': TAGS[1],
  'country': TAGS[1],
  'city': TAGS[1],
};

function parseRecords(data) {
  const result = [];

  const intentStr = Object.keys(data)[0];
  const entries = data[intentStr];
  entries.forEach(entry => {
    const clauses = entry.data;
    const tagArray = [];
    for (const clause of clauses) {
      // Each clause has multiple tokens. They will all share the same tag.
      const tokens = tokenizeSentence(clause.text);
      const entity = clause.entity;
      let tag;
      if (entity == null) {
        tag = TAGS[0];
      } else {
        tag = entityToTag[entity] ? entityToTag[entity] : TAGS[0];
      }

      for (const token of tokens) {
        tagArray.push({token, tag});
      }
    }
    result.push([tagArray, intentStr]);
  });

  return result;
}


function run(outPath) {
  const DATA_FILES = [
    'train_GetWeather_full.json',
  ];

  const INPUT_PATH = './data/raw/';
  DATA_FILES.forEach(fp => {
    const fullPath = path.resolve(__dirname, path.join(INPUT_PATH, fp));
    const loaded = loadJSON(fullPath);
    const parsed = parseRecords(loaded);
    writeNDJson(outPath, parsed);
    console.log('Done.');
  });
}


const parser = new argparse.ArgumentParser();
const defaultOutPath =
    path.resolve(__dirname, './data/intents_tagged_tokens.ndjson');

parser.addArgument('--outPath', {
  type: 'string',
  defaultValue: defaultOutPath,
});

const args = parser.parseArgs();
run(args.outPath);
