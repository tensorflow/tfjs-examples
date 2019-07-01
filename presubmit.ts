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

import * as fs from 'fs';
import {join} from 'path';
import * as shell from 'shelljs';

// Exit if any commands error.
shell.set('-e');
process.on('unhandledRejection', e => {
  throw e;
});

const dir = '.';
const DEPLOYMENT_DIR = 'deployment';
// Testable dirs are all subdirectories of this except those which
// begin with '.', 'node_modules', or the deployment root dir.
const mainDirs = fs.readdirSync(dir)
                 .filter(f => fs.statSync(join(dir, f)).isDirectory())
                 .filter(f => !f.startsWith('.') && f !== 'node_modules' && f !== DEPLOYMENT_DIR);
const deploymentDirs = fs.readdirSync(join(dir, DEPLOYMENT_DIR))
                 .filter(f => fs.statSync(join(dir, f)).isDirectory())
                 .filter(f => !f.startsWith('.') && f !== 'node_modules');
const dirs = mainDirs.concat(deploymentDirs);

dirs.forEach(dir => {
  shell.cd(dir);
  const packageJSON: {} =
      JSON.parse(fs.readFileSync('./package.json', {encoding: 'utf-8'}));
  if (packageJSON['scripts']['test'] != null) {
    console.log(`~~~~~~~~~~~~ Testing ${dir} ~~~~~~~~~~~~`);
    shell.exec('yarn');
    shell.exec('yarn test');
    console.log('\n');
  }

  if (packageJSON['scripts']['lint'] != null) {
    console.log(`~~~~~~~~~~~~ Linting ${dir} ~~~~~~~~~~~~`);
    shell.exec('yarn');
    shell.exec('yarn lint');
    console.log('\n');
  }
  shell.cd('../');
});
