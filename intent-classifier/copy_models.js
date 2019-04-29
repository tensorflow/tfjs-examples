#!/usr/bin/env node
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
 * This script prepares the client for serving by copying trained models
 * in training/models into the dist folder.
 *
 * This will generally be run after training a model. This isn't necessarily
 * how you would serve a model in production, but is provided for convenience
 * in this example.
 */
const fse = require('fs-extra');

try {
  fse.ensureSymlinkSync('./training/models', './dist/models');
} catch (e) {
  console.log('Error linking models folder to dist', e);
  process.exit(1);
}
