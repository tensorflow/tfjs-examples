/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

// This file can be replaced during build by using the `fileReplacements` array.
// `ng build --prod` replaces `environment.ts` with `environment.prod.ts`.
// The list of file replacements can be found in `angular.json`.

import { version } from '../../../../package.json';

export const environment = {
  production: false,
  interactiveVisualizerUrl: `https://storage.googleapis.com/interactive_visualizer/${version}/index.html`,
  models: [
    {
      displayName: 'Birds V1',
      description: 'AIY natural world insects classification model',
      type: 'image classification',
      metadataUrl: 'https://storage.googleapis.com/tfhub-visualizers/google/aiy/vision/classifier/birds_V1/1/metadata.json',
    },
    {
      displayName: 'Insects V1',
      description: 'AIY natural world birds quantized classification model',
      type: 'image classification',
      metadataUrl: 'https://storage.googleapis.com/tfhub-visualizers/google/aiy/vision/classifier/insects_V1/1/metadata.json',
    },
    {
      displayName: 'Mobile Object Localizer V1',
      description: 'Mobile model to localize objects in an image',
      type: 'object detection',
      metadataUrl: 'https://storage.googleapis.com/tfhub-visualizers/google/object_detection/mobile_object_localizer_v1/1/metadata.json',
    },
  ],
};

/*
 * For easier debugging in development mode, you can import the following file
 * to ignore zone related error stack frames such as `zone.run`, `zoneDelegate.invokeTask`.
 *
 * This import should be commented out in production mode because it will have a negative impact
 * on performance if an error is thrown.
 */
// import 'zone.js/dist/zone-error';  // Included with Angular CLI.
