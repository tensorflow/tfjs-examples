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

import { Component } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'playground';
  interactiveVisualizerUrl = 'https://storage.googleapis.com/interactive_visualizer/0.0.1/index.html';
  models = [
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
  ];
  embedUrl: SafeResourceUrl|null = null;

  constructor(private sanitizer: DomSanitizer) {}

  selectModel(index: number): void {
    this.embedUrl = this.sanitizer.bypassSecurityTrustResourceUrl(this.interactiveVisualizerUrl +
                        '?modelMetadataUrl=' + this.models[index].metadataUrl);
  }
}
