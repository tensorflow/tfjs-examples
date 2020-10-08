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

import {Component, OnInit} from '@angular/core';
import * as tf from '@tensorflow/tfjs';

// Messages.
const DOWNLOAD_MESSAGE =
    'This interactive model visualizer will run in the browser: the required TensorFlow.js model files are currently being loaded. Thanks for your patience!';
const NO_MODEL_METADATA_ERROR_MESSAGE =
    'No model metadata URL provided. The visualizer could not start';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})

export class AppComponent implements OnInit {
  title = 'interactive-visualizers';

  // Model related variables.
  modelMetadataUrl: string|null = null;
  modelMetadata: any|null = null;
  modelType: string|null = null;
  model: tf.GraphModel|null = null;
  labelmap: string[] = [];
  defaultScoreThreshold = 0.0;

  // Test data related variables.
  testImagesIndexUrl: string|null = null;
  testImages: Array<{imageUrl: string, thumbnailUrl: string}> = [];
  uploadedImages: string[] = [];

  ngOnInit(): void {
    // Sanity checks on URL query parameters.
    const urlParams = new URLSearchParams(window.location.search);
    if (!urlParams.has('modelMetadataUrl')) {
      throw new Error(NO_MODEL_METADATA_ERROR_MESSAGE);
    }
    const modelMetadataUrl = urlParams.get('modelMetadataUrl') || '';

    this.initApp(modelMetadataUrl);
  }

  async fetchModel(modelUrl: string): Promise<tf.GraphModel> {
    const model = await tf.loadGraphModel(modelUrl);
    return model;
  }

  /**
   * Initializes the app for the provided model metadata URL.
   */
  async initApp(modelMetadataUrl: string): Promise<void> {
    // Load model & metadata.
    this.modelMetadataUrl = modelMetadataUrl;
    const metadataResponse = await fetch(this.modelMetadataUrl);
    this.modelMetadata = await metadataResponse.json();
    const modelUrl = this.getAssetsUrlPrefix() + 'model.json';
    this.model = await this.fetchModel(modelUrl);
    if (this.modelMetadata.tfjs_classifier_model_metadata) {
      this.modelType = 'classifier';
    }

    // Fetch test data if any.
    this.testImagesIndexUrl = this.modelMetadata.test_images_index_path;
    try {
      const testImagesResponse = await fetch(this.testImagesIndexUrl);
      const testImageNames = await testImagesResponse.json();
      this.testImages = [];
      for (const testImageName of testImageNames) {
        const testImageExt =
            testImageName.split('.')[testImageName.split('.').length - 1];
        const testImageNamePrefix = testImageName.split('.' + testImageExt)[0];
        this.testImages.push({
          imageUrl: this.getTestImagesUrlPrefix() + testImageName,
          thumbnailUrl: this.getTestImagesUrlPrefix() + testImageNamePrefix +
              '_thumb.' + testImageExt,
        });
      }

    } catch (error) {
      console.error(`Couldn't fetch test images: ${error}`);
    }
  }

  /**
   * Get the assets URL path, e.g. `http://assets-url-path/` if the metadata URL
   * is `http://assets-url-path/metadata.json`.
   */
  getAssetsUrlPrefix(): string {
    const metadataFileName = this.modelMetadataUrl.split('/').pop();
    return this.modelMetadataUrl.split(metadataFileName)[0];
  }

  /**
   * Get the test images URL path, e.g. `http://test-images-url-path/` if the
   * test images index is `http://test-images-url-path/index.json`.
   */
  getTestImagesUrlPrefix(): string {
    const imageIndexFileName = this.testImagesIndexUrl.split('/').pop();
    return this.testImagesIndexUrl.split(imageIndexFileName)[0];
  }
}
